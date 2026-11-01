"""意图级联接线验收（任务 2.2）：BERT → 大模型 → 规则 → unknown。

盯的是**谁给了结论**（`state.intent.method`），不是"谁更准"：

- BERT 置信 → 采用它的标签，这次大模型调用**少问一个字段**；
- BERT 弃权（`None`）→ 大模型接手；
- 大模型也没拿到（开关关掉 / 词表外 / JSON 坏）→ 规则分类器；
- 三段都弃权 → `unknown` → `generic_clarify`（§7.7 硬规则 2）。

两条最容易写错的接线，各有一条用例守着：

1. **BERT 必须在大模型之前跑，且不依赖大模型可用**——没有 key 的部署恰恰最需要本地分类器，
   所以它不能落在 `self.llm is None` 的早退之后；
2. **先到者的结论优先**——BERT 定过档时，大模型同一次调用里带回的意图不许覆盖它
   （`payload` 里故意给一个不同的标签，断言最终标签仍是 BERT 的）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from base.config import Settings
from base.contracts import EvidenceDecision, FAQHit, IntentResult, SlotResult
from orchestration.chat import ChatPipeline
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult
from understanding.profiles import BASELINE, profile_for

EVIDENCE_TEXT = "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k。"
_UNDERSTAND_MARK = "问题理解"
ANSWER = "顶点式可以通过配方法得到[1]。"
NORMALIZED = "二次函数的顶点式怎么求?"
QUERY = "二次函数的顶点式怎么求？"


# ---------------------------------------------------------------- 替身
def _settings(**overrides: Any) -> Settings:
    base = dict(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        llm_router_model="router",
        chat_support_contact="400-000-0000",
        citation_min_support=0.30,
        total_budget_ms=8000,
        feature_intent=True,
        feature_intent_llm=True,
        feature_query_rewrite=True,
        feature_grade_filter=True,
        understanding_timeout_ms=800,
        intent_bert_timeout_ms=200,
    )
    base.update(overrides)
    return Settings(**base)


class StubLLM:
    """按 system prompt 把"理解"与"生成"两次调用分开计数。"""

    def __init__(
        self,
        *,
        payload: str | None = None,
        answer: str = ANSWER,
        enabled: bool = True,
    ) -> None:
        self.payload = payload if payload is not None else json.dumps(
            {"query": QUERY, "intent": "备课检索", "confidence": 0.9}, ensure_ascii=False
        )
        self.answer = answer
        self.enabled = enabled
        self.understanding_calls = 0
        self.generation_calls = 0
        self.models: list[str | None] = []

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        is_understand = _UNDERSTAND_MARK in messages[0].get("content", "")
        self.models.append(model)
        if is_understand:
            self.understanding_calls += 1
            text = self.payload
        else:
            self.generation_calls += 1
            text = self.answer
        return LLMResult(text=text, model="stub", prompt_tokens=10, completion_tokens=5)


class StubBert:
    """`BertIntentClassifier.classify()` 的替身：返回结论 / 弃权 / 抛异常，三态可控。"""

    def __init__(
        self,
        *,
        result: IntentResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls = 0
        self.queries: list[str] = []
        self.timeouts: list[int] = []

    async def classify(self, query: str, *, timeout_ms: int = 200) -> IntentResult | None:
        self.calls += 1
        self.queries.append(query)
        self.timeouts.append(timeout_ms)
        if self.error is not None:
            raise self.error
        return self.result


class StubFAQ:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        subject: str | None = None,
        *,
        slots: SlotResult | None = None,
        selected_faq_id: int | None = None,
    ) -> FAQHit:
        self.queries.append(query)
        return FAQHit(tier="low", reason="below_threshold", score=0.1)


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.queries.append(query)
        return _retrieval(query)


class StubIntent:
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, query: str) -> IntentResult:
        self.calls += 1
        return IntentResult(label="学习讲解", confidence=0.8, method="rule", profile="")


def _retrieval(query: str = QUERY) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        evidences=[
            Evidence(
                parent_id="doc-1:p3",
                text=EVIDENCE_TEXT,
                score=0.91,
                rerank_score=0.9,
                child_hits=[EVIDENCE_TEXT[:40]],
                title="初中数学教材",
                source="F:/docs/教材.pdf",
                subject="数学",
                doc_type="textbook",
                heading="22.1 二次函数",
                page=12,
            )
        ],
        raw_hits=10,
        cost_ms=20,
    )


def _passed() -> EvidenceDecision:
    return EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)


_DEFAULT_BERT = object()


def _pipeline(
    *,
    llm: StubLLM | None = None,
    bert: Any = _DEFAULT_BERT,
    intent: StubIntent | None = None,
    **settings_overrides: Any,
) -> tuple[ChatPipeline, StubLLM, StubBert | None, StubIntent, RecordingRetriever]:
    llm = llm or StubLLM()
    # 只有**没传**才装一个替身；显式传 `None` 是"不注入 BERT"（回滚口径），不能替它补一个
    if bert is _DEFAULT_BERT:
        bert = StubBert()
    intent = intent or StubIntent()
    retriever = RecordingRetriever()
    settings = _settings(**settings_overrides)
    pipeline = ChatPipeline(
        settings,
        faq_service=StubFAQ(),
        rag={
            "ready": True,
            "retriever": retriever,
            "pipeline": RagPipeline(
                settings, retriever, gate=lambda result, **kw: _passed(), max_repairs=0
            ),
            "gate": None,
            "embedder_name": "stub",
            "rerank_enabled": False,
        },
        llm=llm,
        intent=intent,
        bert=bert,
    )
    return pipeline, llm, bert, intent, retriever


def _ask(pipeline: ChatPipeline, query: str = QUERY) -> Any:
    return asyncio.run(pipeline.ask(query))


def _bert_intent(label: str, confidence: float = 0.94) -> IntentResult:
    return IntentResult(label=label, confidence=confidence, method="bert", profile=profile_for(label).name)


# ---------------------------------------------------------------- ① BERT 定档
def test_confident_bert_wins_over_the_llm_payload() -> None:
    """大模型同一次调用里带回的意图（"备课检索"）不许覆盖 BERT 的结论。

    先到者优先：两个模型给出不一致答案时，若这里允许后者覆盖前者，
    就等于默认"大模型更可信"，而那条规则没有数据支撑——而且日志里永远看不出是谁定的档。
    """
    bert = StubBert(result=_bert_intent("考试出题"))
    pipeline, llm, _bert, intent_stub, _retriever = _pipeline(bert=bert)
    state = _ask(pipeline)

    assert bert.calls == 1
    assert state.intent is not None
    assert state.intent.label == "考试出题"
    assert state.intent.method == "bert"
    assert state.profile is not None and state.profile.name == profile_for("考试出题").name
    # BERT 已定档 → 规则分类器不该再跑
    assert intent_stub.calls == 0


def test_bert_confident_still_pays_only_one_llm_call_for_the_rewrite() -> None:
    """级联省的是"意图字段"，不是整次调用：改写还在同一次往返里做。"""
    bert = StubBert(result=_bert_intent("考试出题"))
    pipeline, llm, _bert, _intent, retriever = _pipeline(bert=bert)
    state = _ask(pipeline)

    assert llm.understanding_calls == 1  # 只为改写
    assert state.rewrite_method == "llm"
    assert state.rewritten == NORMALIZED
    # 改写真的到了执行点
    assert retriever.queries == [NORMALIZED]


def test_bert_confident_with_rewrite_off_sends_zero_llm_calls() -> None:
    """回滚口径：关掉改写 + BERT 定档 → 理解层一次大模型调用都不发。

    「一次理解调用」有**四个**消费者：改写、大模型意图、模型槽位、检索策略。
    不关后三个，BERT 定档之后照样会发一次——所以"零调用"的完整条件是四项都关。
    """
    bert = StubBert(result=_bert_intent("考试出题"))
    pipeline, llm, _bert, _intent, _retriever = _pipeline(
        bert=bert,
        feature_query_rewrite=False,
        feature_slot_llm=False,
        feature_planner_llm=False,
    )
    state = _ask(pipeline)
    assert llm.understanding_calls == 0
    assert state.intent is not None and state.intent.method == "bert"


# ---------------------------------------------------------------- ② 弃权 → 下一段
def test_bert_abstention_hands_over_to_the_llm() -> None:
    bert = StubBert(result=None)
    pipeline, llm, _bert, intent_stub, _retriever = _pipeline(bert=bert)
    state = _ask(pipeline)

    assert bert.calls == 1
    assert llm.understanding_calls == 1
    assert state.intent is not None
    assert state.intent.label == "备课检索" and state.intent.method == "model"
    assert intent_stub.calls == 0
    # 弃权是级联里的正常一步，不该被记成降级（否则每秒几十条的告警噪音）
    assert not [d for d in state.degraded if d.startswith("bert_")]


def test_bert_abstention_and_llm_intent_off_falls_back_to_the_rule_classifier() -> None:
    bert = StubBert(result=None)
    pipeline, llm, _bert, intent_stub, _retriever = _pipeline(
        bert=bert,
        feature_intent_llm=False,
        feature_query_rewrite=False,
        feature_slot_llm=False,
        feature_planner_llm=False,
    )
    state = _ask(pipeline)

    assert llm.understanding_calls == 0
    assert intent_stub.calls == 1
    assert state.intent is not None and state.intent.method == "rule"


# ---------------------------------------------------------------- ③ 降级与开关
def test_bert_runs_even_when_the_llm_is_unavailable() -> None:
    """没有 key 的部署恰恰最需要这条本地分类器。"""
    bert = StubBert(result=_bert_intent("学习规划"))
    pipeline, llm, _bert, _intent, _retriever = _pipeline(
        bert=bert, llm=StubLLM(enabled=False)
    )
    state = _ask(pipeline)

    assert llm.understanding_calls == 0  # 没 key，理解层整段不出门
    assert bert.calls == 1
    assert state.intent is not None and state.intent.method == "bert"
    assert state.profile is not None and state.profile.name == profile_for("学习规划").name


def test_bert_exception_does_not_break_the_request() -> None:
    """分类器自带"永不抛"的契约，但它可以被替换（评测桩 / 别的模型）：任何异常都不许连坐请求。"""
    bert = StubBert(error=RuntimeError("boom"))
    pipeline, _llm, _bert, intent_stub, _retriever = _pipeline(bert=bert)
    state = _ask(pipeline)

    assert bert.calls == 1
    assert state.answer  # 请求照常答完
    assert state.intent is not None and state.intent.method in ("model", "rule")


def test_feature_intent_off_never_calls_bert() -> None:
    """`FEATURE_INTENT=false` 的语义是"不做意图路由"，不是"换个分类器"。"""
    bert = StubBert(result=_bert_intent("考试出题"))
    pipeline, _llm, _bert, _intent, _retriever = _pipeline(bert=bert, feature_intent=False)
    state = _ask(pipeline)

    assert bert.calls == 0
    assert state.profile is not None and state.profile.name == BASELINE.name


def test_no_bert_injected_keeps_the_old_behaviour() -> None:
    """回滚口径：`bert=None` 时与加这条链路之前逐字节一致（大模型 → 规则）。"""
    pipeline, llm, _bert, intent_stub, _retriever = _pipeline(bert=None)
    state = _ask(pipeline)
    assert llm.understanding_calls == 1
    assert state.intent is not None and state.intent.method == "model"
    assert intent_stub.calls == 0


def test_bert_budget_comes_from_its_own_setting() -> None:
    """BERT 是本地前向，但仍要有硬上限：线程池被占时不许拖住首 token（§6.1）。"""
    bert = StubBert(result=None)
    pipeline, _llm, _bert, _intent, _retriever = _pipeline(bert=bert, intent_bert_timeout_ms=50)
    _ask(pipeline)
    assert bert.timeouts == [50]


def test_bert_sees_the_normalized_query() -> None:
    """送到前向的必须是清洗过的串：训练/推理分布一致，否则长问句的标点会把模型带偏。"""
    bert = StubBert(result=None)
    pipeline, _llm, _bert, _intent, _retriever = _pipeline(bert=bert)
    _ask(pipeline, "二次函数　的顶点式怎么求？？？")
    assert bert.queries == ["二次函数 的顶点式怎么求?"]
