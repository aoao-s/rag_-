"""理解层接线验收：归一化 → 一次大模型理解（改写 + 意图）→ 检索。

盯的不是"模型答得好不好"，而是**这三件事有没有真的抵达执行点**：

- 归一化后的问句是不是真的被送进了检索（`retriever.search` 收到的那串字）；
- 大模型给的改写/意图是不是真的改了 `state.rewritten` / `state.intent`，还是被静默丢了；
- 每一项失败时是不是**各自**退回（改写退归一化、意图退规则分类器），而不是一起塌掉。

方法论沿用 `tests/test_rag_store_filter.py`：断言**字段抵达执行点**，不是断言被赋值。
全部用替身，不打 MySQL / Milvus / 真实 LLM。

关于调用次数：理解与生成共用同一个网关，所以替身按 system prompt 分流成
`understanding_calls` / `generation_calls` 两个计数器——只数总数的话，
"FAQ 命中时不生成"这条断言会被理解那一次掩盖掉（阶段 3 之后 FAQ 排在理解之后，
那一次理解调用是**设计如此**：权限判定要读意图，见 `test_faq_high_hit_skips_*`）。
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

EVIDENCE_TEXT = "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k。"
# 理解调用的 system prompt 里的独有标记，用来把两类调用分开计数
_UNDERSTAND_MARK = "问题理解"

ANSWER = "顶点式可以通过配方法得到[1]。"

# 归一化做 NFKC：全角 `？` 折成半角 `?`。检索侧收到的永远是这一种写法——
# 断言里写 `?` 不是笔误，是这条链路的既定形态（大模型改写出来的 `？` 也会被再折一次）。
NORMALIZED = "二次函数的顶点式怎么求?"


# ---------------------------------------------------------------- 替身
def _settings(**overrides: Any) -> Settings:
    base = dict(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        llm_router_model="router",
        chat_support_contact="400-000-0000",
        citation_min_support=0.30,
        total_budget_ms=8000,
        # 本文件就是要验理解层，所以四个开关都显式打开；
        # 需要关的时候由用例自己覆盖，不靠"默认值恰好是 True"
        feature_intent=True,
        feature_intent_llm=True,
        feature_query_rewrite=True,
        feature_grade_filter=True,
        understanding_timeout_ms=800,
    )
    base.update(overrides)
    return Settings(**base)


class StubLLM:
    """同一个网关兼做理解与生成，按 system prompt 分流计数。"""

    def __init__(
        self,
        *,
        payload: str | None = None,
        answer: str = ANSWER,
        enabled: bool = True,
        understand_delay: float = 0.0,
    ) -> None:
        self.payload = payload if payload is not None else json.dumps(
            {"query": "二次函数的顶点式怎么求？", "intent": "学习讲解", "confidence": 0.92},
            ensure_ascii=False,
        )
        self.answer = answer
        self.enabled = enabled
        self.understand_delay = understand_delay
        self.understanding_calls = 0
        self.generation_calls = 0
        self.budgets: list[int] = []
        self.models: list[str | None] = []

    @property
    def calls(self) -> int:
        return self.understanding_calls + self.generation_calls

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        is_understand = _UNDERSTAND_MARK in messages[0].get("content", "")
        self.budgets.append(budget_ms)
        self.models.append(model)
        if is_understand:
            self.understanding_calls += 1
            if self.understand_delay:
                await asyncio.sleep(self.understand_delay)
            text = self.payload
        else:
            self.generation_calls += 1
            text = self.answer
        return LLMResult(text=text, model="stub", prompt_tokens=10, completion_tokens=5)


class StubFAQ:
    def __init__(self, hit: FAQHit | None = None) -> None:
        self.hit = hit if hit is not None else FAQHit(tier="low", reason="below_threshold", score=0.1)
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
        return self.hit


class RecordingRetriever:
    """记下**真正送去检索的那串字**——这是整份接线验收的核心观测点。"""

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
        return IntentResult(label="学习讲解", confidence=0.8, method="rule", profile="kb_stepwise")


def _retrieval(query: str = "二次函数的顶点式怎么求？") -> RetrievalResult:
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


def _pipeline(
    *,
    llm: StubLLM | None = None,
    faq: StubFAQ | None = None,
    retriever: RecordingRetriever | None = None,
    intent: StubIntent | None = None,
    **settings_overrides: Any,
) -> tuple[ChatPipeline, StubLLM, RecordingRetriever]:
    llm = llm or StubLLM()
    retriever = retriever or RecordingRetriever()
    pipeline = ChatPipeline(
        _settings(**settings_overrides),
        faq_service=faq or StubFAQ(),
        rag={
            "ready": True,
            "retriever": retriever,
            "pipeline": RagPipeline(
                _settings(**settings_overrides),
                retriever,
                gate=lambda result, **kw: _passed(),
                max_repairs=0,
            ),
            "gate": None,
            "embedder_name": "stub",
            "rerank_enabled": False,
        },
        llm=llm,
        intent=intent or StubIntent(),
    )
    return pipeline, llm, retriever


def _ask(pipeline: ChatPipeline, query: str = "二次函数的顶点式怎么求？") -> Any:
    return asyncio.run(pipeline.ask(query))


# ---------------------------------------------------------------- ① 高置信仍跳过检索与生成
def test_faq_high_hit_skips_retrieval_and_generation_but_pays_understanding() -> None:
    """阶段 3 时序：会话 → 理解 → **权限** → FAQ → 检索。

    权限判定要读意图（角色 × 意图矩阵），所以理解**必须**排在 FAQ 之前——
    高置信命中不再"零次大模型调用"，它固定付**一次**理解调用（改写 + 意图 + 槽位同一次往返）。
    换来的是：没有任何一条 FAQ 结论能绕过权限闸门。

    仍然成立、也仍然是这条用例真正要守住的两条：**不检索、不生成**。
    FAQ 是人工审核过的逐字答案，不需要证据门，也不需要模型再写一遍。
    """
    high = FAQHit(tier="high", faq_id=7, answer="学费 12800 元。", score=0.95, subject="数学")
    pipeline, llm, retriever = _pipeline(faq=StubFAQ(high))
    state = _ask(pipeline)
    assert state.source == "faq"
    assert llm.understanding_calls == 1  # 一次理解往返：权限要读意图
    assert llm.generation_calls == 0
    assert retriever.queries == []


# ---------------------------------------------------------------- ② 降级
def test_disabled_llm_falls_back_to_normalized_query() -> None:
    """没配 key 是配置状态不是本次降级：不该在每一份 `degraded` 里留痕。"""
    pipeline, llm, retriever = _pipeline(llm=StubLLM(enabled=False))
    state = _ask(pipeline, "二次函数　的顶点式怎么求？？？")
    assert llm.understanding_calls == 0
    # 归一化仍然生效（空白折叠 + 重复标点折叠），这是纯 CPU 的那一半
    assert retriever.queries == ["二次函数 的顶点式怎么求?"]
    assert state.rewrite_method == "normalize"
    assert not [d for d in state.degraded if d.startswith("understanding_")]


def test_understanding_timeout_still_retrieves_with_normalized_query() -> None:
    """改写是增强项：超时只丢增强，不许把检索打挂。"""
    pipeline, _llm, retriever = _pipeline(
        llm=StubLLM(understand_delay=0.5), understanding_timeout_ms=60
    )
    state = _ask(pipeline)
    assert "understanding_timeout" in state.degraded
    assert retriever.queries == [NORMALIZED]
    # 改写没成功，但归一化仍在：`？`→`?` 是确定性变换，不该因为改写失败而丢掉
    assert state.rewrite_method == "normalize"


def test_bad_json_degrades_without_changing_the_query() -> None:
    pipeline, _llm, retriever = _pipeline(llm=StubLLM(payload="我不确定你想问什么"))
    state = _ask(pipeline)
    assert "understanding_bad_json" in state.degraded
    assert retriever.queries == [NORMALIZED]


# ---------------------------------------------------------------- ③ 改写出口校验
def test_rewrite_is_applied_and_reaches_the_retriever() -> None:
    pipeline, llm, retriever = _pipeline()
    state = _ask(pipeline, "二次函数顶点式咋求")
    assert llm.understanding_calls == 1
    assert state.rewritten == NORMALIZED
    assert state.rewrite_method == "llm"
    # 核心断言：改写真的到了执行点，而不是只改了 `state`
    assert retriever.queries == [NORMALIZED]


def test_rewrite_with_newline_is_rejected() -> None:
    payload = json.dumps(
        {"query": "二次函数\n顶点式", "intent": "学习讲解", "confidence": 0.9}, ensure_ascii=False
    )
    pipeline, _llm, retriever = _pipeline(llm=StubLLM(payload=payload))
    state = _ask(pipeline)
    assert "rewrite_rejected" in state.degraded
    assert retriever.queries == [NORMALIZED]


def test_rewrite_that_drops_the_qualifier_is_rejected() -> None:
    """`八年级数学的勾股定理` → `勾股定理` 看起来成功了，召回却丢了约束，比不改更危险。"""
    original = "八年级数学的勾股定理怎么证明"
    payload = json.dumps(
        {"query": "勾股定理", "intent": "学习讲解", "confidence": 0.9}, ensure_ascii=False
    )
    pipeline, _llm, retriever = _pipeline(llm=StubLLM(payload=payload))
    state = _ask(pipeline, original)
    assert "rewrite_rejected" in state.degraded
    assert retriever.queries == [original]


def test_feature_rewrite_off_sends_no_understanding_call() -> None:
    """回滚开关：改写、大模型意图、模型槽位、检索策略四项都关时，一次调用都不该发。"""
    pipeline, llm, retriever = _pipeline(
        feature_query_rewrite=False,
        feature_intent_llm=False,
        feature_slot_llm=False,
        feature_planner_llm=False,
    )
    state = _ask(pipeline)
    assert llm.understanding_calls == 0
    assert retriever.queries == [NORMALIZED]
    assert state.rewrite_method == "normalize"


# ---------------------------------------------------------------- ④ 意图
def test_llm_intent_reaches_the_profile() -> None:
    payload = json.dumps(
        {"query": "二次函数的顶点式怎么求？", "intent": "备课检索", "confidence": 0.9},
        ensure_ascii=False,
    )
    intent_stub = StubIntent()
    pipeline, _llm, _retriever = _pipeline(llm=StubLLM(payload=payload), intent=intent_stub)
    state = _ask(pipeline)
    assert state.intent is not None
    assert state.intent.label == "备课检索" and state.intent.method == "model"
    # 大模型已给结论就不再跑规则分类器：两个结论同时存在会让日志与 prompt 选择各说一套
    assert intent_stub.calls == 0
    assert state.profile is not None and state.profile.name == "doc_lookup"


def test_out_of_vocabulary_intent_falls_back_to_the_rule_classifier() -> None:
    """词表外的标签（"数学题"）当它没给，交回规则分类器，而不是屈尊用它。"""
    payload = json.dumps(
        {"query": "二次函数的顶点式怎么求？", "intent": "数学题", "confidence": 0.95},
        ensure_ascii=False,
    )
    intent_stub = StubIntent()
    pipeline, _llm, _retriever = _pipeline(llm=StubLLM(payload=payload), intent=intent_stub)
    state = _ask(pipeline)
    assert "intent_rejected" in state.degraded
    assert intent_stub.calls == 1
    assert state.intent is not None and state.intent.method == "rule"


def test_low_confidence_intent_is_unknown_not_a_guess() -> None:
    """§3.2 第 2 条：置信度 <0.6 不猜，落 unknown 走 generic_clarify。"""
    payload = json.dumps(
        {"query": "二次函数的顶点式怎么求？", "intent": "学习讲解", "confidence": 0.4},
        ensure_ascii=False,
    )
    intent_stub = StubIntent()
    pipeline, _llm, _retriever = _pipeline(llm=StubLLM(payload=payload), intent=intent_stub)
    state = _ask(pipeline)
    assert state.intent is not None and state.intent.label == "unknown"
    assert state.profile is not None and state.profile.name == "generic_clarify"
    assert intent_stub.calls == 0


def test_retired_six_class_label_is_rejected_not_normalized() -> None:
    """退役的 6 类名不能被"归一化"着接受（任务 2.1 的口径切换）。

    `base/labels.py::normalize_intent_label` 会把 `知识点查询` 顺着别名表映射到新类上，
    那是**给滚动发布期间存量数据**留的兼容；评估与线上判定都不该走它。
    模型若照旧口径吐标签，正确的处理是"当它没给"→ 交回规则分类器，
    否则任务 2.1 最想量的事（大模型到底认不认新口径）永远量不出来。
    """
    payload = json.dumps(
        {"query": "二次函数的顶点式怎么求？", "intent": "知识点查询", "confidence": 0.95},
        ensure_ascii=False,
    )
    intent_stub = StubIntent()
    pipeline, _llm, _retriever = _pipeline(llm=StubLLM(payload=payload), intent=intent_stub)
    state = _ask(pipeline)
    assert "intent_rejected" in state.degraded
    assert intent_stub.calls == 1
    assert state.intent is not None and state.intent.method == "rule"


# ---------------------------------------------------------------- ⑤ 预算
def test_understanding_budget_is_capped_by_its_own_setting() -> None:
    """理解层抢的是同一个总预算，但自己还要有硬上限（§6.1）。"""
    pipeline, llm, _retriever = _pipeline(understanding_timeout_ms=300)
    _ask(pipeline)
    # 第一次调用是理解；上限取 `UNDERSTANDING_TIMEOUT_MS` 与剩余预算的小值
    assert llm.budgets[0] <= 300
    # 模型走的是 `LLM_ROUTER_MODEL` 而不是生成模型
    assert llm.models[0] == "router"


def test_understanding_budget_shrinks_when_the_budget_is_already_spent() -> None:
    """前面步骤慢了就该少花，而不是让生成没时间。"""
    pipeline, llm, _retriever = _pipeline(understanding_timeout_ms=800, total_budget_ms=200)
    _ask(pipeline)
    assert llm.budgets[0] <= 200
