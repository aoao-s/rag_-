"""越界拒答出口（任务 2.1 的 `超出范围` 类，CLAUDE.md §7.7）。

`超出范围` 是七类里**唯一**会拒答的一类，而拒答不该被当成一次失败出门：
它有自己的 `status="refused"` / `source="refusal"` / `error_code="out_of_scope"`，
三者都不许并入 `no_evidence` 或 `error`。理由写在 `_refusal()` 的注释里：
"明天成都天气" 一旦记进 `no_evidence`，教研就会照着这个数去为天气建语料，
而那本来就不是我们要答的题；记进 `error` 则会让 5xx 告警里混进一堆正常的边界行为。

本文件盯四件事，全部用替身，不打 MySQL / Milvus / 真实 LLM：

1. 拒答**真的短路了**——没检索、没生成、没引用；
2. 拒答话术是**固定文案**，不是模型现编的（换个越界问句，答案逐字相同）；
3. 话术**不承诺转人工/客服**（`_OUT_OF_SCOPE` 的注释解释过为什么）；
4. 两条规则管着拒答的边界：判成"闲聊"但带疑问词的**不许**真跳过检索（教学场景词守卫），
   以及 FAQ 排在拒答**之后**——高置信命中不能把一次越界拒答顶掉（阶段 3 时序）。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from fastapi import FastAPI

from base.config import Settings
from base.contracts import EvidenceDecision, FAQHit, IntentResult, SlotResult
from orchestration.api import router as chat_router
from orchestration.chat import ChatPipeline
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult

# 拒答文案的唯一出处。测试里 import 它（而不是另抄一份）是为了让"改话术"
# 只需要改一处——但下面仍然断言它的**性质**，不是断言它等于自己。
from orchestration.chat import _OUT_OF_SCOPE

EVIDENCE_TEXT = "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k。"
ANSWER = "顶点式可以通过配方法得到[1]。"
SUPPORT = "400-000-0000"

# 越界问句：规则分类器判 `超出范围`（高置信），且不含任何教学场景词。
OUT_OF_SCOPE_QUERY = "明天成都天气怎么样"


# ---------------------------------------------------------------- 替身
def _settings(**overrides: Any) -> Settings:
    base = dict(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        llm_router_model="router",
        chat_support_contact=SUPPORT,
        citation_min_support=0.30,
        total_budget_ms=8000,
        feature_intent=True,
        # 默认让**规则**分类器决定意图：本文件验的是编排层对档位的反应，
        # 不是大模型的准确率（那条在 `scripts/eval_intent.py` 里量）。
        feature_intent_llm=False,
        feature_query_rewrite=False,
        # 槽位抽取是"一次理解调用"的第三个消费者（任务 2.3），默认打开。
        # 不关掉它，替身 `StubLLM` 会收到一次理解调用并回空串，`degraded` 里
        # 凭空多出 `understanding_bad_json`——本文件验的是档位与拒答边界，
        # 不该被一条与断言无关的降级码干扰
        feature_slot_llm=False,
        understanding_timeout_ms=800,
    )
    base.update(overrides)
    return Settings(**base)


class StubLLM:
    """一个网关兼做理解与生成；按 system prompt 分流计数。"""

    def __init__(self, *, payload: str | None = None, answer: str = ANSWER) -> None:
        self.payload = payload
        self.answer = answer
        self.enabled = True
        self.understanding_calls = 0
        self.generation_calls = 0

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        if "问题理解" in messages[0].get("content", ""):
            self.understanding_calls += 1
            text = self.payload or ""
        else:
            self.generation_calls += 1
            text = self.answer
        return LLMResult(text=text, model="stub", prompt_tokens=10, completion_tokens=5)


class StubFAQ:
    def __init__(self, hit: FAQHit | None = None) -> None:
        self.hit = hit if hit is not None else FAQHit(tier="low", reason="below_threshold", score=0.1)
        self.calls = 0

    async def search(
        self,
        query: str,
        subject: str | None = None,
        *,
        slots: SlotResult | None = None,
        selected_faq_id: int | None = None,
    ) -> FAQHit:
        self.calls += 1
        return self.hit


class RecordingRetriever:
    def __init__(self) -> None:
        self.calls = 0
        self.queries: list[str] = []

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls += 1
        self.queries.append(query)
        return _retrieval(query)


class StubIntent:
    """可控的规则分类器替身，用于"档位与守卫"而不是"分类准不准"的用例。"""

    def __init__(self, label: str = "学习讲解", confidence: float = 0.8) -> None:
        self.label = label
        self.confidence = confidence
        self.calls = 0

    def classify(self, query: str) -> IntentResult:
        self.calls += 1
        return IntentResult(label=self.label, confidence=self.confidence, method="rule", profile="")


def _retrieval(query: str = "x") -> RetrievalResult:
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


def _passed(result: Any = None, **kwargs: Any) -> EvidenceDecision:
    return EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)


def _pipeline(
    *,
    llm: StubLLM | None = None,
    faq: StubFAQ | None = None,
    retriever: RecordingRetriever | None = None,
    intent: StubIntent | None = None,
    **settings_overrides: Any,
) -> tuple[ChatPipeline, StubLLM, RecordingRetriever, StubFAQ]:
    llm = llm or StubLLM()
    retriever = retriever or RecordingRetriever()
    faq = faq or StubFAQ()
    settings = _settings(**settings_overrides)
    pipeline = ChatPipeline(
        settings,
        faq_service=faq,
        rag={
            "ready": True,
            "retriever": retriever,
            "pipeline": RagPipeline(settings, retriever, gate=_passed, max_repairs=0),
            "gate": None,
            "embedder_name": "stub",
            "rerank_enabled": False,
        },
        llm=llm,
        intent=intent,
    )
    return pipeline, llm, retriever, faq


def _ask(pipeline: ChatPipeline, query: str = OUT_OF_SCOPE_QUERY) -> Any:
    return asyncio.run(pipeline.ask(query))


# ---------------------------------------------------------------- ① 真的短路了
def test_out_of_scope_never_retrieves_or_generates() -> None:
    """拒答发生在检索之前：不该为一个我们明确不答的问题去查一遍知识库。"""
    pipeline, llm, retriever, _faq = _pipeline()
    state = _ask(pipeline)
    assert state.intent is not None and state.intent.label == "超出范围"
    assert state.source == "refusal"
    assert state.error_code == "out_of_scope"
    assert state.answer == _OUT_OF_SCOPE
    assert "intent_out_of_scope" in state.degraded
    # 三条"没做"：没检索、没生成、没引用。
    # 引用尤其重要——给一条引用等于暗示"这条知识库内容与你的问题相关"
    assert retriever.calls == 0
    assert llm.generation_calls == 0
    assert state.citations == []


def test_refusal_does_not_enter_the_no_evidence_bin_on_the_http_layer() -> None:
    """拒答是 200 + `status="refused"`，既不是 `no_evidence` 也不是 `error`。

    `status` 由 `ChatState.resolve_status()` 推导（`source == "refusal"` 优先于一切），
    所以这条断言同时钉住了"没人能在拒答路径上把 status 改成别的"。
    """
    pipeline, _llm, _retriever, _faq = _pipeline()

    async def call() -> httpx.Response:
        app = FastAPI()
        app.include_router(chat_router)
        app.state.chat = pipeline
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/chat", json={"query": OUT_OF_SCOPE_QUERY})

    response = asyncio.run(call())
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "refused"
    assert body["source"] == "refusal"
    assert body["error_code"] == "out_of_scope"
    assert body["citations"] == []
    assert body["answer"] == _OUT_OF_SCOPE
    # 有 request_id 才能把用户截图和日志对上（拒答也不例外）
    assert body["request_id"]


# ---------------------------------------------------------------- ②③ 话术
def test_refusal_wording_is_fixed_not_model_generated() -> None:
    """换个越界问句，答案逐字相同——拒答是代码给的结论，不是模型现编的。"""
    first, _llm, _r1, _f1 = _pipeline()
    second, _llm2, _r2, _f2 = _pipeline()
    a = _ask(first, "明天成都天气怎么样")
    b = _ask(second, "北京到上海的高铁要多久")
    assert a.source == b.source == "refusal"
    assert a.answer == b.answer == _OUT_OF_SCOPE


def test_refusal_wording_makes_no_handoff_promise() -> None:
    """越界问题人工也答不了，所以话术里**不许**出现客服电话或"转人工"。

    这条不是文案偏好：写上电话等于承诺一件我们不做的事，
    还会让用户以为换个渠道就能绕过这条边界（§7.7 `OUT_OF_SCOPE`）。
    """
    assert SUPPORT not in _OUT_OF_SCOPE
    assert "转人工" not in _OUT_OF_SCOPE
    assert "客服" not in _OUT_OF_SCOPE


# ---------------------------------------------------------------- ④ 拒答之前的规则
def test_refusal_decision_precedes_a_high_faq_hit() -> None:
    """阶段 3 定稿的时序：会话 → 理解 → 权限 → FAQ → 检索。

    所以"高置信 FAQ 短路整条流水线"这条旧口径**不再成立**：FAQ 排在理解与权限之后，
    一个被判成 `超出范围` 的问句即便撞上一条高置信 FAQ，也先走固定拒答、**一次 FAQ 都不查**。
    这不是退化——顺序由用户定死"先身份与权限、再 FAQ"，而越界判定属于理解层结论，
    排在 FAQ 之前；让 FAQ 抢先返回等于用一条历史答案盖过"这个问题我们本来就不答"。
    """
    high = FAQHit(tier="high", faq_id=9, answer="学费 12800 元。", score=0.95, subject="数学")
    pipeline, _llm, retriever, faq = _pipeline(faq=StubFAQ(high))
    state = _ask(pipeline)
    assert faq.calls == 0  # 拒答在 FAQ 之前，问都不问
    assert state.source == "refusal" and state.answer == _OUT_OF_SCOPE
    assert retriever.calls == 0


def test_educational_signal_vetoes_the_refusal() -> None:
    """方向相反的守卫：准备拒答时，问句里只要带教学场景词就不许拒。

    误拒的代价是"用户拿不到本来就有的答案"，比多查一轮严重得多，
    所以这条守卫宁可把越界判成 `generic_clarify`（全库检索 + 追问）。

    问句本身取"带教学场景词、但不带指代词"的一条：带"这个/它/上面"的话会先被
    规划层判成 BACKTRACK（无历史时不检索），那样测到的就不是拒答守卫了。
    """
    query = "布置作业怎么设计才合理"
    payload = json.dumps(
        {"query": query, "intent": "超出范围", "confidence": 0.95},
        ensure_ascii=False,
    )
    pipeline, _llm, retriever, _faq = _pipeline(
        llm=StubLLM(payload=payload), feature_intent_llm=True, feature_query_rewrite=True
    )
    state = _ask(pipeline, query)
    assert state.source != "refusal"
    assert "intent_out_of_scope_guarded" in state.degraded
    assert retriever.calls == 1
    assert state.profile is not None and state.profile.name == "generic_clarify"


def test_chitchat_is_answered_not_refused() -> None:
    """`allow_general` 是闲聊与越界之间**唯一**的字段差别：闲聊照常短答，只是不检索。"""
    pipeline, llm, retriever, _faq = _pipeline()
    state = _ask(pipeline, "你好呀")
    assert state.intent is not None and state.intent.label == "闲聊"
    assert state.source != "refusal"
    assert retriever.calls == 0
    assert llm.generation_calls == 1
    assert "intent_no_retrieval" in state.degraded
