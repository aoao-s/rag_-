"""编排层单测：/v1/chat 的三种出口与降级（CLAUDE.md §7.8 验收第 1 条）。

不起 MySQL / Milvus / LLM，全部用替身：这一层要证明的是"分支选对了"，
不是检索质量——后者由黄金集评估负责。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI

from base.config import Settings
from base.contracts import EvidenceDecision, FAQCandidate, FAQHit, SlotResult
from base.errors import RetrievalUnavailable
from orchestration.api import router as chat_router
from orchestration.chat import ChatPipeline
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import PipelineOutcome, RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult

EVIDENCE_TEXT = (
    "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k，其中 h=-b/(2a)，k=(4ac-b²)/(4a)。"
)


def _settings() -> Settings:
    return Settings(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        chat_support_contact="400-000-0000",
        citation_min_support=0.30,
        total_budget_ms=8000,
        # 关掉"一次大模型理解"的**四个**消费者（改写 + 意图 + 槽位抽取 + 检索策略）：
        # 本文件数的是**生成**调用次数，开着任何一个都会让同一个 `StubLLM`
        # 多收一次理解调用。理解路径见 `tests/test_chat_rewrite.py`
        feature_intent_llm=False,
        feature_query_rewrite=False,
        feature_slot_llm=False,
        feature_planner_llm=False,
        # **显式关掉 JWT 分支**：本文件有"带 X-User-Id 就拿会话 id"的用例，走的是
        # 开发期请求头。配了 `JWT_SECRET` 就自动切到 JWT-only（`resolve_identity()`
        # 不允许"配了密钥还认请求头"），不写这一行这个用例验的就是 `.env` 的状态
        jwt_secret="",
    )


class StubFAQ:
    def __init__(self, hit: FAQHit) -> None:
        self.hit = hit
        self.calls = 0

    async def search(self, query: str, subject: str | None = None, *, slots: SlotResult | None = None,
                     selected_faq_id: int | None = None) -> FAQHit:
        self.calls += 1
        return self.hit


class StubRetriever:
    def __init__(self, result: RetrievalResult | None = None, *, boom: bool = False) -> None:
        self.result = result or _retrieval()
        self.boom = boom
        self.calls = 0

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls += 1
        if self.boom:
            raise RuntimeError("milvus down")
        return self.result


class FixedGate:
    """固定结论的替身证据门：这一层验的是"分支选对了"，不是检索质量。"""

    def __init__(self, passed: bool) -> None:
        self.passed = passed
        self.calls = 0

    def __call__(self, result: RetrievalResult, **kwargs: Any) -> EvidenceDecision:
        self.calls += 1
        if self.passed:
            return EvidenceDecision(
                status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6
            )
        return EvidenceDecision(
            status="insufficient",
            reasons=("below_threshold",),
            retry_action="fallback",
            top_score=0.2,
            margin=0.3,
            n_docs=1,
            qcov=0.6,
        )


class StubLLM:
    def __init__(self, text: str = "顶点式可以通过配方法得到[1]。", *, enabled: bool = True) -> None:
        self.text = text
        self.enabled = enabled
        self.calls = 0

    async def complete(self, messages: list[dict], *, budget_ms: int, model: str | None = None) -> LLMResult:
        self.calls += 1
        return LLMResult(text=self.text, model="primary", prompt_tokens=100, completion_tokens=20)


def _evidence() -> Evidence:
    return Evidence(
        parent_id="doc-1:p3",
        text=EVIDENCE_TEXT,
        score=0.91,
        rerank_score=0.9,
        child_hits=[EVIDENCE_TEXT[:60]],
        title="人教版-数学-九年级上册",
        source="F:/docs/人教版-数学-九年级上册.pdf",
        subject="数学",
        doc_type="textbook",
        heading="22.1 二次函数",
        page=12,
    )


def _retrieval() -> RetrievalResult:
    return RetrievalResult(query="顶点式怎么求", evidences=[_evidence()], raw_hits=12, cost_ms=120)


def _rag(
    *, ready: bool = True, pass_gate: bool = True, retriever: StubRetriever | None = None
) -> dict:
    """装配成 `main.Container.load_rag()` 的同一形状：编排层只认 `ready` 与 `pipeline`。"""
    retriever = retriever or StubRetriever()
    gate = FixedGate(pass_gate)
    pipeline = RagPipeline(_settings(), retriever, gate=gate, max_repairs=0)
    return {
        "ready": ready,
        "retriever": retriever,
        "pipeline": pipeline,
        "gate": gate,
        "embedder_name": "stub",
        "rerank_enabled": False,
    }


def _ask(hit: FAQHit, *, rag: dict | None = None, llm: StubLLM | None = None, **kwargs) -> Any:
    pipeline = ChatPipeline(
        _settings(),
        faq_service=StubFAQ(hit),
        rag=rag if rag is not None else _rag(),
        llm=llm or StubLLM(),
    )
    return asyncio.run(pipeline.ask("顶点式怎么求", **kwargs))


def _high_hit() -> FAQHit:
    return FAQHit(tier="high", faq_id=3, answer="学费是 12800 元。", score=0.95, subject="数学")


def _mid_hit() -> FAQHit:
    return FAQHit(
        tier="mid",
        faq_id=7,
        answer=None,
        score=0.6,
        matched_question="二次函数的顶点式怎么写？",
        candidates=(FAQCandidate(7, "二次函数的顶点式怎么写？", 0.6),),
    )


def _low_hit() -> FAQHit:
    return FAQHit(tier="low", reason="below_threshold", score=0.1)


# ---- 三条主干 ----
def test_high_faq_returns_without_retrieval_or_llm() -> None:
    llm = StubLLM()
    retriever = StubRetriever()
    state = _ask(_high_hit(), rag=_rag(retriever=retriever), llm=llm)
    assert state.source == "faq"
    assert state.answer == "学费是 12800 元。"
    assert state.faq_id == 3
    assert retriever.calls == 0
    assert llm.calls == 0


def test_mid_with_failed_gate_clarifies_instead_of_answering_faq() -> None:
    state = _ask(_mid_hit(), rag=_rag(pass_gate=False), llm=StubLLM())
    assert state.source == "clarify"
    assert "二次函数的顶点式怎么写？" in state.answer
    assert state.faq_suggestion is not None
    assert "answer" not in state.faq_suggestion  # 中置信绝不返回 FAQ 原文
    assert state.faq_suggestion["faq_id"] == 7


def test_low_with_failed_gate_falls_back() -> None:
    state = _ask(_low_hit(), rag=_rag(pass_gate=False))
    assert state.source == "fallback"
    assert "400-000-0000" in state.answer
    assert any(d.startswith("gate_") for d in state.degraded)
    assert state.citations == []


# ---- 生成与引用 ----
def test_gate_passed_generates_answer_with_citations() -> None:
    llm = StubLLM("顶点式可以通过配方法得到[1]。")
    state = _ask(_low_hit(), rag=_rag(pass_gate=True), llm=llm)
    assert state.source == "rag"
    assert llm.calls == 1
    assert [c["n"] for c in state.citations] == [1]
    assert state.citations[0]["file_name"] == "人教版-数学-九年级上册.pdf"
    assert state.usage == {"prompt_tokens": 100, "completion_tokens": 20}


def test_mid_with_passed_gate_keeps_faq_card_but_not_answer() -> None:
    state = _ask(_mid_hit(), rag=_rag(pass_gate=True), llm=StubLLM())
    assert state.source == "rag"
    assert state.faq_suggestion is not None and "answer" not in state.faq_suggestion


def test_out_of_range_reference_cannot_support_answer() -> None:
    # 只有 [9]：越界被剔除后答案没有任何引用 → 整段拒答
    state = _ask(_low_hit(), rag=_rag(pass_gate=True), llm=StubLLM("顶点式可以这样求[9]。"))
    assert state.source == "fallback"
    assert "citation_out_of_range" in state.degraded
    assert "citation_missing" in state.degraded


def test_unsupported_sentence_refuses_whole_answer() -> None:
    state = _ask(_low_hit(), rag=_rag(pass_gate=True), llm=StubLLM("今天天气不错适合出门[1]。"))
    assert state.source == "fallback"
    assert "citation_unsupported" in state.degraded


# ---- 降级 ----
def test_rag_not_ready_mid_does_not_upgrade_to_high() -> None:
    state = _ask(_mid_hit(), rag=_rag(ready=False))
    assert state.source == "clarify"
    assert "rag_unavailable" in state.degraded


def test_rag_not_ready_low_falls_back() -> None:
    state = _ask(_low_hit(), rag=_rag(ready=False))
    assert state.source == "fallback"
    assert "rag_unavailable" in state.degraded


def test_llm_disabled_falls_back_even_with_evidence() -> None:
    llm = StubLLM(enabled=False)
    state = _ask(_low_hit(), rag=_rag(pass_gate=True), llm=llm)
    assert state.source == "fallback"
    assert "llm_unavailable" in state.degraded
    assert llm.calls == 0


def test_retriever_failure_falls_back() -> None:
    state = _ask(_low_hit(), rag=_rag(retriever=StubRetriever(boom=True)))
    assert state.source == "fallback"
    assert "retrieve_failed" in state.degraded


def test_require_citation_false_allows_uncited_answer() -> None:
    state = _ask(_low_hit(), rag=_rag(pass_gate=True), llm=StubLLM("顶点式就是配方后的形式。"),
                 require_citation=False)
    assert state.source == "rag"
    assert state.answer == "顶点式就是配方后的形式。"
    assert state.citations == []


# ---- HTTP ----
def _app(
    hit: FAQHit,
    *,
    rag: dict | None = None,
    llm: StubLLM | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    pipeline = ChatPipeline(
        settings or _settings(),
        faq_service=StubFAQ(hit),
        rag=rag if rag is not None else _rag(),
        llm=llm or StubLLM(),
    )
    app = FastAPI()
    app.include_router(chat_router)
    app.state.chat = pipeline
    return app


def _call(app: FastAPI, payload: dict) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/chat", json=payload)

    return asyncio.run(run())


def test_http_high_faq() -> None:
    resp = _call(_app(_high_hit()), {"query": "学费多少"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "faq" and body["faq_id"] == 3
    assert body["citations"] == [] and body["clarify"] is None


def test_http_fallback_is_still_200() -> None:
    resp = _call(_app(_low_hit(), rag=_rag(pass_gate=False)), {"query": "顶点式怎么求"})
    assert resp.status_code == 200
    assert resp.json()["source"] == "fallback"


def test_http_invalid_subject_is_400() -> None:
    resp = _call(_app(_high_hit()), {"query": "学费多少", "subject": "天文学"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "subject_invalid"


def test_http_blank_query_is_rejected() -> None:
    """空问句是**准入层的 400**，不是 pydantic 的 422（阶段 1 收口）。

    422 的语义是"字段类型/长度不合规"，400 的语义是"参数不对"；空问句、超长、
    学科拼错属于同一类产品口径，前端不该为它们分两条分支处理。
    """
    resp = _call(_app(_high_hit()), {"query": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "empty_query"


def test_http_empty_string_is_empty_query_like_blank() -> None:
    """`""` 与 `"   "` 必须是同一个码（2026-10-31 修）。

    `ChatRequest.query` 上写 `min_length=1` 会让 `""` 在 pydantic 那层报
    `invalid_request`，而 `"   "` 走准入层得 `empty_query`——同一个用户问题两个码，
    前端要为此写两条分支，而这正是 `admit()` 存在的理由（口径只有一处）。
    """
    empty = _call(_app(_high_hit()), {"query": ""})
    blank = _call(_app(_high_hit()), {"query": "   "})
    assert empty.status_code == blank.status_code == 400
    assert empty.json()["error_code"] == blank.json()["error_code"] == "empty_query"
    assert empty.json()["detail"]["error"] == "empty_query"


def test_http_bad_token_beats_empty_query() -> None:
    """坏 token + 空问句 → 401：**401 先于一切 400**（§3.4）。

    顺序反了，没凭证的调用方就能借参数错误探出"这份请求体合不合规"。
    pydantic 校验发生在 `resolve_identity()` 之前，所以 `query` 上一旦有长度约束，
    这条顺序就守不住。
    """
    app = _app(_high_hit(), settings=_settings().model_copy(update={"jwt_secret": "s" * 32}))
    resp = _call_headed(app, {"query": ""}, {"Authorization": "Bearer not.a.real.token"})
    assert resp.status_code == 401
    assert resp.json()["error_code"] == "invalid_token"


def test_http_chat_not_ready_is_503() -> None:
    app = FastAPI()
    app.include_router(chat_router)
    resp = _call(app, {"query": "学费多少"})
    assert resp.status_code == 503


# ---- 身份与 session_id（CLAUDE.md §7.9 边界 3）----
def _call_headed(app: FastAPI, payload: dict, headers: dict[str, str]) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/chat", json=payload, headers=headers)

    return asyncio.run(run())


def test_http_without_user_id_gets_no_session_id() -> None:
    """没有归属的会话写进库就再也删不掉（记忆页按 user_id 清），所以干脆不给 id。"""
    body = _call(_app(_high_hit()), {"query": "学费多少"}).json()
    assert body["session_id"] is None


def test_http_with_user_id_gets_and_keeps_session_id() -> None:
    app = _app(_high_hit())
    first = _call_headed(app, {"query": "学费多少"}, {"X-User-Id": "u1"}).json()
    assert first["session_id"]

    again = _call_headed(
        app, {"query": "学费多少", "session_id": first["session_id"]}, {"X-User-Id": "u1"}
    ).json()
    assert again["session_id"] == first["session_id"]  # 客户端带回来就必须沿用，不能另起一个
