"""SSE 流式通道（CLAUDE.md §7.8 第 5 项）：四类帧、逐句引用校验、断连与降级。

这里的替身只验证"分支与帧序列"，不验证答案质量——那是黄金集评估的事。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI

from base.config import Settings
from base.contracts import EvidenceDecision, FAQCandidate, FAQHit, SlotResult
from base.errors import UpstreamUnavailable
from orchestration.api import router as chat_router
from orchestration.chat import ChatPipeline
from rag_system.core.rag_pipeline import RagPipeline
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


class StubStreamLLM:
    """把预设文本按给定片段吐出来，模拟模型的分片。"""

    def __init__(self, chunks: list[str], *, boom_after: int | None = None, enabled: bool = True) -> None:
        self.chunks = chunks
        self.boom_after = boom_after
        self.enabled = enabled
        self.calls = 0

    async def complete(self, messages: list[dict], *, budget_ms: int, model: str | None = None) -> Any:
        raise AssertionError("流式通道不应该调用 complete()")

    async def stream(
        self, messages: list[dict], *, budget_ms: int, model: str | None = None
    ) -> AsyncIterator[str]:
        self.calls += 1
        for i, chunk in enumerate(self.chunks):
            if self.boom_after is not None and i >= self.boom_after:
                raise UpstreamUnavailable("LLM 流式连接失败", node="generate")
            yield chunk


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


class FixedGate:
    """固定结论的替身证据门：只验分支与帧序列，不验检索质量。"""

    def __init__(self, passed: bool) -> None:
        self.passed = passed
        self._decision = (
            EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)
            if passed
            else EvidenceDecision(
                status="insufficient",
                reasons=("below_threshold",),
                retry_action="fallback",
                top_score=0.2,
                margin=0.3,
                n_docs=1,
                qcov=0.6,
            )
        )

    def __call__(self, result: RetrievalResult, **kwargs: Any) -> EvidenceDecision:
        return self._decision


def _rag(*, ready: bool = True, pass_gate: bool = True) -> dict:
    retriever = StubRetriever()
    gate = FixedGate(pass_gate)
    return {
        "ready": ready,
        "retriever": retriever,
        "pipeline": RagPipeline(_settings(), retriever, gate=gate, max_repairs=0),
        "gate": gate,
        "embedder_name": "stub",
        "rerank_enabled": False,
    }


def _high_hit() -> FAQHit:
    return FAQHit(tier="high", faq_id=3, answer="学费是 12800 元。", score=0.95, subject="数学")


def _low_hit() -> FAQHit:
    return FAQHit(tier="low", reason="below_threshold", score=0.1)


def _mid_hit() -> FAQHit:
    return FAQHit(
        tier="mid",
        faq_id=7,
        answer=None,
        score=0.6,
        matched_question="二次函数的顶点式怎么写？",
        candidates=(FAQCandidate(7, "二次函数的顶点式怎么写？", 0.6),),
    )


def _frames(llm: StubStreamLLM, hit: FAQHit, *, rag: dict | None = None, **kwargs) -> list[dict]:
    pipeline = ChatPipeline(
        _settings(), faq_service=StubFAQ(hit), rag=rag if rag is not None else _rag(), llm=llm
    )

    async def run() -> list[dict]:
        return [f async for f in pipeline.ask_stream("顶点式怎么求", **kwargs)]

    return asyncio.run(run())


def _kinds(frames: list[dict]) -> list[str]:
    return [f["type"] for f in frames]


# ---- 提前出口 ----
def test_high_faq_streams_answer_without_llm() -> None:
    llm = StubStreamLLM(["不该被调用"])
    frames = _frames(llm, _high_hit())
    assert _kinds(frames) == ["meta", "delta", "citations", "done"]
    assert frames[0]["source"] == "faq"
    assert frames[1]["text"] == "学费是 12800 元。"
    assert frames[3]["answer"] == "学费是 12800 元。"
    assert frames[3]["faq_id"] == 3
    assert llm.calls == 0


def test_low_with_failed_gate_streams_fallback() -> None:
    frames = _frames(StubStreamLLM(["x"]), _low_hit(), rag=_rag(pass_gate=False))
    assert frames[0]["source"] == "fallback"
    assert "400-000-0000" in frames[3]["answer"]
    assert frames[3]["citations"] == []
    assert any(d.startswith("gate_") for d in frames[3]["degraded"])


def test_mid_with_failed_gate_streams_clarify() -> None:
    frames = _frames(StubStreamLLM(["x"]), _mid_hit(), rag=_rag(pass_gate=False))
    assert frames[0]["source"] == "clarify"
    assert "二次函数的顶点式怎么写？" in frames[1]["text"]
    assert frames[3]["faq_suggestion"] is not None
    assert "answer" not in frames[3]["faq_suggestion"]


def test_llm_disabled_streams_fallback() -> None:
    frames = _frames(StubStreamLLM(["x"], enabled=False), _low_hit(), rag=_rag(pass_gate=True))
    assert frames[0]["source"] == "fallback"
    assert "llm_unavailable" in frames[3]["degraded"]


# ---- 逐句流式 ----
def test_rag_stream_emits_delta_per_sentence_and_final_citations() -> None:
    chunks = ["顶点式可以通过配方法得到[1]。", "配方法的步骤是提取二次项系数[1]。"]
    frames = _frames(StubStreamLLM(chunks), _low_hit(), rag=_rag(pass_gate=True))
    assert _kinds(frames) == ["meta", "delta", "delta", "citations", "done"]
    assert frames[0]["source"] == "rag"
    assert [f["text"] for f in frames[1:3]] == chunks
    assert frames[3]["citations"][0]["n"] == 1
    assert frames[4]["answer"] == "".join(chunks)
    assert frames[4]["source"] == "rag"


def test_sentence_split_across_chunks_is_reassembled() -> None:
    # 引用标记被切成两片时不能在中间断句，否则 [1] 会被当成正文漏过去
    frames = _frames(StubStreamLLM(["顶点式可以通过配方法得到[", "1]。"]), _low_hit(),
                     rag=_rag(pass_gate=True))
    deltas = [f["text"] for f in frames if f["type"] == "delta"]
    assert deltas == ["顶点式可以通过配方法得到[1]。"]
    assert frames[-1]["citations"][0]["n"] == 1


def test_tail_without_period_is_still_emitted() -> None:
    frames = _frames(StubStreamLLM(["顶点式可以先配方[1]，", "再化简"]), _low_hit(),
                     rag=_rag(pass_gate=True))
    assert frames[-1]["answer"] == "顶点式可以先配方[1]，再化简"


def test_unsupported_sentence_is_dropped_but_stream_continues() -> None:
    chunks = ["今天天气不错适合出门[1]。", "顶点式可以通过配方法得到[1]。"]
    frames = _frames(StubStreamLLM(chunks), _low_hit(), rag=_rag(pass_gate=True))
    deltas = [f["text"] for f in frames if f["type"] == "delta"]
    assert deltas == ["顶点式可以通过配方法得到[1]。"]
    done = frames[-1]
    assert done["source"] == "rag"
    assert "citation_unsupported" in done["degraded"]


def test_out_of_range_reference_is_stripped_in_stream() -> None:
    frames = _frames(StubStreamLLM(["顶点式可以通过配方法得到[1]，详见附录[9]。"]), _low_hit(),
                     rag=_rag(pass_gate=True))
    deltas = [f["text"] for f in frames if f["type"] == "delta"]
    assert deltas and "[9]" not in deltas[0]
    assert frames[-1]["degraded"].count("citation_out_of_range") == 1


def test_no_citation_at_all_downgrades_done_to_fallback() -> None:
    """一句话都没对齐上时，done 必须把出口改成兜底——客户端据此丢弃已收到的 delta。"""
    frames = _frames(StubStreamLLM(["顶点式就是配方后的形式。"]), _low_hit(), rag=_rag(pass_gate=True))
    done = frames[-1]
    assert done["source"] == "fallback"
    assert "citation_missing" in done["degraded"]
    assert done["citations"] == []


def test_require_citation_false_keeps_uncited_stream() -> None:
    frames = _frames(StubStreamLLM(["顶点式就是配方后的形式。"]), _low_hit(),
                     rag=_rag(pass_gate=True), require_citation=False)
    assert frames[-1]["source"] == "rag"
    assert frames[-1]["answer"] == "顶点式就是配方后的形式。"


def test_upstream_failure_midstream_downgrades_done() -> None:
    llm = StubStreamLLM(["顶点式可以通过配方法得到[1]。", "第二句"], boom_after=1)
    frames = _frames(llm, _low_hit(), rag=_rag(pass_gate=True))
    done = frames[-1]
    assert done["source"] == "fallback"
    assert "llm_unavailable" in done["degraded"]
    assert "400-000-0000" in done["answer"]


def test_mid_with_passed_gate_streams_faq_card_without_answer() -> None:
    frames = _frames(StubStreamLLM(["顶点式可以通过配方法得到[1]。"]), _mid_hit(), rag=_rag(pass_gate=True))
    suggestion = frames[-1]["faq_suggestion"]
    assert suggestion is not None
    assert "answer" not in suggestion


# ---- HTTP ----
def _app(llm: StubStreamLLM, hit: FAQHit, *, rag: dict | None = None) -> FastAPI:
    pipeline = ChatPipeline(
        _settings(), faq_service=StubFAQ(hit), rag=rag if rag is not None else _rag(), llm=llm
    )
    app = FastAPI()
    app.include_router(chat_router)
    app.state.chat = pipeline
    return app


def _get(app: FastAPI, params: dict) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/v1/chat/stream", params=params)

    return asyncio.run(run())


def test_http_stream_is_sse_with_four_frame_types() -> None:
    resp = _get(_app(StubStreamLLM(["顶点式可以通过配方法得到[1]。"]), _low_hit(),
                     rag=_rag(pass_gate=True)), {"query": "顶点式怎么求"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line[len("data: "):])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [f["type"] for f in frames] == ["meta", "delta", "citations", "done"]
    assert frames[-1]["source"] == "rag"


def test_http_stream_invalid_subject_is_400() -> None:
    resp = _get(_app(StubStreamLLM(["x"]), _high_hit()), {"query": "学费多少", "subject": "天文学"})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "subject_invalid"


def test_http_stream_blank_query_is_400() -> None:
    """SSE 入口与 POST 走同一个 `admit()`，空问句同样是 400（曾经是 422）。"""
    resp = _get(_app(StubStreamLLM(["x"]), _high_hit()), {"query": "   "})
    assert resp.status_code == 400
    assert resp.json()["detail"]["error"] == "empty_query"


def test_http_stream_not_ready_is_503() -> None:
    app = FastAPI()
    app.include_router(chat_router)
    resp = _get(app, {"query": "学费多少"})
    assert resp.status_code == 503


# ---- 会话 id（CLAUDE.md §7.9）----
def test_session_id_is_reported_in_both_meta_and_done() -> None:
    """`meta` 与 `done` 是两次独立读点：只读 `done` 的客户端（重连后重放）也要能继续多轮。"""
    frames = _frames(StubStreamLLM(["x"]), _high_hit(), session_id="s-1")
    assert frames[0]["session_id"] == "s-1"
    assert frames[-1]["session_id"] == "s-1"
