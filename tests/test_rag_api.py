"""rag_system HTTP 层单测：用假 Retriever 顶掉 Milvus 与模型，只验证接口契约。

真链路（Milvus + BGE-M3）的验证在 `scripts/smoke_rag.py` 里做，单测不依赖外部服务。
"""

from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from base.config import Settings
from rag_system.api import router as rag_router
from rag_system.core.evidence_gate import evaluate as gate_evaluate
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult


class _FakeStore:
    def stats(self) -> dict:
        return {"collection": "unit_test", "chunks": 3, "has_collection": True}

    def close(self) -> None:
        pass


class _FakeRetriever:
    def __init__(self, evidences: list[Evidence], reranked: bool = False) -> None:
        self._evidences = evidences
        self._reranked = reranked
        self.store = _FakeStore()
        self.calls: list[dict] = []

    async def search(self, query: str, **kwargs) -> RetrievalResult:
        self.calls.append(kwargs)
        return RetrievalResult(
            query=query,
            evidences=list(self._evidences),
            raw_hits=len(self._evidences) * 2,
            cost_ms=7,
            reranked=self._reranked,
        )


def _ev(parent_id: str, score: float, subject: str = "数学") -> Evidence:
    return Evidence(
        parent_id=parent_id,
        text="直角三角形两直角边平方和等于斜边平方。",
        score=score,
        rerank_score=None,
        child_hits=["a", "b"],
        title="数学-八年级-勾股定理",
        source="datasets/01_textbook/数学.md",
        subject=subject,
        doc_type="textbook",
        heading="第十七章 勾股定理",
        page=3,
    )


def _settings() -> Settings:
    return Settings(rag_evidence_threshold=0.35, rag_evidence_margin=0.03, rag_evidence_qcov=0.0)


def _make_app(evidences: list[Evidence], ready: bool = True) -> FastAPI:
    retriever = _FakeRetriever(evidences)
    settings = _settings()
    app = FastAPI()
    app.include_router(rag_router)
    app.state.rag = (
        {
            "ready": True,
            "reason": "ok",
            "retriever": retriever,
            # 与线上同一条管线：接口是调试台，绕开 pipeline 展示的就不是线上行为
            "pipeline": RagPipeline(settings, retriever, max_repairs=0),
            # 假门的重点在"接口把门的结论透出去"；问题词覆盖度由 test_rag_retriever 单独验证
            "gate": lambda r: gate_evaluate(r, threshold=0.35, margin=0.03, min_qcov=0.0),
            "embedder_name": "fake",
            "rerank_enabled": False,
            "docs": 3,
        }
        if ready
        else {"ready": False, "reason": "milvus_unreachable"}
    )
    return app


def _call(app: FastAPI, url: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(url, **kwargs)

    return asyncio.run(run())


def test_search_returns_citations_and_gate() -> None:
    app = _make_app([_ev("p1", 0.82), _ev("p2", 0.41)])
    resp = _call(app, "/v1/rag/search", params={"q": "勾股定理", "final_n": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_score"] == 0.82
    assert body["evidence"]["passed"] is True
    assert len(body["citations"]) == 2
    # 引用卡片必须能定位到文件与标题，否则前端画不出"引用来源"
    assert body["citations"][0]["file"] == "数学.md"
    assert body["citations"][0]["page"] == 3
    assert body["evidences"][0]["child_hits"] == 2


def test_search_gate_rejects_ambiguous() -> None:
    # 跨文档分差不足 → 歧义拒答（同一文档多块是互相印证，不算歧义）
    app = _make_app([_ev("docA:p1", 0.6), _ev("docB:p1", 0.59)])
    body = _call(app, "/v1/rag/search", params={"q": "二次函数"}).json()
    assert body["evidence"]["passed"] is False
    assert body["evidence"]["reason"] == "ambiguous"


def test_search_not_ready_returns_503() -> None:
    resp = _call(_make_app([], ready=False), "/v1/rag/search", params={"q": "x"})
    assert resp.status_code == 503
    assert "milvus_unreachable" in resp.text


def test_stats_passthrough() -> None:
    body = _call(_make_app([_ev("p1", 0.9)]), "/v1/rag/stats").json()
    assert body["chunks"] == 3
    assert body["embedder"] == "fake"
    assert body["rerank_enabled"] is False


def test_search_passes_filters() -> None:
    app = _make_app([_ev("p1", 0.9)])
    _call(app, "/v1/rag/search", params={"q": "x", "subject": "数学", "doc_type": "textbook,faq"})
    call = app.state.rag["retriever"].calls[0]
    assert call["subject"] == "数学"
    assert call["doc_types"] == ["textbook", "faq"]
