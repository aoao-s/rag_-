"""在线检索与证据门的单测：假 store、假 embedder、假 reranker，不连 Milvus。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np

from base.config import Settings
from rag_system.core import evidence_gate
from rag_system.core.retriever import Retriever
from rag_system.store.milvus_store import Hit


def _settings(tmp_path: Path, **kw) -> Settings:
    base = dict(
        mysql_password="123456",
        milvus_uri="http://127.0.0.1:19541",
        rag_top_k=10,
        rag_final_n=3,
        rag_rerank_enabled="true",
        rag_rerank_candidates=5,
        rag_rerank_chars=100,
        rag_reranker_path=str(tmp_path / "nope"),
    )
    base.update(kw)
    return Settings(**base)


class FakeEmbedder:
    name = "fake"

    def encode_query(self, q: str):
        return [0.1] * 8, {"3": 0.5}


class FakeStore:
    def __init__(self, hits: list[Hit]) -> None:
        self.hits = hits
        self.calls: list[dict] = []

    def search(self, dense, sparse, **kwargs):
        self.calls.append(kwargs)
        return self.hits


class FakeReranker:
    def __init__(self, scores: list[float] | Exception) -> None:
        self.scores = scores

    def score(self, query, passages):
        if isinstance(self.scores, Exception):
            raise self.scores
        return self.scores[: len(passages)]


def _hit(cid: str, pid: str, score: float, text: str = "子块", parent: str = "父块正文") -> Hit:
    return Hit(
        chunk_id=cid,
        doc_id=pid.split(":p")[0],
        parent_id=pid,
        text=text,
        parent_text=parent,
        source=f"F:/x/{pid[:3]}.md",
        title="二次函数",
        subject="数学",
        doc_type="textbook",
        heading="1.1 顶点式",
        page=3,
        score=score,
        meta={},
    )


def test_merge_parents_keeps_best_and_collects_children(tmp_path: Path) -> None:
    hits = [_hit("d:p1:c0", "d:p1", 0.4, "子A"), _hit("d:p1:c1", "d:p1", 0.7, "子B"), _hit("d:p2:c0", "d:p2", 0.5)]
    store = FakeStore(hits)
    r = Retriever(_settings(tmp_path), store=store, embedder=FakeEmbedder(), reranker=FakeReranker([0.9, 0.2, 0.1]))
    res = asyncio.run(r.search("顶点式怎么写", rerank=False))

    assert [e.parent_id for e in res.evidences] == ["d:p1", "d:p2"]
    assert res.evidences[0].score == 0.7
    assert res.evidences[0].child_hits == ["子A", "子B"]
    # 父块正文进证据，而不是子块
    assert res.evidences[0].text == "父块正文"
    assert store.calls[0]["top_k"] == 10


def test_rerank_reorders_and_sets_score(tmp_path: Path) -> None:
    # 融合分把 p2 排在前面；重排分把 p1 顶上来，验证"重排真的重排了"
    hits = [_hit("d:p1:c0", "d:p1", 0.4), _hit("d:p2:c0", "d:p2", 0.5)]
    r = Retriever(
        _settings(tmp_path),
        store=FakeStore(hits),
        embedder=FakeEmbedder(),
        reranker=FakeReranker([0.20, 0.90]),
    )
    res = asyncio.run(r.search("q"))
    assert res.reranked is True
    assert [e.parent_id for e in res.evidences] == ["d:p1", "d:p2"]
    assert res.evidences[0].rerank_score == 0.90
    assert res.evidences[0].score == 0.90


def test_rerank_failure_degrades_to_fusion_score(tmp_path: Path) -> None:
    hits = [_hit("d:p1:c0", "d:p1", 0.4), _hit("d:p2:c0", "d:p2", 0.5)]
    r = Retriever(
        _settings(tmp_path),
        store=FakeStore(hits),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(RuntimeError("CUDA OOM")),
    )
    res = asyncio.run(r.search("q"))
    assert res.reranked is False
    assert "rerank_failed" in res.degraded
    assert res.evidences[0].parent_id == "d:p2"  # 仍按融合分排序


def test_empty_retrieval(tmp_path: Path) -> None:
    r = Retriever(_settings(tmp_path), store=FakeStore([]), embedder=FakeEmbedder(), reranker=FakeReranker([]))
    res = asyncio.run(r.search("q"))
    assert res.evidences == []
    assert res.top_score == 0.0
    assert res.citations() == []


def test_final_n_truncates(tmp_path: Path) -> None:
    hits = [_hit(f"d:p{i}:c0", f"d:p{i}", 0.9 - i / 100) for i in range(6)]
    r = Retriever(
        _settings(tmp_path, rag_final_n=2),
        store=FakeStore(hits),
        embedder=FakeEmbedder(),
        reranker=FakeReranker([0.9] * 6),
    )
    res = asyncio.run(r.search("q", rerank=False))
    assert len(res.evidences) == 2
    assert len(res.citations()) == 2
    assert res.citations()[0]["file"] == "d:p.md" or res.citations()[0]["file"].endswith(".md")


def test_gate_below_threshold() -> None:
    from rag_system.core.retriever import Evidence, RetrievalResult

    res = RetrievalResult(query="q")
    res.evidences = [Evidence("p", "t", 0.2, None, [], "t", "f", "数学", "textbook", "", 0)]
    d = evidence_gate.evaluate(res, threshold=0.35)
    assert d.passed is False and d.reason == "below_threshold"


def test_gate_single_weak_evidence() -> None:
    from rag_system.core.retriever import Evidence, RetrievalResult

    res = RetrievalResult(query="q")
    res.evidences = [Evidence("p", "t", 0.38, None, [], "t", "f", "数学", "textbook", "", 0)]
    d = evidence_gate.evaluate(res, threshold=0.35, single_doc_bonus=0.10)
    assert d.passed is False and d.reason == "single_weak_evidence"


def _evidence(pid: str, score: float, source: str = "f") -> "Evidence":
    from rag_system.core.retriever import Evidence

    return Evidence(pid, "t", score, None, [], "t", source, "数学", "textbook", "", 0)


def test_gate_same_doc_chunks_are_corroboration() -> None:
    """同一篇文档的多块证据分数接近不算歧义——它们是互相印证。"""
    from rag_system.core.retriever import RetrievalResult

    res = RetrievalResult(query="q")
    res.evidences = [_evidence("docA:p1", 0.99), _evidence("docA:p2", 0.9888), _evidence("docA:p3", 0.97)]
    d = evidence_gate.evaluate(res, threshold=0.35, margin=0.03)
    assert d.passed is True and d.reason == "ok"
    assert d.n_docs == 1


def test_gate_ambiguous_when_two_docs_close() -> None:
    """跨文档分差不足才是歧义：两篇资料给出同样可信的答案。"""
    from rag_system.core.retriever import RetrievalResult

    res = RetrievalResult(query="q")
    res.evidences = [_evidence("docA:p1", 0.60), _evidence("docA:p2", 0.585), _evidence("docB:p1", 0.59)]
    d = evidence_gate.evaluate(res, threshold=0.35, margin=0.03)
    assert d.passed is False and d.reason == "ambiguous"
    assert abs(d.margin - 0.01) < 1e-9
    assert d.n_docs == 2


def test_gate_passes_with_clear_winner() -> None:
    from rag_system.core.retriever import RetrievalResult

    res = RetrievalResult(query="q")
    res.evidences = [_evidence("docA:p1", 0.80), _evidence("docB:p1", 0.40)]
    d = evidence_gate.evaluate(res, threshold=0.35, margin=0.03)
    assert d.passed is True and d.reason == "ok"
    assert d.n_docs == 2


def _text_evidence(pid: str, score: float, text: str) -> "Evidence":
    from rag_system.core.retriever import Evidence

    return Evidence(pid, text, score, None, [], text, "f", "数学", "textbook", "", 0)


def test_gate_rejects_when_question_words_absent() -> None:
    """分数很高但证据里没有问句实词——典型"同主题不同知识点"，必须拒答。"""
    from rag_system.core.retriever import RetrievalResult

    res = RetrievalResult(query="有理数的加法法则是什么")
    res.evidences = [
        _text_evidence("docA:p1", 0.92, "教案：板书设计要突出二次函数顶点式的推导过程"),
        _text_evidence("docB:p1", 0.50, "课后作业：一元二次方程求根练习"),
    ]
    d = evidence_gate.evaluate(res, threshold=0.74, margin=0.01, min_qcov=0.20)
    assert d.passed is False and d.reason == "low_query_coverage"
    assert d.qcov == 0.0


def test_gate_passes_when_question_covered() -> None:
    from rag_system.core.retriever import RetrievalResult

    res = RetrievalResult(query="勾股定理的证明方法")
    res.evidences = [
        _text_evidence("docA:p1", 0.92, "勾股定理的证明方法有很多种，常见的是赵爽弦图证明。"),
        _text_evidence("docB:p1", 0.50, "光合作用需要光照。"),
    ]
    d = evidence_gate.evaluate(res, threshold=0.74, margin=0.01, min_qcov=0.20)
    assert d.passed is True and d.reason == "ok"
    assert d.qcov >= 0.5


def test_query_coverage_no_content_words() -> None:
    """问句全是虚词时覆盖度判不了，返回 1.0 而不是把用户挡在门外。"""
    assert evidence_gate.query_coverage("怎么办", "随便一段话") == 1.0
    assert evidence_gate.query_coverage("勾股定理怎么证明", "勾股定理的证明方法") > 0.3


