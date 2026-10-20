"""在线检索：混合召回 → 父块去重 → 可选重排 → 证据门。

设计取舍：
- **召回两路**（dense + sparse）由 Milvus 一次 hybrid_search 完成，不在这里拆成两次调用，
  否则两次 ANN 的 limit 各自生效，融合结果和 Milvus 内部 RRF 不是一回事。
- **去重单位是父块**：子块只是检索的"抓手"，真正喂给 LLM 的是父块正文；同一父块命中
  多个子块时保留分最高的那个，并把每个命中位置标出来，方便生成时强调。
- **重排可选**：CPU 上 bge-reranker 对 80 条候选要 2–4 s，线上必须有 GPU；没有 GPU 时
  按融合分排序（`RAG_RERANK_ENABLED=false`），质量降级但不报错。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from base.logger import get_logger
from rag_system.store.milvus_store import ChunkStore, Hit

log = get_logger(__name__)


@dataclass(slots=True)
class Evidence:
    """一条可引用证据 = 一个父块 + 它在本次检索中的最好成绩。"""

    parent_id: str
    text: str
    score: float
    rerank_score: float | None
    child_hits: list[str]
    title: str
    source: str
    subject: str
    doc_type: str
    heading: str
    page: int

    def citation(self) -> dict:
        from pathlib import Path

        return {
            "parent_id": self.parent_id,
            "title": self.title or Path(self.source).stem,
            "file": Path(self.source).name,
            "subject": self.subject or None,
            "heading": self.heading or None,
            "page": self.page or None,
        }


@dataclass(slots=True)
class RetrievalResult:
    query: str
    evidences: list[Evidence] = field(default_factory=list)
    raw_hits: int = 0
    cost_ms: int = 0
    reranked: bool = False
    degraded: list[str] = field(default_factory=list)

    @property
    def top_score(self) -> float:
        return self.evidences[0].score if self.evidences else 0.0

    def citations(self) -> list[dict]:
        return [e.citation() for e in self.evidences]


class Retriever:
    def __init__(
        self,
        settings,
        *,
        store: ChunkStore | None = None,
        embedder=None,
        reranker=None,
    ) -> None:
        self.settings = settings
        self._store = store
        self._embedder = embedder
        self._reranker = reranker

    @property
    def store(self) -> ChunkStore:
        if self._store is None:
            self._store = ChunkStore(self.settings)
        return self._store

    @property
    def embedder(self):
        if self._embedder is None:
            from rag_system.embedding.bge_m3 import build_embedder

            self._embedder = build_embedder(self.settings)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from rag_system.core.reranker import build_reranker

            self._reranker = build_reranker(self.settings)
        return self._reranker

    async def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        final_n: int | None = None,
        subject: str | None = None,
        doc_types: list[str] | None = None,
        rerank: bool | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        s = self.settings
        top_k = top_k or s.rag_top_k
        final_n = final_n or s.rag_final_n
        use_rerank = s.rerank_enabled if rerank is None else rerank
        result = RetrievalResult(query=query)

        dense, sparse = await asyncio.to_thread(self.embedder.encode_query, query)
        hits = await asyncio.to_thread(
            self.store.search,
            dense,
            sparse,
            top_k=top_k,
            subject=subject,
            doc_types=doc_types,
        )
        result.raw_hits = len(hits)
        evidences = self._merge_parents(hits)
        if not evidences:
            result.cost_ms = int((time.perf_counter() - started) * 1000)
            log.info("rag_search_empty", q_len=len(query), top_k=top_k, cost_ms=result.cost_ms)
            return result

        if use_rerank:
            try:
                evidences, scores = await self._rerank(query, evidences)
                for e, sc in zip(evidences, scores):
                    e.rerank_score = sc
                    e.score = sc
                result.reranked = True
            except Exception as exc:  # noqa: BLE001 - 重排失败降级为融合分
                result.degraded.append("rerank_failed")
                log.warning("rag_rerank_failed", error=str(exc), fallback="fusion_score")

        result.evidences = evidences[:final_n]
        result.cost_ms = int((time.perf_counter() - started) * 1000)
        log.info(
            "rag_search",
            q_len=len(query),
            raw_hits=result.raw_hits,
            parents=len(evidences),
            kept=len(result.evidences),
            top_score=round(result.top_score, 4),
            reranked=result.reranked,
            degraded=",".join(result.degraded),
            cost_ms=result.cost_ms,
        )
        return result

    async def _rerank(
        self, query: str, evidences: list[Evidence]
    ) -> tuple[list[Evidence], list[float]]:
        limit = min(len(evidences), self.settings.rag_rerank_candidates)
        head, tail = evidences[:limit], evidences[limit:]
        scores = await asyncio.to_thread(
            self.reranker.score,
            query,
            [e.text[: self.settings.rag_rerank_chars] for e in head],
        )
        order = sorted(range(len(head)), key=lambda i: scores[i], reverse=True)
        ranked = [head[i] for i in order] + tail
        return ranked, [scores[i] for i in order] + [0.0] * len(tail)

    @staticmethod
    def _merge_parents(hits: list[Hit]) -> list[Evidence]:
        """同一父块的多条子块命中合并成一条证据，按最好分数排序。"""
        grouped: dict[str, Evidence] = {}
        for hit in hits:
            pid = hit.parent_id or hit.chunk_id
            cur = grouped.get(pid)
            if cur is None:
                grouped[pid] = Evidence(
                    parent_id=pid,
                    text=hit.parent_text or hit.text,
                    score=hit.score,
                    rerank_score=None,
                    child_hits=[hit.text] if hit.text else [],
                    title=hit.title,
                    source=hit.source,
                    subject=hit.subject,
                    doc_type=hit.doc_type,
                    heading=hit.heading,
                    page=hit.page,
                )
            else:
                cur.child_hits.append(hit.text)
                if hit.score > cur.score:
                    cur.score = hit.score
                    cur.heading = hit.heading or cur.heading
                    cur.page = hit.page or cur.page
        return sorted(grouped.values(), key=lambda e: e.score, reverse=True)
