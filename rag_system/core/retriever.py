"""在线召回：混合检索 → 父块去重 → 可选重排。

设计取舍：
- **召回两路**（dense + sparse）由 Milvus 一次 `hybrid_search` 完成，不在这里拆成两次调用，
  否则两次 ANN 的 limit 各自生效，融合结果和 Milvus 内部 `WeightedRanker` 不是一回事。
  代价是**没有 `recall_dense` / `recall_sparse` 两段独立耗时**——那就不要编，
  统一记 `recall`，其余阶段按实测填（§7.2.7-D 要求的是实测值）。
- **去重单位是父块**：子块只是检索的"抓手"，真正喂给 LLM 的是父块正文；同一父块命中
  多个子块时保留分最高的那个，并把每个命中位置标出来，方便生成时强调。
- **重排可选**：CPU 上 bge-reranker 对 80 条候选要 2–4 s，线上必须有 GPU；没有 GPU 时
  按融合分排序（`RAG_RERANK_ENABLED=false`），质量降级但不报错。
- **安全过滤前置**：`filters` 直接交给 store 拼 expr，本层不做任何"查后过滤"。
  检索不可用抛 `RetrievalUnavailable`，与"检索成功但没证据"分开（§7.2.7 错误码）。
"""

from __future__ import annotations

import asyncio
import time

from base.contracts import Evidence, RetrievalFilters, RetrievalResult
from base.errors import RetrievalUnavailable
from base.logger import get_logger
from rag_system.store.milvus_store import ChunkStore, Hit

log = get_logger(__name__)

# 兼容旧导入路径：`Evidence` / `RetrievalResult` 的唯一出处是 `base.contracts`
__all__ = ["Evidence", "RetrievalResult", "Retriever"]


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
        filters: RetrievalFilters | None = None,
        task_id: str = "",
        strategy=None,
        stages: dict[str, int] | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        s = self.settings
        top_k = top_k or s.rag_top_k
        final_n = final_n or s.rag_final_n
        use_rerank = s.rerank_enabled if rerank is None else rerank
        result = RetrievalResult(query=query, stages=stages if stages is not None else {})
        if strategy is not None:
            result.strategy = strategy

        stage = time.perf_counter()
        try:
            dense, sparse = await asyncio.to_thread(self.embedder.encode_query, query)
        except Exception as exc:  # noqa: BLE001 - 向量化失败属于检索不可用
            raise RetrievalUnavailable(f"向量化失败：{exc}", node="retriever.embed") from exc
        result.stages["query_embed"] = _ms(stage)

        stage = time.perf_counter()
        try:
            hits = await asyncio.to_thread(
                self.store.search,
                dense,
                sparse,
                top_k=top_k,
                subject=subject,
                doc_types=doc_types,
                filters=filters,
            )
        except RetrievalUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - Milvus 连接/超时统一映射
            raise RetrievalUnavailable(f"向量检索失败：{exc}", node="retriever.recall") from exc
        result.stages["recall"] = _ms(stage)
        result.raw_hits = len(hits)

        stage = time.perf_counter()
        evidences = self._merge_parents(hits, task_id=task_id)
        result.stages["parent_expand"] = _ms(stage)
        result.stages["fusion"] = 0  # 融合在 Milvus 内部完成，本层无独立阶段

        if not evidences:
            result.cost_ms = _ms(started)
            result.stages["total"] = result.cost_ms
            log.info("rag_search_empty", q_len=len(query), top_k=top_k, cost_ms=result.cost_ms)
            return result

        if use_rerank:
            stage = time.perf_counter()
            try:
                evidences, scores = await self._rerank(query, evidences)
                for e, sc in zip(evidences, scores):
                    e.rerank_score = sc
                    e.score = sc
                result.reranked = True
            except Exception as exc:  # noqa: BLE001 - 重排失败降级为融合分
                result.degraded.append("rerank_failed")
                log.warning("rag_rerank_failed", error=str(exc), fallback="fusion_score")
            result.stages["rerank"] = _ms(stage)

        result.evidences = evidences[:final_n]
        result.cost_ms = _ms(started)
        result.stages["total"] = result.cost_ms
        log.info(
            "rag_search",
            q_len=len(query),
            raw_hits=result.raw_hits,
            parents=len(evidences),
            kept=len(result.evidences),
            top_score=round(result.top_score, 4),
            reranked=result.reranked,
            task_id=task_id,
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
    def _merge_parents(hits: list[Hit], *, task_id: str = "") -> list[Evidence]:
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
                    task_id=task_id,
                    child_id=hit.chunk_id,
                    stage=hit.stage,
                    grade=hit.grade,
                )
            else:
                if hit.text:
                    cur.child_hits.append(hit.text)
                if hit.score > cur.score:
                    cur.score = hit.score
                    cur.child_id = hit.chunk_id or cur.child_id
                    cur.heading = hit.heading or cur.heading
                    cur.page = hit.page or cur.page
        return sorted(grouped.values(), key=lambda e: e.score, reverse=True)


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)
