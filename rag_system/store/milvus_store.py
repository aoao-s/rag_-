"""Milvus 存储层：建库建表、写入、删除、混合检索。

设计要点（对应 CLAUDE.md §8.1）：
1. **一张表存子块 + 父块正文**，检索一次拿全，不必二次查库；
2. **dense + sparse 双索引**，`hybrid_search(WeightedRanker(0.7, 0.3))` 融合；
3. **按 `source` 删除**：重跑入库时先删同一文件的历史块，避免"改了内容旧块还在"；
4. 标量字段留 `subject` / `doc_type` / `page`，为"按学科过滤""引用卡片显示页码"预留。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from base.config import Settings, get_settings
from base.logger import get_logger
from rag_system.text_splitters.parent_child import Chunk

log = get_logger(__name__)

# VARCHAR 的 max_length 按字符算；父块上限 1200 字，留两倍余量给超长模板
_F_TEXT = 4096
_F_PARENT = 8192


@dataclass(slots=True)
class Hit:
    """检索结果。`score` 是融合后的分数，不同查询之间不可比。"""

    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    parent_text: str
    source: str
    title: str
    subject: str
    doc_type: str
    heading: str
    page: int
    score: float
    meta: dict[str, Any]


class ChunkStore:
    def __init__(self, settings: Settings | None = None) -> None:
        from pymilvus import MilvusClient

        self.settings = settings or get_settings()
        s = self.settings
        self._client = MilvusClient(uri=s.milvus_uri, token=s.milvus_token or None, timeout=30)
        self._ensure_database()
        self._client = MilvusClient(
            uri=s.milvus_uri,
            token=s.milvus_token or None,
            db_name=s.milvus_database,
            timeout=30,
        )

    # ---- 建库建表 ----
    def _ensure_database(self) -> None:
        name = self.settings.milvus_database
        if name not in self._client.list_databases():
            self._client.create_database(db_name=name)
            log.info("milvus_database_created", database=name)

    @property
    def collection(self) -> str:
        return self.settings.milvus_collection

    def has_collection(self) -> bool:
        return self._client.has_collection(self.collection)

    def ensure_collection(self, *, recreate: bool = False) -> None:
        from pymilvus import DataType

        if recreate and self.has_collection():
            self._client.drop_collection(self.collection)
            log.info("milvus_collection_dropped", collection=self.collection)
        if self.has_collection():
            self._client.load_collection(self.collection)
            return

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=160, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=160)
        schema.add_field("text", DataType.VARCHAR, max_length=_F_TEXT)
        schema.add_field("parent_text", DataType.VARCHAR, max_length=_F_PARENT)
        # 标量字段：来源与引用
        schema.add_field("source", DataType.VARCHAR, max_length=512)
        schema.add_field("title", DataType.VARCHAR, max_length=256)
        schema.add_field("subject", DataType.VARCHAR, max_length=16)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
        schema.add_field("heading", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT32)
        schema.add_field("chunk_index", DataType.INT32)
        schema.add_field("ts", DataType.INT64)
        schema.add_field("embedder", DataType.VARCHAR, max_length=32)
        # 向量字段
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.settings.milvus_dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        index = self._client.prepare_index_params()
        index.add_index(
            field_name="dense",
            index_type="HNSW",
            index_name="dense_idx",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_idx",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )
        self._client.create_collection(
            collection_name=self.collection, schema=schema, index_params=index
        )
        self._client.load_collection(self.collection)
        log.info("milvus_collection_created", collection=self.collection, dim=self.settings.milvus_dim)

    # ---- 写 ----
    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense: Sequence[list[float]],
        sparse: Sequence[dict[int, float]],
        *,
        source: str,
        subject: str | None,
        doc_type: str,
        title: str,
        embedder: str,
        batch_size: int = 64,
    ) -> int:
        if not chunks:
            return 0
        ts = int(time.time())
        rows: list[dict[str, Any]] = []
        for chunk, dv, sv in zip(chunks, dense, sparse):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "parent_id": chunk.parent_id,
                    "text": chunk.text[: _F_TEXT - 1],
                    "parent_text": chunk.parent_text[: _F_PARENT - 1],
                    "source": source[:511],
                    "title": (title or "")[:255],
                    "subject": (subject or "")[:15],
                    "doc_type": (doc_type or "other")[:31],
                    "heading": (chunk.heading or "")[:511],
                    "page": int(chunk.meta.get("page") or 0),
                    "chunk_index": chunk.index,
                    "ts": ts,
                    "embedder": embedder[:31],
                    "dense": dv,
                    "sparse": sv,
                }
            )

        written = 0
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._client.upsert(collection_name=self.collection, data=batch)
            written += len(batch)
        return written

    def delete_by_source(self, source: str) -> int:
        """删除同一文件的历史块：入库幂等的关键一步。"""
        expr = f'source == "{_escape(source)}"'
        res = self._client.delete(collection_name=self.collection, filter=expr)
        count = int(res.get("delete_count", 0)) if isinstance(res, dict) else 0
        if count:
            log.info("milvus_deleted_by_source", source=source, count=count)
        return count

    def delete_by_doc(self, doc_id: str) -> int:
        expr = f'doc_id == "{_escape(doc_id)}"'
        res = self._client.delete(collection_name=self.collection, filter=expr)
        return int(res.get("delete_count", 0)) if isinstance(res, dict) else 0

    # ---- 读 ----
    def search(
        self,
        dense: list[float],
        sparse: dict[int, float],
        *,
        top_k: int = 20,
        subject: str | None = None,
        doc_types: Sequence[str] | None = None,
        weights: tuple[float, float] = (0.7, 0.3),
    ) -> list[Hit]:
        from pymilvus import AnnSearchRequest, WeightedRanker

        expr = self._build_expr(subject, doc_types)
        reqs = [
            AnnSearchRequest(data=[dense], anns_field="dense", param={"ef": 64}, limit=top_k),
            AnnSearchRequest(
                data=[sparse], anns_field="sparse", param={"drop_ratio_search": 0.2}, limit=top_k
            ),
        ]
        results = self._client.hybrid_search(
            collection_name=self.collection,
            reqs=reqs,
            ranker=WeightedRanker(*weights),
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            filter=expr or "",
        )
        return [_to_hit(hit) for hit in (results[0] if results else [])]

    def count(self) -> int:
        try:
            res = self._client.query(
                collection_name=self.collection, filter="", output_fields=["count(*)"]
            )
            return int(res[0]["count(*)"]) if res else 0
        except Exception as exc:  # noqa: BLE001 - 统计失败不该影响主流程
            log.warning("milvus_count_failed", error=str(exc))
            return 0

    def stats(self) -> dict:
        return {"collection": self.collection, "chunks": self.count(), "has_collection": self.has_collection()}

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _build_expr(subject: str | None, doc_types: Sequence[str] | None) -> str:
        conds: list[str] = []
        if subject:
            conds.append(f'subject == "{_escape(subject)}"')
        if doc_types:
            types = ", ".join(f'"{_escape(t)}"' for t in doc_types)
            conds.append(f"doc_type in [{types}]")
        return " and ".join(conds)


_OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "parent_id",
    "text",
    "parent_text",
    "source",
    "title",
    "subject",
    "doc_type",
    "heading",
    "page",
]


def _to_hit(hit: dict) -> Hit:
    entity = hit.get("entity", {})
    return Hit(
        chunk_id=entity.get("chunk_id", ""),
        doc_id=entity.get("doc_id", ""),
        parent_id=entity.get("parent_id", ""),
        text=entity.get("text", ""),
        parent_text=entity.get("parent_text", ""),
        source=entity.get("source", ""),
        title=entity.get("title", ""),
        subject=entity.get("subject", ""),
        doc_type=entity.get("doc_type", ""),
        heading=entity.get("heading", ""),
        page=int(entity.get("page") or 0),
        score=float(hit.get("distance", 0.0)),
        meta={},
    )


def _escape(value: str) -> str:
    """Milvus 表达式里的字符串转义：反斜杠与双引号必须转，否则路径带引号就会炸表达式。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')
