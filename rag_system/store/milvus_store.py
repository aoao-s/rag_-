"""Milvus 存储层：建库建表、写入、删除、混合检索。

设计要点（对应 CLAUDE.md §8.1）：
1. **一张表存子块 + 父块正文**，检索一次拿全，不必二次查库；
2. **dense + sparse 双索引**，`hybrid_search(WeightedRanker(0.7, 0.3))` 融合；
3. **按 `source` 删除**：重跑入库时先删同一文件的历史块，避免"改了内容旧块还在"；
4. 标量字段留 `subject` / `doc_type` / `page`，为"按学科过滤""引用卡片显示页码"预留；
5. **安全三元组 `tenant_id` / `acl_tags` / `kb_version` 是 schema 字段**（§7.2.7）：
   只有写进 schema 才能做前置标量过滤——"先检索再在 Python 里筛租户"是权限泄漏，
   而且会让 recall 指标虚高（被过滤掉的块其实占用了 top_k 名额）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from base.config import Settings, get_settings
from base.contracts import RetrievalFilters
from base.logger import get_logger
from rag_system.text_splitters.parent_child import Chunk

log = get_logger(__name__)

# VARCHAR 的 `max_length` 在本版本 Milvus 里按 **UTF-8 字节**校验（不是字符）。
# 原注释写"按字符算"，于是按字符截断对中文毫无保护：2026-10-29 实测一个 1 401 汉字的子块
# = 4 203 字节，被 `max_length=4096` 整份拒收，18 本教材入库全失败。
# 下面所有写库字段统一走 `_clip()`（按字节截断），父块上限 1 200 字 × 3 字节仍有余量。
_F_TEXT = 4096
_F_PARENT = 8192
# ACL 单文档标签上限（数组容量必须显式给出，Milvus 不会自动扩）
_ACL_CAPACITY = 16

# ---- 按 `doc_type` 分物理分区（2026-10-29 定案）----
# 一个集合、一份 `kb_version`、一个 alias（拆成 4 个集合会让版本切换与回滚变成 4 份同步动作，
# 中途失败就是"教材新版 + 试卷旧版"的混合态），但写入按 `doc_type` 落到各自分区，检索可只搜指定分区：
# 分区是**物理**隔离，被排除的分区连段都不会被扫到；`doc_type in [...]` 的标量表达式只是 bitset 剪枝。
DOC_TYPE_PARTITIONS: tuple[str, ...] = ("textbook", "lesson_plan", "exam", "question_bank")

# **默认检索不进题库**：题库按唯一内容折算约 1.8 万块（≈47%），题干词面与教材高度重合；
# 不收窄 `doc_type` 的档位（`generic_clarify` 占真实流量 42%）会让教材证据被两万道真题稀释。
# 只有意图档位显式点名 `question_bank`（`kb_stepwise`）时才搜它。
EXCLUDED_BY_DEFAULT: tuple[str, ...] = ("question_bank",)


def partition_of(doc_type: str | None) -> str:
    """`doc_type` → 分区名。未知类型各占一个同名分区，不并进 `_default`。

    并进 `_default` 的代价：默认分区会被"显式分区检索"漏掉（传了 `partition_names` 就不再搜
    `_default`），于是那些块写进去了却永远查不到——比多开一个分区糟得多。
    """
    return (doc_type or "").strip() or "other"


def search_partitions(
    doc_types: Sequence[str] | None, existing: Sequence[str]
) -> list[str] | None:
    """本次检索要落哪些分区；`None` 表示"不限制"（Milvus 会搜全部）。

    - 给了 `doc_types`：只搜这些分区；一个都不认识时返回 `None`——
      宁可全库检索，也不要因为拼错类型名搜一个空结果集。
    - 没给：全部**除** `EXCLUDED_BY_DEFAULT`（题库），这是"默认排除题库"的落点。
    """
    known = [p for p in existing]
    if not known:
        return None
    if doc_types:
        wanted = set(doc_types)
        picked = [p for p in known if p in wanted]
        return picked or None
    picked = [p for p in known if p not in EXCLUDED_BY_DEFAULT]
    return picked or None


def _clip(text: str, max_bytes: int) -> str:
    """按 UTF-8 字节截断，保证写库不会因子段超长整份失败。

    留 2 字节余量（`max_bytes - 1` 之后按字符对齐）而不是精确等于上限：
    多字节字符被切一半会抛错，`errors="ignore"` 会静默吞掉它，不如一开始就退一格。
    """
    raw = (text or "").encode("utf-8")
    if len(raw) <= max_bytes:
        return text or ""
    return raw[: max_bytes - 1].decode("utf-8", errors="ignore")


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
    tenant_id: str = ""
    kb_version: str = ""
    acl_tags: list[str] | None = None
    # 学段 / 年级：引用卡片要能区分"高一物理"与"高二物理"，过滤也靠它们
    stage: str = ""
    grade: str = ""


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
        # 分区名缓存：每次写入都问一遍 `list_partitions` 是白花的往返
        self._partitions: list[str] | None = None

    # ---- 建库建表 ----
    def _ensure_database(self) -> None:
        name = self.settings.milvus_database
        if name not in self._client.list_databases():
            self._client.create_database(db_name=name)
            log.info("milvus_database_created", database=name)

    @property
    def collection(self) -> str:
        return self.settings.milvus_collection

    @property
    def client(self) -> Any:
        """只读暴露底层 `MilvusClient`，给组合根复用（`main.py` 把它递给记忆模块的
        `MilvusAtomIndex`）。

        为什么允许这样伸手：`memory/` 自己不许 `MilvusClient(...)`（§7.0 硬边界 2），
        而再建一个连接只是多一份要维护的连接数——同一个库、同一个账号，没必要开两条。
        这里**只给引用不给设置方法**：调用方能拿到 client，但改不了本 store 的连接。
        """
        return self._client

    def has_collection(self) -> bool:
        return self._client.has_collection(self.collection)

    def ensure_collection(self, *, recreate: bool = False) -> None:
        from pymilvus import DataType

        if recreate and self.has_collection():
            self._client.drop_collection(self.collection)
            log.info("milvus_collection_dropped", collection=self.collection)
        if self.has_collection():
            self._assert_security_fields()
            self._assert_grade_fields()
            self._ensure_partitions()
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
        # 学段 / 年级：新语料 `阶段/学科/年级/` 三层目录的后两层。
        # 写进 schema 而不是塞进 `title`，是因为"按年级过滤"必须是 Milvus 前置标量过滤，
        # 不能先检索再在 Python 里筛（§7.2.7 DoD 第 5 条同一教训）。
        schema.add_field("stage", DataType.VARCHAR, max_length=16)
        schema.add_field("grade", DataType.VARCHAR, max_length=16)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=32)
        schema.add_field("heading", DataType.VARCHAR, max_length=512)
        schema.add_field("page", DataType.INT32)
        schema.add_field("chunk_index", DataType.INT32)
        schema.add_field("ts", DataType.INT64)
        schema.add_field("embedder", DataType.VARCHAR, max_length=32)
        # 安全三元组：检索前置过滤，**任何 retry / 放宽业务过滤都不能移除**（§7.2.7-C）
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=64)
        schema.add_field("kb_version", DataType.VARCHAR, max_length=32)
        schema.add_field(
            "acl_tags",
            DataType.ARRAY,
            element_type=DataType.VARCHAR,
            max_capacity=_ACL_CAPACITY,
            max_length=64,
        )
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
        self._ensure_partitions()
        self._client.load_collection(self.collection)
        log.info("milvus_collection_created", collection=self.collection, dim=self.settings.milvus_dim)

    # ---- 分区 ----
    def _existing_partitions(self) -> list[str]:
        """当前集合的分区名；读不到就当"没有分区"（检索退回全库，不为统计把查询打挂）。"""
        if self._partitions is not None:
            return self._partitions
        try:
            self._partitions = list(self._client.list_partitions(collection_name=self.collection))
        except Exception as exc:  # noqa: BLE001
            log.warning("milvus_list_partitions_failed", error=str(exc))
            return []
        return self._partitions

    def _ensure_partition(self, name: str) -> None:
        if name in self._existing_partitions():
            return
        try:
            self._client.create_partition(collection_name=self.collection, partition_name=name)
            log.info("milvus_partition_created", partition=name)
        except Exception as exc:  # noqa: BLE001 - 并发重建时可能已被别人建好
            log.warning("milvus_create_partition_failed", partition=name, error=str(exc))
        self._partitions = None  # 让下次重新读一遍，不缓存"猜的分区表"

    def _ensure_partitions(self) -> None:
        for name in DOC_TYPE_PARTITIONS:
            self._ensure_partition(name)

    def partition_counts(self) -> dict[str, int]:
        """各分区块数：重建后"某一类整个没进去"看这里最快。"""
        out: dict[str, int] = {}
        for name in self._existing_partitions():
            try:
                res = self._client.query(
                    collection_name=self.collection,
                    filter="",
                    output_fields=["count(*)"],
                    partition_names=[name],
                )
                out[name] = int(res[0]["count(*)"]) if res else 0
            except Exception as exc:  # noqa: BLE001
                log.warning("milvus_partition_count_failed", partition=name, error=str(exc))
                out[name] = -1
        return out

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
        stage: str | None = None,
        grade: str | None = None,
        tenant_id: str = "default",
        kb_version: str = "v1",
        acl_tags: Sequence[str] = (),
        batch_size: int = 64,
    ) -> int:
        if not chunks:
            return 0
        ts = int(time.time())
        tags = [_clip(t, 64) for t in list(dict.fromkeys(acl_tags))[:_ACL_CAPACITY]]
        rows: list[dict[str, Any]] = []
        for chunk, dv, sv in zip(chunks, dense, sparse):
            rows.append(
                {
                    "chunk_id": _clip(chunk.chunk_id, 160),
                    "doc_id": _clip(chunk.doc_id, 64),
                    "parent_id": _clip(chunk.parent_id, 160),
                    "text": _clip(chunk.text, _F_TEXT),
                    "parent_text": _clip(chunk.parent_text, _F_PARENT),
                    "source": _clip(source, 512),
                    "title": _clip(title or "", 256),
                    "subject": _clip(subject or "", 16),
                    "stage": _clip(stage or "", 16),
                    "grade": _clip(grade or "", 16),
                    "doc_type": _clip(doc_type or "other", 32),
                    "heading": _clip(chunk.heading or "", 512),
                    "page": int(chunk.meta.get("page") or 0),
                    "chunk_index": chunk.index,
                    "ts": ts,
                    "embedder": _clip(embedder, 32),
                    "tenant_id": _clip(tenant_id or "default", 64),
                    "kb_version": _clip(kb_version or "v1", 32),
                    "acl_tags": tags,
                    "dense": dv,
                    "sparse": sv,
                }
            )

        written = 0
        target = partition_of(doc_type)
        self._ensure_partition(target)
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            self._client.upsert(
                collection_name=self.collection, data=batch, partition_name=target
            )
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
        filters: RetrievalFilters | None = None,
        weights: tuple[float, float] = (0.7, 0.3),
    ) -> list[Hit]:
        from pymilvus import AnnSearchRequest, WeightedRanker

        # `filters` 优先；`subject` / `doc_types` 是给老调用方（诊断接口、脚本）留的便捷入口，
        # 二者同时给时以 filters 为准——绝不合并，免得出现两处过滤谁生效说不清的情况。
        expr = self._build_expr(
            subject if filters is None else None,
            doc_types if filters is None else None,
            filters,
        )
        # **过滤表达式只能挂在子请求上**：`MilvusClient.hybrid_search` 没有 `filter` 参数
        # （本机实装 pymilvus 3.0.2 实测；`requirements.txt` 钉的是 2.4.9，两版都不该赌）——
        # 传进去只落进 `**kwargs` 被静默丢弃：2026-10-29 实测 `subject=地理` 返回道德与法治/历史，
        # 且换成不存在的学科也照样有结果，说明**整个标量过滤从未生效**：`subject` 之外，
        # `tenant_id` / `kb_version` / `acl_tags` 也一并失效（跨租户可读到别人的资料）。
        # 原实现不报错、不告警，是最难查的那种。
        # 两路必须给**同一份** expr：只挂一路的话，融合时另一路仍然全库召回，
        # 权重 0.3 也足够把不该出现的块带进 top-k。
        expr = expr or None
        reqs = [
            AnnSearchRequest(
                data=[dense], anns_field="dense", param={"ef": 64}, limit=top_k, expr=expr
            ),
            AnnSearchRequest(
                data=[sparse],
                anns_field="sparse",
                param={"drop_ratio_search": 0.2},
                limit=top_k,
                expr=expr,
            ),
        ]
        results = self._client.hybrid_search(
            collection_name=self.collection,
            reqs=reqs,
            ranker=WeightedRanker(*weights),
            limit=top_k,
            output_fields=_OUTPUT_FIELDS,
            # 分区路由：`doc_types` 给了就只搜那几类，没给就搜"全部除题库"（见 `search_partitions`）
            partition_names=search_partitions(
                filters.doc_types if filters is not None else doc_types,
                self._existing_partitions(),
            ),
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
    def _build_expr(
        subject: str | None,
        doc_types: Sequence[str] | None,
        filters: RetrievalFilters | None = None,
    ) -> str:
        """拼 Milvus 过滤表达式。

        顺序固定为**安全 → 业务**：安全三项一定在表达式的与项里，`filters.relaxed()`
        只清业务项，所以"放宽业务过滤"永远动不到租户/ACL/版本。这是 §7.2.7 DoD 第 5 条
        的落点，单测直接断言 `_build_expr(relaxed)` 里仍有 tenant_id / kb_version。
        """
        conds: list[str] = []
        if filters is not None:
            conds.append(f'tenant_id == "{_escape(filters.tenant_id)}"')
            conds.append(f'kb_version == "{_escape(filters.kb_version)}"')
            if filters.acl_tags:
                tags = ", ".join(f'"{_escape(t)}"' for t in filters.acl_tags)
                # 无标签的公共文档对所有租户可见；有标签的必须命中其一
                conds.append(f"(array_length(acl_tags) == 0 or array_contains_any(acl_tags, [{tags}]))")
            if len(filters.subjects) == 1:
                conds.append(f'subject == "{_escape(filters.subjects[0])}"')
            elif len(filters.subjects) > 1:
                subs = ", ".join(f'"{_escape(s)}"' for s in filters.subjects)
                conds.append(f"subject in [{subs}]")
            if filters.doc_types:
                types = ", ".join(f'"{_escape(t)}"' for t in filters.doc_types)
                conds.append(f"doc_type in [{types}]")
            conds.extend(_enum_conds("stage", filters.stages))
            conds.extend(_enum_conds("grade", filters.grades))
            return " and ".join(conds)

        if subject:
            conds.append(f'subject == "{_escape(subject)}"')
        if doc_types:
            types = ", ".join(f'"{_escape(t)}"' for t in doc_types)
            conds.append(f"doc_type in [{types}]")
        return " and ".join(conds)

    def _assert_security_fields(self) -> None:
        """老 collection 没有安全字段时**宁可让 RAG 起不来**，也不能静默跳过过滤。

        缺字段时前置过滤等于不存在，跨租户读到别人的资料是数据事故；而 `main.py` 已经把
        RAG 装载失败处理为 `rag=false`（FAQ 照常），所以这里 fail-fast 的代价可控。
        """
        try:
            desc = self._client.describe_collection(self.collection)
            fields = {f.get("name") for f in desc.get("fields", [])}
        except Exception as exc:  # noqa: BLE001 - 描述失败同样当作未就绪
            raise RuntimeError(f"无法读取 collection 结构（{self.collection}）：{exc}") from exc
        missing = {"tenant_id", "kb_version", "acl_tags"} - fields
        if missing:
            raise RuntimeError(
                f"collection {self.collection} 缺少安全字段 {sorted(missing)}："
                "前置过滤无法生效。请执行 `python scripts/init_milvus.py --recreate` "
                "并重新入库（数据在 datasets/，可重建）。"
            )

    def _assert_grade_fields(self) -> None:
        """`stage` / `grade` 是 **schema 版本** 的一部分，不是可选列。

        它们出现在 `_OUTPUT_FIELDS` 里：老集合没有这两列，检索会在 Milvus 端直接报
        "field not exist"，而失败点离原因很远（看起来像检索挂了）。这里在装载时就说清楚
        ——和 `_assert_security_fields` 同一个处理口径：宁可 RAG 起不来，也不要带着
        一个"查不到年级"的库对外服务。改造代价可控：`main.py` 把 RAG 装载失败处理成
        `rag=false`，FAQ 照常。
        """
        try:
            desc = self._client.describe_collection(self.collection)
            fields = {f.get("name") for f in desc.get("fields", [])}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"无法读取 collection 结构（{self.collection}）：{exc}") from exc
        missing = {"stage", "grade"} - fields
        if missing:
            raise RuntimeError(
                f"collection {self.collection} 缺少学段/年级字段 {sorted(missing)}："
                "语料已改为 `阶段/学科/年级` 三层结构，这些列是检索输出与引用的必需字段。"
                "请执行 `python -m rag_system.ingest --recreate` 重建并重新入库。"
            )


_OUTPUT_FIELDS = [
    "chunk_id",
    "doc_id",
    "parent_id",
    "text",
    "parent_text",
    "source",
    "title",
    "subject",
    "stage",
    "grade",
    "doc_type",
    "heading",
    "page",
    "tenant_id",
    "kb_version",
    "acl_tags",
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
        tenant_id=entity.get("tenant_id", ""),
        kb_version=entity.get("kb_version", ""),
        acl_tags=list(entity.get("acl_tags") or []),
        stage=entity.get("stage", ""),
        grade=entity.get("grade", ""),
    )


def _enum_conds(field: str, values: Sequence[str]) -> list[str]:
    """枚举型业务过滤：单值用 `==`，多值用 `in [...]`，空则不过滤。"""
    if not values:
        return []
    if len(values) == 1:
        return [f'{field} == "{_escape(values[0])}"']
    joined = ", ".join(f'"{_escape(v)}"' for v in values)
    return [f"{field} in [{joined}]"]


def _escape(value: str) -> str:
    """Milvus 表达式里的字符串转义：反斜杠与双引号必须转，否则路径带引号就会炸表达式。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')
