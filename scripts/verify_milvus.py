"""重建后的知识库体检：块数、`doc_type` / `subject` / `stage` / `grade` 分布、抽样检索。

为什么单独一个脚本：入库脚本只报"写了多少条"，不报"写进去的东西能不能查"。
重建 collection 后要回答的是三个问题——

1. 总量对不对（有没有半截失败）；
2. 分类分布均不均匀（某个学科整本没进去，看这里最快）；
3. 真的能检索吗（`/v1/rag/search` 依赖的那条链路）。

只读，不建集合、不写库、不加载重排模型（重排在本机是 CPU 跳过态，算进延迟没有意义）。

用法：

    python scripts/verify_milvus.py
    python scripts/verify_milvus.py --probe "二次函数的顶点式怎么求"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.config import get_settings  # noqa: E402
from base.contracts import RetrievalFilters  # noqa: E402
from base.logger import configure  # noqa: E402
from rag_system.store.milvus_store import (  # noqa: E402
    DOC_TYPE_PARTITIONS,
    EXCLUDED_BY_DEFAULT,
    ChunkStore,
    partition_of,
)


def _distribution(store: ChunkStore, field: str, total: int) -> Counter:
    """拉全量该字段做计数。

    Milvus 2.4 的 `query` 不支持 `group_by`，只能取回来自己数。字段是短字符串，
    几万行的量级可以接受；真到百万级应改成 `count(*)` + `expr` 逐值查。
    """
    rows = store._client.query(  # noqa: SLF001 - 同一仓库的诊断脚本
        collection_name=store.collection,
        filter="",
        output_fields=[field],
        limit=total,
    )
    return Counter(r.get(field) or "(空)" for r in rows)


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure(settings.log_level, log_path=settings.log_path)

    p = argparse.ArgumentParser(description="重建后的 Milvus 知识库体检")
    p.add_argument("--probe", default="二次函数的顶点式怎么求", help="检索冒烟用的问句")
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="只做统计、不加载向量化模型（入库进程还在跑时用这个，免得抢 CPU）",
    )
    args = p.parse_args(argv)

    store = ChunkStore(settings)
    if not store.has_collection():
        print(f"[FAIL] 集合 {store.collection} 不存在")
        return 2

    total = store.count()
    print(f"集合 {store.collection}：{total} 块")
    if total == 0:
        print("[FAIL] 集合是空的")
        return 2

    # `stage` / `grade` 是 2026-10-29 新语料（`阶段/学科/年级/`）带来的字段，
    # 与 doc_type/subject 一起看才能发现"某个年级整本没进去"或"学段全空"
    by_type: Counter = Counter()
    for field in ("doc_type", "subject", "stage", "grade", "kb_version"):
        counts = _distribution(store, field, total)
        if field == "doc_type":
            by_type = counts
        print(f"\n-- {field} --")
        for key, n in counts.most_common():
            print(f"   {key:<16} {n:>7}")

    # 分区（写入落点）。它必须与 `doc_type` 分布逐个对齐：
    # 区块按 `doc_type` 落分区，`_default` 里还有东西就说明有写入绕过了 `upsert` 的分区参数
    # （例如重建前后混跑了两版代码）——那些块不会被任何 `partition_names` 搜到，属于静默丢数据。
    print("\n-- 分区（写入落点）--")
    parts = store.partition_counts()
    for name, n in sorted(parts.items(), key=lambda kv: -kv[1]):
        marker = "  ← 默认检索排除" if name in EXCLUDED_BY_DEFAULT else ""
        print(f"   {name:<16} {n:>7}{marker}")
    stray = parts.get("_default", 0)
    if stray:
        print(f"[WARN] `_default` 还有 {stray} 块：显式分区检索看不到它们，需重跑该文档")
    empty = [p for p in DOC_TYPE_PARTITIONS if parts.get(p, 0) == 0]
    if empty:
        print(f"[WARN] 空分区：{', '.join(empty)}（该 doc_type 一条都没入库？）")
    for doc_type, n in by_type.most_common():
        if doc_type in ("(空)",):
            continue
        got = parts.get(partition_of(doc_type), 0)
        if got != n:
            print(f"[WARN] {doc_type}：doc_type 计 {n} 块，分区 {got} 块，不一致")

    if args.no_probe:
        store.close()
        return 0

    # 检索冒烟：只看召回，不看重排（本机 CPU 重排是跳过态，算进延迟没有意义）
    print(f"\n-- 检索冒烟：{args.probe} --")
    try:
        import asyncio

        from rag_system.core.retriever import Retriever

        filters = RetrievalFilters(tenant_id=settings.tenant_id, kb_version=settings.kb_version)
        retriever = Retriever(settings, store=store)
        result = asyncio.run(retriever.search(args.probe, top_k=5, rerank=False, filters=filters))
        for ev in result.evidences:
            # 引用卡片要能区分"高一物理"与"高二物理"，所以学段/年级一起打出来
            print(
                f"   {ev.score:6.3f} [{ev.stage}{ev.grade} {ev.subject} {ev.doc_type}] "
                f"{Path(ev.source).name} p{ev.page} {ev.text[:40]}"
            )
        if not result.evidences:
            print("   （无召回）")
        print(f"   阶段耗时：{result.stages}")
    except Exception as exc:  # noqa: BLE001 - 冒烟失败不该让体检没有结果
        print(f"[FAIL] 检索冒烟：{exc}")
        store.close()
        return 1

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
