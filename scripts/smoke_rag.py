"""rag_system 冒烟：入库少量文件 → 检索 → 证据门，全程打印可核对的信息。

用法（先用 hash 后端验证管线，再用真模型验证质量）：

    # 管线验证：不加载模型，秒级跑完
    python scripts/smoke_rag.py --backend hash --paths datasets/01_textbook/math --limit 2

    # 真实质量：需要 torch + BGE-M3 权重
    python scripts/smoke_rag.py --paths datasets/01_textbook/math --limit 2 --query 顶点式怎么写

退出码：0 全部正常；1 有步骤失败（CI 可据此拦住坏提交）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from base.config import get_settings  # noqa: E402
from base.logger import configure, get_logger  # noqa: E402
from rag_system.core import evidence_gate  # noqa: E402
from rag_system.core.retriever import Retriever  # noqa: E402
from rag_system.ingest import IngestPipeline  # noqa: E402
from rag_system.store.milvus_store import ChunkStore  # noqa: E402

log = get_logger(__name__)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--paths", nargs="*", default=[])
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--backend", choices=["auto", "hash", "bge"], default=None)
    p.add_argument("--query", action="append", default=[])
    p.add_argument("--keep", action="store_true", help="保留已入库数据（默认先清空集合）")
    return p.parse_args()


async def _main() -> int:
    args = _parse()
    if args.backend:
        os.environ["RAG_EMBED_BACKEND"] = args.backend
    settings = get_settings()
    if args.backend:
        settings.rag_embed_backend = args.backend
    configure(settings.log_level, log_path=settings.log_path, file_level=settings.log_file_level)

    targets = [Path(p) for p in args.paths] or settings.rag_docs_paths[:1]
    print(f"== backend={settings.rag_embed_backend} model={settings.embed_model_path.exists()} 目标={targets}")

    store = ChunkStore(settings)
    t0 = time.perf_counter()
    pipe = IngestPipeline(settings, store=store)
    report = pipe.run(targets, recreate=not args.keep, limit=args.limit)
    print(
        f"== 入库：文件 {report.files}（成功 {report.docs_ok} / 失败 {report.docs_failed}）"
        f" 子块 {report.chunks} 写入 {report.written} 用时 {int((time.perf_counter() - t0) * 1000)} ms"
    )
    print(f"== 集合块数：{store.count()}")
    if report.chunks == 0:
        print("!! 没有可检索的块，后续检索无法验证")
        return 1

    retriever = Retriever(settings, store=store)
    queries = args.query or ["顶点式怎么写", "勾股定理的证明方法"]
    failures = 0
    for q in queries:
        res = await retriever.search(q, rerank=settings.rag_rerank_enabled)
        decision = evidence_gate.evaluate(
            res,
            threshold=settings.rag_evidence_threshold,
            margin=settings.rag_evidence_margin,
            min_docs=settings.rag_evidence_min_docs,
        )
        print(f"\n-- 查询：{q}")
        print(f"   命中父块 {len(res.evidences)}/{res.raw_hits} 条子块，耗时 {res.cost_ms} ms，重排={res.reranked}")
        for i, e in enumerate(res.evidences[:3], 1):
            print(f"   [{i}] {e.score:.3f} {e.title} / {e.heading} / {Path(e.source).name}")
            print(f"       {e.text[:60].replace(chr(10), ' ')}...")
        print(f"   证据门：{decision.as_dict()}")
        if not decision.passed:
            print("   → 该查询会拒答（转人工/RAG 兜底），这是设计行为，不是失败")
    if failures:
        return 1
    print("\n== 冒烟完成")
    return 0


if __name__ == "__main__":
    configure(get_settings().log_level, log_path=get_settings().log_path)
    sys.exit(asyncio.run(_main()))
