"""切分质检：按文档统计父/子块数量与长度分布，先看结构再决定要不要调入参。

为什么单独做工具而不是塞进 ingest：
- 入库只关心"写成功没有"，但切分好不好要看**分布**（有没有 4000 字的巨块、
  有没有一句话一块、标题是否继承）。这些数字在每次改参数后都要复核一遍。

用法：
    python scripts/chunk_stats.py                       # 用 RAG_DOCS_DIR
    python scripts/chunk_stats.py --paths datasets/01_textbook
    python scripts/chunk_stats.py --paths ... --json out.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from base.config import get_settings  # noqa: E402
from base.logger import configure  # noqa: E402
from rag_system.document_loaders.thy_factory import DocumentLoaderFactory  # noqa: E402
from rag_system.text_splitters.parent_child import ParentChildSplitter, SplitterConfig  # noqa: E402


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure(settings.log_level, log_path=settings.log_path, console_level="WARNING")

    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", nargs="*", default=[])
    ap.add_argument("--json", default="", help="把明细写到指定文件")
    ap.add_argument("--min-size", type=int, default=None, help="覆盖 RAG_CHILD_SIZE 之外的短块下限")
    args = ap.parse_args(argv)

    targets = [Path(p) for p in args.paths] or settings.rag_docs_paths
    targets = [t if t.is_absolute() else (BASE_DIR / t) for t in targets]

    cfg = SplitterConfig(
        parent_size=settings.rag_parent_size,
        parent_overlap=settings.rag_parent_overlap,
        child_size=settings.rag_child_size,
        child_overlap=settings.rag_child_overlap,
    )
    splitter = ParentChildSplitter(cfg)
    factory = DocumentLoaderFactory(settings)

    rows: list[dict] = []
    failures: list[dict] = []
    all_sizes: list[int] = []
    for path, item in factory.iter_documents(targets):
        if isinstance(item, Exception):
            failures.append({"path": str(path), "error": f"{type(item).__name__}: {item}"})
            continue
        chunks = splitter.split(item)
        child_sizes = [len(c.text) for c in chunks]
        all_sizes.extend(child_sizes)
        parents = {c.parent_id for c in chunks}
        rows.append(
            {
                "file": Path(item.source).name,
                "doc_type": item.doc_type,
                "subject": item.subject or "",
                "chars": len(item.text),
                "parents": len(parents),
                "children": len(chunks),
                "min": min(child_sizes, default=0),
                "p50": int(statistics.median(child_sizes)) if child_sizes else 0,
                "p95": _percentile(child_sizes, 0.95),
                "max": max(child_sizes, default=0),
                "no_heading": sum(1 for c in chunks if not c.heading),
                "avg_children_per_parent": round(len(chunks) / len(parents), 2) if parents else 0.0,
            }
        )

    if not rows:
        print("!! 没有可切分的文档，检查路径是否正确")
        return 1

    header = f"{'文件':<34}{'类型':<12}{'字数':>8}{'父块':>6}{'子块':>6}{'min':>5}{'p50':>5}{'p95':>5}{'max':>6}{'无标题':>7}"
    print("== 切分明细 ==")
    print(header)
    for r in sorted(rows, key=lambda x: -x["children"]):
        print(
            f"{r['file'][:32]:<34}{r['doc_type']:<12}{r['chars']:>8}{r['parents']:>6}{r['children']:>6}"
            f"{r['min']:>5}{r['p50']:>5}{r['p95']:>5}{r['max']:>6}{r['no_heading']:>7}"
        )

    total_children = sum(r["children"] for r in rows)
    total_parents = sum(r["parents"] for r in rows)
    print("\n== 汇总 ==")
    print(f"文档 {len(rows)} 篇（失败 {len(failures)}）  父块 {total_parents}  子块 {total_children}")
    print(
        f"子块长度（全量 {len(all_sizes)} 块） min/p50/p95/max = "
        f"{min(all_sizes, default=0)}/{_percentile(all_sizes, 0.5)}/"
        f"{_percentile(all_sizes, 0.95)}/{max(all_sizes, default=0)}"
    )
    print(f"无标题子块占比 = {sum(r['no_heading'] for r in rows) / max(1, total_children):.1%}")

    # 三条硬指标：巨块、碎块、无标题块
    too_long = [r["file"] for r in rows if r["max"] > cfg.child_size * 2.2]
    too_short = [r["file"] for r in rows if r["min"] < cfg.min_child_size]
    print(f"超长块（> {int(cfg.child_size * 2.2)} 字）的文档：{too_long or '无'}")
    print(f"过短块（< {cfg.min_child_size} 字）的文档：{too_short or '无'}")

    for f in failures:
        print(f"  [FAIL] {Path(f['path']).name}: {f['error']}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"明细已写入 {out}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
