"""临时脚本：重建前试算——按 doc_type 统计文件数与子块数，并核对 stage/grade/subject。

不属于交付物，跑完即删。dry-run 不连 Milvus、不做向量化，只为回答"这次重建要写多少块"。
为了不与正在跑的 MinerU 预热抢网络，跳过尚未缓存的 PDF。
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.config import get_settings
from base.logger import configure
from rag_system.document_loaders.loader_pdf_mineru import content_hash
from rag_system.document_loaders.thy_factory import DocumentLoaderFactory
from rag_system.ingest import IngestPipeline

settings = get_settings()
configure("WARNING", log_path=settings.log_path)

factory = DocumentLoaderFactory.build()
cache = Path("data/mineru")

files = [p for p in sorted(settings.rag_docs_paths[0].rglob("*")) if p.is_file() and factory.is_supported(p)]
skipped: list[str] = []
ready: list[Path] = []
for p in files:
    if p.suffix.lower() == ".pdf" and not (cache / content_hash(p) / "content.md").exists():
        skipped.append(p.name)
        continue
    ready.append(p)

print(f"支持文件 {len(files)}：待解析 {len(ready)}，跳过未缓存 PDF {len(skipped)}", flush=True)

# dedup 走全量：近重复是"这批要入多少块"的一部分，试算里不能被绕过
pipe = IngestPipeline(settings)
report = pipe.run(ready, dry_run=True)

by_type: Counter = Counter()
chunks_by_type: Counter = Counter()
by_subject: Counter = Counter()
by_stage: Counter = Counter()
by_grade: Counter = Counter()
per_doc: list[tuple[str, int, str]] = []

# 文件级明细直接从 report.results 拿不到 doc_type，重新按 loader 的元数据归一次
for p in ready:
    try:
        doc = factory._load_one(p)  # noqa: SLF001 - 临时脚本
    except Exception as exc:  # noqa: BLE001
        print(f"  [PARSE-FAIL] {p.name}: {type(exc).__name__}: {exc}")
        continue
    by_type[doc.doc_type or "(空)"] += 1
    by_subject[doc.subject or "(空)"] += 1
    by_stage[doc.stage or "(空)"] += 1
    by_grade[doc.grade or "(空)"] += 1
    per_doc.append((str(p.relative_to(settings.rag_docs_paths[0])), len(doc.text), doc.doc_type or ""))


def _show(title: str, c: Counter) -> None:
    print(f"\n-- {title} --")
    for k, n in c.most_common():
        print(f"   {k:<16} {n:>4}")


print("\n==== 试算结果 ====")
print(f"文件 {report.files}（成功 {report.docs_ok} / 失败 {report.docs_failed} / 近重复 {report.duplicates}）")
print(f"子块合计 {report.chunks}，文件字符合计 {sum(c for _, c, _ in per_doc)}")
_show("doc_type（文件数）", by_type)
_show("subject（文件数）", by_subject)
_show("stage（文件数）", by_stage)
_show("grade（文件数）", by_grade)

print("\n-- 最大的 10 个文件（字符数）--")
for name, chars, dt in sorted(per_doc, key=lambda x: -x[1])[:10]:
    print(f"   {chars:>8}  {dt:<14} {name}")

if report.duplicates:
    print("\n-- 近重复被拒 --")
    for r in report.results:
        if r.status == "duplicate":
            print(f"   {Path(r.path).name}")
