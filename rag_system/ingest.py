"""入库流水线：文件 → 加载 → 父块/子块切分 → 向量化 → Milvus。

入口（也是运维脚本）：

    python -m rag_system.ingest                        # 用 RAG_DOCS_DIR 配置的目录
    python -m rag_system.ingest --paths datasets/01_textbook --pdf datasets/09_textbooks/x.pdf
    python -m rag_system.ingest --recreate --dry-run   # 只切分不入库，看块数与耗时

幂等策略：每个文件入库前按 `source` 删旧块。这样"改了一篇教案再跑一次"不会留下
两份互相矛盾的旧块，也不需要在 Milvus 里维护版本表。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from base.config import BASE_DIR, Settings, get_settings
from base.logger import configure, get_logger
from rag_system.document_loaders.base import LoadedDoc
from rag_system.document_loaders.thy_factory import DocumentLoaderFactory
from rag_system.embedding.bge_m3 import build_embedder
from rag_system.store.milvus_store import ChunkStore
from rag_system.text_splitters.parent_child import ParentChildSplitter, SplitterConfig

log = get_logger(__name__)


@dataclass(slots=True)
class FileResult:
    path: str
    status: str  # ok / failed / skipped
    chunks: int = 0
    chars: int = 0
    cost_ms: int = 0
    error: str = ""


@dataclass(slots=True)
class IngestReport:
    files: int = 0
    docs_ok: int = 0
    docs_failed: int = 0
    chunks: int = 0
    written: int = 0
    deleted: int = 0
    elapsed_ms: int = 0
    results: list[FileResult] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "files": self.files,
            "docs_ok": self.docs_ok,
            "docs_failed": self.docs_failed,
            "chunks": self.chunks,
            "written": self.written,
            "deleted": self.deleted,
            "elapsed_ms": self.elapsed_ms,
            "failures": [{"path": r.path, "error": r.error} for r in self.results if r.status == "failed"],
        }


class IngestPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: ChunkStore | None = None,
        embedder=None,
        factory: DocumentLoaderFactory | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        s = self.settings
        self.splitter = ParentChildSplitter(
            SplitterConfig(
                parent_size=s.rag_parent_size,
                parent_overlap=s.rag_parent_overlap,
                child_size=s.rag_child_size,
                child_overlap=s.rag_child_overlap,
            )
        )
        self._store = store
        self._embedder = embedder
        self._factory = factory or DocumentLoaderFactory.build(s)

    # 惰性创建：dry-run 不连 Milvus，也不加载 2GB 模型
    @property
    def store(self) -> ChunkStore:
        if self._store is None:
            self._store = ChunkStore(self.settings)
        return self._store

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = build_embedder(self.settings)
        return self._embedder

    def run(
        self,
        targets: list[Path],
        *,
        recreate: bool = False,
        dry_run: bool = False,
        limit: int | None = None,
    ) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport()
        if not dry_run:
            self.store.ensure_collection(recreate=recreate)

        for path, item in self._factory.iter_documents(targets):
            if limit is not None and report.files >= limit:
                break
            report.files += 1
            if isinstance(item, Exception):
                report.docs_failed += 1
                report.results.append(FileResult(str(path), "failed", error=f"{type(item).__name__}: {item}"))
                log.warning("ingest_file_failed", path=str(path), error=str(item))
                continue
            report.results.append(self._ingest_one(item, report, dry_run=dry_run))

        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        log.info("ingest_done", **report.summary())
        return report

    def _ingest_one(self, doc: LoadedDoc, report: IngestReport, *, dry_run: bool) -> FileResult:
        t0 = time.perf_counter()
        try:
            chunks = self.splitter.split(doc)
            if len(chunks) > self.settings.rag_max_chunks_per_doc:
                chunks = chunks[: self.settings.rag_max_chunks_per_doc]
                log.warning(
                    "ingest_truncated",
                    path=doc.source,
                    limit=self.settings.rag_max_chunks_per_doc,
                )
            if not chunks:
                report.docs_ok += 1
                return FileResult(doc.source, "skipped", error="切分后为空")
            if dry_run:
                report.docs_ok += 1
                report.chunks += len(chunks)
                return FileResult(doc.source, "ok", len(chunks), len(doc.text), int((time.perf_counter() - t0) * 1000))

            deleted = self.store.delete_by_source(doc.source)
            report.deleted += deleted
            dense, sparse = self.embedder.encode([c.text for c in chunks])
            written = self.store.upsert(
                chunks,
                dense,
                sparse,
                source=doc.source,
                subject=doc.subject,
                doc_type=doc.doc_type,
                title=doc.title,
                embedder=self.embedder.name,
            )
            report.docs_ok += 1
            report.chunks += len(chunks)
            report.written += written
            cost = int((time.perf_counter() - t0) * 1000)
            log.info(
                "ingest_file_ok",
                path=doc.source,
                chunks=len(chunks),
                chars=len(doc.text),
                written=written,
                cost_ms=cost,
            )
            return FileResult(doc.source, "ok", len(chunks), len(doc.text), cost)
        except Exception as exc:  # noqa: BLE001 - 单文件失败不拖垮整批
            report.docs_failed += 1
            log.error("ingest_file_error", path=doc.source, error=str(exc))
            return FileResult(doc.source, "failed", error=f"{type(exc).__name__}: {exc}")

    def write_manifest(self, report: IngestReport, path: Path | None = None) -> Path:
        """上次入库的快照，用于排查"某篇为什么没进来"。"""
        out = path or (BASE_DIR / "data" / "ingest_manifest.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), **report.summary()}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="EduRAG 文档切分与入库")
    p.add_argument("--paths", nargs="*", default=[], help="文件或目录，可多个")
    p.add_argument("--pdf", nargs="*", default=[], help="额外的 PDF（走 MinerU）")
    p.add_argument("--recreate", action="store_true", help="先删集合再重建（全量重灌）")
    p.add_argument("--dry-run", action="store_true", help="只切分不向量化、不入库")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件（试探用）")
    p.add_argument("--defaults", action="store_true", help=f"使用 RAG_DOCS_DIR（{s.rag_docs_dir}）")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    configure(
        settings.log_level,
        log_path=settings.log_path,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
        file_level=settings.log_file_level,
    )
    targets = [Path(p) for p in args.paths] + [Path(p) for p in args.pdf]
    if args.defaults or not targets:
        targets = settings.rag_docs_paths
    targets = [t if t.is_absolute() else (BASE_DIR / t) for t in targets]

    pipeline = IngestPipeline(settings)
    report = pipeline.run(targets, recreate=args.recreate, dry_run=args.dry_run, limit=args.limit)
    manifest = pipeline.write_manifest(report)

    print("\n==== 入库结果 ====")
    print(f"文件 {report.files}（成功 {report.docs_ok} / 失败 {report.docs_failed}）")
    print(f"子块 {report.chunks}，写入 {report.written}，删除旧块 {report.deleted}")
    print(f"耗时 {report.elapsed_ms} ms" + ("（dry-run，未写库）" if args.dry_run else ""))
    print(f"清单 {manifest}")
    for r in report.results:
        if r.status == "failed":
            print(f"  [FAIL] {Path(r.path).name}: {r.error}")
    return 0 if report.docs_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
