"""入库流水线：文件 → 加载 → 父块/子块切分 → 向量化 → Milvus。

入口（也是运维脚本）：

    python -m rag_system.ingest                        # 用 RAG_DOCS_DIR 配置的目录
    python -m rag_system.ingest --paths datasets/01_textbook --pdf datasets/09_textbooks/x.pdf
    python -m rag_system.ingest --recreate --dry-run   # 只切分不入库，看块数与耗时
    python -m rag_system.ingest --exclude 题库.json     # 分两趟：先教材/教案/试卷，题库单独一趟

幂等策略：每个文件入库前按 `source` 删旧块。这样"改了一篇教案再跑一次"不会留下
两份互相矛盾的旧块，也不需要在 Milvus 里维护版本表。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from base.config import BASE_DIR, Settings, get_settings
from base.logger import configure, get_logger
from rag_system.document_loaders.base import LoadedDoc
from rag_system.document_loaders.dedup import DedupIndex
from rag_system.document_loaders.thy_factory import DocumentLoaderFactory
from rag_system.embedding.bge_m3 import build_embedder
from rag_system.store.milvus_store import ChunkStore
from rag_system.text_splitters.parent_child import ParentChildSplitter, SplitterConfig

log = get_logger(__name__)

# 文档指纹在磁盘上的位置：跨批次共用，增量入库也能拦住上一批里已经进过的那份
FINGERPRINT_PATH = BASE_DIR / "data" / "doc_fingerprints.json"


@dataclass(slots=True)
class FileResult:
    path: str
    status: str  # ok / failed / skipped / duplicate
    chunks: int = 0
    chars: int = 0
    cost_ms: int = 0
    error: str = ""


@dataclass(slots=True)
class IngestReport:
    tenant_id: str = ""
    kb_version: str = ""
    acl_tags: list[str] = field(default_factory=list)
    files: int = 0
    docs_ok: int = 0
    docs_failed: int = 0
    duplicates: int = 0
    chunks: int = 0
    written: int = 0
    deleted: int = 0
    elapsed_ms: int = 0
    results: list[FileResult] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "kb_version": self.kb_version,
            "acl_tags": list(self.acl_tags),
            "files": self.files,
            "docs_ok": self.docs_ok,
            "docs_failed": self.docs_failed,
            # 近重复被拒的条数与明细：静默跳过等于"库里少了两本书"却没人知道
            "duplicates": self.duplicates,
            "chunks": self.chunks,
            "written": self.written,
            "deleted": self.deleted,
            "elapsed_ms": self.elapsed_ms,
            "failures": [{"path": r.path, "error": r.error} for r in self.results if r.status == "failed"],
            "skipped_duplicates": [
                {"path": r.path, "reason": r.error} for r in self.results if r.status == "duplicate"
            ],
        }


class IngestPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: ChunkStore | None = None,
        embedder=None,
        factory: DocumentLoaderFactory | None = None,
        dedup: DedupIndex | None = None,
        fingerprint_path: Path | None = None,
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
        self._dedup = dedup
        self.fingerprint_path = fingerprint_path or FINGERPRINT_PATH

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

    @property
    def dedup(self) -> DedupIndex:
        """查重表：首次访问时按配置建好并灌入历史指纹。

        历史指纹很重要——增量入库一次只跑几个文件，不读盘就拦不住上一批已经进过的那份。
        指纹文件损坏时 `load()` 返回 0（当空表），宁可漏拦也不能让入库整体失败。
        """
        if self._dedup is None:
            s = self.settings
            self._dedup = DedupIndex(
                enabled=s.rag_dedup_enabled,
                distance=s.rag_dedup_distance,
                prefix_chars=s.rag_dedup_prefix_chars,
            )
            self._dedup.load(self.fingerprint_path)
        return self._dedup

    def run(
        self,
        targets: list[Path],
        *,
        recreate: bool = False,
        dry_run: bool = False,
        limit: int | None = None,
        allow_duplicate: bool = False,
        exclude: tuple[str, ...] = (),
        tenant_id: str | None = None,
        kb_version: str | None = None,
        acl_tags: tuple[str, ...] | list[str] = (),
    ) -> IngestReport:
        started = time.perf_counter()
        tenant_id = (tenant_id if tenant_id is not None else self.settings.tenant_id).strip()
        kb_version = (kb_version if kb_version is not None else self.settings.kb_version).strip()
        acl_tags = tuple(dict.fromkeys(str(tag).strip() for tag in acl_tags if str(tag).strip()))
        if not tenant_id:
            raise ValueError("tenant_id 不能为空；请通过 --tenant 或 TENANT_ID 指定")
        if not kb_version:
            raise ValueError("kb_version 不能为空；请通过 --kb-version 或 KB_VERSION 指定")
        if len(tenant_id.encode("utf-8")) > 64:
            raise ValueError("tenant_id UTF-8 字节长度不能超过 64")
        if len(kb_version.encode("utf-8")) > 32:
            raise ValueError("kb_version UTF-8 字节长度不能超过 32")
        if len(acl_tags) > 16 or any(len(tag.encode("utf-8")) > 64 for tag in acl_tags):
            raise ValueError("ACL 标签最多 16 个，且每个标签 UTF-8 字节长度不能超过 64")
        report = IngestReport(tenant_id=tenant_id, kb_version=kb_version, acl_tags=list(acl_tags))
        log.info("ingest_scope", tenant_id=tenant_id, kb_version=kb_version, acl_tags=list(acl_tags))
        if not dry_run:
            self.store.ensure_collection(recreate=recreate)

        for path, item in self._factory.iter_documents(targets):
            if limit is not None and report.files >= limit:
                break
            # 按文件名排除：分两趟入库时用（题库体积占七成，先让教材/教案/试卷可用）
            if exclude and any(fnmatch(Path(path).name, pat) for pat in exclude):
                continue
            report.files += 1
            if isinstance(item, Exception):
                report.docs_failed += 1
                report.results.append(FileResult(str(path), "failed", error=f"{type(item).__name__}: {item}"))
                log.warning("ingest_file_failed", path=str(path), error=str(item))
                continue
            report.results.append(
                self._ingest_one(
                    item,
                    report,
                    dry_run=dry_run,
                    allow_duplicate=allow_duplicate,
                    tenant_id=tenant_id,
                    kb_version=kb_version,
                    acl_tags=acl_tags,
                )
            )

        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        if not dry_run and self.settings.rag_dedup_enabled:
            try:
                saved = self.dedup.save(self.fingerprint_path)
                log.info("ingest_fingerprints_saved", count=len(self.dedup), path=str(saved))
            except OSError as exc:  # 指纹存不下不影响本次入库，只是下次拦不住
                log.warning("ingest_file_failed", path=str(self.fingerprint_path), error=f"指纹保存失败: {exc}")
        log.info("ingest_done", **report.summary())
        return report

    def _ingest_one(
        self,
        doc: LoadedDoc,
        report: IngestReport,
        *,
        dry_run: bool,
        allow_duplicate: bool = False,
        tenant_id: str,
        kb_version: str,
        acl_tags: tuple[str, ...],
    ) -> FileResult:
        t0 = time.perf_counter()
        try:
            # 查重放在切分之前：被拒的文档不该付切分与向量化的代价
            if not allow_duplicate:
                duplicate = self.dedup.check_and_add(
                    doc.source, doc.text, scope=f"{tenant_id}\x1f{kb_version}"
                )
                if duplicate is not None:
                    report.duplicates += 1
                    reason = f"doc_duplicate: 与 {duplicate.kept_source} 近重复（{duplicate.reason}）"
                    log.warning(
                        "ingest_duplicate",
                        path=doc.source,
                        kept_source=duplicate.kept_source,
                        reason=duplicate.reason,
                        distance=duplicate.distance,
                    )
                    return FileResult(doc.source, "duplicate", chars=len(doc.text), error=reason)

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

            deleted = self.store.delete_by_source(
                doc.source, tenant_id=tenant_id, kb_version=kb_version
            )
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
                stage=doc.stage,
                grade=doc.grade,
                embedder=self.embedder.name,
                tenant_id=tenant_id,
                kb_version=kb_version,
                acl_tags=acl_tags,
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
    p.add_argument("--tenant", default=None, help=f"租户 ID（默认 TENANT_ID={s.tenant_id}）")
    p.add_argument("--kb-version", default=None, help=f"知识库版本（默认 KB_VERSION={s.kb_version}）")
    p.add_argument(
        "--acl",
        nargs="*",
        default=None,
        help="本批次文档的 ACL 标签；不传或传空表示公共文档，例如 --acl teacher admin",
    )
    p.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="关闭近重复检测（对照实验用；正常入库不要开）",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="按文件名 glob 跳过（如 --exclude 题库.json），分两趟入库时用",
    )
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
    report = pipeline.run(
        targets,
        recreate=args.recreate,
        dry_run=args.dry_run,
        limit=args.limit,
        allow_duplicate=args.allow_duplicate,
        exclude=tuple(args.exclude),
        tenant_id=args.tenant,
        kb_version=args.kb_version,
        acl_tags=tuple(args.acl or ()),
    )
    manifest = pipeline.write_manifest(report)

    print("\n==== 入库结果 ====")
    print(f"租户 {report.tenant_id} / KB {report.kb_version} / ACL {report.acl_tags or ['public']}")
    print(f"文件 {report.files}（成功 {report.docs_ok} / 失败 {report.docs_failed} / 近重复跳过 {report.duplicates}）")
    print(f"子块 {report.chunks}，写入 {report.written}，删除旧块 {report.deleted}")
    print(f"耗时 {report.elapsed_ms} ms" + ("（dry-run，未写库）" if args.dry_run else ""))
    print(f"清单 {manifest}")
    for r in report.results:
        if r.status == "failed":
            print(f"  [FAIL] {Path(r.path).name}: {r.error}")
        elif r.status == "duplicate":
            # 近重复要显式打出来：它是"数据没问题但库少了一本书"的唯一现场
            print(f"  [DUP ] {Path(r.path).name}: {r.error}")
    return 0 if report.docs_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
