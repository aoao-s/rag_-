"""按后缀把文件分派给对应加载器（旧项目 `DocumentLoaderFactory` 的收敛版）。

和旧项目的差别：
1. 加载失败**抛异常**并带文件名，由入库脚本汇总成"失败清单"，而不是打一行日志继续；
2. `doc_type` / `subject` 从目录名与文件名推断，注入到文档元数据——检索时按学科过滤靠它；
3. MinerU 客户端**惰性创建**：没有 PDF 的批次根本不读 token，本地跑 md 不需要联网；
4. PDF 有**双解析器**（`RAG_PDF_PARSER=auto|mineru|local`）：云解析失败自动退本地 PyMuPDF，
   并打 `pdf_parser_fallback`；退不了就把两条错误一起抛出，不假装成功。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from base.config import Settings, get_settings
from base.logger import get_logger
from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError
from rag_system.document_loaders.loader_image import ImageLoader
from rag_system.document_loaders.loader_office import DocxLoader, PptxLoader
from rag_system.document_loaders.loader_pdf_local import LocalPdfLoader
from rag_system.document_loaders.loader_pdf_mineru import MinerUClient, PdfLoader
from rag_system.document_loaders.loader_text import SubtitleLoader, TextLoader

log = get_logger(__name__)

# 目录名 → doc_type。命名沿用 datasets 的既有约定，不要求用户改目录
_DOC_TYPE_BY_DIR = {
    "01_textbook": "textbook",
    "02_lesson_plans": "lesson_plan",
    "03_question_bank": "question_bank",
    "04_faq": "faq",
    "05_sft": "other",
    "08_subtitles": "subtitle",
    "09_textbooks": "textbook",
    "docs": "textbook",
}

# 文件名关键词 → doc_type（目录名认不出时的兜底）
_DOC_TYPE_BY_NAME = (
    ("教案", "lesson_plan"),
    ("教学设计", "lesson_plan"),
    ("大纲", "syllabus"),
    ("试卷", "exam"),
    ("真题", "exam"),
    ("习题", "exam"),
    ("字幕", "subtitle"),
)


class DocumentLoaderFactory:
    def __init__(self, settings: Settings, mineru: MinerUClient | None = None) -> None:
        self._settings = settings
        self._mineru = mineru
        self._text = TextLoader()
        self._subtitle = SubtitleLoader()
        self._docx = DocxLoader(ocr_images=settings.rag_ocr_images)
        self._pptx = PptxLoader(ocr_images=settings.rag_ocr_images)
        self._image = ImageLoader()
        self._pdf_local = LocalPdfLoader()
        self._pdf_mode = (settings.rag_pdf_parser or "auto").strip().lower()
        self._loaders: tuple[BaseDocumentLoader, ...] = (
            self._text,
            self._subtitle,
            self._docx,
            self._pptx,
            self._image,
        )

    @classmethod
    def build(cls, settings: Settings | None = None) -> "DocumentLoaderFactory":
        s = settings or get_settings()
        mineru = MinerUClient(
            s.mineru_token,
            base=s.mineru_base,
            model=s.mineru_model,
            poll_interval=s.mineru_poll_interval,
            timeout_s=s.mineru_timeout_s,
            cache_dir=s.mineru_cache_path,
        ) if s.mineru_token else None
        return cls(s, mineru)

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        exts = {".pdf"}
        for loader in self._loaders:
            exts.update(loader.extensions)
        return tuple(sorted(exts))

    # ---- 分派 ----
    def get_loader(self, path: Path) -> BaseDocumentLoader:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._pdf_loader(path)
        for loader in self._loaders:
            if loader.supports(path):
                return loader
        raise LoaderError(f"不支持的文件类型 [{suffix}]：{path}")

    def _pdf_loader(self, path: Path) -> BaseDocumentLoader:
        if self._pdf_mode == "local":
            return self._pdf_local
        if self._pdf_mode == "mineru":
            if self._mineru is None:
                raise LoaderError(f"RAG_PDF_PARSER=mineru 但没有 MINERU_TOKEN：{path.name}")
            return PdfLoader(self._mineru)
        # auto：有 token 就先云端，没有就直接本地
        if self._mineru is None:
            log.info("pdf_parser_local", reason="no_mineru_token", file=path.name)
            return self._pdf_local
        return _MinerUWithLocalFallback(
            PdfLoader(self._mineru), self._pdf_local, path_name=path.name
        )

    def is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    # ---- 批量 ----
    def iter_documents(self, targets: list[Path]) -> Iterator[tuple[Path, LoadedDoc | Exception]]:
        """逐个产出 (路径, 文档或异常)：失败不打断整批，由调用方决定怎么记账。"""
        for target in targets:
            files = self._expand(target)
            for f in files:
                try:
                    doc = self._load_one(f)
                except Exception as exc:  # noqa: BLE001 - 失败要记到清单里，不能静默跳过
                    yield f, exc
                    continue
                yield f, doc

    def _expand(self, target: Path) -> list[Path]:
        if target.is_file():
            return [target] if self.is_supported(target) else []
        if not target.exists():
            return []
        return sorted(p for p in target.rglob("*") if p.is_file() and self.is_supported(p))

    def _load_one(self, path: Path) -> LoadedDoc:
        loader = self.get_loader(path)
        doc = loader.load(path)
        doc.doc_type = _doc_type(path, doc.doc_type)
        if not doc.title:
            doc.title = path.stem
        return doc


class _MinerUWithLocalFallback(BaseDocumentLoader):
    """先云解析，失败退本地；两次都失败时把两个原因一起抛出。

    为什么要单独一个类：`iter_documents` 只把整个文件的成败记一次，
    降级逻辑必须包在"一次 load"里面，才能保证同一文件不会既算成功又算失败。
    """

    extensions = (".pdf",)

    def __init__(self, cloud: BaseDocumentLoader, local: BaseDocumentLoader, *, path_name: str) -> None:
        self._cloud = cloud
        self._local = local
        self._name = path_name

    def load(self, path: str | Path) -> LoadedDoc:
        try:
            return self._cloud.load(path)
        except Exception as cloud_exc:  # noqa: BLE001 - 任何云侧失败都退本地
            log.warning(
                "pdf_parser_fallback",
                file=self._name,
                reason=f"{type(cloud_exc).__name__}: {cloud_exc}",
                fallback="pymupdf",
            )
            try:
                doc = self._local.load(path)
            except Exception as local_exc:  # noqa: BLE001
                raise LoaderError(
                    f"PDF 云解析与本地解析都失败：{self._name}；"
                    f"mineru={type(cloud_exc).__name__}: {cloud_exc}；"
                    f"local={type(local_exc).__name__}: {local_exc}"
                ) from local_exc
            doc.meta["parser_fallback"] = "pymupdf"
            doc.meta["mineru_error"] = f"{type(cloud_exc).__name__}: {cloud_exc}"
            return doc


def _doc_type(path: Path, default: str) -> str:
    for part in reversed(path.parts[:-1]):
        if part in _DOC_TYPE_BY_DIR:
            return _DOC_TYPE_BY_DIR[part]
    name = path.name
    for keyword, doc_type in _DOC_TYPE_BY_NAME:
        if keyword in name:
            return doc_type
    return default
