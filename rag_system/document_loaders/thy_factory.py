"""按后缀把文件分派给对应加载器（旧项目 `DocumentLoaderFactory` 的收敛版）。

和旧项目的差别：
1. 加载失败**抛异常**并带文件名，由入库脚本汇总成"失败清单"，而不是打一行日志继续；
2. `doc_type` / `subject` 从目录名与文件名推断，注入到文档元数据——检索时按学科过滤靠它；
3. MinerU 客户端**惰性创建**：没有 PDF 的批次根本不读 token，本地跑 md 不需要联网。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from base.config import Settings, get_settings
from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError
from rag_system.document_loaders.loader_pdf_mineru import MinerUClient, PdfLoader
from rag_system.document_loaders.loader_text import SubtitleLoader, TextLoader

# 目录名 → doc_type。命名沿用 datasets 的既有约定，不要求用户改目录
_DOC_TYPE_BY_DIR = {
    "01_textbook": "textbook",
    "02_lesson_plans": "lesson_plan",
    "03_question_bank": "question_bank",
    "08_subtitles": "subtitle",
    "09_textbooks": "textbook",
    "docs": "textbook",
}


class DocumentLoaderFactory:
    def __init__(self, settings: Settings, mineru: MinerUClient | None = None) -> None:
        self._settings = settings
        self._mineru = mineru
        self._text = TextLoader()
        self._subtitle = SubtitleLoader()

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

    # ---- 分派 ----
    def get_loader(self, path: Path) -> BaseDocumentLoader:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            if self._mineru is None:
                raise LoaderError(f"PDF 需要 MINERU_TOKEN，请先在 .env 配置：{path.name}")
            return PdfLoader(self._mineru)
        if self._subtitle.supports(path):
            return self._subtitle
        if self._text.supports(path):
            return self._text
        raise LoaderError(f"不支持的文件类型 [{suffix}]：{path}")

    def is_supported(self, path: Path) -> bool:
        return path.suffix.lower() in (".pdf", ".txt", ".md", ".markdown", ".srt")

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
            return [target]
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


def _doc_type(path: Path, default: str) -> str:
    for part in reversed(path.parts[:-1]):
        if part in _DOC_TYPE_BY_DIR:
            return _DOC_TYPE_BY_DIR[part]
    name = path.name
    if "教案" in name:
        return "lesson_plan"
    if "大纲" in name:
        return "syllabus"
    return default
