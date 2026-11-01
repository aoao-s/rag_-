"""纯文本 / Markdown / SRT 字幕加载器。

三者可以共用一条读取路径（都是 UTF-8/GBK 文本），差别只在后处理：
- Markdown：保留 `#` 标题，切分器靠它找语义边界，所以这里不能"清洗"；
- 纯文本：原样；
- SRT：去掉序号与时间轴，保留台词（时间轴对回答没用，还会污染向量）。
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_system.document_loaders.base import LoadedDoc, BaseDocumentLoader, guess_subject, read_text

_SRT_TIME = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->")
_SRT_INDEX = re.compile(r"^\d+$")
_FRONT_MATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_front_matter(text: str) -> str:
    """去掉 Markdown 头部的 YAML Front Matter（Hugo/MkDocs 常带）。"""
    return _FRONT_MATTER.sub("", text, count=1)


class TextLoader(BaseDocumentLoader):
    """`.txt` / `.md` 通用加载器。"""

    extensions = (".txt", ".md", ".markdown")

    def __init__(self, doc_type: str = "other") -> None:
        self.doc_type = doc_type

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path).resolve()
        raw = read_text(p)
        if p.suffix.lower() in (".md", ".markdown"):
            raw = strip_front_matter(raw)
        text = raw.strip()
        if not text:
            raise ValueError(f"文件内容为空：{p}")
        return LoadedDoc(
            text=text,
            source=str(p),
            doc_type=self.doc_type,
            subject=guess_subject(str(p)),
            title=p.stem,
            meta={"suffix": p.suffix.lower(), "chars": len(text)},
        )


class SubtitleLoader(BaseDocumentLoader):
    """SRT 字幕 → 纯台词。"""

    extensions = (".srt",)

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path).resolve()
        lines: list[str] = []
        for line in read_text(p).splitlines():
            s = line.strip()
            if not s or _SRT_INDEX.match(s) or _SRT_TIME.match(s):
                continue
            lines.append(s)
        text = "\n".join(lines)
        if not text:
            raise ValueError(f"字幕没有可用台词：{p}")
        return LoadedDoc(
            text=text,
            source=str(p),
            doc_type="subtitle",
            subject=guess_subject(str(p)),
            title=p.stem,
            meta={"suffix": ".srt", "chars": len(text)},
        )
