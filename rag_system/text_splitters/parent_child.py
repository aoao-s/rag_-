"""父块-子块（parent-child）切分。

为什么不是"整篇按 300 字等分"：
- 检索命中的是短块，但生成要连贯上下文。小块进模型会把公式/表格切成两半，
  大块进模型则检索不准——所以同一个知识点要两份：**子块用来命中，父块用来回答**；
- 本文件的子块会带 `parent_text` 一起入库，检索后直接拿父块拼 prompt，
  不需要第二次查库（代价是存储冗余，教材量级完全可接受）。

切分顺序：Markdown 标题 → 段落 → 句子，逐级降级；同级里按字数装箱，
超长段落才真的按字数硬切。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from rag_system.document_loaders.base import LoadedDoc

# 句子边界：中文句号/问号/感叹号/分号，英文同样；换行也算（教材一行一句很常见）
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;\n])")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(slots=True)
class Chunk:
    """一个可检索单元（子块）+ 它的父块正文。"""

    chunk_id: str
    doc_id: str
    parent_id: str
    text: str
    parent_text: str
    heading: str = ""
    index: int = 0
    meta: dict = field(default_factory=dict)


@dataclass(slots=True)
class SplitterConfig:
    parent_size: int = 1200
    parent_overlap: int = 100
    child_size: int = 300
    child_overlap: int = 20
    # 低于这个长度的子块并入前一块，避免"一句话一块"污染索引
    min_child_size: int = 40


def split_sentences(text: str) -> list[str]:
    """按标点切句，保留标点；空行单独保留（段落边界）。"""
    parts = [p for p in _SENT_SPLIT.split(text) if p]
    return [p for p in parts if p.strip()]


def _pack(sentences: list[str], size: int, overlap: int, min_size: int) -> list[str]:
    """把句子贪婪装箱到 size 字符；相邻块之间留 overlap 字符的句子重叠。"""
    if not sentences:
        return []
    chunks: list[str] = []
    buf: list[str] = []
    length = 0
    for sent in sentences:
        if length + len(sent) > size and buf:
            chunks.append("".join(buf).strip())
            # 从尾部回退若干句作为重叠，保证跨块的问题在两边都能命中
            back: list[str] = []
            back_len = 0
            for s in reversed(buf):
                if back_len + len(s) > overlap:
                    break
                back.append(s)
                back_len += len(s)
            buf = list(reversed(back))
            length = sum(len(s) for s in buf)
        buf.append(sent)
        length += len(sent)
    if buf:
        tail = "".join(buf).strip()
        if tail:
            chunks.append(tail)

    # 太短的尾块并入上一块：单独成块会让 BM25/向量的分数被"半句话"稀释
    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < min_size:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    return [c for c in merged if c.strip()]


def _sections(markdown: str) -> list[tuple[str, str]]:
    """按标题切段，返回 [(标题路径, 正文)]。

    标题路径只保留最近两级（"三、函数 > 3.2 二次函数"），全路径会把元数据撑爆。
    """
    lines = markdown.splitlines()
    stack: list[tuple[int, str]] = []
    title = ""
    buf: list[str] = []
    out: list[tuple[str, str]] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            out.append((title, body))

    for line in lines:
        m = _HEADING.match(line.strip())
        if m:
            flush()
            buf = []
            level = len(m.group(1))
            stack = [s for s in stack if s[0] < level]
            stack.append((level, m.group(2).strip()))
            title = " > ".join(t for _, t in stack[-2:])
        else:
            buf.append(line)
    flush()
    return out or [("", markdown)]


class ParentChildSplitter:
    """入口：`split(doc) -> list[Chunk]`。"""

    def __init__(self, config: SplitterConfig | None = None) -> None:
        self.cfg = config or SplitterConfig()

    def split(self, doc: LoadedDoc) -> list[Chunk]:
        doc_id = doc_id_for(doc)
        parents: list[tuple[str, str]] = []  # (heading, parent_text)
        for heading, body in _sections(doc.text):
            for block in _pack(split_sentences(body), self.cfg.parent_size, self.cfg.parent_overlap, 0):
                parents.append((heading, block))

        chunks: list[Chunk] = []
        for pi, (heading, parent_text) in enumerate(parents):
            parent_id = f"{doc_id}:p{pi}"
            children = _pack(
                split_sentences(parent_text),
                self.cfg.child_size,
                self.cfg.child_overlap,
                self.cfg.min_child_size,
            )
            for ci, child in enumerate(children):
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent_id}:c{ci}",
                        doc_id=doc_id,
                        parent_id=parent_id,
                        text=child,
                        parent_text=parent_text,
                        heading=heading,
                        index=len(chunks),
                        meta={"doc_type": doc.doc_type, "title": doc.title},
                    )
                )
        return chunks


def doc_id_for(doc: LoadedDoc) -> str:
    """文档身份 = 路径 + 正文哈希。

    正文变了就换 id，旧块靠 `source` 删除——只按路径取 id 会让"改了内容但 id 不变"
    的旧向量留在库里，改完反而查不到新内容。
    """
    h = hashlib.sha1(f"{doc.source}|{doc.text[:2000]}|{len(doc.text)}".encode()).hexdigest()[:12]
    return f"{hashlib.sha1(doc.source.encode()).hexdigest()[:8]}_{h}"
