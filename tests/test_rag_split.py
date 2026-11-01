"""加载器与切分器的单测：不连库、不下载模型，纯函数级验证。

覆盖的三件事：
1. 编码兜底（GBK 文件不抛、乱码不进库）；
2. 字幕清洗（时间轴与序号不能出现在正文里，否则会污染向量）；
3. 父块/子块关系（每个子块必须有父块正文、父块字符数不超过上限的 1.2 倍）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_system.document_loaders.base import LoadedDoc
from rag_system.document_loaders.loader_text import SubtitleLoader, TextLoader, strip_front_matter
from rag_system.text_splitters.parent_child import ParentChildSplitter, SplitterConfig, split_sentences


def test_text_loader_gbk(tmp_path: Path) -> None:
    p = tmp_path / "数学-九年级-二次函数.md"
    p.write_text("# 二次函数\n顶点式 y=a(x-h)^2+k。", encoding="gbk")
    doc = TextLoader().load(p)
    assert "二次函数" in doc.text
    assert doc.subject == "数学"
    assert doc.doc_type == "other"  # 目录推断交给工厂，loader 不猜


def test_text_loader_strips_front_matter(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    p.write_text("---\ntitle: x\n---\n\n正文内容。\n", encoding="utf-8")
    assert strip_front_matter(p.read_text(encoding="utf-8")).strip() == "正文内容。"


def test_subtitle_loader_drops_timeline(tmp_path: Path) -> None:
    p = tmp_path / "数学-二次函数入门-字幕.srt"
    p.write_text(
        "1\n00:00:01,000 --> 00:00:03,000\n今天讲二次函数\n\n"
        "2\n00:00:03,500 --> 00:00:06,000\n先看顶点式\n",
        encoding="utf-8",
    )
    doc = SubtitleLoader().load(p)
    assert "00:00:01" not in doc.text
    assert "今天讲二次函数" in doc.text
    assert doc.doc_type == "subtitle"
    assert doc.subject == "数学"


def test_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.md"
    p.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        TextLoader().load(p)


def test_split_sentences_keeps_punct() -> None:
    parts = split_sentences("第一句。第二句！第三句？")
    assert parts == ["第一句。", "第二句！", "第三句？"]


def test_parent_child_relationship() -> None:
    doc = LoadedDoc(
        text="# 一、函数\n" + "函数是描述两个变量之间关系的数学对象。" * 40 + "\n## 1.1 顶点式\n" + "顶点式写作 y=a(x-h)^2+k。" * 30,
        source="F:/x/数学-二次函数.md",
        doc_type="textbook",
        subject="数学",
        title="二次函数",
    )
    splitter = ParentChildSplitter(SplitterConfig(parent_size=300, parent_overlap=50, child_size=80, child_overlap=10))
    chunks = splitter.split(doc)
    assert len(chunks) > 3
    parents = {c.parent_id: c.parent_text for c in chunks}
    for c in chunks:
        assert c.chunk_id.startswith(c.parent_id + ":c")
        assert c.parent_text == parents[c.parent_id]
        assert len(c.text) <= 80 + 40  # 装箱不会无限超长
        assert c.parent_id.startswith(c.doc_id)
    # 子块编号在父块内从 0 递增，父块之间各自重置
    from itertools import groupby

    for _pid, group in groupby(chunks, key=lambda c: c.parent_id):
        idx = [int(c.chunk_id.rsplit(":c", 1)[1]) for c in group]
        assert idx == list(range(len(idx)))


def test_child_block_inherits_heading() -> None:
    doc = LoadedDoc(text="# 一、函数\n函数定义。\n## 1.2 图像\n图像是抛物线。", source="F:/x/a.md", title="a")
    chunks = ParentChildSplitter(SplitterConfig(parent_size=200, child_size=100)).split(doc)
    assert any("1.2 图像" in c.heading for c in chunks)


def test_unpunctuated_block_is_hard_split() -> None:
    """没有标点的长串（MinerU 的表格行、公式行）必须被硬切到 size 以内。

    2026-10-29 实测：教材 PDF 经 MinerU 解析后切出过 **12 140 字**的子块，
    被 Milvus 的 `text`（max_length=4096）整份拒收，18 本教材入库全失败。
    """
    body = "|" + "1" * 1200 + "|"  # 一句 1202 字，没有任何句末标点
    doc = LoadedDoc(text="# 一、函数\n" + body, source="F:/x/a.md", title="a")
    chunks = ParentChildSplitter(
        SplitterConfig(parent_size=300, parent_overlap=30, child_size=80, child_overlap=10)
    ).split(doc)
    assert chunks
    for c in chunks:
        assert len(c.text) <= 80 + 40
        assert len(c.parent_text) <= 300 + 40


def test_hard_split_keeps_overlap_and_length() -> None:
    from rag_system.text_splitters.parent_child import _hard_split

    pieces = _hard_split("x" * 250, size=100, overlap=20)
    assert [len(p) for p in pieces] == [100, 100, 90]
    assert pieces[0][-20:] == pieces[1][:20]  # 相邻片段真的重叠
    assert _hard_split("短句。", size=100, overlap=20) == ["短句。"]


def test_hard_split_survives_overlap_larger_than_size() -> None:
    """overlap ≥ size 时不能原地打转（step 会退化成 0，循环写不完）。"""
    from rag_system.text_splitters.parent_child import _hard_split

    pieces = _hard_split("y" * 100, size=10, overlap=50)
    assert len("".join(pieces)) > 0
    assert "".join(pieces).startswith("y" * 10)
