"""加载器基类与统一产物。

为什么不用 langchain 的 BaseLoader：
1. 我们只需要"文件 → 文本 + 元数据"，langchain 的 Document 会带上大量用不到的字段，
   写进 Milvus 还要再剥一层；
2. 旧项目里 `lazy_load` 的 `yield` 一旦前置校验失败就 `return`（生成器里等价于空迭代），
   调用方拿不到任何错误信号——这里改成抛异常，让入库脚本能明确记 failed。

产物 `LoadedDoc` 是"一篇文档"，不是"一个 chunk"：切分在 splitter 里做。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from base.labels import SUBJECTS, SUBJECT_ALIASES


class LoaderError(RuntimeError):
    """加载失败（文件不存在 / 解析超时 / 编码不可识别）。"""


@dataclass(slots=True)
class LoadedDoc:
    """一篇原始文档。

    Attributes:
        text: 正文（PDF 走 MinerU，是 Markdown）。
        source: 绝对路径字符串，入库后作为"删除/覆盖同一篇文档"的依据。
        doc_type: textbook / lesson_plan / subtitle / exam / other，供检索时按类型过滤。
        subject: 学科（数学/物理/...）；识别不出时为 None，不硬猜。
        title: 文档标题，用于引用卡片展示。
        meta: 其余元数据（PDF 的页数、MinerU 的 batch_id 等）。
        stage / grade: 学段（初中/高中）与年级（七年级…高三）。新语料按
            `阶段/学科/年级/` 组织，这两层是**检索过滤与引用定位的必需字段**；
            认不出时留 None，不硬猜（与 `subject` 同一口径）。
    """

    text: str
    source: str
    doc_type: str = "other"
    subject: str | None = None
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    stage: str | None = None
    grade: str | None = None

    @property
    def name(self) -> str:
        return Path(self.source).name


class BaseDocumentLoader:
    """所有加载器的接口：`load(path) -> LoadedDoc`。"""

    extensions: tuple[str, ...] = ()

    def load(self, path: str | Path) -> LoadedDoc:  # pragma: no cover - 抽象方法
        raise NotImplementedError

    def supports(self, path: str | Path) -> bool:
        return Path(path).suffix.lower() in self.extensions


def read_text(path: Path, encodings: Iterable[str] = ("utf-8-sig", "utf-8", "gbk", "gb18030")) -> str:
    """按候选编码依次尝试；全部失败才抛错。

    旧项目用 `errors="ignore"` 兜底，会把乱码静默吞掉——对 RAG 是灾难（脏 chunk 进了库
    还查不出来），这里宁可失败。
    """
    last_error: Exception | None = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = exc
    raise LoaderError(f"无法识别文件编码 {path}：{last_error}")


def guess_subject(name: str, subjects: Iterable[str] | None = None) -> str | None:
    """从文件名/路径里认出学科，返回 `SUBJECTS` 里的规范名。

    三级匹配，从确定到宽松：

    1. 按分隔符切词、找**完整命中**的 token（`数学-九年级-二次函数.md`、`初中-数学-教材.pdf`）；
    2. 整串子串匹配，**长写法优先**（"思想政治" 必须先于 "政治"，否则永远先命中别名前缀）；
    3. 都认不出返回 `None`——交给入库统计暴露，而不是默认成某个学科。

    `subjects=None` 时用 `base/labels.py` 的词表与别名表（学科写法的唯一出处）；
    调用方传自定义词表时只认该表，别名仍会做归一但必须落在表内。

    为什么接受整条路径而不只是文件名：新语料是 `初中/数学/教材.pdf` 这种
    "学科在目录、文件名是通用词"的结构，只看 `p.name` 会让 54 篇文档全部 subject=None。
    """
    candidates: tuple[str, ...] = tuple(subjects) if subjects is not None else SUBJECTS

    def canon(token: str) -> str | None:
        token = token.strip().removesuffix("学科").strip()
        mapped = SUBJECT_ALIASES.get(token, token)
        return mapped if mapped in candidates else None

    for token in re.split(r"[-_/\\\s.,，、（）()【】]+", name):
        hit = canon(token)
        if hit:
            return hit

    vocabulary = set(candidates) | set(SUBJECT_ALIASES) if subjects is None else set(candidates)
    for word in sorted(vocabulary, key=len, reverse=True):
        if word in name:
            hit = canon(word)
            if hit:
                return hit
    return None
