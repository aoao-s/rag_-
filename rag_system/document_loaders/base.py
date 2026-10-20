"""加载器基类与统一产物。

为什么不用 langchain 的 BaseLoader：
1. 我们只需要"文件 → 文本 + 元数据"，langchain 的 Document 会带上大量用不到的字段，
   写进 Milvus 还要再剥一层；
2. 旧项目里 `lazy_load` 的 `yield` 一旦前置校验失败就 `return`（生成器里等价于空迭代），
   调用方拿不到任何错误信号——这里改成抛异常，让入库脚本能明确记 failed。

产物 `LoadedDoc` 是"一篇文档"，不是"一个 chunk"：切分在 splitter 里做。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


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
    """

    text: str
    source: str
    doc_type: str = "other"
    subject: str | None = None
    title: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

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


def guess_subject(name: str, subjects: Iterable[str]) -> str | None:
    """从文件名里认出学科。

    先按 '-'/'_'/' ' 切词找完整命中的 token（数据集是"数学-九年级-二次函数.md"这种格式），
    再退回整串子串匹配（"二次函数入门-字幕.srt"这种省略了学科前缀的命名）。
    都认不出返回 None——交给入库统计暴露，而不是默认成某个学科。
    """
    subjects = tuple(subjects)
    for token in name.replace("_", "-").replace(" ", "-").split("-"):
        token = token.strip()
        if token in subjects:
            return token
    for subject in subjects:
        if subject in name:
            return subject
    return None
