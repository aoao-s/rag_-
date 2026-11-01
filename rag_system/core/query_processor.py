"""查询预处理：归一化 + 入参校验（CLAUDE.md §7.2.7-B 第 2 步）。

**不在这里做敏感词拦截**：那是接入层的职责（§7.7 前置节点），放到检索包里等于让
"能不能问"取决于"走没走检索"——闲聊档不检索，词表就失效了。这里只做两件事：
把问句整理成适合向量化的形态，以及把非法入参挡在检索之前（`InvalidRequest`）。

与 `qa_system.res.qa_normalize` 的关系：**故意不复用**。FAQ 的归一化服务于"归一化串精确相等"，
动一个标点就会改变缓存 key 与命中结果；检索侧的归一化只影响向量质量，两者目标不同，
共用一份会让任何一方都不敢改（§7.1.2 归一化指纹那条教训）。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from base.errors import InvalidRequest

# 连续空白/换行压成一个空格；检索模型对换行敏感，而换行几乎总是粘贴带进来的
_WS = re.compile(r"\s+")
# 成串的重复标点（"？？？""。。。"）对语义无用
_REPEAT_PUNCT = re.compile(r"([，。！？；：、,.!?;:])\1+")

MAX_QUERY_CHARS = 400


@dataclass(frozen=True, slots=True)
class ProcessedQuery:
    original: str
    normalized: str
    degraded: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return self.normalized != self.original


def normalize(query: str) -> str:
    """NFKC（全角→半角）+ 空白折叠 + 重复标点折叠。**不删词**，改写交给理解层。"""
    text = unicodedata.normalize("NFKC", query or "")
    text = _WS.sub(" ", text).strip()
    return _REPEAT_PUNCT.sub(r"\1", text)


def process(query: str) -> ProcessedQuery:
    raw = (query or "").strip()
    if not raw:
        raise InvalidRequest("query 不能为空", node="query_processor")
    if len(raw) > MAX_QUERY_CHARS:
        # 截断而不是报错：超长问句多半是粘贴了整段材料，截断后仍可检索
        return ProcessedQuery(raw, normalize(raw)[:MAX_QUERY_CHARS], ("query_truncated",))
    return ProcessedQuery(raw, normalize(raw))
