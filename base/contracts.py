"""跨模块契约：三模块之间只允许通过这些数据类通信。

规则（见 CLAUDE.md §7.0）：
- qa_system / rag_system / 记忆模块 平级，互不 import；
- 任何跨模块返回值必须在本文档中定义，禁止用 dict 传递业务字段。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # Python 3.10（edu_rag 环境）没有 StrEnum，这里补一个等价实现

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)


class Strategy(StrEnum):
    """检索策略。faq 是短路分支，不进入多路检索。"""

    FAQ = "faq"
    DIRECT = "direct"
    HYDE = "hyde"
    SUB_QUERY = "sub_query"
    BACKTRACK = "backtrack"


# 槽位来源：上游"怎么得到这个学科"的唯一枚举（CLAUDE.md §7.1.8）。
# not_provided（没人做槽位提取）与 unknown（做了但抽不出）必须分开——
# 排障时是两条完全不同的线索。
SubjectSource = Literal[
    "explicit",
    "inferred",
    "context",
    "multi",
    "conflict",
    "invalid",
    "empty",
    "unknown",
    "not_provided",
]

# 本次实际检索范围：filtered=单学科索引、multi=多学科合并、full=全库、empty=学科无数据。
# invalid/conflict/unknown 属于"为什么落到这个范围"，记在 FAQHit.degraded 里。
SubjectState = Literal["filtered", "multi", "full", "empty"]

# FAQ 置信档（CLAUDE.md §7.1.9）：high=唯一允许直接作答；mid=只给候选与追问；
# low=交棒 RAG。档位只在 qa_system 里算，编排层只读它决定动作，不自己再比阈值。
Tier = Literal["high", "mid", "low"]


@dataclass(frozen=True, slots=True)
class SlotResult:
    """槽位抽取结果。qa_system / rag_system / 记忆模块共享，谁都不许另立字段。"""

    subject: str | None = None
    subjects: tuple[str, ...] = ()  # 多学科候选；非空时 subject 取第一个
    grade: str | None = None  # 只记日志，暂不参与过滤（FAQ 表无年级字段）
    subject_source: SubjectSource = "not_provided"
    subject_confidence: float = 0.0  # 规则=1.0；推断=0.6；未知/非法=0

    @property
    def scope(self) -> tuple[str, ...]:
        """检索要用到的学科集合（已去重、保持顺序）。空元组表示全库。"""
        if self.subjects:
            return tuple(dict.fromkeys(self.subjects))
        return (self.subject,) if self.subject else ()


@dataclass(frozen=True, slots=True)
class FetchFilter:
    """检索前置过滤条件，全部下推到 Milvus expr / SQL WHERE，禁止查后过滤。"""

    tenant_id: str = "default"
    kb_version: str = "v1"
    subject: str | None = None
    acl_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FAQCandidate:
    """未命中时交给 RAG 做 query 扩展的候选问题。"""

    faq_id: int
    question: str
    score: float


@dataclass(frozen=True, slots=True)
class FAQHit:
    """qa_system 对外的唯一返回结构。

    三档（§7.1.9）：`tier` 是唯一结论字段，`hit` 由它派生（`hit == (tier == "high")`），
    不提供第三种组合——否则调用方要同时判断两个可能互相矛盾的布尔。
    """

    # 高置信：可直接作答（answer 非空）；中置信：只给候选与追问（answer 恒为 None）；
    # 低置信：交棒 RAG。见 CLAUDE.md §7.1.9。
    tier: Tier = "low"
    faq_id: int | None = None
    answer: str | None = None
    score: float = 0.0
    subject: str | None = None
    matched_question: str | None = None
    alias_hit: bool = False
    candidates: tuple[FAQCandidate, ...] = ()
    reason: str = ""
    latency_ms: float = 0.0
    # 命中来自哪一层：cache（Redis/进程内）、index（BM25/精确哈希）、none。
    # 调用方（以及排障的人）需要它来区分"服务慢"与"缓存没生效"。
    source: str = "index"
    cache: str = ""
    # 本次实际检索范围（§7.1.8 路由表）；"学科非法/冲突/主动推断"这些由来记在 degraded，
    # 调用方不必再从一堆字段里猜"刚才到底查了全库还是某个学科"。
    subject_state: SubjectState = "full"
    degraded: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        """是否允许直接作答。派生而非独立字段：两处各写一遍必然对不上。"""
        return self.tier == "high"

    def as_dict(self) -> dict[str, object]:
        return {
            "hit": self.hit,
            "tier": self.tier,
            "faq_id": self.faq_id,
            "answer": self.answer,
            "score": round(self.score, 4),
            "subject": self.subject,
            "matched_question": self.matched_question,
            "alias_hit": self.alias_hit,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "source": self.source,
            "cache": self.cache,
            "subject_state": self.subject_state,
            "degraded": list(self.degraded),
            "candidates": [
                {"faq_id": c.faq_id, "question": c.question, "score": round(c.score, 4)}
                for c in self.candidates
            ],
        }


@dataclass(frozen=True, slots=True)
class Chunk:
    """父子块检索结果（rag_system）。"""

    child_id: str
    text: str
    parent_id: str
    parent_content: str
    doc_id: str
    file_name: str
    page_no: int = 0
    section: str = ""
    subject: str | None = None
    score_dense: float = 0.0
    score_sparse: float = 0.0
    score_rerank: float = 0.0


@dataclass(frozen=True, slots=True)
class Evidence:
    """证据门判定结果。ok=False 时禁止调用生成模型。"""

    ok: bool
    top_score: float = 0.0
    coverage: float = 0.0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """记忆模块注入 prompt 的背景信息，不得作为引用来源。"""

    profile: str = ""
    summary: str = ""
    atoms: tuple[str, ...] = ()
    tokens: int = 0
    degraded: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        """拼成 prompt 的"记忆"段；空记忆返回空串，调用方据此跳过该段。"""
        parts: list[str] = []
        if self.profile:
            parts.append(f"【学员画像】{self.profile}")
        if self.summary:
            parts.append(f"【历史摘要】{self.summary}")
        if self.atoms:
            parts.append("【长期记忆】" + "；".join(self.atoms))
        return "\n".join(parts)
