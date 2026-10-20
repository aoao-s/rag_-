"""跨模块契约：三模块之间只允许通过这些数据类通信。

规则（见 CLAUDE.md §7.0）：
- qa_system / rag_system / 记忆模块 平级，互不 import；
- 任何跨模块返回值必须在本文档中定义，禁止用 dict 传递业务字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Strategy(StrEnum):
    """检索策略。faq 是短路分支，不进入多路检索。"""

    FAQ = "faq"
    DIRECT = "direct"
    HYDE = "hyde"
    SUB_QUERY = "sub_query"
    BACKTRACK = "backtrack"


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
    """qa_system 对外的唯一返回结构。"""

    hit: bool
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

    def as_dict(self) -> dict[str, object]:
        return {
            "hit": self.hit,
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
