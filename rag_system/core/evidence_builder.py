"""证据装配：去重 → 排序 → token 预算裁剪 → 冲突探测（CLAUDE.md §7.2.7-B 第 6 步）。

三件事：

1. **去重与排序**：父块为唯一单位（同一父块的多个子块命中已经在 `Retriever` 里合并）。
2. **预算裁剪**：知识段硬上限 `MAX_CONTEXT_CHARS`（≈6 000 token，中文 1 字≈1 token），
   按分数从高到低装，装不下就停——**不截断单条证据**，因为截半个父块会让模型引用半句话。
3. **冲突探测**：两篇文档对同一个"量"给出不同的数（学费 12800 / 15800、学制 6 个月 / 12 个月），
   这是**必须并列返回**的情形，不能按分数挑一条（§7.2.7-C）。没有称职的 NLI 模型时，
   只做保守的"同单位数值冲突"判定：宁可漏报成 ambiguous，也不要把冲突静默抹平。
"""

from __future__ import annotations

import re
from typing import Sequence

from base.contracts import Evidence, RetrievalFilters, RetrievalResult

# 知识段预算（字符≈token）。§3.7：知识 6 000 token 是硬预算的一部分
MAX_CONTEXT_CHARS = 6000
DEFAULT_FINAL_N = 6
MIN_FINAL_N = 1

# "数字 + 单位"：学费 12 800 元 / 6 个月 / 120 课时 / 98%
_QUANTITY = re.compile(
    r"(\d+(?:\.\d+)?)\s*(元|块钱|块|万元|个月|年|天|课时|小时|分钟|人|%|分|折)"
)


def build(
    results: Sequence[RetrievalResult],
    *,
    final_n: int = DEFAULT_FINAL_N,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> list[Evidence]:
    """按分数装配最终进 prompt 的证据列表。"""
    from rag_system.core.fusion import fuse

    ranked = fuse(results)
    if not ranked:
        return []

    keep_limit = max(MIN_FINAL_N, final_n)
    kept: list[Evidence] = []
    used = 0
    seen: set[str] = set()
    for ev in ranked:
        if ev.parent_id in seen:
            continue
        cost = len(ev.text or "")
        if kept and (len(kept) >= keep_limit or used + cost > max_chars):
            break
        seen.add(ev.parent_id)
        kept.append(ev)
        used += cost
    return kept


def detect_conflicts(
    evidences: Sequence[Evidence],
    *,
    threshold: float = 0.0,
    limit: int = 3,
) -> tuple[str, ...]:
    """找出"两篇文档对同一个量给出不同数值"的情况。

    只比较**跨文档**且都达到 `threshold` 的证据（低于阈值的证据本来就不可信，
    拿它判冲突等于让噪音否决主证据）。返回 `conflicting_values:<单位>` 形态的原因串。
    """
    if len(evidences) < 2:
        return ()
    top = [e for e in evidences[:limit] if e.score >= threshold]
    if len({e.doc_id for e in top}) < 2:
        return ()

    by_unit: dict[str, dict[str, set[str]]] = {}
    for ev in top:
        for value, unit in _QUANTITY.findall(ev.text or ""):
            by_unit.setdefault(unit, {}).setdefault(value, set()).add(ev.doc_id)

    reasons: list[str] = []
    for unit, values in by_unit.items():
        if len(values) >= 2 and len({doc for docs in values.values() for doc in docs}) >= 2:
            reasons.append(f"conflicting_values:{unit}")
    return tuple(reasons[:3])


def best_by_doc(evidences: Sequence[Evidence]) -> dict[str, float]:
    """每篇文档的最好分数（证据门的跨文档 margin 用同一口径）。"""
    out: dict[str, float] = {}
    for e in evidences:
        if e.score > out.get(e.doc_id, float("-inf")):
            out[e.doc_id] = e.score
    return out


def describe_filters(filters: RetrievalFilters | None) -> str:
    """日志里一行说清"这次查了多大范围"，避免排障时靠猜。"""
    if filters is None:
        return "none"
    bits = [f"tenant={filters.tenant_id}", f"kb={filters.kb_version}"]
    if filters.subjects:
        bits.append("subject=" + "|".join(filters.subjects))
    if filters.doc_types:
        bits.append("doc_type=" + "|".join(filters.doc_types))
    if filters.acl_tags:
        bits.append(f"acl={len(filters.acl_tags)}")
    return " ".join(bits)
