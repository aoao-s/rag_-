"""命中判定：把"原始 BM25 分"变成"可跨学科解释的概率"。

旧项目直接用 `best_score >= 8.0` 判定，问题是：原始分随语料规模、文档长度、学科分布漂移，
换个学科阈值就失效，日志里出现过"最高得分 7.16（阈值 8.0）"这种差一点就全错的场景。

本模块只做纯函数计算，无状态、无 IO，便于单测穷举边界。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MatchPolicy:
    """命中策略。参数基线见 CLAUDE.md §7.1.2。"""

    threshold: float = 0.62
    margin: float = 0.05
    norm_k: float = 12.0
    candidate_n: int = 3


@dataclass(frozen=True, slots=True)
class MatchResult:
    hit: bool
    faq_id: int | None = None
    score: float = 0.0
    margin: float = 0.0
    reason: str = ""
    candidates: tuple[tuple[int, float], ...] = ()


def normalize_score(raw: float, norm_k: float = 12.0) -> float:
    """原始 BM25 分 → (0,1)。用有理函数而非 min-max：不需要预先知道分数上界。"""
    if raw <= 0 or norm_k <= 0:
        return 0.0
    return raw / (raw + norm_k)


def decide(scored: list[tuple[int, float]], policy: MatchPolicy = MatchPolicy()) -> MatchResult:
    """三条件判定：绝对分够、领先第二名够多、（学科一致性由调用方保证）。

    `scored` 必须已按原始分降序，元素为 (faq_id, raw_score)。
    """
    if not scored:
        return MatchResult(hit=False, reason="empty_corpus")

    top_id, top_raw = scored[0]
    top_score = normalize_score(top_raw, policy.norm_k)
    second_score = normalize_score(scored[1][1], policy.norm_k) if len(scored) > 1 else 0.0
    margin = top_score - second_score

    if top_score < policy.threshold:
        return MatchResult(
            hit=False,
            faq_id=top_id,
            score=top_score,
            margin=margin,
            reason="below_threshold",
        )
    if margin < policy.margin:
        return MatchResult(
            hit=False,
            faq_id=top_id,
            score=top_score,
            margin=margin,
            reason="ambiguous",
        )
    return MatchResult(hit=True, faq_id=top_id, score=top_score, margin=margin, reason="ok")
