"""命中判定：把"原始 BM25 分"变成"可跨学科解释的概率"。

旧项目直接用 `best_score >= 8.0` 判定，问题是：原始分随语料规模、文档长度、学科分布漂移，
换个学科阈值就失效，日志里出现过"最高得分 7.16（阈值 8.0）"这种差一点就全错的场景。

判定四条（CLAUDE.md §7.1.2，本模块负责 ①②③，④ 学科状态由路由层给结论）：
① 绝对门槛 `score_norm ≥ threshold`（先绝对后相对，避免小分数把相对分差放大）；
② 候选数 ≥ `min_candidates`（只有 1 条时"领先第二名"没有意义）；
③ 领先幅度 ≥ `margin`，可工作在三种空间：相对分差（默认目标）/ 归一化分差 / 原始分差。

本模块只做纯函数计算，无状态、无 IO，便于单测穷举边界。

三档（CLAUDE.md §7.1.9）的**基准档**也在这里算：过 ①②③ 且 `margin ≥ high_margin` 才是 high，
落在 `[margin, high_margin)` 的是 mid（"够格被考虑、但还不够格直接答"）。基准档放在这里，
离线标定脚本与线上服务才复用同一份判定；`qa_search` 只能按学科状态把它往下压，不许往上抬。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from base.contracts import Tier

MarginMode = Literal["relative", "normalized", "raw"]


@dataclass(frozen=True, slots=True)
class MatchPolicy:
    """命中策略。参数基线见 CLAUDE.md §7.1.2 / §6.3。"""

    threshold: float = 0.25
    margin: float = 0.10  # 按 margin_mode 解释的空间里的阈值
    norm_k: float = 12.0
    candidate_n: int = 3
    min_candidates: int = 2
    # 三种 margin 在同一评估集上逐一重扫后择优；重标定完成前保持 normalized（现网基线）
    margin_mode: MarginMode = "normalized"
    # 三档的高置信下限（与 margin 同一空间）。None = 与 margin 相同 = 单档（旧行为）：
    # 默认走单档，三档必须由配置显式打开，免得"忘了设"就悄悄改了线上判定。
    high_margin: float | None = None


@dataclass(frozen=True, slots=True)
class MatchResult:
    """判定结果。`tier` 是唯一结论，`hit` 由它派生（`hit == (tier == "high")`）。"""

    tier: Tier = "low"
    faq_id: int | None = None
    score: float = 0.0
    margin: float = 0.0  # 按 policy.margin_mode 解释的那个值
    margin_relative: float = 0.0
    margin_normalized: float = 0.0
    margin_raw: float = 0.0
    candidate_count: int = 0
    cross_delta: float = 0.0  # 多学科场景：与另一学科 best 的归一化分差
    reason: str = ""
    candidates: tuple[tuple[int, float], ...] = ()

    @property
    def hit(self) -> bool:
        return self.tier == "high"


def resolve_high_margin(policy: MatchPolicy) -> float:
    """高置信下限。`None` 退回 `margin`，即精确复现单档行为（灰度/回滚的唯一手段）。"""
    return policy.margin if policy.high_margin is None else policy.high_margin


def normalize_score(raw: float, norm_k: float = 12.0) -> float:
    """原始 BM25 分 → (0,1)。用有理函数而非 min-max：不需要预先知道分数上界。"""
    if raw <= 0 or norm_k <= 0:
        return 0.0
    return raw / (raw + norm_k)


def relative_margin(top_raw: float, second_raw: float) -> float:
    """相对领先幅度 `(raw1 − raw2)/raw1`，落在 [0,1)。

    跨学科比较时它是**唯一**保留"相对差距"含义的量：同一学科索引内原始分可比，
    归一化分差则会把高分段的差距压扁（0.95 vs 0.90 与 0.30 vs 0.25 不可同日而语）。
    """
    if top_raw <= 0:
        return 0.0
    return max(0.0, (top_raw - second_raw) / top_raw)


def decide(scored: list[tuple[int, float]], policy: MatchPolicy = MatchPolicy()) -> MatchResult:
    """单检索范围判定。`scored` 必须已按原始分降序，元素为 (faq_id, raw_score)。"""
    if not scored:
        return MatchResult(tier="low", reason="empty_corpus")

    top_id, top_raw = scored[0]
    second_raw = scored[1][1] if len(scored) > 1 else 0.0

    top_score = normalize_score(top_raw, policy.norm_k)
    second_score = normalize_score(second_raw, policy.norm_k)
    m_norm = top_score - second_score
    m_rel = relative_margin(top_raw, second_raw)
    m_raw = top_raw - second_raw
    margin = {"relative": m_rel, "normalized": m_norm, "raw": m_raw}.get(policy.margin_mode, m_norm)

    base = MatchResult(
        tier="low",
        faq_id=top_id,
        score=top_score,
        margin=margin,
        margin_relative=m_rel,
        margin_normalized=m_norm,
        margin_raw=m_raw,
        candidate_count=len(scored),
    )

    if top_score < policy.threshold:
        return replace(base, reason="below_threshold")
    # 单候选时 top2 不存在，"领先"无从谈起；原实现让它 margin=top1 从而必过
    if len(scored) < policy.min_candidates:
        return replace(base, reason="single_candidate")
    if margin < policy.margin:
        return replace(base, reason="ambiguous")
    # 过 ①②③ 只是"够格被考虑"：够不够格直接答，还要看是否越过高置信线
    if margin < resolve_high_margin(policy):
        return replace(base, tier="mid", reason="mid_confidence")
    return replace(base, tier="high", reason="ok")


def decide_multi(
    per_subject: dict[str, list[tuple[int, float]]],
    policy: MatchPolicy = MatchPolicy(),
    *,
    cross_delta: float = 0.0,
) -> MatchResult:
    """多学科判定：先在**学科之内**按单学科规则判，再要求跨学科领先。

    原始分在不同学科索引里不可比（IDF 与文档长度分布都不同），所以跨学科只在
    归一化空间比；"领先同一学科内 top2"仍交给 `decide()`，避免两套 margin 口径。
    """
    tops = {
        subject: normalize_score(ranked[0][1], policy.norm_k)
        for subject, ranked in per_subject.items()
        if ranked
    }
    if not tops:
        return MatchResult(tier="low", reason="empty_corpus")

    best_subject = max(tops, key=lambda s: tops[s])
    second_best = max((v for s, v in tops.items() if s != best_subject), default=0.0)
    delta = tops[best_subject] - second_best

    inner = decide(per_subject[best_subject], policy)
    inner = replace(inner, cross_delta=delta)
    if not inner.hit:
        return inner
    # 跨学科打平：不是"排序没找到"，是"两个学科都说得通"——降到 mid 交给追问，
    # 而不是当 low 丢掉（档位只降不升，从 high 降到 mid 是允许的那一档）。
    if delta < cross_delta:
        return replace(inner, tier="mid", reason="cross_subject_ambiguous")
    return inner
