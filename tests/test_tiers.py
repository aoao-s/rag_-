"""三档路由单测（CLAUDE.md §7.1.9：high / mid / low）。

分两层测，因为两层的责任不同：
- **判定层**（`decide()` / `decide_multi()`）：基准档怎么切出来。纯函数，可以精确构造 margin，
  所以在这里验"一根旋钮切三段""high_margin=None 精确回退单档"。
- **服务层**（`FAQService`）：档位只降不升（学科来源 / 多学科封顶）、mid 不返回答案、
  确认通道、缓存按 tier 复原、缺 tier 的存量条目按 low 覆盖写。

铁律：`hit == (tier == "high")`。任何一条用例都不允许出现"hit 为真但 tier 不是 high"。
"""

from __future__ import annotations

import asyncio

from base.contracts import SlotResult
from qa_system.res.qa_cache import InMemoryFAQCache
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository
from qa_system.res.qa_matcher import MatchPolicy, decide, decide_multi, resolve_high_margin
from qa_system.res.qa_search import FAQService

# 相对分差空间最适合写用例：margin = (raw1 − raw2) / raw1，构造出来的数就是人读的数
_RELATIVE = MatchPolicy(threshold=0.25, margin=0.10, high_margin=0.22, margin_mode="relative")


# ------------------------------------------------------------------ 判定层：基准档
def test_one_knob_cuts_three_tiers() -> None:
    # 三条都过绝对门槛（raw 10 → 0.4545 ≥ 0.25），只差领先幅度
    high = decide([(1, 10.0), (2, 5.0)], _RELATIVE)  # margin 0.50
    mid = decide([(1, 10.0), (2, 8.0)], _RELATIVE)  # margin 0.20
    low = decide([(1, 10.0), (2, 9.5)], _RELATIVE)  # margin 0.05

    assert (high.tier, high.reason) == ("high", "ok")
    assert (mid.tier, mid.reason) == ("mid", "mid_confidence")
    assert (low.tier, low.reason) == ("low", "ambiguous")
    # hit 由 tier 派生，不允许两处各判一次
    assert [r.hit for r in (high, mid, low)] == [True, False, False]


def test_mid_still_needs_the_absolute_threshold() -> None:
    # 领先幅度再大，没过绝对门槛也进不了 mid——三档不是"绕过阈值"的新通道
    result = decide([(1, 2.0), (2, 1.0)], _RELATIVE)
    assert result.tier == "low" and result.reason == "below_threshold"


def test_single_candidate_never_reaches_mid() -> None:
    # 判据 ② 先于 margin：只有一条候选时"领先第二名"没有意义（§7.1.2）
    result = decide([(1, 10.0)], _RELATIVE)
    assert result.tier == "low" and result.reason == "single_candidate"


def test_high_margin_none_is_exactly_single_tier() -> None:
    # 回滚开关（§7.1.9）：high_margin=None 与 high_margin == margin 必须逐字段等价
    single = MatchPolicy(threshold=0.25, margin=0.10, margin_mode="relative")
    rolled_back = MatchPolicy(threshold=0.25, margin=0.10, high_margin=0.10, margin_mode="relative")
    assert resolve_high_margin(single) == single.margin

    for scored in (
        [(1, 10.0), (2, 5.0)],
        [(1, 10.0), (2, 8.0)],
        [(1, 10.0), (2, 9.5)],
        [(1, 10.0)],
        [(1, 2.0), (2, 1.0)],
    ):
        a, b = decide(scored, single), decide(scored, rolled_back)
        assert (a.tier, a.reason, a.margin) == (b.tier, b.reason, b.margin)
        assert a.tier in ("high", "low")  # 单档下不存在 mid


def test_multi_subject_tie_is_capped_at_mid() -> None:
    # 跨学科只在归一化分上比，CROSS_SUBJECT_DELTA 没经过精度标定，不能让这条路直接作答
    per_subject = {"数学": [(1, 10.0), (2, 3.0)], "英语": [(3, 9.9), (4, 1.0)]}
    tied = decide_multi(per_subject, _RELATIVE, cross_delta=0.05)
    assert tied.tier == "mid" and tied.reason == "cross_subject_ambiguous"
    assert tied.hit is False

    clear = decide_multi(per_subject, _RELATIVE, cross_delta=0.0)
    assert clear.tier == "high" and clear.faq_id == 1


# ------------------------------------------------------------------ 服务层
def _records() -> list[FAQRecord]:
    return [
        FAQRecord(1, "数学", "勾股定理的内容是什么？", "直角三角形两直角边平方和等于斜边平方。"),
        FAQRecord(2, "数学", "勾股定理怎么证明？", "常用面积法（赵爽弦图）。"),
        FAQRecord(3, "英语", "怎么报名", "在官网填写报名表并缴费。"),
    ]


def _service(
    *,
    high_margin: float | None = None,
    cache: InMemoryFAQCache | None = None,
    cache_ttl: int = 3600,
    cache_negative_ttl: int = 30,
) -> FAQService:
    """默认 high_margin=None：任何过 ①②③ 的命中都是 high（旧行为），封顶用例才好构造。"""
    service = FAQService(
        InMemoryFAQRepository(_records()),
        policy=MatchPolicy(
            threshold=0.05,
            margin=0.0,
            candidate_n=3,
            margin_mode="relative",
            high_margin=high_margin,
        ),
        top_k=10,
        cache=cache,
        cache_ttl=cache_ttl,
        cache_negative_ttl=cache_negative_ttl,
    )
    asyncio.run(service.load())
    return service


def test_exact_is_high_even_when_subject_is_inferred() -> None:
    # 归一化串完全相同是字面事实，不受"学科是猜的"封顶（§7.1.9 精确命中恒 high）
    service = _service()
    slots = SlotResult(subject="数学", subject_source="inferred", subject_confidence=0.6)
    hit = asyncio.run(service.search("勾股定理的内容是什么？", slots=slots))
    assert hit.tier == "high" and hit.answer
    assert hit.reason == "exact"


def test_exact_stays_high_in_multi_scope() -> None:
    service = _service()
    slots = SlotResult(
        subject="数学", subjects=("数学", "英语"), subject_source="multi", subject_confidence=1.0
    )
    hit = asyncio.run(service.search("勾股定理的内容是什么？", slots=slots))
    assert hit.tier == "high" and hit.answer
    assert hit.subject_state == "multi"


def test_inferred_bm25_hit_is_capped_to_mid() -> None:
    # 基准档是 high（margin=0），封顶由学科来源决定：推断出来的学科不能直接答
    service = _service()
    slots = SlotResult(subject="数学", subject_source="inferred", subject_confidence=0.6)
    hit = asyncio.run(service.search("勾股定理的证明方法", slots=slots))

    assert (hit.tier, hit.reason) == ("mid", "subject_inferred_handoff")
    assert hit.hit is False
    assert hit.answer is None  # 中置信不返回答案：正文必须走 RAG 或确认通道
    assert hit.faq_id == 2 and hit.matched_question  # 追问对象还在
    assert hit.candidates


def test_multi_scope_bm25_hit_is_capped_to_mid() -> None:
    service = _service()
    slots = SlotResult(
        subject="数学", subjects=("数学", "英语"), subject_source="multi", subject_confidence=1.0
    )
    hit = asyncio.run(service.search("勾股定理的证明方法", slots=slots))

    assert (hit.tier, hit.reason) == ("mid", "multi_scope_handoff")
    assert hit.subject_state == "multi" and hit.answer is None


def test_mid_from_margin_band_hides_answer() -> None:
    # 相对 margin 恒 < 1，把高置信线设到 0.99 就能让任何 BM25 命中落在 mid 段
    service = _service(high_margin=0.99)
    hit = asyncio.run(service.search("勾股定理的证明方法"))

    assert (hit.tier, hit.reason) == ("mid", "mid_confidence")
    assert hit.hit is False and hit.answer is None
    assert hit.faq_id == 2 and hit.matched_question == "勾股定理怎么证明？"


def test_low_tier_hands_off_with_candidates() -> None:
    service = _service(high_margin=0.99)
    hit = asyncio.run(service.search("量子纠缠"))
    assert hit.tier == "low" and hit.reason == "no_match"
    assert hit.hit is False and hit.answer is None


# ------------------------------------------------------------------ 确认通道
def test_confirmation_channel_answers_without_touching_shared_cache() -> None:
    cache = InMemoryFAQCache()
    service = _service(high_margin=0.99, cache=cache)

    first = asyncio.run(service.search("勾股定理的证明方法"))
    assert first.tier == "mid"

    confirmed = asyncio.run(service.search("勾股定理的证明方法", selected_faq_id=2))
    assert (confirmed.tier, confirmed.reason) == ("high", "user_confirmed")
    assert confirmed.faq_id == 2 and confirmed.answer
    assert confirmed.source == "index"  # 不走缓存：结论只对这次点击成立

    # 共享缓存里仍然是 mid——不能因为一个人点了"是"，就把答案发给所有人
    again = asyncio.run(service.search("勾股定理的证明方法"))
    assert again.tier == "mid" and again.source == "cache"


def test_confirmation_with_unknown_id_falls_through_without_4xx() -> None:
    # 用户点的是旧卡片（已下线/不存在）：忽略参数走正常链路，不是参数错误
    service = _service(high_margin=0.99)
    hit = asyncio.run(service.search("勾股定理的证明方法", selected_faq_id=999))
    assert hit.tier == "mid"
    assert "faq_selected_invalid" in hit.degraded


# ------------------------------------------------------------------ 缓存
def test_cache_restores_tier_instead_of_redeciding() -> None:
    cache = InMemoryFAQCache()
    service = _service(high_margin=0.99, cache=cache)

    first = asyncio.run(service.search("勾股定理的证明方法"))
    second = asyncio.run(service.search("勾股定理的证明方法"))

    assert (first.source, second.source) == ("index", "cache")
    assert second.tier == "mid" and second.answer is None
    assert second.matched_question == first.matched_question  # 追问对象没丢
    assert [c.faq_id for c in second.candidates] == [c.faq_id for c in first.candidates]


def test_legacy_cache_entry_without_tier_is_low_and_overwritten() -> None:
    cache = InMemoryFAQCache()
    service = _service(cache=cache)  # 基准档 high

    asyncio.run(service.search("勾股定理的证明方法"))
    ((entry, _, _),) = tuple(cache._data.values())  # 只有一次查询，缓存里就一条
    entry.tier = None  # 模拟滚动发布期间旧版本写入的存量条目

    second = asyncio.run(service.search("勾股定理的证明方法"))
    assert second.source == "index"  # 缺 tier 按 low 丢弃，重算一遍
    assert second.tier == "high"

    third = asyncio.run(service.search("勾股定理的证明方法"))
    assert (third.source, third.tier) == ("cache", "high")  # 覆盖写生效，不会每次都重算


def test_tier_decides_cache_ttl() -> None:
    # 高置信走正缓存（1 h），中/低与负缓存同寿（30 s）：mid 是"待确认"的临时结论
    cache = InMemoryFAQCache()
    service = _service(high_margin=0.99, cache=cache, cache_ttl=3600, cache_negative_ttl=7)

    asyncio.run(service.search("勾股定理的证明方法"))  # mid
    asyncio.run(service.search("勾股定理的内容是什么？"))  # exact → high
    assert sorted(item[2] for item in cache._data.values()) == [7, 3600]


# ------------------------------------------------------------------ 观测
def test_stats_expose_three_tiers_and_hit_tracks_high() -> None:
    service = _service(high_margin=0.99)
    hits = [
        asyncio.run(service.search("勾股定理的证明方法")),  # mid
        asyncio.run(service.search("勾股定理的内容是什么？")),  # exact → high
        asyncio.run(service.search("量子纠缠")),  # low
    ]
    assert [h.tier for h in hits] == ["mid", "high", "low"]
    assert [h.hit for h in hits] == [False, True, False]

    stats = service.stats()
    # 三档恒齐备：缺一个键，面板上就分不清"没发生"与"埋点漏了"
    assert set(stats["tiers"]) == {"high", "mid", "low"}
    assert stats["tiers"] == {"high": 1, "mid": 1, "low": 1}
    assert stats["hits"] == 1 and stats["misses"] == 2
