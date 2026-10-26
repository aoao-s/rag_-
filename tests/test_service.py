"""FAQService 端到端单测（内存仓库，不需要 MySQL / Redis）。"""

from __future__ import annotations

import asyncio

import pytest

from qa_system.res.qa_matcher import MatchPolicy
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository
from qa_system.res.qa_search import FAQService


def _records() -> list[FAQRecord]:
    return [
        FAQRecord(1, "数学", "勾股定理的内容是什么？", "直角三角形两直角边平方和等于斜边平方。"),
        FAQRecord(2, "数学", "勾股定理怎么证明？", "常用面积法（赵爽弦图）。"),
        FAQRecord(3, "英语", "怎么报名", "在官网填写报名表并缴费。"),
        FAQRecord(4, "物理", "牛顿第二定律是什么", "F=ma。"),
    ]


def _service() -> FAQService:
    repo = InMemoryFAQRepository(_records())
    service = FAQService(
        repo,
        policy=MatchPolicy(threshold=0.05, margin=0.0, candidate_n=3),
        top_k=10,
    )
    asyncio.run(service.load())
    return service


def test_exact_hit() -> None:
    service = _service()
    hit = asyncio.run(service.search("勾股定理的内容是什么？"))
    assert hit.hit and hit.faq_id == 1
    assert hit.reason == "exact"
    assert hit.answer and hit.latency_ms >= 0


def test_polite_prefix_still_exact() -> None:
    service = _service()
    hit = asyncio.run(service.search("请问一下，勾股定理的内容是什么？"))
    assert hit.hit and hit.faq_id == 1


def test_alias_hit_flagged() -> None:
    repo = InMemoryFAQRepository(_records())
    repo.insert(FAQRecord(5, "英语", "学费多少", "按课时计费。"), aliases=["多少钱"])
    service = FAQService(repo, policy=MatchPolicy(threshold=0.3, margin=0.0))
    asyncio.run(service.load())

    hit = asyncio.run(service.search("多少钱"))
    assert hit.hit and hit.faq_id == 5
    assert hit.alias_hit is True
    assert hit.reason == "alias_exact"


def test_fuzzy_hit_by_bm25() -> None:
    service = _service()
    hit = asyncio.run(service.search("勾股定理的证明方法"))
    assert hit.hit and hit.faq_id == 2


def test_subject_scoping_does_not_lose_hit() -> None:
    service = _service()
    hit = asyncio.run(service.search("牛顿第二定律是什么", subject="物理"))
    assert hit.hit and hit.faq_id == 4


def test_unknown_subject_falls_back_to_full_scan_not_direct_answer() -> None:
    # 学科参数不在词表里 = invalid：内部兜底成全库（不能让调用方以为过滤生效了），
    # 但**不许直接答**——"没认出来"不等于"确认了"（§7.1.8 路由表 + §7.1.9 封顶）。
    # 用 BM25 命中来验：精确命中是字面事实，恒为 high，不在这条封顶的管辖范围里。
    service = _service()
    hit = asyncio.run(service.search("勾股定理的证明方法", subject="天文学"))
    assert hit.subject_state == "full"
    assert hit.degraded == ("subject_invalid_fallback",)
    assert (hit.tier, hit.reason) == ("mid", "subject_invalid_handoff")
    assert hit.faq_id == 2 and hit.answer is None


def test_no_match_when_no_token_overlap() -> None:
    service = _service()
    hit = asyncio.run(service.search("一元二次方程的解法"))
    assert hit.hit is False and hit.reason == "no_match"


def test_below_threshold_returns_candidates() -> None:
    repo = InMemoryFAQRepository(_records())
    service = FAQService(repo, policy=MatchPolicy(threshold=0.99, margin=0.0, candidate_n=3), top_k=10)
    asyncio.run(service.load())
    hit = asyncio.run(service.search("勾股定理证明思路"))
    assert hit.hit is False and hit.reason == "below_threshold"
    assert len(hit.candidates) >= 1  # 未命中也要给 RAG 留 top3 扩展线索


def test_empty_query_not_hit() -> None:
    service = _service()
    hit = asyncio.run(service.search("？？？"))
    assert hit.hit is False and hit.reason == "empty_query"


def test_publish_and_rebuild_visible() -> None:
    repo = InMemoryFAQRepository([FAQRecord(1, "数学", "甲问题", "甲答案", status="draft")])
    service = FAQService(repo, policy=MatchPolicy(threshold=0.3, margin=0.0))
    asyncio.run(service.load())
    assert service.state()["docs"] == 0

    asyncio.run(service.publish(1))
    assert service.state()["docs"] == 1
    hit = asyncio.run(service.search("甲问题"))
    assert hit.hit and hit.faq_id == 1


def test_deprecate_removes_from_index() -> None:
    service = _service()
    asyncio.run(service.deprecate(1))
    hit = asyncio.run(service.search("勾股定理的内容是什么？"))
    assert not (hit.hit and hit.faq_id == 1)


def test_publish_unknown_id_raises() -> None:
    service = _service()
    with pytest.raises(LookupError):
        asyncio.run(service.publish(999))


def test_stats_track_hit_rate() -> None:
    service = _service()
    asyncio.run(service.search("勾股定理的内容是什么？"))
    asyncio.run(service.search("量子纠缠"))
    stats = service.stats()
    assert stats["searches"] == 2 and stats["hits"] == 1
    assert stats["exact_hits"] == 1 and stats["misses"] == 1


def test_empty_repository_not_loaded() -> None:
    service = FAQService(InMemoryFAQRepository())
    asyncio.run(service.load())
    hit = asyncio.run(service.search("勾股定理"))
    assert hit.hit is False and hit.reason == "not_loaded"


# ---------------------------------------------------------------- 语料重复合并
# 契约见 CLAUDE.md §7.1.7「语料重复合并契约」：同学科 + 答案逐字相同 + 去框架词后 Jaccard ≥0.8 才合并。
_O2_ANSWER = "查装置气密性、装药品、固定、加热、收集、先移导管再熄灯。"


def _near_dup_records() -> list[FAQRecord]:
    """一对"同一事实的两种问法"（去框架词后 Jaccard = 1.0）+ 两条同科但不同题的行。

    另外两条的存在是刻意的：判据 ② 要求候选数 ≥2，语料里只剩一条候选时
    哪怕分数再高也不会命中（§7.1.2），那会把"能不能答第三种问法"的结论搅浑。
    """
    return [
        FAQRecord(1, "化学", "什么是用高锰酸钾制氧气的步骤和注意事项？", _O2_ANSWER),
        FAQRecord(2, "化学", "用高锰酸钾制氧气的步骤和注意事项是什么？", _O2_ANSWER),
        FAQRecord(3, "化学", "如何书写化学方程式？", "左写反应物，右写生成物，配平后标条件。"),
        FAQRecord(4, "化学", "用高锰酸钾制氧气时试管口为什么放棉花？", "防止高锰酸钾粉末随气流进入导管。"),
    ]


def test_near_duplicate_questions_merge_into_one_entry() -> None:
    repo = InMemoryFAQRepository(_near_dup_records())
    service = FAQService(repo, policy=MatchPolicy(threshold=0.05, margin=0.0), top_k=10)
    asyncio.run(service.load())

    state = service.state()
    assert state["docs"] == 4
    assert state["merged_duplicates"] == 1
    assert state["indexed"] == 3  # 被合并的那条只作别名，不进 BM25
    assert service.canonical_id(1) == 1 and service.canonical_id(2) == 1

    # 被合并的原话仍可直接答：它登记成了指向代表条目的精确命中
    hit = asyncio.run(service.search("用高锰酸钾制氧气的步骤和注意事项是什么？"))
    assert hit.hit and hit.faq_id == 1 and hit.reason == "exact"

    # 第三种问法：合并前双胞胎分数几乎相同、margin 必然被拒；合并后能走 BM25 答出
    fuzzy = asyncio.run(service.search("用高锰酸钾制氧气需要哪些步骤和注意事项"))
    assert fuzzy.hit and fuzzy.faq_id == 1
    assert fuzzy.reason not in ("exact", "alias_exact")


def test_similar_questions_with_different_answers_not_merged(caplog: pytest.LogCaptureFixture) -> None:
    # 去框架词后 Jaccard ≈0.86（≥0.8）但答案不同：同一件事写了两份答案，交教研，不自动合并
    repo = InMemoryFAQRepository(
        [
            FAQRecord(1, "数学", "一元二次方程的应用题的解题步骤是什么？", "设未知数、列方程、解方程、检验。"),
            FAQRecord(2, "数学", "一元二次方程应用题的步骤是什么？", "先判题型，再套利润/增长率/面积模型。"),
        ]
    )
    service = FAQService(repo, policy=MatchPolicy(threshold=0.05, margin=0.0), top_k=10)
    asyncio.run(service.load())

    state = service.state()
    assert state["merged_duplicates"] == 0 and state["indexed"] == 2
    assert service.canonical_id(2) == 2
    # 冲突必须被看见：这条告警是教研核对"同一件事写了两份答案"的唯一入口
    assert "faq_near_dup_answer_conflict" in caplog.text
    hit = asyncio.run(service.search("一元二次方程应用题的步骤是什么？"))
    assert hit.hit and hit.faq_id == 2


def test_near_duplicate_merge_can_be_disabled() -> None:
    # FAQ_DEDUPE_NEAR_DUP=false 时保持原样，用于 A/B 对照实验（报告见 docs/analysis/archive/）
    repo = InMemoryFAQRepository(_near_dup_records())
    service = FAQService(
        repo,
        policy=MatchPolicy(threshold=0.05, margin=0.0),
        top_k=10,
        dedupe_near_dup=False,
    )
    asyncio.run(service.load())

    state = service.state()
    assert state["merged_duplicates"] == 0 and state["indexed"] == 4
    hit = asyncio.run(service.search("用高锰酸钾制氧气的步骤和注意事项是什么？"))
    assert hit.hit and hit.faq_id == 2  # 两条都在索引里，各答各的
