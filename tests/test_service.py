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


def test_unknown_subject_falls_back_to_all() -> None:
    service = _service()
    hit = asyncio.run(service.search("牛顿第二定律是什么", subject="天文学"))
    assert hit.hit and hit.faq_id == 4


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




