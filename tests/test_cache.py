"""FAQ 二级缓存单测：内存缓存语义 + 服务层命中/回填/版本失效。

不依赖 Redis：`InMemoryFAQCache` 与 `RedisFAQCache` 行为一致（同样的 get/put/slide 契约），
真实 Redis 只在集成验证里跑。
"""

from __future__ import annotations

import asyncio
import time

from qa_system.res.qa_cache import CacheEntry, InMemoryFAQCache
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository
from qa_system.res.qa_matcher import MatchPolicy
from qa_system.res.qa_search import FAQService


def _records() -> list[FAQRecord]:
    return [
        FAQRecord(1, "数学", "勾股定理的内容是什么？", "直角三角形两直角边平方和等于斜边平方。"),
        FAQRecord(2, "数学", "勾股定理怎么证明？", "常用面积法（赵爽弦图）。"),
        FAQRecord(3, "英语", "怎么报名", "在官网填写报名表并缴费。"),
    ]


def _service(cache: InMemoryFAQCache | None = None) -> FAQService:
    service = FAQService(
        InMemoryFAQRepository(_records()),
        policy=MatchPolicy(threshold=0.05, margin=0.0, candidate_n=3),
        top_k=10,
        cache=cache,
        cache_ttl=60,
        cache_negative_ttl=5,
    )
    asyncio.run(service.load())
    return service


# ---------------------------------------------------------------- 缓存原语
def test_put_get_and_slide_extends_remaining() -> None:
    cache = InMemoryFAQCache(max_ttl=3600)
    entry = CacheEntry(faq_id=1, score=1.0, reason="exact")

    asyncio.run(cache.put("k", entry, ttl=30))
    got = asyncio.run(cache.get("k"))
    assert got is not None and got.faq_id == 1
    before = cache.remaining("k")

    asyncio.run(cache.slide("k", ttl=30))
    assert cache.remaining("k") >= before - 0.05  # 允许毫秒级时钟抖动
    assert cache.slides == ["k"]


def test_absolute_ttl_wins_over_slide() -> None:
    """滑动续命不能无限延长：超过 max_ttl 的条目视为失效（否则改了答案也换不掉）。"""
    cache = InMemoryFAQCache(max_ttl=10)
    stale = CacheEntry(faq_id=1, score=1.0, reason="exact", born=time.time() - 100)
    asyncio.run(cache.put("k", stale, ttl=3600))
    assert asyncio.run(cache.get("k")) is None


def test_missing_key_returns_none() -> None:
    assert asyncio.run(InMemoryFAQCache().get("nope")) is None


# ---------------------------------------------------------------- 服务层集成
def test_second_identical_query_hits_cache() -> None:
    cache = InMemoryFAQCache()
    service = _service(cache)

    first = asyncio.run(service.search("勾股定理的内容是什么？"))
    second = asyncio.run(service.search("勾股定理的内容是什么？"))

    assert first.hit and first.reason == "exact"
    assert second.hit and second.faq_id == 1
    assert second.reason == "cache_exact"
    stats = service.stats()
    assert stats["cache_hits"] == 1
    assert stats["cache_fills"] >= 1


def test_miss_is_negatively_cached_to_protect_index() -> None:
    cache = InMemoryFAQCache()
    service = _service(cache)

    miss1 = asyncio.run(service.search("量子纠缠的退相干时间"))
    miss2 = asyncio.run(service.search("量子纠缠的退相干时间"))

    assert not miss1.hit and not miss2.hit
    # 第二次直接命中负缓存，不再重新打分
    assert miss2.reason.startswith("cached_")
    assert miss2.source == "cache"


def test_publish_bumps_version_and_bypasses_old_cache() -> None:
    cache = InMemoryFAQCache()
    service = _service(cache)

    asyncio.run(service.search("勾股定理的内容是什么？"))
    asyncio.run(service.publish(2))  # 版本 +1，旧 key 不再可达
    again = asyncio.run(service.search("勾股定理的内容是什么？"))

    assert again.hit
    assert not again.reason.startswith("cache_")
    assert service.state()["version"] == 2


def test_cache_disabled_still_works() -> None:
    service = _service(None)
    first = asyncio.run(service.search("怎么报名"))
    second = asyncio.run(service.search("怎么报名"))

    assert first.hit and second.hit
    assert service.stats()["cache_hits"] == 0
