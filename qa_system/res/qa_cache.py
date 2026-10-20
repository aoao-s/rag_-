"""Redis 客户端工厂 + FAQ 二级缓存 + 单飞。

key 规范见 CLAUDE.md §6.4：`edurag:{env}:{用途}:...`。
socket_timeout 必须显式设置——没有超时的 Redis 调用会把整个请求拖死。

FAQ 二级缓存的三条设计约束（都踩过坑）：
1. **key 带语料版本号**：发布/下线后 `version+1`，旧 key 自然不可达并随 TTL 过期，
   不需要在重建时遍历删 key（几万条时 DEL 扫描会把 Redis 打满）。
2. **滑动续命有上限**：命中续 TTL 能让热数据常驻，但若无上限，一条写错的答案会一直活着。
   因此 value 里记 `born`，超过 `max_ttl` 视为失效，重新走一次索引并覆盖写。
3. **失败开放**：Redis 不可用只降级为"没有缓存"，绝不能拒服务。所有异常在此吞掉并打日志。
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from base.config import Settings, get_settings
from base.logger import get_logger

log = get_logger(__name__)


def create_redis(settings: Settings | None = None) -> Any:
    """返回 redis.Redis（decode_responses=True）。构造不连库。"""
    import redis

    s = settings or get_settings()
    return redis.Redis(
        host=s.redis_host,
        port=s.redis_port,
        password=s.redis_password or None,
        db=s.redis_db,
        decode_responses=True,
        socket_timeout=1.0,
        socket_connect_timeout=1.0,
        health_check_interval=30,
    )


def cache_key(settings: Settings, kind: str, *parts: str) -> str:
    """统一 key 构造，禁止业务代码手拼字符串。"""
    tail = ":".join(parts)
    return f"edurag:{settings.app_env}:{kind}:{tail}" if tail else f"edurag:{settings.app_env}:{kind}"


@dataclass(slots=True)
class CacheEntry:
    """缓存内容。只存结论与溯源信息，答案正文由快照按 faq_id 取，避免正文被缓存后改不动。"""

    faq_id: int | None
    score: float
    reason: str
    subject: str | None = None
    matched_question: str | None = None
    # 负缓存也要带候选：未命中要交棒 RAG，候选列表是 RAG 做 query 扩展的输入，
    # 缓存命中却把候选丢了，等于"省钱省掉了下游的线索"（m01712 复盘）。
    candidates: tuple[tuple[int, float], ...] = ()
    born: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "faq_id": self.faq_id,
                "score": self.score,
                "reason": self.reason,
                "subject": self.subject,
                "matched_question": self.matched_question,
                "candidates": [list(item) for item in self.candidates],
                "born": self.born,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "CacheEntry | None":
        try:
            data = json.loads(raw)
            return cls(
                faq_id=data.get("faq_id"),
                score=float(data.get("score") or 0.0),
                reason=str(data.get("reason") or ""),
                subject=data.get("subject"),
                matched_question=data.get("matched_question"),
                candidates=tuple(
                    (int(item[0]), float(item[1]))
                    for item in (data.get("candidates") or [])
                    if isinstance(item, (list, tuple)) and len(item) == 2
                ),
                born=float(data.get("born") or 0.0),
            )
        except Exception:
            return None


class BaseFAQCache:
    """缓存接口：内存实现用于单测，Redis 实现用于线上。"""

    async def get(self, key: str) -> CacheEntry | None:  # pragma: no cover - 抽象
        raise NotImplementedError

    async def put(self, key: str, entry: CacheEntry, ttl: int) -> None:  # pragma: no cover
        raise NotImplementedError

    async def slide(self, key: str, ttl: int) -> None:  # pragma: no cover
        raise NotImplementedError


class InMemoryFAQCache(BaseFAQCache):
    """进程内缓存（单测/无 Redis 环境）。行为与 Redis 版一致，含绝对寿命上限。"""

    def __init__(self, *, max_ttl: int = 6 * 3600) -> None:
        self._data: dict[str, tuple[CacheEntry, float, int]] = {}
        self._max_ttl = max_ttl
        self.slides: list[str] = []

    async def get(self, key: str) -> CacheEntry | None:
        item = self._data.get(key)
        if item is None:
            return None
        entry, expire_at, _ = item
        if time.time() >= expire_at or time.time() - entry.born > self._max_ttl:
            self._data.pop(key, None)
            return None
        return entry

    async def put(self, key: str, entry: CacheEntry, ttl: int) -> None:
        # born 未填时补上：绝对寿命判定依赖它，漏填会让条目"一写就过期"
        entry.born = entry.born or time.time()
        self._data[key] = (entry, time.time() + ttl, ttl)

    async def slide(self, key: str, ttl: int) -> None:
        item = self._data.get(key)
        if item is None:
            return
        entry = item[0]
        self._data[key] = (entry, time.time() + ttl, ttl)
        self.slides.append(key)

    def remaining(self, key: str) -> float:
        item = self._data.get(key)
        return 0.0 if item is None else max(0.0, item[1] - time.time())


class RedisFAQCache(BaseFAQCache):
    """Redis 实现：get/put/slide 全部 fail-open，异常只记日志。"""

    def __init__(
        self,
        client: Any,
        *,
        max_ttl: int = 6 * 3600,
        slide_ratio: float = 0.5,
    ) -> None:
        self._redis = client
        self._max_ttl = max_ttl
        self._slide_ratio = slide_ratio

    async def get(self, key: str) -> CacheEntry | None:
        try:
            raw = await asyncio.to_thread(self._redis.get, key)
        except Exception as exc:
            log.warning("faq_cache_get_failed", error=str(exc))
            return None
        if raw is None:
            return None
        entry = CacheEntry.from_json(raw)
        if entry is None:
            log.warning("faq_cache_bad_value", key=key)
            return None
        age = time.time() - entry.born
        if age > self._max_ttl:
            # 超过绝对寿命：当作未命中，让上层重新检索并覆盖写（保证答案能改得动）
            log.info("faq_cache_expired_absolute", key=key, age_s=round(age, 1))
            return None
        return entry

    async def put(self, key: str, entry: CacheEntry, ttl: int) -> None:
        entry.born = entry.born or time.time()
        try:
            await asyncio.to_thread(self._redis.set, key, entry.to_json(), ex=ttl)
        except Exception as exc:
            log.warning("faq_cache_put_failed", error=str(exc))

    async def slide(self, key: str, ttl: int) -> None:
        """续命：仅当剩余寿命低于阈值时才写，避免每次命中都产生一次 Redis 写。"""
        try:
            remaining = await asyncio.to_thread(self._redis.ttl, key)
            if remaining is None or remaining < 0:
                return
            if remaining <= max(1.0, ttl * self._slide_ratio):
                await asyncio.to_thread(self._redis.expire, key, ttl)
                log.info("faq_cache_slide", key=key, remaining_s=remaining, new_ttl=ttl)
        except Exception as exc:
            log.warning("faq_cache_slide_failed", error=str(exc))


class SingleFlight:
    """同一 key 并发只算一次（缓存击穿保护）。

    没有它，热门问题在下线/过期的瞬间会有 N 个请求同时穿透到索引与 DB——
    索引是内存的还能扛，等 PR3 换成向量检索就是 N 次 GPU 推理。
    """

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._guard = asyncio.Lock()

    async def lock(self, key: str) -> asyncio.Lock:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    async def release(self, key: str) -> None:
        async with self._guard:
            lock = self._locks.get(key)
            if lock is not None and not lock.locked():
                self._locks.pop(key, None)
