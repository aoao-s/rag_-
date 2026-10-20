"""Redis 客户端工厂：缓存、会话、限流、单飞锁共用。

key 规范见 CLAUDE.md §6.4：`edurag:{env}:{用途}:...`。
socket_timeout 必须显式设置——没有超时的 Redis 调用会把整个请求拖死。
"""

from __future__ import annotations

from typing import Any

from base.config import Settings, get_settings


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

