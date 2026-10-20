"""MySQL 连接池（唯一允许建连接的地方，业务层只拿 repository）。

设计要点：
- PooledDB 懒加载：导入本模块不会连库，测试与纯离线脚本因此可以直接跑；
- DictCursor 统一返回 dict，避免下游再拼元组下标；
- 连接池参数来自 Settings，OOM/高并发时调 MYSQL_POOL_MAX。
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from base.config import Settings, get_settings
from base.logger import get_logger

log = get_logger(__name__)


class MySQL:
    """线程安全的连接池封装；构造时不连库。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._pool: Any = None
        self._lock = threading.Lock()

    def _build_pool(self) -> Any:
        import pymysql
        from dbutils.pooled_db import PooledDB

        s = self._settings
        return PooledDB(
            creator=pymysql,
            maxconnections=s.mysql_pool_max,
            mincached=2,
            maxcached=min(8, s.mysql_pool_max),
            blocking=True,
            ping=1,
            charset="utf8mb4",
            host=s.mysql_host,
            port=s.mysql_port,
            user=s.mysql_user,
            password=s.mysql_password,
            database=s.mysql_database,
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=True,
        )

    def pool(self) -> Any:
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    self._pool = self._build_pool()
                    log.info("mysql_pool_ready", host=self._settings.mysql_host, db=self._settings.mysql_database)
        return self._pool

    @contextmanager
    def cursor(self) -> Iterator[Any]:
        conn = self.pool().connection()
        try:
            cur = conn.cursor()
            try:
                yield cur
            finally:
                cur.close()
        finally:
            conn.close()

    def query(self, sql: str, args: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        with self.cursor() as cur:
            cur.execute(sql, args or ())
            return list(cur.fetchall() or [])

    def execute(self, sql: str, args: tuple[Any, ...] | None = None) -> int:
        with self.cursor() as cur:
            return cur.execute(sql, args or ())

    def executemany(self, sql: str, seq: list[tuple[Any, ...]]) -> int:
        if not seq:
            return 0
        with self.cursor() as cur:
            return cur.executemany(sql, seq)

    def ping(self) -> bool:
        try:
            self.query("SELECT 1 AS ok")
            return True
        except Exception as exc:  # 健康检查不允许抛异常
            log.warning("mysql_ping_failed", error=str(exc))
            return False

    def close(self) -> None:
        if self._pool is not None:
            try:
                self._pool.close()
            finally:
                self._pool = None

