"""FAQ 持久化：MySQL 实现 + 内存实现（测试 / 无库演示）。

为什么用 Protocol 而不是基类：qa_system 只依赖"能取到 published 语料"这一契约，
MySQL、内存、未来的 ES/CSV 都能替换，且测试不需要起数据库。
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Protocol

from base.labels import normalize_subject
from base.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FAQRecord:
    faq_id: int
    subject: str | None
    question: str
    answer: str
    status: str = "published"
    content_hash: str = ""

    def with_hash(self) -> FAQRecord:
        return replace(self, content_hash=question_hash(self.question))


def question_hash(question: str) -> str:
    return hashlib.md5(question.strip().encode("utf-8")).hexdigest()


class FAQRepository(Protocol):
    async def fetch_published(self) -> list[FAQRecord]: ...

    async def fetch_aliases(self) -> list[tuple[int, str]]: ...

    async def fetch_by_id(self, faq_id: int) -> FAQRecord | None: ...

    async def deprecate(self, faq_id: int, operator: str) -> bool: ...

    async def publish(self, faq_id: int, operator: str) -> bool: ...


class InMemoryFAQRepository:
    """内存实现：单测与本地无库演示用；线程安全，行为与 MySQL 版本保持一致。"""

    def __init__(self, records: list[FAQRecord] | None = None) -> None:
        self._lock = threading.RLock()
        self._rows: dict[int, FAQRecord] = {}
        self._aliases: list[tuple[int, str]] = []
        self._next_id = 1
        for rec in records or []:
            self.insert(rec)

    def insert(self, record: FAQRecord, aliases: list[str] | None = None) -> int:
        with self._lock:
            faq_id = record.faq_id or self._next_id
            self._next_id = max(self._next_id, faq_id + 1)
            self._rows[faq_id] = replace(record.with_hash(), faq_id=faq_id)
            if aliases:
                self._aliases.extend((faq_id, alias) for alias in aliases if alias)
            return faq_id

    async def fetch_published(self) -> list[FAQRecord]:
        with self._lock:
            return [row for row in self._rows.values() if row.status == "published"]

    async def fetch_aliases(self) -> list[tuple[int, str]]:
        with self._lock:
            published = {row.faq_id for row in self._rows.values() if row.status == "published"}
            return [(faq_id, alias) for faq_id, alias in self._aliases if faq_id in published]

    async def fetch_by_id(self, faq_id: int) -> FAQRecord | None:
        with self._lock:
            return self._rows.get(faq_id)

    async def deprecate(self, faq_id: int, operator: str) -> bool:
        with self._lock:
            row = self._rows.get(faq_id)
            if row is None:
                return False
            self._rows[faq_id] = replace(row, status="deprecated")
            return True

    async def publish(self, faq_id: int, operator: str) -> bool:
        with self._lock:
            row = self._rows.get(faq_id)
            if row is None:
                return False
            self._rows[faq_id] = replace(row, status="published")
            return True

    async def upsert_many(self, records: list[FAQRecord]) -> int:
        return sum(1 for rec in records if self.insert(rec))


class MySQLFAQRepository:
    """生产实现。只允许在启动/发布/重建路径调用（同步驱动包在 async 里跑）。"""

    def __init__(self, mysql: Any) -> None:
        self._db = mysql

    def _query(self, op: str, sql: str, params: tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
        """统一出口：所有 MySQL 访问都带 op/rows/cost 日志。

        没有这行日志时，"命中来自 MySQL 还是缓存"在故障复盘里只能靠猜；
        慢查询（例如 faq 表无索引时的 LIKE）也只能看到接口变慢、看不到慢在哪。
        """
        started = time.perf_counter()
        try:
            rows = self._db.query(sql, params)
        except Exception as exc:
            log.error(
                "faq_db_query_failed",
                op=op,
                cost_ms=round((time.perf_counter() - started) * 1000, 1),
                error=str(exc),
            )
            raise
        cost_ms = (time.perf_counter() - started) * 1000
        log.info("faq_db_query", op=op, rows=len(rows), cost_ms=round(cost_ms, 1))
        return rows

    async def fetch_published(self) -> list[FAQRecord]:
        rows = self._query(
            "fetch_published",
            "SELECT id, subject_name, question, answer, status, content_hash "
            "FROM faq WHERE status = 'published' ORDER BY id",
        )
        return [
            FAQRecord(
                faq_id=int(row["id"]),
                subject=normalize_subject(row.get("subject_name")),
                question=row["question"] or "",
                answer=row["answer"] or "",
                status=row.get("status") or "published",
                content_hash=row.get("content_hash") or "",
            )
            for row in rows
        ]

    async def fetch_aliases(self) -> list[tuple[int, str]]:
        rows = self._query(
            "fetch_aliases",
            "SELECT a.faq_id, a.alias FROM faq_alias a JOIN faq f ON f.id = a.faq_id "
            "WHERE f.status = 'published'",
        )
        return [(int(row["faq_id"]), row["alias"]) for row in rows]

    async def fetch_by_id(self, faq_id: int) -> FAQRecord | None:
        rows = self._query(
            "fetch_by_id",
            "SELECT id, subject_name, question, answer, status, content_hash FROM faq WHERE id = %s",
            (faq_id,),
        )
        if not rows:
            return None
        row = rows[0]
        return FAQRecord(
            faq_id=int(row["id"]),
            subject=normalize_subject(row.get("subject_name")),
            question=row["question"] or "",
            answer=row["answer"] or "",
            status=row.get("status") or "draft",
            content_hash=row.get("content_hash") or "",
        )

    async def deprecate(self, faq_id: int, operator: str) -> bool:
        affected = self._db.execute(
            "UPDATE faq SET status = 'deprecated', updated_at = NOW(), updated_by = %s WHERE id = %s",
            (operator, faq_id),
        )
        log.info("faq_db_deprecate", faq_id=faq_id, operator=operator, affected=affected)
        return affected > 0

    async def publish(self, faq_id: int, operator: str) -> bool:
        affected = self._db.execute(
            "UPDATE faq SET status = 'published', updated_at = NOW(), updated_by = %s "
            "WHERE id = %s AND status IN ('draft', 'pending', 'deprecated')",
            (operator, faq_id),
        )
        log.info("faq_db_publish", faq_id=faq_id, operator=operator, affected=affected)
        return affected > 0

