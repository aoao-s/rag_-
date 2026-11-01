"""记忆与会话数据的物理清理（CLAUDE.md §7.9 / 实施手册 §8.4，M2）。

用法：
    python scripts/purge_memory.py                  # 真删（窗口取 MEM_PURGE_AFTER_DAYS）
    python scripts/purge_memory.py --dry-run        # 只出条数，一行都不删
    python scripts/purge_memory.py --days 7         # 临时用别的窗口（运维排障/演练）
    python scripts/purge_memory.py --table memory_atom

三张表，三种到期口径——**它们不是同一件事**，所以别合成一条 SQL：

| 表 | 到期依据 | 为什么是这个 |
|---|---|---|
| `memory_atom` | `deleted_at` 非空且超窗口 | 只有**用户要求删掉**的才会写 `deleted_at`（`forget` 的两种模式、`clear_all`）。`valid_to` 是 supersede 留下的历史链，链视图 `GET /v1/memory/chains/{id}` 要能看——按它清理等于把"这条事实以前是什么"抹掉 |
| `chat_message` | `created_at` 超窗口 | 逐字历史（L0 真相源），也是唯一会无限增长的表 |
| `session_summary` | `updated_at` 超窗口 | 会话停止更新后才连摘要一起到期；用 `created_at` 会把刚压缩过摘要的老会话提前删掉 |

**不清 `memory_audit`**：审计的意义就是比数据活得久。删除行为的审计跟着数据一起被删，
"谁在什么时候删了这条记忆"就再也查不出来——那等于没有审计。

**不在请求路径上跑**（手册 §5.1 的 M2 边界）：每日定时或在维护窗口手动执行；
`--dry-run` 先出条数，确认量级再真删。

性能注意：`chat_message` 上只有 `(session_id, id)` 与 `(tenant_id, user_id, created_at)` 两个索引，
按 `created_at` 单列比较走不到前缀，大表上是全表扫。目前表还小（记忆模块刚落地），
真到了百万行量级再补 `created_at` 索引——那时是一次 DDL，不该现在顺手加进 M2。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.config import get_settings  # noqa: E402
from base.db import MySQL  # noqa: E402
from base.logger import get_logger  # noqa: E402

log = get_logger("scripts.purge_memory")


@dataclass(frozen=True, slots=True)
class Target:
    """一张表的清理口径。`where` 里**只能有一个** `%s`，就是窗口天数。"""

    table: str
    where: str
    reason: str


TARGETS: tuple[Target, ...] = (
    Target(
        "memory_atom",
        "deleted_at IS NOT NULL AND deleted_at < NOW() - INTERVAL %s DAY",
        "软删超期",
    ),
    Target("chat_message", "created_at < NOW() - INTERVAL %s DAY", "逐字历史超期"),
    Target("session_summary", "updated_at < NOW() - INTERVAL %s DAY", "会话摘要超期"),
)

_TABLES = tuple(t.table for t in TARGETS)


def select(targets: tuple[Target, ...], table: str | None) -> tuple[Target, ...]:
    """按 `--table` 取子集；名字写错直接报错，不能静默什么都不删。"""
    if not table:
        return targets
    picked = tuple(t for t in targets if t.table == table)
    if not picked:
        raise SystemExit(f"未知表名：{table}；可选：{' / '.join(_TABLES)}")
    return picked


def count(mysql: MySQL, target: Target, days: int) -> int:
    rows = mysql.query(
        f"SELECT COUNT(*) AS n FROM `{target.table}` WHERE {target.where}", (days,)
    )
    return int(rows[0]["n"]) if rows else 0


def purge(mysql: MySQL, target: Target, days: int, *, batch: int, dry_run: bool) -> int:
    """分批删除，返回删掉的条数。

    **必须带 `LIMIT`**：一条 `DELETE` 删十万行会长时间持锁，把同一个库上的 FAQ 读与记忆写一起卡住；
    分批之后每次只锁一小段，中间让出。
    """
    if dry_run:
        rows = count(mysql, target, days)
        log.info("memory_purge_planned", table=target.table, rows=rows, days=days, reason=target.reason)
        return rows

    total = 0
    while True:
        # 批大小走常量拼接而不是占位符：pymysql 的 `LIMIT %s` 会渲染成带引号的
        # `LIMIT '2000'`，在严格模式下是语法错误。数值已经过 int() 校验，无注入面。
        deleted = mysql.execute(
            f"DELETE FROM `{target.table}` WHERE {target.where} LIMIT {int(batch)}", (days,)
        )
        total += deleted
        if deleted < batch:
            break
    log.info("memory_purge_done", table=target.table, rows=total, days=days, reason=target.reason)
    return total


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="记忆/会话数据的物理清理（幂等，可重复跑）")
    parser.add_argument("--dry-run", action="store_true", help="只统计待删条数，不执行删除")
    parser.add_argument("--days", type=int, default=None, help="覆盖 MEM_PURGE_AFTER_DAYS")
    parser.add_argument("--batch", type=int, default=None, help="覆盖 MEM_PURGE_BATCH")
    parser.add_argument("--table", default=None, help=f"只清理一张表：{' / '.join(_TABLES)}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    settings = get_settings()

    days = settings.mem_purge_after_days if args.days is None else args.days
    batch = settings.mem_purge_batch if args.batch is None else args.batch
    # 窗口 ≤ 0 等于"删掉包括今天在内的全部"：这不是运维参数写错，是数据事故
    if days <= 0:
        log.error("config_problem", detail=f"清理窗口必须为正数，当前 {days}")
        return 2
    if batch <= 0:
        log.error("config_problem", detail=f"批大小必须为正数，当前 {batch}")
        return 2

    targets = select(TARGETS, args.table)
    mysql = MySQL(settings)
    try:
        results: list[tuple[str, int]] = []
        failed: list[str] = []
        for target in targets:
            try:
                results.append((target.table, purge(mysql, target, days, batch=batch, dry_run=args.dry_run)))
            except Exception as exc:  # 一张表失败不拦其余的表，但退出码必须非 0
                failed.append(target.table)
                log.error("memory_purge_failed", table=target.table, error=str(exc))
    finally:
        mysql.close()

    verb = "待删" if args.dry_run else "已删"
    summary = ", ".join(f"{table} {n}" for table, n in results) or "无"
    print(f"[{'dry-run' if args.dry_run else 'ok'}] 窗口={days}天 {verb}: {summary}")
    if failed:
        print(f"[fail] 清理失败的表: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
