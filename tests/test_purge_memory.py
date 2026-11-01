"""物理清理单测（CLAUDE.md §7.9 / 实施手册 §8.4，M2）。

清理是**不可逆**操作，所以这里盯的四件事都是"错了会悄无声息"的：
1. `memory_atom` 只清 `deleted_at`，**绝不碰 `valid_to`** —— 后者是 supersede 留下的历史链；
2. `memory_audit` 不在清理范围内 —— 审计必须比数据活得久；
3. `--dry-run` 只 `SELECT COUNT`，一行都不删；
4. 窗口 ≤ 0 直接拒绝 —— "删掉包括今天在内的全部"不是参数写错，是数据事故。

用假 MySQL（只记录 SQL 与参数）：这一层证明的是"发出去的语句对不对"，
不是"MySQL 真删了没有"。断言参数里带得动窗口天数，与 §7.2.7-G 同一条教训——
字符串看着对、值没到达，是最难发现的那种错。
"""

from __future__ import annotations

import pytest

from scripts import purge_memory
from scripts.purge_memory import TARGETS, count, purge, select


class _FakeMySQL:
    """记录每条 SQL；`rowcounts` 按调用顺序回放受影响行数。"""

    def __init__(self, *, count_n: int = 0, rowcounts: list[int] | None = None) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self._count_n = count_n
        self._rowcounts = list(rowcounts or [])
        self._raise_on: dict[str, Exception] = {}
        self.closed = False

    def fail_on(self, table: str, exc: Exception) -> None:
        self._raise_on[table] = exc

    def query(self, sql: str, args: tuple[object, ...] = ()) -> list[dict[str, object]]:
        self.calls.append((sql, tuple(args)))
        for table, exc in self._raise_on.items():
            if f"`{table}`" in sql:
                raise exc
        return [{"n": self._count_n}]

    def execute(self, sql: str, args: tuple[object, ...] = ()) -> int:
        self.calls.append((sql, tuple(args)))
        for table, exc in self._raise_on.items():
            if f"`{table}`" in sql:
                raise exc
        return self._rowcounts.pop(0) if self._rowcounts else 0

    def close(self) -> None:
        self.closed = True


def _target(table: str) -> purge_memory.Target:
    return next(t for t in TARGETS if t.table == table)


# ------------------------------------------------------------------ 口径
def test_atom_purge_is_keyed_on_deleted_at_not_valid_to() -> None:
    """`valid_to` 是"被新事实取代"，不是"用户要求删"。按它清理等于抹掉链视图。"""
    atom = _target("memory_atom")
    assert "deleted_at IS NOT NULL" in atom.where
    assert "valid_to" not in atom.where
    # 三张表都不许拿 valid_to 当到期依据（其余两张根本没有这一列，写错就是 SQL 报错）
    assert not any("valid_to" in t.where for t in TARGETS)


def test_audit_table_is_never_purged() -> None:
    """审计的价值就是比数据活得久：删掉"谁删了这条记忆"的记录等于没有审计。"""
    tables = {t.table for t in TARGETS}
    assert "memory_audit" not in tables
    assert tables == {"memory_atom", "chat_message", "session_summary"}


def test_session_expiry_uses_updated_at() -> None:
    """摘要按 `updated_at`：用 `created_at` 会把刚压缩过摘要的老会话提前删掉。"""
    assert "updated_at" in _target("session_summary").where
    assert "created_at" in _target("chat_message").where


# ------------------------------------------------------------------ 预演
def test_dry_run_only_counts_and_deletes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMySQL(count_n=4)
    monkeypatch.setattr(purge_memory, "MySQL", lambda settings=None: fake)

    assert purge_memory.main(["--dry-run", "--days", "7"]) == 0
    assert fake.calls, "dry-run 至少该查一次条数"
    assert all(sql.startswith("SELECT COUNT(*)") for sql, _ in fake.calls)
    assert not any("DELETE" in sql for sql, _ in fake.calls)
    assert fake.closed  # 连接池要关，脚本跑完不能挂着连接


def test_dry_run_reports_the_window_it_used() -> None:
    fake = _FakeMySQL(count_n=2)
    target = _target("chat_message")
    assert purge_memory.purge(fake, target, 7, batch=100, dry_run=True) == 2

    sql, params = fake.calls[-1]
    assert "INTERVAL %s DAY" in sql and params == (7,)  # 窗口真的到达了 SQL


# ------------------------------------------------------------------ 真删
def test_real_run_batches_until_a_short_batch() -> None:
    """分批删到"不足一批"为止；一次删十万行会把同库的写入全卡住。"""
    fake = _FakeMySQL(rowcounts=[2000, 2000, 7])
    deleted = purge(fake, _target("memory_atom"), 30, batch=2000, dry_run=False)

    assert deleted == 4007
    assert len(fake.calls) == 3  # 4007 行分成 3 批，没有白跑第 4 次
    for sql, params in fake.calls:
        assert sql.startswith("DELETE FROM `memory_atom`")
        assert "LIMIT 2000" in sql
        assert params == (30,)


def test_empty_table_makes_exactly_one_delete_attempt() -> None:
    fake = _FakeMySQL(rowcounts=[0])
    assert purge(fake, _target("chat_message"), 30, batch=500, dry_run=False) == 0
    assert len(fake.calls) == 1


def test_count_reads_the_single_rowshape_mysql_returns() -> None:
    """`COUNT(*) AS n` + DictCursor 给的是 `[{"n": k}]`；取错 key 会变成静默 0。"""
    assert count(_FakeMySQL(count_n=9), _target("chat_message"), 30) == 9


# ------------------------------------------------------------------ 失败与拒绝
@pytest.mark.parametrize("days", [0, -1])
def test_non_positive_window_refuses_before_touching_the_db(
    monkeypatch: pytest.MonkeyPatch, days: int
) -> None:
    built: list[object] = []
    monkeypatch.setattr(purge_memory, "MySQL", lambda settings=None: built.append(1))
    assert purge_memory.main(["--days", str(days)]) == 2
    assert built == [], "参数不合理时不该连库"


def test_unknown_table_name_fails_loudly() -> None:
    with pytest.raises(SystemExit) as excinfo:
        select(TARGETS, "memory_audit")
    assert "memory_audit" in str(excinfo.value)


def test_one_failing_table_does_not_block_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    """一张表报错不该让另外两张的清理半途而废，但退出码必须非 0（cron 要能发现）。"""
    fake = _FakeMySQL(rowcounts=[0, 0, 0])
    fake.fail_on("chat_message", RuntimeError("lock wait timeout"))
    monkeypatch.setattr(purge_memory, "MySQL", lambda settings=None: fake)

    assert purge_memory.main(["--days", "30"]) == 1
    touched = {sql.split("`")[1] for sql, _ in fake.calls if sql.startswith(("DELETE", "SELECT COUNT"))}
    assert touched == {"memory_atom", "chat_message", "session_summary"}
