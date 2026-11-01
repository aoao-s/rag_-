"""会话窗口单测（CLAUDE.md §7.9 L0：双上限 + 摘要滚动）。

这一层只有一件事会出错，而且出错时很难被发现：**窗口无限长**。
它不会报错，只会让 prompt 慢慢超预算，直到某一天生成开始 400。
所以每条用例都盯着"压了没有、压完之后摘要还记不记得"，而不是"存进去了没有"。
"""

from __future__ import annotations

import asyncio

from base.config import Settings
from memory.session import SessionStore, approx_tokens


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "mem_window_turns": 4,
        "mem_window_tokens": 10_000,
        "mem_compress_batch_turns": 2,
        "mem_summary_max_tokens": 400,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _store(**overrides: object) -> SessionStore:
    # redis=None → 进程内实现（与生产同一条代码路径，只换了存取后端）
    return SessionStore(_settings(**overrides))


def test_turn_cap_compresses_oldest_batch() -> None:
    store = _store()

    async def run() -> None:
        for i in range(5):  # 上限 4 轮，压 2 轮
            await store.append("s1", "user", f"第{i}个问题")

        turns = await store.window("s1")
        state = await store.state("s1")
        # 留 4 轮，但刚从 5 压到 3 再追加过；唯一硬约束是不能超过上限
        assert len(turns) <= 4
        assert state.compressed >= 2
        assert "第0个问题" in state.summary  # 压掉的进了摘要，不是删掉
        assert all(t.content != "第0个问题" for t in turns)

    asyncio.run(run())


def test_token_cap_triggers_before_turn_cap() -> None:
    # 轮数还很富裕，但单轮很长：先到先压（§3.7 第 2 条）
    store = _store(mem_window_turns=40, mem_window_tokens=100)

    async def run() -> None:
        long_turn = "勾" * 60
        for _ in range(3):
            await store.append("s2", "user", long_turn)
        state = await store.state("s2")
        assert state.compressed > 0 and state.turns
        assert sum(approx_tokens(t.content) for t in state.turns) <= 100 + len(long_turn)

    asyncio.run(run())


def test_summary_is_written_to_truth_source_only_when_compressed() -> None:
    """每轮都写摘要等于把同一份内容覆盖 40 次；只有真压过才落库。"""

    class ChatLog:
        def __init__(self) -> None:
            self.turns: list[tuple[str, str, str, str]] = []
            self.summaries: list[tuple[str, str, str, int]] = []

        def append_turn(self, sid: str, user_id: str, role: str, content: str) -> None:
            self.turns.append((sid, user_id, role, content))

        def save_summary(self, sid: str, user_id: str, summary: str, compressed: int) -> None:
            self.summaries.append((sid, user_id, summary, compressed))

    store = _store()
    log = ChatLog()
    store._chat_log = log  # 替身：只验证"什么时候调用"，不验证 SQL

    async def run() -> None:
        await store.append("s3", "user", "短问题", user_id="u1")
        assert log.turns and not log.summaries  # 第一轮没压 → 不写摘要

        for i in range(6):
            await store.append("s3", "user", f"第{i}个问题", user_id="u1")
        assert log.summaries and log.summaries[-1][3] > 0

    asyncio.run(run())


def test_unowned_session_is_not_persisted() -> None:
    """没有 user_id 的会话写进 MySQL 就再也删不掉（记忆页按 user_id 清），一律不落库。"""

    class ChatLog:
        def __init__(self) -> None:
            self.turns = 0

        def append_turn(self, *args: object) -> None:
            self.turns += 1

        def save_summary(self, *args: object) -> None:
            raise AssertionError("无主会话不该写摘要")

    store = _store()
    log = ChatLog()
    store._chat_log = log

    async def run() -> None:
        await store.append("s4", "user", "勾股定理怎么证明")
        assert log.turns == 0

    asyncio.run(run())


def test_persist_failure_is_fail_open() -> None:
    """MySQL 抖动不能让人拿不到回答：写库失败只记日志。"""

    class Boom:
        def append_turn(self, *args: object) -> None:
            raise RuntimeError("mysql down")

        def save_summary(self, *args: object) -> None:
            raise RuntimeError("mysql down")

    store = _store()
    store._chat_log = Boom()

    async def run() -> None:
        state = await store.append("s5", "user", "勾股定理怎么证明", user_id="u1")
        assert state.turns  # 窗口照常写进 Redis/内存

    asyncio.run(run())


def test_clear_removes_the_window() -> None:
    store = _store()

    async def run() -> None:
        await store.append("s6", "user", "勾股定理怎么证明")
        await store.clear("s6")
        assert await store.window("s6") == []

    asyncio.run(run())


def test_corrupt_value_degrades_to_empty_window() -> None:
    """缓存值损坏当作空窗口，不能把整个请求打挂（与 FAQ 缓存同一条 fail-open）。"""

    class Broken:
        def get(self, key: str) -> str:
            return "{not json"

        def set(self, *args: object, **kwargs: object) -> None:
            return None

        def delete(self, key: str) -> None:
            return None

    store = SessionStore(_settings(), redis=Broken())

    async def run() -> None:
        assert await store.window("s7") == []

    asyncio.run(run())
