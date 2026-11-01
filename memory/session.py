"""会话 L0 工作记忆（CLAUDE.md §7.9）：当前 session 的逐字对话窗口。

**双上限**：`MEM_WINDOW_TURNS`（40 轮）**或** `MEM_WINDOW_TOKENS`（28 000），先到先压（§3.7 第 2 条）。
超限时把最旧的 `MEM_COMPRESS_BATCH_TURNS`（10 轮）压进摘要，而不是一次性砍掉——
"窗口 + 摘要"是两级结构，砍掉不压等于让模型失忆。

存储：Redis 单 key 存 JSON（`edurag:{env}:session:{sid}`），TTL `MEM_SESSION_TTL_S`；
Redis 不可用时退化为进程内字典（单副本仍可用，重启即丢）——与 FAQ 缓存的 fail-open 同一条原则。

**并发说明**：单 key 读改写，多 worker 并发写同一 session 时是"后写赢"。
会话天然按用户串行（同一用户不会同时发两条），所以先不做 WATCH/事务；
真要做多标签页并发再说，不提前上乐观锁。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from base.config import Settings
from base.logger import get_logger

log = get_logger(__name__)

Role = Literal["user", "assistant"]


def approx_tokens(text: str) -> int:
    """粗估 token 数。中文按 1 字 ≈ 1 token 估（偏保守，宁可早压不可晚压）。"""
    return len(text or "")


@dataclass(frozen=True, slots=True)
class Turn:
    role: Role
    content: str
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SessionState:
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    compressed: int = 0  # 已被压进摘要的轮数，用于观测"压了多少"
    # 会话归属人（阶段 1 §3.4）。**先到者胜**，由 `SessionStore.claim()` 写入，之后只读。
    # 放在会话状态里而不是另建一张表，是因为它和窗口同生命周期——
    # 会话过期了归属也没有意义，分开存只会多一处会漂移的真相
    owner_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [t.to_dict() for t in self.turns],
            "summary": self.summary,
            "compressed": self.compressed,
            "owner_id": self.owner_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        return cls(
            turns=[
                Turn(role=t.get("role", "user"), content=t.get("content", ""), ts=float(t.get("ts") or 0.0))
                for t in data.get("turns") or []
                if t.get("content")
            ],
            summary=str(data.get("summary") or ""),
            compressed=int(data.get("compressed") or 0),
            owner_id=str(data.get("owner_id") or ""),
        )


class SessionStore:
    """会话窗口的唯一入口。`redis=None` 时用进程内实现（测试与无 Redis 的开发机）。

    `chat_log` 不为空时，每次追加都会**顺带**写一份到 MySQL（`chat_message`）。
    这不是缓存：Redis 的 key 会过期、会被 `down -v` 清掉，而"这个学员上次问了什么"
    是用户数据，丢了没法重算（§5.2）。写库是 best-effort——失败只记日志，
    不能因为 MySQL 抖动就让用户拿不到回答。
    """

    def __init__(self, settings: Settings, redis: Any = None, chat_log: Any = None) -> None:
        self.settings = settings
        self._redis = redis
        self._chat_log = chat_log
        self._local: dict[str, SessionState] = {}

    # ---- 存储层 ----
    def _key(self, sid: str) -> str:
        return f"edurag:{self.settings.app_env}:session:{sid}"

    async def _load(self, sid: str) -> SessionState:
        if self._redis is None:
            return self._local.get(sid) or SessionState()
        import asyncio

        try:
            raw = await asyncio.to_thread(self._redis.get, self._key(sid))
        except Exception as exc:  # 缓存读写必须 fail-open（§5.2）
            log.warning("session_read_failed", error=str(exc))
            return SessionState()
        if not raw:
            return SessionState()
        try:
            return SessionState.from_dict(json.loads(raw))
        except (ValueError, TypeError):
            log.warning("session_bad_value", session_id=sid)
            return SessionState()

    async def _save(self, sid: str, state: SessionState) -> None:
        if self._redis is None:
            self._local[sid] = state
            return
        import asyncio

        try:
            await asyncio.to_thread(
                self._redis.set,
                self._key(sid),
                json.dumps(state.to_dict(), ensure_ascii=False),
                ex=self.settings.mem_session_ttl_s,
            )
        except Exception as exc:
            log.warning("session_write_failed", error=str(exc))

    # ---- 对外 ----
    async def window(self, sid: str) -> list[Turn]:
        return (await self._load(sid)).turns

    async def summary(self, sid: str) -> str:
        return (await self._load(sid)).summary

    async def state(self, sid: str) -> SessionState:
        return await self._load(sid)

    async def owner_of(self, sid: str) -> str | None:
        """会话归属人；`None` 表示**还没有人认领**（首轮，允许绑定）。

        读不到（Redis 抖动 / 值损坏）时 `_load()` 已经 fail-open 成空状态，
        这里因此返回 `None`——那是"当成没认领"，由准入层的 `session_owner_unchecked`
        降级码标记出来，而不是把用户挡在门外。
        """
        return (await self._load(sid)).owner_id or None

    async def claim(self, sid: str, user_id: str) -> str:
        """把会话绑定到首个使用它的用户，返回当前归属人（**已绑定则不覆盖**）。

        幂等：同一个用户重复认领不写库，省掉每轮一次 Redis 往返。
        """
        if not sid or not user_id:
            return ""
        state = await self._load(sid)
        if state.owner_id:
            return state.owner_id
        state.owner_id = user_id
        await self._save(sid, state)
        return user_id

    async def append(
        self, sid: str, role: Role, content: str, *, user_id: str = ""
    ) -> SessionState:
        """追加一轮并按双上限裁剪。返回裁剪后的状态（调用方可能要用摘要）。"""
        state = await self._load(sid)
        state.turns.append(Turn(role=role, content=content, ts=time.time()))
        before = state.compressed
        self._compress(state)
        await self._save(sid, state)
        await self._log_turn(sid, role, content, user_id)
        if state.compressed != before:
            # 摘要只在压过之后写库：每轮都写会把同一份摘要覆盖 40 次，纯属浪费
            await self._log_summary(sid, state, user_id)
        return state

    async def set_summary(self, sid: str, text: str) -> None:
        state = await self._load(sid)
        state.summary = text[: self.settings.mem_summary_max_tokens]
        await self._save(sid, state)

    async def clear(self, sid: str) -> None:
        if self._redis is None:
            self._local.pop(sid, None)
            return
        import asyncio

        try:
            await asyncio.to_thread(self._redis.delete, self._key(sid))
        except Exception as exc:
            log.warning("session_clear_failed", error=str(exc))

    async def _log_turn(self, sid: str, role: Role, content: str, user_id: str) -> None:
        if self._chat_log is None or not user_id:
            return
        import asyncio

        try:
            await asyncio.to_thread(self._chat_log.append_turn, sid, user_id, role, content)
        except Exception as exc:  # 落库失败只丢真相源，不影响本轮对话
            log.warning("chat_message_persist_failed", error=str(exc))

    async def _log_summary(self, sid: str, state: SessionState, user_id: str) -> None:
        if self._chat_log is None or not user_id:
            return
        import asyncio

        try:
            await asyncio.to_thread(
                self._chat_log.save_summary, sid, user_id, state.summary, state.compressed
            )
        except Exception as exc:
            log.warning("session_summary_persist_failed", error=str(exc))

    # ---- 压缩 ----
    def _compress(self, state: SessionState) -> None:
        """先到先压：轮数或 token 任一超限就吃掉最旧的 N 轮，滚动进摘要。

        压缩是**抽取式**的（见 `memory/summary.py`）：不调模型也能跑，
        有 LLM 时由调用方另行覆盖摘要。这里只保证"窗口不会无限长"。
        """
        from memory.summary import digest

        s = self.settings
        while len(state.turns) > s.mem_window_turns or _tokens(state.turns) > s.mem_window_tokens:
            batch = state.turns[: s.mem_compress_batch_turns]
            if not batch:
                break
            state.turns = state.turns[s.mem_compress_batch_turns :]
            state.summary = digest(
                state.summary, batch, max_chars=s.mem_summary_max_tokens
            )
            state.compressed += len(batch)


def _tokens(turns: list[Turn]) -> int:
    return sum(approx_tokens(t.content) for t in turns)
