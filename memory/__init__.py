"""记忆模块（CLAUDE.md §7.3 / §7.9）。

四层记忆里本包负责 L0/L1/L2/L3 的存取，**不负责**把它们拼进 prompt（那是编排层的事）：

- L0 工作记忆 → `session.SessionStore`（Redis 热 + MySQL 真相源）
- L1 会话摘要 → `summary.digest`（抽取式）/ `SessionStore._compress`
- L2 原子记忆 → `service.MemoryService` + `store.AtomRepo`
- L3 用户画像 → `MemoryService.recall()` 里由身份类事实渲染

边界（§7.0）：本包不 import `qa_system` / `rag_system`，不检索知识库，
也不参与事实性回答——它只提供背景。
"""

from memory.extractor import Candidate, extract, gated
from memory.service import MemoryService
from memory.session import SessionState, SessionStore, Turn
from memory.store import Atom, ChatLogRepo, InMemoryAtomRepo, MySQLAtomRepo, new_atom_id

__all__ = [
    "Atom",
    "Candidate",
    "ChatLogRepo",
    "InMemoryAtomRepo",
    "MemoryService",
    "MySQLAtomRepo",
    "SessionState",
    "SessionStore",
    "Turn",
    "extract",
    "gated",
    "new_atom_id",
]
