"""记忆的持久化层（CLAUDE.md §7.9）：两张表的读写 + 两个实现。

为什么 MySQL 与内存各一份：`qa_system` 就是这么做的（`qa_db.py`），
测试与离线脚本不该被数据库拖住，而生产必须落 MySQL——**Redis 只放可重算的东西，
会话与记忆是用户数据，不能只活在会过期的缓存里**（§5.2）。

时间列统一用 `DATETIME`，Python 侧用 epoch 秒（float）。转换只在本文件做，
上层拿到的时间戳永远是同一种东西，免得到处 `datetime` / `str` 混着比。
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, replace
from typing import Any, Protocol

from base.db import MySQL
from base.logger import get_logger

log = get_logger(__name__)

FACT_TYPES = ("preference", "identity", "constraint", "decision", "progress")


@dataclass(frozen=True, slots=True)
class Atom:
    """一条原子记忆。`valid_to` / `deleted_at` 都为空才是"活着的"。"""

    id: str
    user_id: str
    fact_type: str
    fact_key: str
    content: str
    confidence: float
    mention_count: int = 1
    source_sid: str = ""
    source_turn: int = 0
    source_span: str = ""
    valid_from: float = 0.0
    valid_to: float | None = None
    superseded_by: str | None = None
    chain_root_id: str | None = None
    deleted_at: float | None = None
    # 结构化坐标，取值域是 `base/labels` 的 SUBJECTS / GRADES；空串 = 未归类。
    # 放在末尾且带默认值，是为了不动现有的构造点（`_merge` / `mark_*` 都靠 `to_dict()` 展开）。
    subject: str = ""
    grade: str = ""

    @property
    def alive(self) -> bool:
        return self.valid_to is None and self.deleted_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "fact_type": self.fact_type,
            "fact_key": self.fact_key,
            "content": self.content,
            "confidence": self.confidence,
            "mention_count": self.mention_count,
            "source_sid": self.source_sid,
            "source_turn": self.source_turn,
            "source_span": self.source_span,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "superseded_by": self.superseded_by,
            "chain_root_id": self.chain_root_id,
            "deleted_at": self.deleted_at,
            "subject": self.subject,
            "grade": self.grade,
        }


def new_atom_id() -> str:
    return uuid.uuid4().hex[:24]


class AtomRepo(Protocol):
    def list_alive(self, user_id: str) -> list[Atom]: ...
    def list_all(self, user_id: str) -> list[Atom]: ...
    def get(self, atom_id: str) -> Atom | None: ...
    def insert(self, atom: Atom) -> None: ...
    def update(self, atom: Atom) -> None: ...
    def count_alive(self, user_id: str) -> int: ...
    def chain(self, root_id: str) -> list[Atom]: ...


class InMemoryAtomRepo:
    """测试与离线用。行为与 MySQL 版逐条对齐（同一批单测断言两边）。"""

    def __init__(self, atoms: list[Atom] | None = None) -> None:
        self._atoms: dict[str, Atom] = {a.id: a for a in atoms or []}
        # 审计行按写入顺序攒着：M4 的验收项是"所有写操作落 memory_audit"，
        # 内存版没有这张表，就把行留在属性里——否则这条 DoD 只能连库才测得出来。
        self.audits: list[dict[str, Any]] = []

    def list_alive(self, user_id: str) -> list[Atom]:
        return [a for a in self._atoms.values() if a.user_id == user_id and a.alive]

    def list_all(self, user_id: str) -> list[Atom]:
        return [a for a in self._atoms.values() if a.user_id == user_id]

    def get(self, atom_id: str) -> Atom | None:
        return self._atoms.get(atom_id)

    def insert(self, atom: Atom) -> None:
        self._atoms[atom.id] = atom

    def update(self, atom: Atom) -> None:
        self._atoms[atom.id] = atom

    def count_alive(self, user_id: str) -> int:
        return len(self.list_alive(user_id))

    def chain(self, root_id: str) -> list[Atom]:
        return sorted(
            (a for a in self._atoms.values() if (a.chain_root_id or a.id) == root_id),
            key=lambda a: a.valid_from,
        )

    def all(self) -> list[Atom]:
        return list(self._atoms.values())

    def audit(
        self, atom_id: str, action: str, user_id: str, detail: str = "", operator: str = ""
    ) -> None:
        """与 `MySQLAtomRepo.audit` 同参同序。**不吞异常**的差别是有意的：
        内存版没有网络、没有约束，写不进去说明代码错了，应该在测试里炸出来。"""
        self.audits.append(
            {
                "atom_id": atom_id,
                "action": action,
                "user_id": user_id,
                "operator": operator or user_id,
                "detail": detail[:500],
            }
        )


_COLS = (
    "id, tenant_id, user_id, fact_type, fact_key, content, confidence, mention_count, "
    "source_sid, source_turn, source_span, valid_from, valid_to, superseded_by, "
    "chain_root_id, deleted_at, subject, grade"
)


class MySQLAtomRepo:
    """`memory_atom` 表（§7.3.3 DDL）。`tenant_id` 现在固定 `default`，接入层鉴权后从 token 取。"""

    def __init__(self, mysql: MySQL, tenant_id: str = "default") -> None:
        self.mysql = mysql
        self.tenant_id = tenant_id

    def list_alive(self, user_id: str) -> list[Atom]:
        rows = self.mysql.query(
            f"SELECT {_COLS} FROM memory_atom "
            "WHERE tenant_id=%s AND user_id=%s AND valid_to IS NULL AND deleted_at IS NULL "
            "ORDER BY valid_from DESC",
            (self.tenant_id, user_id),
        )
        return [_row_to_atom(r) for r in rows]

    def get(self, atom_id: str) -> Atom | None:
        rows = self.mysql.query(
            f"SELECT {_COLS} FROM memory_atom WHERE id=%s", (atom_id,)
        )
        return _row_to_atom(rows[0]) if rows else None

    def list_all(self, user_id: str) -> list[Atom]:
        rows = self.mysql.query(
            f"SELECT {_COLS} FROM memory_atom WHERE tenant_id=%s AND user_id=%s "
            "ORDER BY valid_from DESC",
            (self.tenant_id, user_id),
        )
        return [_row_to_atom(r) for r in rows]

    def insert(self, atom: Atom) -> None:
        self.mysql.execute(
            "INSERT INTO memory_atom (id, tenant_id, user_id, fact_type, fact_key, content, "
            "confidence, mention_count, source_sid, source_turn, source_span, valid_from, "
            "valid_to, superseded_by, chain_root_id, deleted_at, subject, grade, created_at) VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),"
            "CASE WHEN %s IS NULL THEN NULL ELSE FROM_UNIXTIME(%s) END,"
            "%s,%s,CASE WHEN %s IS NULL THEN NULL ELSE FROM_UNIXTIME(%s) END,%s,%s,NOW())",
            (
                atom.id,
                self.tenant_id,
                atom.user_id,
                atom.fact_type,
                atom.fact_key,
                atom.content,
                atom.confidence,
                atom.mention_count,
                atom.source_sid,
                atom.source_turn,
                atom.source_span,
                atom.valid_from,
                atom.valid_to,
                atom.valid_to,
                atom.superseded_by,
                atom.chain_root_id,
                atom.deleted_at,
                atom.deleted_at,
                atom.subject,
                atom.grade,
            ),
        )

    def update(self, atom: Atom) -> None:
        self.mysql.execute(
            "UPDATE memory_atom SET content=%s, confidence=%s, mention_count=%s, "
            "valid_to=CASE WHEN %s IS NULL THEN NULL ELSE FROM_UNIXTIME(%s) END, "
            "superseded_by=%s, deleted_at=CASE WHEN %s IS NULL THEN NULL ELSE FROM_UNIXTIME(%s) END, "
            "subject=%s, grade=%s WHERE id=%s",
            (
                atom.content,
                atom.confidence,
                atom.mention_count,
                atom.valid_to,
                atom.valid_to,
                atom.superseded_by,
                atom.deleted_at,
                atom.deleted_at,
                atom.subject,
                atom.grade,
                atom.id,
            ),
        )

    def count_alive(self, user_id: str) -> int:
        rows = self.mysql.query(
            "SELECT COUNT(*) AS n FROM memory_atom WHERE tenant_id=%s AND user_id=%s "
            "AND valid_to IS NULL AND deleted_at IS NULL",
            (self.tenant_id, user_id),
        )
        return int(rows[0]["n"]) if rows else 0

    def chain(self, root_id: str) -> list[Atom]:
        rows = self.mysql.query(
            f"SELECT {_COLS} FROM memory_atom WHERE COALESCE(chain_root_id, id)=%s "
            "ORDER BY valid_from",
            (root_id,),
        )
        return [_row_to_atom(r) for r in rows]

    # ---- 审计 ----
    def audit(self, atom_id: str, action: str, user_id: str, detail: str = "", operator: str = "") -> None:
        try:
            self.mysql.execute(
                "INSERT INTO memory_audit (atom_id, action, tenant_id, user_id, operator, detail) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (atom_id, action, self.tenant_id, user_id, operator or user_id, detail[:500]),
            )
        except Exception as exc:  # 审计失败不该阻断用户操作（删记忆是用户权利）
            log.warning("memory_audit_failed", error=str(exc), action=action)


def _row_to_atom(row: dict[str, Any]) -> Atom:
    return Atom(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        fact_type=str(row["fact_type"]),
        fact_key=str(row["fact_key"]),
        content=str(row["content"]),
        confidence=float(row["confidence"]),
        mention_count=int(row["mention_count"] or 1),
        source_sid=str(row.get("source_sid") or ""),
        source_turn=int(row.get("source_turn") or 0),
        source_span=str(row.get("source_span") or ""),
        valid_from=_epoch(row.get("valid_from")),
        valid_to=_epoch(row.get("valid_to")) or None,
        superseded_by=row.get("superseded_by"),
        chain_root_id=row.get("chain_root_id"),
        deleted_at=_epoch(row.get("deleted_at")) or None,
        subject=str(row.get("subject") or ""),
        grade=str(row.get("grade") or ""),
    )


def _epoch(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, dt.datetime):
        return value.timestamp()
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


# ---- 会话落库（真相源）-----------------------------------------------------
class ChatLogRepo:
    """`chat_message` + `session_summary`（§6.4）。Redis 是热副本，这里才是真相源。"""

    def __init__(self, mysql: MySQL, tenant_id: str = "default") -> None:
        self.mysql = mysql
        self.tenant_id = tenant_id

    def append_turn(self, session_id: str, user_id: str, role: str, content: str) -> None:
        try:
            self.mysql.execute(
                "INSERT INTO chat_message (session_id, tenant_id, user_id, role, content) "
                "VALUES (%s,%s,%s,%s,%s)",
                (session_id, self.tenant_id, user_id, role, content[:4000]),
            )
        except Exception as exc:  # 落库失败不能影响回答（回答已经产出了）
            log.warning("chat_message_persist_failed", error=str(exc))

    def save_summary(self, session_id: str, user_id: str, summary: str, compressed: int) -> None:
        try:
            self.mysql.execute(
                "INSERT INTO session_summary (session_id, tenant_id, user_id, summary, compressed_turns) "
                "VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE summary=VALUES(summary), "
                "compressed_turns=VALUES(compressed_turns), updated_at=NOW()",
                (session_id, self.tenant_id, user_id, summary, compressed),
            )
        except Exception as exc:
            log.warning("session_summary_persist_failed", error=str(exc))


def mark_superseded(old: Atom, new_id: str, now: float) -> Atom:
    return replace(old, valid_to=now, superseded_by=new_id)


def mark_deleted(atom: Atom, now: float) -> Atom:
    return replace(atom, deleted_at=now)


def revive(atom: Atom) -> Atom:
    """恢复一条被 supersede 的记忆：清掉 `valid_to` 与 `superseded_by`。"""
    return replace(atom, valid_to=None, superseded_by=None)


# ---- 顾问只读授权（§3.7 第 7 条 / M4）-------------------------------------
@dataclass(frozen=True, slots=True)
class Grant:
    """一条"某人可以读我记忆"的授权。**只有读**（`scope` 目前恒为 `read`）。

    `status` 与 `revoked_at` 是两件不同的事，写在这里是因为它们容易混：
    `status='revoked'` 是"这条授权已被撤销"的**事实**，`revoked_at` 是"何时撤销"。
    判活只看 `status`，`revoked_at` 供记忆页显示；重新授权时两者一起被覆盖。
    """

    id: str
    owner_id: str
    grantee_id: str
    scope: str = "read"
    status: str = "active"
    granted_by: str = ""
    granted_at: float = 0.0
    expires_at: float | None = None
    revoked_at: float | None = None

    def active_at(self, now: float) -> bool:
        """`now` 时刻是否有效。**过期必须真的判**——授权不带过期检查等于永久后门。"""
        if self.status != "active" or self.revoked_at is not None:
            return False
        return self.expires_at is None or self.expires_at > now

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "grantee_id": self.grantee_id,
            "scope": self.scope,
            "status": self.status,
            "granted_by": self.granted_by,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "revoked_at": self.revoked_at,
        }


def new_grant_id() -> str:
    return uuid.uuid4().hex[:24]


class GrantRepo(Protocol):
    def get(self, owner_id: str, grantee_id: str) -> Grant | None: ...
    def list_by_owner(self, owner_id: str) -> list[Grant]: ...
    def list_by_grantee(self, grantee_id: str) -> list[Grant]: ...
    def upsert(self, grant: Grant) -> None: ...


class InMemoryGrantRepo:
    """测试与无库模式用。与 MySQL 版同一套语义：同 (owner, grantee) 只有一行，重新授权是覆盖。"""

    def __init__(self, grants: list[Grant] | None = None) -> None:
        self._grants: dict[tuple[str, str], Grant] = {
            (g.owner_id, g.grantee_id): g for g in grants or []
        }

    def get(self, owner_id: str, grantee_id: str) -> Grant | None:
        return self._grants.get((owner_id, grantee_id))

    def list_by_owner(self, owner_id: str) -> list[Grant]:
        return [g for g in self._grants.values() if g.owner_id == owner_id]

    def list_by_grantee(self, grantee_id: str) -> list[Grant]:
        return [g for g in self._grants.values() if g.grantee_id == grantee_id]

    def upsert(self, grant: Grant) -> None:
        self._grants[(grant.owner_id, grant.grantee_id)] = grant


_GRANT_COLS = (
    "id, owner_id, grantee_id, scope, status, granted_by, granted_at, expires_at, revoked_at"
)


class MySQLGrantRepo:
    """`memory_grant` 表。`UNIQUE KEY (tenant_id, owner_id, grantee_id)` 让"重新授权"是一次覆盖。"""

    def __init__(self, mysql: MySQL, tenant_id: str = "default") -> None:
        self.mysql = mysql
        self.tenant_id = tenant_id

    def get(self, owner_id: str, grantee_id: str) -> Grant | None:
        rows = self.mysql.query(
            f"SELECT {_GRANT_COLS} FROM memory_grant "
            "WHERE tenant_id=%s AND owner_id=%s AND grantee_id=%s",
            (self.tenant_id, owner_id, grantee_id),
        )
        return _row_to_grant(rows[0]) if rows else None

    def list_by_owner(self, owner_id: str) -> list[Grant]:
        rows = self.mysql.query(
            f"SELECT {_GRANT_COLS} FROM memory_grant WHERE tenant_id=%s AND owner_id=%s "
            "ORDER BY granted_at DESC",
            (self.tenant_id, owner_id),
        )
        return [_row_to_grant(r) for r in rows]

    def list_by_grantee(self, grantee_id: str) -> list[Grant]:
        rows = self.mysql.query(
            f"SELECT {_GRANT_COLS} FROM memory_grant WHERE tenant_id=%s AND grantee_id=%s "
            "ORDER BY granted_at DESC",
            (self.tenant_id, grantee_id),
        )
        return [_row_to_grant(r) for r in rows]

    def upsert(self, grant: Grant) -> None:
        self.mysql.execute(
            "INSERT INTO memory_grant (id, tenant_id, owner_id, grantee_id, scope, status, "
            "granted_by, granted_at, expires_at, revoked_at) VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,FROM_UNIXTIME(%s),"
            "CASE WHEN %s IS NULL THEN NULL ELSE FROM_UNIXTIME(%s) END,"
            "CASE WHEN %s IS NULL THEN NULL ELSE FROM_UNIXTIME(%s) END) "
            "ON DUPLICATE KEY UPDATE scope=VALUES(scope), status=VALUES(status), "
            "granted_by=VALUES(granted_by), granted_at=VALUES(granted_at), "
            "expires_at=VALUES(expires_at), revoked_at=VALUES(revoked_at)",
            (
                grant.id,
                self.tenant_id,
                grant.owner_id,
                grant.grantee_id,
                grant.scope,
                grant.status,
                grant.granted_by,
                grant.granted_at,
                grant.expires_at,
                grant.expires_at,
                grant.revoked_at,
                grant.revoked_at,
            ),
        )


def _row_to_grant(row: dict[str, Any]) -> Grant:
    return Grant(
        id=str(row["id"]),
        owner_id=str(row["owner_id"]),
        grantee_id=str(row["grantee_id"]),
        scope=str(row.get("scope") or "read"),
        status=str(row.get("status") or "active"),
        granted_by=str(row.get("granted_by") or ""),
        granted_at=_epoch(row.get("granted_at")),
        expires_at=_epoch(row.get("expires_at")) or None,
        revoked_at=_epoch(row.get("revoked_at")) or None,
    )


# ---- L3 用户画像（§3.7 第 1 条 / §7.3.1）---------------------------------
@dataclass(frozen=True, slots=True)
class Profile:
    """一张"这个学员是谁"的稳定描述，常驻 300 token。

    为什么它要单独一张表，而不是继续从 identity 类原子记忆拼：
    拼出来的画像**用户改不了**。学员说得最多的是"我上初三、数学不太好"，
    而真正应该常驻的是"目标中考、每周两次课、家长电话尾号 88"这类顾问确认过的东西。
    两者来源不同（`source="derived"` vs `"manual"`），优先级也不同——人工确认的覆盖推导的。
    """

    user_id: str
    content: str
    source: str = "manual"  # manual=用户在记忆页写的；derived=从身份类原子记忆推的（不落库）
    updated_at: float = 0.0
    updated_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "content": self.content,
            "source": self.source,
            "updated_at": self.updated_at,
            "updated_by": self.updated_by,
        }


class ProfileRepo(Protocol):
    def get(self, user_id: str) -> Profile | None: ...
    def upsert(self, profile: Profile) -> None: ...
    def delete(self, user_id: str) -> None: ...


class InMemoryProfileRepo:
    def __init__(self, profiles: list[Profile] | None = None) -> None:
        self._profiles: dict[str, Profile] = {p.user_id: p for p in profiles or []}

    def get(self, user_id: str) -> Profile | None:
        return self._profiles.get(user_id)

    def upsert(self, profile: Profile) -> None:
        self._profiles[profile.user_id] = profile

    def delete(self, user_id: str) -> None:
        self._profiles.pop(user_id, None)


class MySQLProfileRepo:
    """`user_profile` 表。主键 `(tenant_id, user_id)`：一人一行，写入是覆盖。"""

    def __init__(self, mysql: MySQL, tenant_id: str = "default") -> None:
        self.mysql = mysql
        self.tenant_id = tenant_id

    def get(self, user_id: str) -> Profile | None:
        rows = self.mysql.query(
            "SELECT user_id, content, source, updated_by, updated_at FROM user_profile "
            "WHERE tenant_id=%s AND user_id=%s",
            (self.tenant_id, user_id),
        )
        if not rows:
            return None
        row = rows[0]
        return Profile(
            user_id=str(row["user_id"]),
            content=str(row.get("content") or ""),
            source=str(row.get("source") or "manual"),
            updated_at=_epoch(row.get("updated_at")),
            updated_by=str(row.get("updated_by") or ""),
        )

    def upsert(self, profile: Profile) -> None:
        self.mysql.execute(
            "INSERT INTO user_profile (tenant_id, user_id, content, source, updated_by) "
            "VALUES (%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE content=VALUES(content), "
            "source=VALUES(source), updated_by=VALUES(updated_by), updated_at=NOW()",
            (self.tenant_id, profile.user_id, profile.content, profile.source, profile.updated_by),
        )

    def delete(self, user_id: str) -> None:
        self.mysql.execute(
            "DELETE FROM user_profile WHERE tenant_id=%s AND user_id=%s",
            (self.tenant_id, user_id),
        )


# ---- L2 原子记忆的向量索引（§3.7 第 8 条，M1）-------------------------------
def _clip(text: str, max_bytes: int) -> str:
    """按 **UTF-8 字节**截断。Milvus 的 `max_length` 是字节不是字符（§7.2.8 的教训：
    1 400 汉字 = 4 200 字节，按字符算毫无保护，写库直接 `invalid length string`）。"""
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", "ignore")


def _expr_str(value: str) -> str:
    """把字符串安全地嵌进 Milvus 过滤表达式。

    表达式是拼出来的字符串、没有参数化，所以引号与反斜杠必须转义：
    `user_id` 里带一个 `"` 就能让 `user_id == "a" or user_id == "b"` 变成两个条件。
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


class AtomIndex(Protocol):
    """原子记忆的向量索引。

    `search()` 的 `None` 与 `[]` **必须分开**：`None` = 索引不可用（调用失败），
    调用方该退回字面排序；`[]` = 索引是好的、只是没有命中，那就是"没有相关记忆"。
    混成一个空列表，会让"索引挂了"永远表现为"这个用户没有记忆"。
    """

    def upsert(self, atom: Atom) -> None: ...
    def delete(self, atom_id: str) -> None: ...
    def search(self, user_id: str, query: str, top_k: int = ...) -> list[str] | None: ...


class MilvusAtomIndex:
    """bge-m3 稠密向量 + Milvus top-k（§3.7 第 8 条）。

    **client 与 embedder 都是注入的**，不是自己建：
    `memory/` 不许 import `rag_system`、也不许自己 `MilvusClient(...)`（§7.0 硬边界 1、2）。
    组合根 `main.py` 把 RAG 已经在用的那个 client/embedder 递进来，
    顺带避免第二次装载 2.5 GB 的 bge-m3。

    写穿语义：MySQL 是真相源，这里只是副本。**任何一次索引调用失败都只记日志**
    （fail-open）——索引挂了应该表现为"召回退化成字面排序"，而不是"记忆功能不可用"。

    时间列用 INT64 epoch，`NULL` 落成 `0`：Milvus 没有 NULL，用 0 表示"这条没有
    失效时间/没有被删"，于是可见条件变成 `valid_to == 0 and deleted_at == 0`。
    用 0 而不是 -1 是因为 epoch 0 永远不会是一条真实记忆的 `valid_to`。
    """

    def __init__(
        self, client: Any, embedder: Any, settings: Any, *, tenant_id: str = "default"
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.settings = settings
        self.tenant_id = tenant_id
        self.name = getattr(embedder, "name", "unknown")

    @property
    def collection(self) -> str:
        return self.settings.milvus_memory_collection

    @property
    def dim(self) -> int:
        return int(getattr(self.embedder, "dim", 0) or self.settings.milvus_dim)

    def ensure(self) -> bool:
        """建集合与索引。**已存在就直接 load**，不 drop——集合里是持久化的记忆副本。"""
        from pymilvus import DataType

        if self.client.has_collection(self.collection):
            self.client.load_collection(self.collection)
            return True

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, max_length=40, is_primary=True)
        schema.add_field("tenant_id", DataType.VARCHAR, max_length=32)
        schema.add_field("user_id", DataType.VARCHAR, max_length=64)
        schema.add_field("fact_type", DataType.VARCHAR, max_length=16)
        schema.add_field("fact_key", DataType.VARCHAR, max_length=128)
        schema.add_field("content", DataType.VARCHAR, max_length=1600)
        schema.add_field("subject", DataType.VARCHAR, max_length=32)
        schema.add_field("grade", DataType.VARCHAR, max_length=16)
        schema.add_field("confidence", DataType.DOUBLE)
        schema.add_field("valid_to", DataType.INT64)
        schema.add_field("deleted_at", DataType.INT64)
        schema.add_field("dense", DataType.FLOAT_VECTOR, dim=self.dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(
            collection_name=self.collection, schema=schema, index_params=index_params
        )
        self.client.load_collection(self.collection)
        log.info("memory_vector_collection_created", collection=self.collection, dim=self.dim)
        return True

    def upsert(self, atom: Atom) -> None:
        """写一条（或删一条）记忆副本。不 alive 的条目**从索引里删掉**，
        不是 upsert 成 `valid_to=now`——索引里少一行，比留一行靠过滤条件挡着更难出错。"""
        if not atom.alive:
            self.delete(atom.id)
            return
        try:
            dense, _ = self.embedder.encode([atom.content])
            self.client.upsert(
                collection_name=self.collection, data=[self._row(atom, dense[0])]
            )
        except Exception as exc:
            log.warning("memory_vector_upsert_failed", atom_id=atom.id, error=str(exc)[:200])

    def delete(self, atom_id: str) -> None:
        try:
            self.client.delete(
                collection_name=self.collection, filter=f'id == "{_expr_str(atom_id)}"'
            )
        except Exception as exc:
            log.warning("memory_vector_delete_failed", atom_id=atom_id, error=str(exc)[:200])

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[str] | None:
        if not user_id or not query.strip():
            return []
        try:
            dense, _ = self.embedder.encode([query])
            res = self.client.search(
                collection_name=self.collection,
                data=dense,
                anns_field="dense",
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=max(top_k, 1),
                filter=self._visible(user_id),
                output_fields=["id"],
            )
        except Exception as exc:
            log.warning("memory_vector_search_failed", error=str(exc)[:200])
            return None
        return [str(hit["id"]) for hits in res for hit in hits if "id" in hit]

    def count(self) -> int:
        """索引里的行数。启动日志用它对照 MySQL，"索引空但库里有记忆"要能一眼看出来。"""
        try:
            res = self.client.query(
                collection_name=self.collection, filter="", output_fields=["count(*)"]
            )
        except Exception as exc:
            log.warning("memory_vector_count_failed", error=str(exc)[:200])
            return 0
        return int(res[0]["count(*)"]) if res else 0

    # ---- 内部 ----
    def _visible(self, user_id: str) -> str:
        """召回的唯一可见条件（§7.3.3）：`valid_to` / `deleted_at` 为空且置信度过门槛。

        **三个条件少一个就会出现"已删除记忆还在回答"**，所以它写在索引侧而不是
        召回侧的内存过滤——top_k 是有限名额，让不该出现的条目占掉名额同样有害。
        """
        return (
            f'tenant_id == "{_expr_str(self.tenant_id)}" '
            f'and user_id == "{_expr_str(user_id)}" '
            "and valid_to == 0 and deleted_at == 0 "
            f"and confidence >= {float(self.settings.mem_min_confidence)}"
        )

    def _row(self, atom: Atom, vector: list[float]) -> dict[str, Any]:
        return {
            "id": _clip(atom.id, 40),
            "tenant_id": _clip(self.tenant_id, 32),
            "user_id": _clip(atom.user_id, 64),
            "fact_type": _clip(atom.fact_type, 16),
            "fact_key": _clip(atom.fact_key, 128),
            "content": _clip(atom.content, 1600),
            "subject": _clip(atom.subject, 32),
            "grade": _clip(atom.grade, 16),
            "confidence": float(atom.confidence),
            "valid_to": int(atom.valid_to or 0),
            "deleted_at": int(atom.deleted_at or 0),
            "dense": vector,
        }
