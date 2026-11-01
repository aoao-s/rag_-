"""用户的持久化：`sys_user` / `auth_token` 两表（DDL 见 `scripts/init_db.py`）。

与 `memory/store.py` 同一套写法：`Protocol` + 内存实现 + MySQL 实现，
**仓储是同步的**，由 `auth/service.py` 用 `asyncio.to_thread` 包住。
好处是测试不必起 MySQL 就能跑同一份业务逻辑（两个实现互为参照物），
坏处是"忘了 `to_thread` 就阻塞事件循环"——所以同步方法一律以 `_` 之外的
动名词命名并在服务层显式出现在 `to_thread(` 里，code review 时看得见。

**两张表的分工**：

| 表 | 存什么 | 为什么 |
|---|---|---|
| `sys_user` | 账号、口令哈希、**角色**、租户、ACL | 角色与权限的唯一真相源。登录接口**只认这里的 role**，绝不从请求体或旧 token 里取 |
| `auth_token` | 已签发的 refresh 的 `jti` 与吊销时间 | JWT 本身不可撤销；没有这张表，"登出"只是一句安慰话 |

access token **不入库**：它只有 10 分钟寿命，为它每次请求查一次库，
等于用一次数据库往返换"提前撤销"这个几乎用不到的能力。登出后 access 靠 TTL 自然失效，
这条取舍写在 `auth/service.py::logout()` 的注释里。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from base.db import MySQL
from base.logger import get_logger

log = get_logger(__name__)

KIND_REFRESH = "refresh"
KIND_ACCESS = "access"


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class User:
    """一个账号。`password_hash` 只在这里与仓储里出现，`public()` 是它唯一的出口。"""

    user_id: str
    username: str
    password_hash: str
    nickname: str = ""
    avatar: str = ""
    role: str = "STUDENT"
    tenant_id: str = "default"
    acl_tags: tuple[str, ...] = ()
    status: str = "active"
    last_login_at: float | None = None
    created_at: float | None = None

    @property
    def active(self) -> bool:
        return self.status == "active"

    def public(self) -> dict[str, Any]:
        """对外形状（登录响应里的 `user`、`/me` 的 `data.user`）。

        **没有 `password_hash`**：不是"忘了放"，是刻意不给。响应体最终会进日志、
        进浏览器缓存、进用户截图，任何一处都不该有口令哈希。
        """
        return {
            "user_id": self.user_id,
            "username": self.username,
            "nickname": self.nickname or self.username,
            "avatar": self.avatar,
            "role": self.role,
            "tenant_id": self.tenant_id,
            "acl_tags": list(self.acl_tags),
        }

    def with_login(self, at: float) -> "User":
        return replace(self, last_login_at=at)


@dataclass(frozen=True, slots=True)
class TokenRecord:
    """一条**已签发**的令牌记录。目前只记 refresh（见模块注释）。"""

    jti: str
    user_id: str
    tenant_id: str = "default"
    kind: str = KIND_REFRESH
    expires_at: float = 0.0
    revoked_at: float | None = None
    created_at: float = field(default_factory=time.time)

    def usable(self, now: float | None = None) -> bool:
        """还能不能拿来换新令牌。三件事都成立才行：没吊销、没过期、类型对。"""
        moment = time.time() if now is None else now
        return self.revoked_at is None and self.kind == KIND_REFRESH and self.expires_at > moment


# ---------------------------------------------------------------------------
# 协议
# ---------------------------------------------------------------------------


class UserRepo(Protocol):
    def by_username(self, tenant_id: str, username: str) -> User | None: ...

    def by_user_id(self, tenant_id: str, user_id: str) -> User | None: ...

    def create(self, user: User) -> None:
        """用户名已存在时抛 `UsernameTaken`（两个实现必须一致）。"""
        ...

    def touch_login(self, tenant_id: str, user_id: str) -> None: ...

    def update_password(self, tenant_id: str, user_id: str, password_hash: str) -> None:
        """换口令哈希。只被"参数升级"调用（Argon2 换参后静默改写），
        不做旧口令校验——那是改密码接口的事，本版本还没有。"""
        ...


class TokenRepo(Protocol):
    def save(self, record: TokenRecord) -> None: ...

    def get(self, jti: str) -> TokenRecord | None: ...

    def revoke(self, jti: str) -> bool:
        """返回"这次是真的吊销了一条"还是"本来就没有/已经吊销过"。"""
        ...

    def revoke_user(self, tenant_id: str, user_id: str, kind: str = KIND_REFRESH) -> int:
        """登出：把该用户所有未吊销的 refresh 一次作废，返回条数。"""
        ...


class UsernameTaken(Exception):
    """用户名已存在。

    它不是 `EduRAGError`：这只在仓储层发生，翻译成什么 HTTP 码由服务层决定
    （409 `username_taken`）。仓储不认识 HTTP。
    """


# ---------------------------------------------------------------------------
# 内存实现（测试 / 非 MySQL 后端）
# ---------------------------------------------------------------------------


class InMemoryUserRepo:
    """进程内账号表。`backend != mysql` 时用它：能注册能登录，但**重启即失**。

    这不是"临时凑合"——它与 `main.py::build_memory()` 用同一条规则
    （FAQ 在库里才把记忆放库里），避免出现"FAQ 在内存、账号在库"这种
    一半持久化一半不持久化的状态：那种状态下排障最难，因为"重启后有些东西还在"。
    """

    def __init__(self) -> None:
        self._users: dict[tuple[str, str], User] = {}

    def by_username(self, tenant_id: str, username: str) -> User | None:
        return self._users.get((tenant_id, username))

    def by_user_id(self, tenant_id: str, user_id: str) -> User | None:
        for user in self._users.values():
            if user.tenant_id == tenant_id and user.user_id == user_id:
                return user
        return None

    def create(self, user: User) -> None:
        key = (user.tenant_id, user.username)
        if key in self._users:
            raise UsernameTaken(user.username)
        self._users[key] = user

    def touch_login(self, tenant_id: str, user_id: str) -> None:
        for key, user in self._users.items():
            if user.tenant_id == tenant_id and user.user_id == user_id:
                self._users[key] = user.with_login(time.time())

    def update_password(self, tenant_id: str, user_id: str, password_hash: str) -> None:
        for key, user in self._users.items():
            if user.tenant_id == tenant_id and user.user_id == user_id:
                self._users[key] = replace(user, password_hash=password_hash)

    def clear(self) -> None:
        """只给测试用。"""
        self._users.clear()

    def count(self) -> int:
        return len(self._users)


class InMemoryTokenRepo:
    def __init__(self) -> None:
        self._rows: dict[str, TokenRecord] = {}

    def save(self, record: TokenRecord) -> None:
        self._rows[record.jti] = record

    def get(self, jti: str) -> TokenRecord | None:
        return self._rows.get(jti)

    def revoke(self, jti: str) -> bool:
        record = self._rows.get(jti)
        if record is None or record.revoked_at is not None:
            return False
        self._rows[jti] = replace(record, revoked_at=time.time())
        return True

    def revoke_user(self, tenant_id: str, user_id: str, kind: str = KIND_REFRESH) -> int:
        now = time.time()
        count = 0
        for jti, record in list(self._rows.items()):
            if (
                record.tenant_id == tenant_id
                and record.user_id == user_id
                and record.kind == kind
                and record.revoked_at is None
            ):
                self._rows[jti] = replace(record, revoked_at=now)
                count += 1
        return count

    def clear(self) -> None:
        self._rows.clear()


# ---------------------------------------------------------------------------
# MySQL 实现
# ---------------------------------------------------------------------------

_USER_COLS = (
    "user_id, username, password_hash, nickname, avatar, role, tenant_id, "
    "status, acl_tags, last_login_at, created_at"
)
_TOKEN_COLS = "jti, user_id, tenant_id, kind, expires_at, revoked_at, created_at"


def _parse_acl(raw: Any) -> tuple[str, ...]:
    text = (raw or "").strip() if isinstance(raw, str) else ""
    return tuple(tag.strip() for tag in text.split(",") if tag.strip())


def _epoch(value: Any) -> float | None:
    """DATETIME → epoch 秒。`None` 原样返回（`last_login_at` 可以为空）。"""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return value.timestamp()
    except AttributeError:  # 驱动把它当字符串返回时
        return None


def _row_to_user(row: dict[str, Any]) -> User:
    return User(
        user_id=row["user_id"],
        username=row["username"],
        password_hash=row["password_hash"],
        nickname=row.get("nickname") or "",
        avatar=row.get("avatar") or "",
        role=row.get("role") or "STUDENT",
        tenant_id=row.get("tenant_id") or "default",
        acl_tags=_parse_acl(row.get("acl_tags")),
        status=row.get("status") or "active",
        last_login_at=_epoch(row.get("last_login_at")),
        created_at=_epoch(row.get("created_at")),
    )


def _row_to_token(row: dict[str, Any]) -> TokenRecord:
    return TokenRecord(
        jti=row["jti"],
        user_id=row["user_id"],
        tenant_id=row.get("tenant_id") or "default",
        kind=row.get("kind") or KIND_REFRESH,
        expires_at=_epoch(row.get("expires_at")) or 0.0,
        revoked_at=_epoch(row.get("revoked_at")),
        created_at=_epoch(row.get("created_at")) or 0.0,
    )


class MySQLUserRepo:
    """`sys_user`。构造不连库（`MySQL` 本身懒连接）。"""

    def __init__(self, mysql: MySQL) -> None:
        self.mysql = mysql

    def by_username(self, tenant_id: str, username: str) -> User | None:
        rows = self.mysql.query(
            f"SELECT {_USER_COLS} FROM sys_user WHERE tenant_id=%s AND username=%s",
            (tenant_id, username),
        )
        return _row_to_user(rows[0]) if rows else None

    def by_user_id(self, tenant_id: str, user_id: str) -> User | None:
        rows = self.mysql.query(
            f"SELECT {_USER_COLS} FROM sys_user WHERE tenant_id=%s AND user_id=%s",
            (tenant_id, user_id),
        )
        return _row_to_user(rows[0]) if rows else None

    def create(self, user: User) -> None:
        try:
            self.mysql.execute(
                "INSERT INTO sys_user (user_id, tenant_id, username, nickname, password_hash, "
                "role, status, acl_tags, avatar, created_at, updated_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),NOW())",
                (
                    user.user_id,
                    user.tenant_id,
                    user.username,
                    user.nickname,
                    user.password_hash,
                    user.role,
                    user.status,
                    ",".join(user.acl_tags),
                    user.avatar,
                ),
            )
        except Exception as exc:  # noqa: BLE001 —— 只把唯一键冲突翻译成业务异常，其余照抛
            # 靠捕获主键/唯一键冲突而不是"先查再插"：先查后插在并发下会漏
            # （两个请求同时查到"不存在"，然后一个插入成功、另一个炸 500）。
            if _is_duplicate(exc):
                raise UsernameTaken(user.username) from None
            raise

    def touch_login(self, tenant_id: str, user_id: str) -> None:
        self.mysql.execute(
            "UPDATE sys_user SET last_login_at=NOW() WHERE tenant_id=%s AND user_id=%s",
            (tenant_id, user_id),
        )

    def update_password(self, tenant_id: str, user_id: str, password_hash: str) -> None:
        self.mysql.execute(
            "UPDATE sys_user SET password_hash=%s, updated_at=NOW() WHERE tenant_id=%s AND user_id=%s",
            (password_hash, tenant_id, user_id),
        )


class MySQLTokenRepo:
    """`auth_token`。只写 refresh（access 不落库，见模块注释）。"""

    def __init__(self, mysql: MySQL) -> None:
        self.mysql = mysql

    def save(self, record: TokenRecord) -> None:
        self.mysql.execute(
            "INSERT INTO auth_token (jti, user_id, tenant_id, kind, expires_at, revoked_at, created_at) "
            "VALUES (%s,%s,%s,%s,FROM_UNIXTIME(%s),"
            "CASE WHEN %s IS NULL THEN NULL ELSE FROM_UNIXTIME(%s) END,NOW())",
            (
                record.jti,
                record.user_id,
                record.tenant_id,
                record.kind,
                int(record.expires_at),
                record.revoked_at,
                record.revoked_at,
            ),
        )

    def get(self, jti: str) -> TokenRecord | None:
        rows = self.mysql.query(
            f"SELECT {_TOKEN_COLS} FROM auth_token WHERE jti=%s", (jti,)
        )
        return _row_to_token(rows[0]) if rows else None

    def revoke(self, jti: str) -> bool:
        return self.mysql.execute(
            "UPDATE auth_token SET revoked_at=NOW() WHERE jti=%s AND revoked_at IS NULL", (jti,)
        ) > 0

    def revoke_user(self, tenant_id: str, user_id: str, kind: str = KIND_REFRESH) -> int:
        return self.mysql.execute(
            "UPDATE auth_token SET revoked_at=NOW() "
            "WHERE tenant_id=%s AND user_id=%s AND kind=%s AND revoked_at IS NULL",
            (tenant_id, user_id, kind),
        )


def _is_duplicate(exc: Exception) -> bool:
    """`pymysql` 的 1062 = Duplicate entry。用错误码而不是消息文本：
    消息随语言与版本变，码不变。"""
    args = getattr(exc, "args", ())
    return bool(args) and args[0] == 1062


def new_user_id() -> str:
    """`user_id` 用 32 位十六进制字符串（uuid4），不用自增主键：
    自增 id 会泄露"我们是第几个用户"，而且将来做多租户合并时两边 id 会撞。"""
    return uuid.uuid4().hex
