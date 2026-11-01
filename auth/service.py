"""认证业务：注册 / 登录 / 刷新 / 登出 / 我是谁。

**这一层是"角色从哪来"的唯一答案**：`role` / `tenant_id` / `acl_tags` 只有在
`sys_user` 里读到的那一份算数。请求体里带的、旧 token 里带的，一律不读——
那是用户自己写的，不是身份。

四条与安全直接相关、且都写成了代码（不是注释里的良好愿望）：

1. **口令只进哈希**。明文除了 `verify_password` 的参数，不落库、不落日志、不进响应。
2. **账号枚举防护**：查无此人时也跑一次 Argon2（`burn_equivalent_time`），
   并把"用户名不存在"与"密码错误"合并成同一句 `invalid_credentials`。
3. **refresh 可撤销、access 不可**。refresh 的 `jti` 落 `auth_token`，
   登出/轮换时作废；access 只有 10 分钟寿命，靠 TTL 自然死亡。
   为 access 每次请求查一次库，是用一次往返换一个几乎用不到的能力。
4. **轮换即作废旧 jti**（refresh rotation）。一条 refresh 被用过就废，
   这样"令牌被抄走了"这件事会以"合法用户下次刷新失败"的形式暴露，而不是静默共用。

**所有仓储调用都走 `asyncio.to_thread`**：仓储是同步的（见 `auth/store.py`），
Argon2 也是刻意算得慢的。协程里直接调它们不是"慢一点"，是把整个事件循环卡住。
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any, Final

from auth import passwords, tokens
from auth.store import (
    KIND_REFRESH,
    TokenRecord,
    TokenRepo,
    User,
    UsernameTaken,
    UserRepo,
    new_user_id,
)
from base.config import Settings
from base.errors import EduRAGError, Forbidden, InvalidRequest, Unauthorized
from base.labels import ROLES, ROLE_ANONYMOUS, normalize_role
from base.logger import get_logger

log = get_logger(__name__)

# 用户名口径：ASCII 字母数字与 `_ . @ -`，3–64 位。
# **不收中文**不是懒：中文字符串在日志、URL、邮件往来里都容易出同形字问题
# （`〇` 与 `○`、全角半角），而账号是排障时唯一的对齐锚点。真人姓名放 `nickname`。
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.@-]{3,64}$")

TOKEN_TYPE = "Bearer"
_TYP_ACCESS = "access"
_TYP_REFRESH = "refresh"

# 自助注册可选的**声明角色**。学生与老师在注册时选定，落 `sys_user.role`；
# 之后所有判定（`orchestration/admission.py::permissions_for` 的"角色 × 意图"）
# 只读库里那一份，请求体与旧 token 里的一律不认。
#
# ADMIN **不可自助注册**：管理员账号由运维创建（改库 + 审计）。放开的代价不是
# "多一个管理员"这么简单——`permissions_for("ADMIN")` 的拒绝集为空，一次 HTTP 请求
# 就能拿到全部意图（教案制作 / 备课检索）的权限，整套角色矩阵等于没写。
REGISTER_ROLES: Final[tuple[str, ...]] = ("STUDENT", "TEACHER")
DEFAULT_REGISTER_ROLE: Final[str] = "STUDENT"
ADMIN_ROLE: Final[str] = "ADMIN"


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class AuthError(EduRAGError):
    """认证失败的基类。子类各自带对外错误码与 HTTP 状态。"""

    code = "auth_error"
    http_status = 400


class InvalidCredentials(AuthError):
    """用户名或密码错误。**两种情况共用这一句**：区分开就等于送出一本用户名目录。"""

    code = "invalid_credentials"
    http_status = 401


class UserDisabled(AuthError):
    """账号被停用。只有在**口令正确之后**才会出现，否则它也是一个探测手段。"""

    code = "user_disabled"
    http_status = 403


class UsernameTakenError(AuthError):
    code = "username_taken"
    http_status = 409


class InvalidRoleError(AuthError):
    """角色不在 `ROLES` 里（拼错、或用了 `ANONYMOUS` 这种"不是角色"的取值）。"""

    code = "invalid_role"
    http_status = 400


class RoleNotAllowedError(AuthError):
    """角色合法但**不能自助注册**（目前只有 ADMIN）。

    单独一个码，不与 `invalid_role` 合并：前端对这两种情况的处置不同——
    "拼错了"改一个字重试，"想注册管理员"应当提示联系管理员。
    """

    code = "role_not_allowed"
    http_status = 400


class InvalidToken(AuthError):
    """令牌不可用（结构/签名/过期/已吊销/类型不对）。一律 401，前端据此跳登录。"""

    code = "invalid_token"
    http_status = 401


class AuthNotConfigured(AuthError):
    """服务端没配 `JWT_SECRET`，签不出可信令牌。

    503 而不是 500：这是**部署没做完**，不是代码出错；给一句怎么修比给一个堆栈有用。
    """

    code = "auth_not_configured"
    http_status = 503


class InvalidPassword(InvalidRequest):
    code = "invalid_password"


class InvalidUsername(InvalidRequest):
    code = "invalid_username"


# ---------------------------------------------------------------------------
# 返回值
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IssuedTokens:
    access_token: str
    refresh_token: str
    token_type: str
    expires_in: int  # access 的有效秒数，即响应里的 `expires_in`
    refresh_expires_in: int


@dataclass(frozen=True, slots=True)
class AuthSession:
    user: User
    tokens: IssuedTokens


@dataclass(frozen=True, slots=True)
class MintedTokens:
    """`_mint()` 的产出。带 `refresh_jti` 是因为**落库要同一个 jti**：
    再解一次自己刚签的字符串去取它，是把"签名"当成"往返编码"来用——
    多一处能忘的地方，也多一次无谓的解析。"""

    access: str
    refresh: str
    refresh_ttl: int
    refresh_jti: str


# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------


class AuthService:
    """认证服务。仓储由 `main.py::build_auth()` 注入（MySQL 或内存）。"""

    def __init__(
        self,
        *,
        settings: Settings,
        users: UserRepo,
        tokens: TokenRepo,
    ) -> None:
        self.settings = settings
        self.users = users
        self.tokens = tokens

    # -- 基础设施 ----------------------------------------------------------

    @property
    def configured(self) -> bool:
        """空密钥等于签不出可信令牌（详见 `auth/tokens.py::encode_token`）。"""
        return bool(self.settings.jwt_secret.strip())

    def _require_configured(self) -> str:
        secret = self.settings.jwt_secret.strip()
        if not secret:
            raise AuthNotConfigured(
                "未配置 JWT_SECRET，认证接口不可用。"
                "请在 .env 里设置一个长度 ≥32 的随机串后重启服务"
            )
        return secret

    @property
    def default_tenant(self) -> str:
        """新账号默认租户。公网注册走部署配置的租户，不从请求体取——
        让客户端自报 `tenant_id` 等于让任何人把自己注册进别人的学校。"""
        return (self.settings.tenant_id or "default").strip() or "default"

    @staticmethod
    def _normalize_username(raw: str) -> str:
        """小写 + 去空白。

        转小写是**为了两个实现对同一件事给出同一答案**：MySQL 的 `utf8mb4_0900_ai_ci`
        默认大小写不敏感（`Student001` 与 `student001` 撞唯一键），而内存实现
        比较的是字符串本身。服务层统一归一后，两边行为一致，登录也不会出现
        "注册时写的大写、登录时打的小写查不到"。
        """
        return (raw or "").strip().lower()

    # -- 注册 --------------------------------------------------------------

    async def register(
        self,
        username: str,
        password: str,
        *,
        nickname: str = "",
        role: str | None = None,
    ) -> AuthSession:
        """注册并**直接登录**（返回令牌）。前端少一次往返，少一个"注册成功但没登上"的断点。

        `role` 是注册时的**声明角色**，只接受 `REGISTER_ROLES`。不传即学生。
        角色一经落库就由库说了算：之后改角色只能改库（或将来由管理员接口改），
        手里的旧令牌换不出新角色的 access（`refresh` 里重新读库，见下）。
        """
        secret = self._require_configured()

        name = self._normalize_username(username)
        if not _USERNAME_RE.match(name):
            raise InvalidUsername("用户名只能包含字母、数字与 _ . @ -，长度 3–64")

        reason = passwords.validate_password(password)
        if reason:
            raise InvalidPassword(reason)

        chosen = normalize_role(role) or DEFAULT_REGISTER_ROLE
        if chosen not in ROLES:
            raise InvalidRoleError(f"角色不合法：{role}（可选：{' / '.join(REGISTER_ROLES)}）")
        if chosen == ADMIN_ROLE:
            # 管理员不能自助注册：`permissions_for("ADMIN")` 的拒绝集为空，
            # 放开的代价是"一次 HTTP 请求拿到全部意图的权限"。
            raise RoleNotAllowedError("管理员账号不能自助注册，请联系管理员创建")

        hashed = await asyncio.to_thread(passwords.hash_password, password)
        user = User(
            user_id=new_user_id(),
            username=name,
            password_hash=hashed,
            nickname=(nickname or "").strip()[:64],
            role=chosen,
            tenant_id=self.default_tenant,
            status="active",
        )
        try:
            await asyncio.to_thread(self.users.create, user)
        except UsernameTaken:
            # 注册接口**必须**能说"这个名字被占了"：不说的话用户会反复重试，
            # 而攻击者从这里能得到的只有"这个名字存在"——注册成功本来就公之于众。
            raise UsernameTakenError("用户名已被占用") from None

        issued = await self._issue(user, secret, remember=False)
        log.info("auth_register", user_id=user.user_id, tenant_id=user.tenant_id, role=user.role)
        return AuthSession(user=user, tokens=issued)

    # -- 登录 --------------------------------------------------------------

    async def login(
        self,
        username: str,
        password: str,
        *,
        remember_me: bool = False,
    ) -> AuthSession:
        """校验口令并签发令牌。

        **校验顺序是有讲究的**：先比口令、再看账号状态。
        反过来（先看 `status`）会让"这个账号存在但被停用"变成一个不需要知道密码
        就能探知的事实；而"密码对不对"本来就该是看到账号状态的唯一门槛。
        """
        secret = self._require_configured()
        name = self._normalize_username(username)
        user = await asyncio.to_thread(self.users.by_username, self.default_tenant, name)

        if user is None:
            # 查无此人也要花掉一次 Argon2 的时间：否则响应快 80 ms 的那个用户名
            # 就是"不存在"，登录接口变成用户名目录。
            await asyncio.to_thread(passwords.burn_equivalent_time, password or "")
            log.warning("auth_login_failed", reason="no_such_user", username=_safe_name(name))
            raise InvalidCredentials("用户名或密码错误")

        ok = await asyncio.to_thread(passwords.verify_password, user.password_hash, password or "")
        if not ok:
            log.warning(
                "auth_login_failed", reason="bad_password", user_id=user.user_id, username=_safe_name(name)
            )
            raise InvalidCredentials("用户名或密码错误")

        if not user.active:
            log.warning("auth_login_failed", reason="disabled", user_id=user.user_id)
            raise UserDisabled("账号已停用，请联系管理员")

        # 参数升级：库升级后老哈希仍能验过，这里顺手换成新参数。
        # 不做的话 `passwords.needs_rehash` 永远没人调用，等于白写。
        if await asyncio.to_thread(passwords.needs_rehash, user.password_hash):
            fresh = await asyncio.to_thread(passwords.hash_password, password)
            await asyncio.to_thread(self.users.update_password, user.tenant_id, user.user_id, fresh)

        issued = await self._issue(user, secret, remember=remember_me)
        await asyncio.to_thread(self.users.touch_login, user.tenant_id, user.user_id)

        log.info("auth_login", user_id=user.user_id, tenant_id=user.tenant_id, role=user.role)
        return AuthSession(user=user, tokens=issued)

    # -- 刷新 --------------------------------------------------------------

    async def refresh(self, refresh_token: str) -> AuthSession:
        """用 refresh 换一对新令牌。**轮换**：旧 jti 立刻作废。

        轮换的代价是"客户端的刷新请求没拿到响应就再也刷不上"，
        收益是"被拷走的令牌只能用一次"。后者是安全事件，前者是体验问题——
        所以选择轮换，并把"旧令牌被重复使用"记进日志（那是泄漏的信号）。
        """
        secret = self._require_configured()
        claims = self._decode(refresh_token, expect=_TYP_REFRESH)

        jti = str(claims.get("jti") or "")
        record = await asyncio.to_thread(self.tokens.get, jti) if jti else None
        if record is None:
            log.warning("auth_token_rejected", reason="unknown_jti")
            raise InvalidToken("刷新令牌不存在，请重新登录")
        if not record.usable():
            # 已吊销却被再次使用 = 要么前端重放了旧令牌，要么令牌被抄走。
            # 两种情况都只能让用户重新登录，并留下线索。
            log.warning("auth_token_rejected", reason="revoked_or_expired", user_id=record.user_id)
            raise InvalidToken("刷新令牌已失效，请重新登录")

        user = await asyncio.to_thread(self.users.by_user_id, record.tenant_id, record.user_id)
        if user is None:
            log.warning("auth_token_rejected", reason="user_gone", user_id=record.user_id)
            raise InvalidToken("用户不存在，请重新登录")
        if not user.active:
            raise UserDisabled("账号已停用，请联系管理员")

        # 角色以库里的为准，不从旧 token 抄。账号被降级后，
        # 手里的旧 refresh 不该还能换出一个旧角色的新 access。
        await asyncio.to_thread(self.tokens.revoke, jti)

        issued = await self._issue(user, secret, remember=bool(claims.get("rmb")))
        log.info("auth_refresh", user_id=user.user_id, tenant_id=user.tenant_id)
        return AuthSession(user=user, tokens=issued)

    # -- 登出 --------------------------------------------------------------

    async def logout(self, refresh_token: str | None) -> int:
        """作废该用户的**全部** refresh，返回作废条数。

        **幂等且不抛错**：令牌已经过期/签名坏了/压根没传，都算"现在没有有效会话"，
        返回 0。登出接口如果能 401，前端就得为"清本地存储"写一条异常分支，
        而用户按下登出键时想要的恰恰是"别再问我了"。

        **access 不在这里作废**（10 分钟内仍有效）：JWT 无状态，为它每次请求查库
        不值得（见 `auth/store.py` 模块注释）。这条取舍必须写在契约里，
        否则会有人以为登出后令牌立刻全线失效。
        """
        if not refresh_token:
            return 0
        try:
            claims = tokens.decode_token(
                refresh_token, secret=self.settings.jwt_secret, leeway_s=self.settings.jwt_leeway_s
            )
        except tokens.TokenError:
            # 解不开就没法知道是谁——但调用方要的"登出"已经成立（本地删掉了）。
            log.info("auth_logout", reason="unparsable_token", revoked=0)
            return 0

        user_id = str(claims.get("sub") or "")
        tenant = str(claims.get("tenant") or self.default_tenant)
        jti = str(claims.get("jti") or "")
        revoked = 0
        if user_id:
            revoked = await asyncio.to_thread(self.tokens.revoke_user, tenant, user_id, KIND_REFRESH)
        elif jti:
            revoked = 1 if await asyncio.to_thread(self.tokens.revoke, jti) else 0
        log.info("auth_logout", user_id=user_id or "-", tenant_id=tenant, revoked=revoked)
        return revoked

    # -- 我是谁 ------------------------------------------------------------

    async def me(self, access_token: str) -> User:
        """用 access 换回**库里的**用户资料。

        不直接从 token 的 claim 拼一个 `User` 返回：那份 claim 里没有 `nickname` / `avatar`，
        而且它是 10 分钟前的快照——用户刚改了昵称，`/me` 该显示新的。
        """
        claims = self._decode(access_token, expect=_TYP_ACCESS)
        user_id = str(claims.get("sub") or "")
        tenant = str(claims.get("tenant") or self.default_tenant)
        if not user_id:
            raise InvalidToken("令牌缺少用户标识")
        user = await asyncio.to_thread(self.users.by_user_id, tenant, user_id)
        if user is None:
            raise InvalidToken("用户不存在")
        if not user.active:
            raise UserDisabled("账号已停用，请联系管理员")
        return user

    # -- 内部 --------------------------------------------------------------

    async def _issue(self, user: User, secret: str, *, remember: bool) -> IssuedTokens:
        """签发一对令牌，并把 refresh 的 jti 落库。

        **注册 / 登录 / 刷新三条路共用这一处**：漏掉一次 `save()` 的表现是
        "登录成功但一刷新就掉线"，而且只在某一条入口上出现——分三份写迟早会漏一份。
        """
        minted = self._mint(user, secret, remember=remember)
        await asyncio.to_thread(
            self.tokens.save,
            TokenRecord(
                jti=minted.refresh_jti,
                user_id=user.user_id,
                tenant_id=user.tenant_id,
                kind=KIND_REFRESH,
                expires_at=time.time() + minted.refresh_ttl,
            ),
        )
        return IssuedTokens(
            access_token=minted.access,
            refresh_token=minted.refresh,
            token_type=TOKEN_TYPE,
            expires_in=self.settings.auth_access_ttl_s,
            refresh_expires_in=minted.refresh_ttl,
        )

    def _decode(self, token: str, *, expect: str) -> dict[str, Any]:
        """验签 + 类型校验。`TokenError` 一律翻成 401 `invalid_token`。

        类型校验不能省：拿 access 去刷新（或拿 refresh 去调业务接口）都必须失败，
        否则"短寿命 access"这条设计就没有意义——refresh 能当 access 用，
        等于把 10 分钟的窗口变成 30 天。
        """
        secret = self._require_configured()
        try:
            claims = tokens.decode_token(token, secret=secret, leeway_s=self.settings.jwt_leeway_s)
        except tokens.TokenError as exc:
            log.warning("auth_token_rejected", reason="decode", detail=exc.message)
            raise InvalidToken("令牌无效或已过期") from None
        if str(claims.get("typ") or "") != expect:
            log.warning("auth_token_rejected", reason="wrong_type", expected=expect)
            raise InvalidToken("令牌类型不正确")
        return claims

    def _mint(self, user: User, secret: str, *, remember: bool) -> "MintedTokens":
        """签一对令牌。

        claim 的形状与 `orchestration/admission.py::_identity_from_token` 对齐：
        `sub` / `role` / `tenant` / `acl`。**`kb_version` 不写**——索引版本由服务端运行期
        决定，写进令牌会让"切索引后旧令牌查旧库"变成可能。
        """
        role = normalize_role(user.role)
        if role == ROLE_ANONYMOUS:
            # 库里的角色不在 `ROLES` 里：拒绝签发，而不是发一个下游会 401 的令牌。
            # 出现即数据被手改过，该有人去看。
            raise Forbidden(f"账号角色不合法：{user.role}", code="invalid_role")

        access_ttl = max(1, int(self.settings.auth_access_ttl_s))
        refresh_ttl = int(
            self.settings.auth_refresh_ttl_remember_s if remember else self.settings.auth_refresh_ttl_s
        )
        refresh_ttl = max(access_ttl, refresh_ttl)

        shared = {
            "sub": user.user_id,
            "role": role,
            "tenant": user.tenant_id,
            "acl": list(user.acl_tags),
        }
        refresh_jti = tokens.new_jti()
        access = tokens.encode_token(
            {**shared, "typ": _TYP_ACCESS, "jti": tokens.new_jti()},
            secret=secret,
            expires_in_s=access_ttl,
        )
        refresh = tokens.encode_token(
            # `rmb` 一起签进去：轮换时要保持"记住我"的 30 天还是 7 天，
            # 而 claim 是服务端签名过的、改不动，读它比自己再存一列更省事也更不容易分叉
            {**shared, "typ": _TYP_REFRESH, "jti": refresh_jti, "rmb": bool(remember)},
            secret=secret,
            expires_in_s=refresh_ttl,
        )
        return MintedTokens(
            access=access,
            refresh=refresh,
            refresh_ttl=refresh_ttl,
            refresh_jti=refresh_jti,
        )


def _safe_name(username: str) -> str:
    """日志里的用户名。账号本身不是密码，但它常常是学号/手机号，
    所以只留前两位 + 长度，够定位一个人、不够凑出一份名单。"""
    if not username:
        return "-"
    head = username[:2]
    return f"{head}***({len(username)})"
