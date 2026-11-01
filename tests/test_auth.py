"""认证接口单测：`/api/v1/auth/*`（正常 / 超时 / 降级三条路径，CLAUDE.md §5.4）。

不起 MySQL：`InMemoryUserRepo` / `InMemoryTokenRepo` 与 MySQL 实现是同一个 Protocol
的两个实现，业务逻辑（`AuthService`）在两边完全一致，所以这一层能证明的东西是
"分支与码选对了"，持久化细节由 `auth/store.py` 自己负责。

四条断言值得单独说：

1. **角色只认库里那一份**。注册时可以**声明一次** `role`（`STUDENT` / `TEACHER`，缺省
   `STUDENT`，`ADMIN` 与拼错的名字都是 400），声明后由库说了算：登录、刷新、`/me`
   都读 `sys_user.role`。`tenant_id` / `user_id` / `permissions` / `acl_tags` 传了一律
   400（`_Body` 的 `extra="forbid"`）。
2. **"查无此人"与"密码错误"不可区分**（同 401、同码、同句），但**"账号停用"必须排在
   口令校验之后**——否则不需要知道密码就能探知"这个账号存在但被停用"。
3. **登出只作废 refresh**（用户决定：access 10 分钟自然死亡）。所以"登出后 access 仍能
   调 `/me`"是契约的一部分，用一条测试把它钉住，将来有人改长 `AUTH_ACCESS_TTL_S`
   时能看见自己改的是什么。
4. **没有可用身份时 401，绝不降级为匿名**；服务没装载是 503（部署问题），不是 500。
"""

from __future__ import annotations

import asyncio
import time

import httpx
from fastapi import FastAPI

from auth import passwords, tokens
from auth.api import router as auth_router
from auth.service import AuthService
from auth.store import InMemoryTokenRepo, InMemoryUserRepo, User, new_user_id
from base.config import Settings

# ≥32 位的测试密钥。真实密钥只在 `.env`（不进库、不进本文件）
SECRET = "auth-test-secret-0123456789abcdef01234567"
PASSWORD = "Passw0rd!ok"


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def _settings(**overrides: object) -> Settings:
    """显式给全本文件依赖的每一项，避免读到本机 `.env` 造成"这台机器上过、那台不过"。"""
    base: dict[str, object] = {
        "jwt_secret": SECRET,
        "jwt_leeway_s": 60,
        "max_request_bytes": 65536,
        "tenant_id": "default",
        "auth_access_ttl_s": 600,
        "auth_refresh_ttl_s": 604800,
        "auth_refresh_ttl_remember_s": 2592000,
    }
    base.update(overrides)
    return Settings(**base)


def _service(**overrides: object) -> AuthService:
    return AuthService(
        settings=_settings(**overrides),
        users=InMemoryUserRepo(),
        tokens=InMemoryTokenRepo(),
    )


def _app(service: object | None) -> FastAPI:
    app = FastAPI()
    app.include_router(auth_router)
    if service is not None:
        app.state.auth = service
    return app


def _call(app: FastAPI, method: str, path: str, **kwargs: object) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(run())


def _register(
    app: FastAPI,
    username: str = "student001",
    password: str = PASSWORD,
    nickname: str = "小明",
    role: str | None = None,
) -> httpx.Response:
    body: dict[str, object] = {"username": username, "password": password, "nickname": nickname}
    if role is not None:
        body["role"] = role
    return _call(app, "POST", "/api/v1/auth/register", json=body)


def _login(
    app: FastAPI,
    username: str = "student001",
    password: str = PASSWORD,
    **extra: object,
) -> httpx.Response:
    return _call(
        app, "POST", "/api/v1/auth/login", json={"username": username, "password": password, **extra}
    )


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_user(
    repo: InMemoryUserRepo,
    *,
    username: str = "student001",
    role: str = "STUDENT",
    status: str = "active",
    nickname: str = "小明",
    tenant_id: str = "default",
    acl_tags: tuple[str, ...] = (),
) -> User:
    """直接塞一条账号（不经过注册接口）：用来造"注册路径造不出来"的状态，如 disabled。"""
    user = User(
        user_id=new_user_id(),
        username=username,
        password_hash=passwords.hash_password(PASSWORD),
        nickname=nickname,
        role=role,
        tenant_id=tenant_id,
        acl_tags=acl_tags,
        status=status,
    )
    repo.create(user)
    return user


def _assert_fail(response: httpx.Response, status: int, error: str) -> dict[str, object]:
    """失败信封的统一形状：`code == HTTP 状态`，机器码在 `data.error`。"""
    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == status
    assert body["data"]["error"] == error
    assert isinstance(body["message"], str) and body["message"]
    return body


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------


def test_register_returns_envelope_and_student_role() -> None:
    app = _app(_service())
    response = _register(app)

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["message"] == "注册成功"

    data = body["data"]
    assert data["token_type"] == "Bearer"
    assert data["expires_in"] == 600  # AUTH_ACCESS_TTL_S（用户决定：10 分钟）
    assert data["access_token"] and data["refresh_token"]
    assert data["access_token"] != data["refresh_token"]
    # 不声明角色即 STUDENT（缺省值，不是"没设置"）
    assert data["user"]["role"] == "STUDENT"
    assert data["user"]["username"] == "student001"  # 已归一为小写
    assert data["user"]["nickname"] == "小明"

    # 口令与哈希都不得出现在响应里（响应体最终会进日志、浏览器缓存、用户截图）
    assert PASSWORD not in response.text
    assert "$argon2" not in response.text
    assert "password_hash" not in data["user"]


def test_register_rejects_client_supplied_identity() -> None:
    """除注册时的 `role` 声明外，身份字段传了就 400。

    `role` 是**唯一**合法的自报字段（见 `test_register_role_declaration`），
    其余四个与租户/授权相关的字段一律走 `extra="forbid"`。
    """
    service = _service()
    app = _app(service)

    for field, value in (
        ("tenant_id", "other-school"),
        ("user_id", "u-123"),
        ("permissions", ["exam:create"]),
        ("acl_tags", ["secret"]),
    ):
        response = _call(
            app,
            "POST",
            "/api/v1/auth/register",
            json={"username": "student002", "password": PASSWORD, field: value},
        )
        _assert_fail(response, 400, "invalid_request")

    # 被拒的请求一条都没落库
    assert service.users.count() == 0


def test_register_role_declaration() -> None:
    """注册时声明 TEACHER：落库、响应回显、令牌 claim 带同一个角色。

    这三处必须一致——只改响应不改库，用户登录一次就变回 STUDENT；
    只改库不改 claim，聊天侧的 `permissions_for(role, intent)` 仍然按学生算。
    """
    app = _app(_service())
    response = _register(app, username="teacher001", role="teacher")  # 大小写归一

    assert response.status_code == 200
    data = response.json()["data"]
    assert data["user"]["role"] == "TEACHER"

    claims = tokens.decode_token(data["access_token"], secret=SECRET, leeway_s=60)
    assert claims["role"] == "TEACHER"


def test_register_admin_and_bad_role_are_rejected() -> None:
    """`ADMIN` 与拼错的名字都是 400，但码不同：一个"不许"，一个"写错了"。"""
    service = _service()
    app = _app(service)

    _assert_fail(_register(app, role="ADMIN"), 400, "role_not_allowed")
    _assert_fail(_register(app, role="principle"), 400, "invalid_role")
    assert service.users.count() == 0


def test_register_duplicate_username_is_409() -> None:
    app = _app(_service())
    _register(app)
    response = _register(app)
    _assert_fail(response, 409, "username_taken")


def test_register_rejects_weak_password_and_bad_username() -> None:
    app = _app(_service())
    _assert_fail(_register(app, password="short"), 400, "invalid_password")
    _assert_fail(_register(app, username="小明"), 400, "invalid_username")


def test_register_then_login_is_case_insensitive() -> None:
    app = _app(_service())
    _register(app, username="Student001")
    response = _login(app, username="STUDENT001")

    assert response.status_code == 200
    assert response.json()["message"] == "登录成功"


def test_register_tenant_comes_from_settings_not_body() -> None:
    app = _app(_service(tenant_id="school-a"))
    body = _register(app).json()
    assert body["data"]["user"]["tenant_id"] == "school-a"


# ---------------------------------------------------------------------------
# 登录
# ---------------------------------------------------------------------------


def test_login_unknown_user_and_wrong_password_are_indistinguishable() -> None:
    app = _app(_service())
    _register(app)

    wrong_password = _login(app, password="WrongPassw0rd!")
    unknown_user = _login(app, username="nobody001")

    assert wrong_password.status_code == 401
    assert unknown_user.status_code == 401
    # 同状态、同码、同句：区分开就等于把登录接口变成一本用户名目录
    assert wrong_password.json()["data"]["error"] == "invalid_credentials"
    assert wrong_password.json()["data"]["error"] == unknown_user.json()["data"]["error"]
    assert wrong_password.json()["message"] == unknown_user.json()["message"]


def test_login_disabled_user_is_403_only_after_password_matches() -> None:
    repo = InMemoryUserRepo()
    _seed_user(repo, username="blocked", status="disabled")
    app = _app(AuthService(settings=_settings(), users=repo, tokens=InMemoryTokenRepo()))

    # 口令正确 → 才暴露"账号已停用"
    _assert_fail(_login(app, username="blocked"), 403, "user_disabled")
    # 口令错误 → 仍然是 401：先看 status 的话，不需要密码就能探知账号存在
    _assert_fail(_login(app, username="blocked", password="WrongPassw0rd!"), 401, "invalid_credentials")


def test_login_remember_me_extends_refresh_ttl() -> None:
    app = _app(_service())
    _register(app)

    normal = _login(app).json()["data"]
    remembered = _login(app, remember_me=True).json()["data"]

    assert normal["refresh_expires_in"] == 604800  # 7 天
    assert remembered["refresh_expires_in"] == 2592000  # 30 天
    assert normal["expires_in"] == remembered["expires_in"] == 600  # access 不变


def test_access_token_claims_carry_identity_but_not_kb_version() -> None:
    repo = InMemoryUserRepo()
    _seed_user(repo, username="teacher01", role="TEACHER", acl_tags=("class-a",))
    app = _app(AuthService(settings=_settings(), users=repo, tokens=InMemoryTokenRepo()))

    access = _login(app, username="teacher01").json()["data"]["access_token"]
    claims = tokens.decode_token(access, secret=SECRET, leeway_s=60)

    assert claims["typ"] == "access"
    assert claims["role"] == "TEACHER"
    assert claims["tenant"] == "default"
    assert claims["acl"] == ["class-a"]
    # kb_version 由服务端运行期决定：写进令牌会让"切索引后旧令牌查旧库"变成可能
    assert "kb_version" not in claims
    assert "password" not in claims


# ---------------------------------------------------------------------------
# /me
# ---------------------------------------------------------------------------


def test_me_returns_profile_from_store() -> None:
    repo = InMemoryUserRepo()
    user = _seed_user(repo, username="xiaoming", nickname="小明", acl_tags=("grade-9",))
    app = _app(AuthService(settings=_settings(), users=repo, tokens=InMemoryTokenRepo()))

    access = _login(app, username="xiaoming").json()["data"]["access_token"]
    response = _call(app, "GET", "/api/v1/auth/me", headers=_bearer(access))

    assert response.status_code == 200
    data = response.json()["data"]["user"]
    assert data["user_id"] == user.user_id
    assert data["nickname"] == "小明"
    assert data["acl_tags"] == ["grade-9"]
    # 令牌里带的是登录那一刻的快照，/me 必须回库里的那一份（改了昵称立刻可见）
    assert "password_hash" not in data


def test_me_rejects_missing_or_broken_credentials() -> None:
    app = _app(_service())
    access = _register(app).json()["data"]["access_token"]

    _assert_fail(_call(app, "GET", "/api/v1/auth/me"), 401, "invalid_token")
    _assert_fail(
        _call(app, "GET", "/api/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"}),
        401,
        "invalid_token",
    )
    # 带了凭证却验不过，是 401 而不是"按匿名放行"
    _assert_fail(_call(app, "GET", "/api/v1/auth/me", headers=_bearer(access + "x")), 401, "invalid_token")


def test_me_denies_disabled_user_after_login() -> None:
    from dataclasses import replace

    repo = InMemoryUserRepo()
    user = _seed_user(repo, username="xiaohong")
    app = _app(AuthService(settings=_settings(), users=repo, tokens=InMemoryTokenRepo()))
    access = _login(app, username="xiaohong").json()["data"]["access_token"]

    # 运维在库里停用了这个账号；手里的 access 还没过期
    repo._users[(user.tenant_id, user.username)] = replace(user, status="disabled")  # noqa: SLF001

    _assert_fail(_call(app, "GET", "/api/v1/auth/me", headers=_bearer(access)), 403, "user_disabled")


# ---------------------------------------------------------------------------
# 刷新与令牌类型
# ---------------------------------------------------------------------------


def test_refresh_rotates_and_invalidates_the_old_token() -> None:
    app = _app(_service())
    first = _register(app).json()["data"]
    old_refresh = first["refresh_token"]

    rotated = _call(app, "POST", "/api/v1/auth/refresh", json={"refresh_token": old_refresh})
    assert rotated.status_code == 200
    new_refresh = rotated.json()["data"]["refresh_token"]
    assert new_refresh != old_refresh

    # 新 refresh 能用；旧 refresh 已经作废（"被抄走的令牌只能用一次"）
    assert _call(
        app, "POST", "/api/v1/auth/refresh", json={"refresh_token": new_refresh}
    ).status_code == 200
    _assert_fail(
        _call(app, "POST", "/api/v1/auth/refresh", json={"refresh_token": old_refresh}),
        401,
        "invalid_token",
    )
    # 新的 access 立刻可用（换的是新令牌，不是同一份再发一遍）
    access = rotated.json()["data"]["access_token"]
    assert _call(app, "GET", "/api/v1/auth/me", headers=_bearer(access)).status_code == 200


def test_token_types_cannot_be_swapped() -> None:
    app = _app(_service())
    data = _register(app).json()["data"]

    # refresh 当 access 用 → 401（否则 10 分钟的窗口就变成 30 天）
    _assert_fail(
        _call(app, "GET", "/api/v1/auth/me", headers=_bearer(data["refresh_token"])), 401, "invalid_token"
    )
    # access 当 refresh 用 → 401
    _assert_fail(
        _call(app, "POST", "/api/v1/auth/refresh", json={"refresh_token": data["access_token"]}),
        401,
        "invalid_token",
    )


def test_expired_access_is_rejected_and_leeway_is_the_knob() -> None:
    """超时路径：access 只有 `AUTH_ACCESS_TTL_S` 秒，过点即 401。"""
    # 两个服务共用同一份仓储：换的只是配置（TTL / 时钟容忍），不是账号
    users, token_repo = InMemoryUserRepo(), InMemoryTokenRepo()
    strict = _app(
        AuthService(
            settings=_settings(auth_access_ttl_s=1, jwt_leeway_s=0),
            users=users,
            tokens=token_repo,
        )
    )
    data = _register(strict).json()["data"]

    time.sleep(1.1)

    _assert_fail(_call(strict, "GET", "/api/v1/auth/me", headers=_bearer(data["access_token"])), 401, "invalid_token")

    # 同一枚令牌、同一份仓储，只把 `JWT_LEEWAY_S` 放到 60：时钟偏移容忍内仍然接受。
    # 这条同时证明"过期判定真的在读配置"，不是写死的。
    lenient = _app(
        AuthService(
            settings=_settings(auth_access_ttl_s=1, jwt_leeway_s=60),
            users=users,
            tokens=token_repo,
        )
    )
    assert _call(lenient, "GET", "/api/v1/auth/me", headers=_bearer(data["access_token"])).status_code == 200


# ---------------------------------------------------------------------------
# 登出
# ---------------------------------------------------------------------------


def test_logout_revokes_all_refresh_and_is_idempotent() -> None:
    app = _app(_service())
    _register(app)  # 注册即登录：这里已经签发了一条 refresh
    # 再加两次登录 = 两个设备，共三条 refresh
    first = _login(app).json()["data"]["refresh_token"]
    second = _login(app).json()["data"]["refresh_token"]

    logout = _call(app, "POST", "/api/v1/auth/logout", json={"refresh_token": first})
    assert logout.status_code == 200
    assert logout.json()["message"] == "已退出登录"
    # 该用户**全部** refresh（注册那条也在内），不只传进来的这一条
    assert logout.json()["data"]["revoked"] == 3

    # 幂等：再登出一次是 0，不是错误
    again = _call(app, "POST", "/api/v1/auth/logout", json={"refresh_token": first})
    assert again.status_code == 200
    assert again.json()["data"]["revoked"] == 0

    # 两条 refresh 都换不出新令牌了
    for token in (first, second):
        _assert_fail(_call(app, "POST", "/api/v1/auth/refresh", json={"refresh_token": token}), 401, "invalid_token")


def test_logout_never_fails_on_missing_or_garbage_token() -> None:
    app = _app(_service())
    _register(app)

    no_body = _call(app, "POST", "/api/v1/auth/logout")
    assert no_body.status_code == 200
    assert no_body.json()["data"]["revoked"] == 0

    empty = _call(app, "POST", "/api/v1/auth/logout", json={})
    assert empty.status_code == 200

    garbage = _call(app, "POST", "/api/v1/auth/logout", json={"refresh_token": "not-a-jwt"})
    assert garbage.status_code == 200
    assert garbage.json()["data"]["revoked"] == 0


def test_logout_leaves_access_usable_until_expiry() -> None:
    """用户决定（2026-10-31）：登出只作废 refresh，access 靠 10 分钟 TTL 自然死亡。

    这条**不是缺陷**，是把"每次请求查一次库"换掉所付的代价。写成测试而不是注释，
    是为了让将来调长 `AUTH_ACCESS_TTL_S` 的人看见自己实际放宽的是什么。
    """
    app = _app(_service())
    data = _register(app).json()["data"]

    _call(app, "POST", "/api/v1/auth/logout", json={"refresh_token": data["refresh_token"]})

    assert _call(
        app, "GET", "/api/v1/auth/me", headers=_bearer(data["access_token"])
    ).status_code == 200
    _assert_fail(
        _call(app, "POST", "/api/v1/auth/refresh", json={"refresh_token": data["refresh_token"]}),
        401,
        "invalid_token",
    )


# ---------------------------------------------------------------------------
# 降级：没配密钥 / 服务没装载 / 内部异常 / 请求体
# ---------------------------------------------------------------------------


def test_empty_jwt_secret_degrades_to_503() -> None:
    service = _service(jwt_secret="")
    assert service.configured is False
    app = _app(service)

    # 注册、登录、刷新、/me 全部 503 auth_not_configured（不是 500：这是部署没做完）
    _assert_fail(_register(app), 503, "auth_not_configured")
    _assert_fail(_login(app), 503, "auth_not_configured")
    _assert_fail(
        _call(app, "POST", "/api/v1/auth/refresh", json={"refresh_token": "x"}), 503, "auth_not_configured"
    )
    _assert_fail(_call(app, "GET", "/api/v1/auth/me", headers=_bearer("x")), 503, "auth_not_configured")


def test_missing_service_is_503_not_500() -> None:
    _assert_fail(_login(_app(None)), 503, "auth_not_ready")


def test_unexpected_error_is_500_envelope_without_leaking_details() -> None:
    class _Broken:
        def __init__(self, settings: Settings) -> None:
            self.settings = settings

        async def login(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("boom: password=leaked-value")

    _assert_fail(_login(_app(_Broken(_settings()))), 500, "internal_error")
    body = _login(_app(_Broken(_settings()))).text
    assert "boom" not in body
    assert "leaked-value" not in body
    assert "RuntimeError" not in body


def test_malformed_and_oversized_bodies_are_rejected_before_parsing() -> None:
    app = _app(_service())
    _assert_fail(
        _call(app, "POST", "/api/v1/auth/login", content=b"{not json", headers={"content-type": "application/json"}),
        400,
        "invalid_request",
    )
    _assert_fail(
        _call(app, "POST", "/api/v1/auth/login", content=b'["not","an","object"]'),
        400,
        "invalid_request",
    )

    tiny = _app(_service(max_request_bytes=1024))
    big = b'{"username":"a","password":"' + b"x" * 2000 + b'"}'
    _assert_fail(_call(tiny, "POST", "/api/v1/auth/login", content=big), 413, "request_too_large")
