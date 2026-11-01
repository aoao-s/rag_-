"""记忆权限与合规单测（CLAUDE.md §3.7 第 7 条 / 实施手册 M4）。

这一层要证明的**不是**"记忆能不能读"，而是三件权限语义：

1. **无 grant 的顾问读记忆返回 404，不是 403**——"某学员有这条记忆"本身就是隐私，
   403 会承认资源存在；
2. **有 grant 但只能读**：写操作返回 403，而不是静默成功，也不是 404
   （对方已经知道记忆存在，这时候承认存在不额外泄漏任何东西）；
3. **每一步都留痕**：拒绝的读取尝试、成功的顾问读取、授权与撤销全部落 `memory_audit`，
   学员要能在记忆页里回答"谁看过我的记忆"。

另有两组与身份头相关的断言（`X-User-Id` 生产前置条件）：

- 白名单环境（dev/test…）才认客户端自报的 `X-User-Id`；
- 其余环境只认网关注入的 `request.state.user_id`，否则 401——**不回落**到请求头，
  否则任何人在生产改个请求头就能读别人的记忆。

`X-Memory-Owner-Id` 是**目标资源**，不是身份：它带不来任何权限，只是"我要看谁的"。
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import httpx
from fastapi import FastAPI

from base.config import Settings
from memory.api import router as memory_router
from memory.service import MemoryAccessDenied, MemoryReadOnly, MemoryService
from memory.session import SessionStore
from memory.store import (
    Atom,
    Grant,
    InMemoryAtomRepo,
    InMemoryGrantRepo,
    new_atom_id,
    new_grant_id,
)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "feature_memory": True,
        "mem_min_confidence": 0.6,
        "mem_recall_top_k": 5,
        "mem_profile_max_tokens": 300,
        "mem_summary_max_tokens": 400,
        "mem_atom_max_tokens": 300,
        "mem_recall_budget_ms": 120,
        "app_env": "dev",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _atom(user_id: str = "u1", *, content: str = "我数学不好") -> Atom:
    return Atom(
        id=new_atom_id(),
        user_id=user_id,
        fact_type="constraint",
        fact_key="subject_weak:数学",
        content=content,
        confidence=0.85,
        valid_from=time.time(),
    )


def _service(
    atom: Atom | None = None, *, grants: InMemoryGrantRepo | None = None
) -> tuple[MemoryService, InMemoryAtomRepo, InMemoryGrantRepo]:
    settings = _settings()
    repo = InMemoryAtomRepo([atom or _atom()])
    store = grants if grants is not None else InMemoryGrantRepo()
    service = MemoryService(settings, repo, SessionStore(settings), grants=store)
    return service, repo, store


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def _actions(repo: InMemoryAtomRepo) -> list[str]:
    return [row["action"] for row in repo.audits]


# ---- 服务层：授权语义 ------------------------------------------------------
def test_owner_reads_own_memory_without_grant_and_no_advisor_audit() -> None:
    service, repo, _ = _service()
    body = _run(service.list_atoms("u1"))
    assert body["total"] == 1
    assert "advisor_read" not in _actions(repo)  # 自己看自己不算"顾问读取"


def test_stranger_read_raises_access_denied() -> None:
    service, _, _ = _service()
    try:
        _run(service.list_atoms("u1", viewer="u2"))
    except MemoryAccessDenied:
        pass
    else:  # pragma: no cover - 走到这里说明闸门没生效
        raise AssertionError("无 grant 的跨用户读取必须被拒")


def test_denied_read_is_audited_with_the_viewer() -> None:
    service, repo, _ = _service()
    with_raises = False
    try:
        _run(service.list_atoms("u1", viewer="u2"))
    except MemoryAccessDenied:
        with_raises = True
    assert with_raises
    rows = [r for r in repo.audits if r["action"] == "advisor_read_denied"]
    assert len(rows) == 1
    assert rows[0]["user_id"] == "u1" and rows[0]["operator"] == "u2"


def test_granted_advisor_reads_and_the_read_is_audited() -> None:
    service, repo, _ = _service()
    _run(service.grant_read("u1", "advisor"))
    body = _run(service.list_atoms("u1", viewer="advisor"))
    assert body["total"] == 1
    assert "advisor_read" in _actions(repo)
    # 授权本身也要留痕，否则"曾经授权过给谁"只能靠记忆
    assert "grant_read" in _actions(repo)


def test_expired_grant_blocks_read() -> None:
    """过期判定必须真的执行：一条永不过期的授权等于永久后门。"""
    service, _, store = _service()
    grant = Grant(
        id=new_grant_id(),
        owner_id="u1",
        grantee_id="advisor",
        expires_at=time.time() - 1,
    )
    store.upsert(grant)
    assert _run(service.authorize_read("advisor", "u1")) is False


def test_grant_active_at_边界() -> None:
    now = 1_000.0
    live = Grant(id="g", owner_id="u1", grantee_id="a", expires_at=now + 1)
    dead = Grant(id="g", owner_id="u1", grantee_id="a", expires_at=now)
    assert live.active_at(now) is True
    assert dead.active_at(now) is False  # 到点即失效，不给"整秒"留后门
    assert Grant(id="g", owner_id="u1", grantee_id="a").active_at(now) is True
    assert Grant(id="g", owner_id="u1", grantee_id="a", status="revoked").active_at(now) is False


def test_revoked_grant_blocks_read_and_revoke_is_audited() -> None:
    service, repo, _ = _service()
    _run(service.grant_read("u1", "advisor"))
    assert _run(service.revoke_read("u1", "advisor"))["revoked"] is True
    assert "revoke_read" in _actions(repo)
    with_raises = False
    try:
        _run(service.list_atoms("u1", viewer="advisor"))
    except MemoryAccessDenied:
        with_raises = True
    assert with_raises


def test_regrant_after_revoke_overwrites_instead_of_duplicating() -> None:
    service, _, _ = _service()
    first = _run(service.grant_read("u1", "advisor"))
    _run(service.revoke_read("u1", "advisor"))
    second = _run(service.grant_read("u1", "advisor"))
    assert second["id"] == first["id"]  # 同一行覆盖，不堆行
    body = _run(service.list_grants("u1"))
    assert body["total"] == 1
    assert body["items"][0]["status"] == "active"
    assert body["items"][0]["revoked_at"] is None


def test_grant_list_both_directions() -> None:
    service, _, _ = _service()
    _run(service.grant_read("u1", "advisor"))
    _run(service.grant_read("u3", "advisor"))
    as_owner = _run(service.list_grants("u1", direction="owner"))
    as_grantee = _run(service.list_grants("advisor", direction="grantee"))
    assert as_owner["total"] == 1 and as_owner["items"][0]["grantee_id"] == "advisor"
    assert as_grantee["total"] == 2


def test_self_grant_is_rejected() -> None:
    service, _, _ = _service()
    for bad in (("u1", "u1"), ("u1", "  ")):
        with_raises = False
        try:
            _run(service.grant_read(*bad))
        except ValueError:
            with_raises = True
        assert with_raises
    assert _run(service.list_grants("u1"))["total"] == 0


def test_advisor_with_read_grant_cannot_write() -> None:
    service, repo, _ = _service()
    atom_id = repo.all()[0].id
    _run(service.grant_read("u1", "advisor"))
    for call in (
        lambda: service.patch("u1", atom_id, content="改掉", actor="advisor"),
        lambda: service.forget("u1", atom_id, "purge_chain", actor="advisor"),
        lambda: service.clear_all("u1", actor="advisor"),
    ):
        try:
            _run(call())
        except MemoryReadOnly:
            continue
        raise AssertionError("只读授权不能写")
    assert repo.all()[0].content == "我数学不好"  # 一条都没改
    assert "advisor_write_denied" in _actions(repo)


def test_stranger_write_raises_access_denied_not_readonly() -> None:
    """没有授权的人写：连"存在"都不该确认，所以是 AccessDenied（对外 404）。"""
    service, repo, _ = _service()
    atom_id = repo.all()[0].id
    try:
        _run(service.patch("u1", atom_id, content="改掉", actor="u2"))
    except MemoryAccessDenied:
        pass
    else:  # pragma: no cover
        raise AssertionError("无授权写入必须被拒")
    # 拒写也要留痕：谁试图改过这条记忆，和谁真的改过一样重要
    assert "advisor_write_denied" in _actions(repo)


def test_owner_writes_are_audited() -> None:
    """M4 验收项：所有写操作落 `memory_audit`。"""
    service, repo, _ = _service()
    atom_id = repo.all()[0].id
    _run(service.patch("u1", atom_id, content="数学很难"))
    _run(service.forget("u1", atom_id, "purge_chain"))
    _run(service.clear_all("u1"))
    actions = _actions(repo)
    for expected in ("update", "purge_chain", "clear_all"):
        assert expected in actions
    assert all(row["user_id"] == "u1" for row in repo.audits)


def test_service_without_grant_repo_denies_cross_user_by_default() -> None:
    """`grants=None` 不是"人人可读"，是"这个实例不提供授权"（默认拒绝）。"""
    settings = _settings()
    repo = InMemoryAtomRepo([_atom()])
    service = MemoryService(settings, repo, SessionStore(settings))
    assert _run(service.authorize_read("u2", "u1")) is False
    assert _run(service.authorize_read("u1", "u1")) is True
    assert _run(service.list_grants("u1"))["total"] == 0


# ---- HTTP 层：状态码与身份头 ----------------------------------------------
def _make_app(
    *, app_env: str = "dev", state_user: str | None = None, with_grant: bool = False
) -> tuple[FastAPI, InMemoryAtomRepo]:
    settings = _settings(app_env=app_env)
    repo = InMemoryAtomRepo([_atom()])
    grants = InMemoryGrantRepo()
    if with_grant:
        grants.upsert(Grant(id=new_grant_id(), owner_id="u1", grantee_id="advisor"))
    service = MemoryService(settings, repo, SessionStore(settings), grants=grants)

    app = FastAPI()

    @app.middleware("http")
    async def _inject_identity(request, call_next):  # type: ignore[no-untyped-def]
        if state_user:
            # 模拟网关注入：生产环境唯一可信的身份来源
            request.state.user_id = state_user
        return await call_next(request)

    app.include_router(memory_router)
    app.state.memory = service
    app.state.container = SimpleNamespace(settings=settings)
    return app, repo


def _call(app: FastAPI, method: str, url: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def test_missing_identity_is_401_and_never_anonymous() -> None:
    """空身份 401——退化成共享 anonymous 会让所有未登录用户共用一份长期记忆。"""
    app, _ = _make_app()
    resp = _call(app, "GET", "/v1/memory/atoms")
    assert resp.status_code == 401


def test_trusted_env_accepts_x_user_id_header() -> None:
    app, _ = _make_app(app_env="dev")
    resp = _call(app, "GET", "/v1/memory/atoms", headers={"X-User-Id": "u1"})
    assert resp.status_code == 200
    assert resp.json()["total"] == 1


def test_untrusted_env_ignores_x_user_id_header() -> None:
    """生产不认客户端自报的身份：改个请求头就能读别人记忆的洞必须堵死。"""
    app, _ = _make_app(app_env="prod")
    resp = _call(app, "GET", "/v1/memory/atoms", headers={"X-User-Id": "u1"})
    assert resp.status_code == 401


def test_untrusted_env_accepts_gateway_injected_identity() -> None:
    app, _ = _make_app(app_env="prod", state_user="u1")
    resp = _call(app, "GET", "/v1/memory/atoms")  # 不带任何身份头
    assert resp.status_code == 200


def test_stranger_read_over_http_is_404_not_403() -> None:
    app, _ = _make_app(app_env="dev", with_grant=False)
    resp = _call(
        app,
        "GET",
        "/v1/memory/atoms",
        headers={"X-User-Id": "stranger", "X-Memory-Owner-Id": "u1"},
    )
    assert resp.status_code == 404  # 404 而不是 403：不确认"u1 有记忆"这件事


def test_granted_advisor_read_over_http_is_200() -> None:
    app, repo = _make_app(app_env="dev", with_grant=True)
    resp = _call(
        app,
        "GET",
        "/v1/memory/atoms",
        headers={"X-User-Id": "advisor", "X-Memory-Owner-Id": "u1"},
    )
    assert resp.status_code == 200
    assert "advisor_read" in _actions(repo)


def test_advisor_write_over_http_is_403() -> None:
    app, _ = _make_app(app_env="dev", with_grant=True)
    atom_id = "whatever"
    resp = _call(
        app,
        "PATCH",
        f"/v1/memory/atoms/{atom_id}",
        headers={"X-User-Id": "advisor", "X-Memory-Owner-Id": "u1"},
        json={"content": "改掉"},
    )
    # 有 grant 的人写：403（承认记忆存在但只读）；无 grant 的人写会得到 404
    assert resp.status_code == 403


def test_advisor_clear_all_over_http_is_403() -> None:
    app, _ = _make_app(app_env="dev", with_grant=True)
    resp = _call(
        app,
        "DELETE",
        "/v1/memory/all",
        params={"confirm": "true"},
        headers={"X-User-Id": "advisor", "X-Memory-Owner-Id": "u1"},
    )
    assert resp.status_code == 403


def test_export_and_chain_are_guarded_like_list() -> None:
    app, _ = _make_app(app_env="dev")
    headers = {"X-User-Id": "stranger", "X-Memory-Owner-Id": "u1"}
    assert _call(app, "POST", "/v1/memory/export", headers=headers).status_code == 404
    assert _call(app, "GET", "/v1/memory/chains/abc", headers=headers).status_code == 404


def test_grant_endpoints_are_owner_only() -> None:
    """顾问不能给自己发授权（那等于自己开门），管理员也不行。"""
    app, _ = _make_app(app_env="dev")
    resp = _call(
        app,
        "POST",
        "/v1/memory/grants",
        headers={"X-User-Id": "advisor", "X-Memory-Owner-Id": "u1"},
        json={"grantee_id": "advisor"},
    )
    assert resp.status_code == 403


def test_grant_create_list_revoke_roundtrip() -> None:
    app, _ = _make_app(app_env="dev")
    headers = {"X-User-Id": "u1"}
    created = _call(app, "POST", "/v1/memory/grants", headers=headers, json={"grantee_id": "advisor"})
    assert created.status_code == 200 and created.json()["grantee_id"] == "advisor"

    listed = _call(app, "GET", "/v1/memory/grants", headers=headers)
    assert listed.json()["total"] == 1

    revoked = _call(app, "DELETE", "/v1/memory/grants/advisor", headers=headers)
    assert revoked.status_code == 200 and revoked.json()["revoked"] is True

    again = _call(app, "DELETE", "/v1/memory/grants/advisor", headers=headers)
    assert again.status_code == 404  # 已撤销：不是"又撤了一次成功"


def test_grant_with_past_expiry_is_400() -> None:
    app, _ = _make_app(app_env="dev")
    resp = _call(
        app,
        "POST",
        "/v1/memory/grants",
        headers={"X-User-Id": "u1"},
        json={"grantee_id": "advisor", "expires_at": time.time() - 60},
    )
    assert resp.status_code == 400


def test_memory_api_returns_503_when_not_wired() -> None:
    app = FastAPI()
    app.include_router(memory_router)
    resp = _call(app, "GET", "/v1/memory/atoms", headers={"X-User-Id": "u1"})
    assert resp.status_code == 503
