"""登录 / 登出接口（`/api/v1/auth/*`）。

**这一层只做搬运**：读请求体 → 交给 `AuthService` → 翻成 HTTP。身份、角色、租户、
ACL 的判定全在服务层，这里一个字都不判——判定写两处，两处迟早不一致。

## 响应体：本模块自成一套 `{code, message, data}`

问答接口用 `ChatResponse`（23 字段的扁平结构），认证接口**不复用它**。两者读者不同：
问答的读者是渲染答案的页面，它关心 `answer / citations / status`；
认证的读者是登录表单，它只需要"成没成功、令牌、以及一句能直接显示给用户的话"。
硬塞进同一个结构，会让登录表单去关心 `citations` 这种和它毫无关系的字段。

- 成功：`{"code": 200, "message": "登录成功", "data": {"access_token": ..., ...}}`
- 失败：`{"code": 401, "message": "用户名或密码错误", "data": {"error": "invalid_credentials"}}`

`code` **恒等于外层 HTTP 状态码**（不做自定义业务码）。这样前端既可以用 `fetch` 的
`response.ok` 判成功，也可以只看 `body.code`，两条路指向同一个结论；而监控按 HTTP 状态
报警的规则不会被"业务码另起一套"打乱。

失败时 `data` 里放**机器码**（`invalid_credentials` / `user_disabled` / `username_taken` /
`invalid_token` / `auth_not_configured` …）。没有它，前端就只能拿中文 `message` 做字符串
匹配来区分"密码错了"和"账号被停用"——而那句话随时会为了措辞更好听被改掉。

## 三条协议级的安全约定

1. **令牌是唯一身份**。本文件的 `/me` 只认 `Authorization: Bearer`，不认任何自报请求头；
   认证接口**不看** `X-User-Id`（那是聊天的开发期身份，与登录无关）。
2. **`tenant_id` / `acl` / `user_id` / `permissions` / `kb_version` 传了就 400**。四个请求体都是
   `extra="forbid"`：多一个字段直接拒绝，而不是"悄悄忽略"。忽略的问题在于用户会以为
   "我传了租户，服务端应该按它算"，而实际上从来没生效过；拒绝则当场把这条误解掐掉。
   唯一的例外是**注册时的 `role`**：它是"我是学生还是老师"的声明（且只接受
   `STUDENT` / `TEACHER`，`ADMIN` 不可自助注册），落了库之后由库说了算——
   登录、刷新、`/me` 都从库读角色，任何请求体都改不了它。
3. **口令只进哈希**。本文件**不打印任何请求体**，日志里出现的只有 `user_id` 与打码后的用户名
   （`auth/service.py::_safe_name`）。

## 关于 `captcha`

请求体里保留 `captcha` 字段，但本版本**没有接入验证码服务**，传了也不会被校验——这是
有意的"字段先行"，不是遗漏：前端契约现在定下来，将来接上提供方时不必再改一次接口。
⚠ 因此**当前没有任何防爆破能力**（账号枚举靠 `AuthService` 的耗时拉平挡，撞库挡不住），
上线前必须补限流或验证码。这句话写在这里，是因为它最容易被"字段都收下了"误导成"已经做了"。
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auth.service import AuthService, AuthSession, InvalidToken
from base.errors import EduRAGError, InternalError, InvalidRequest
from base.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# 请求体
# ---------------------------------------------------------------------------


class _Body(BaseModel):
    """所有请求体的共同配置：`extra="forbid"`。

    用户明确要求"客户端不能传 `role` / `tenant_id` / `acl` / `user_id` / `kb_version` /
    `permissions`"。**不是靠文档提醒，是靠解析器拒绝**——提醒过的东西总有人忘，
    而拒绝了的东西没人能绕过。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class RegisterBody(_Body):
    username: str = Field(min_length=1, max_length=64, description="3–64 位，字母数字与 _ . @ -")
    password: str = Field(min_length=1, max_length=128, description="至少 8 位")
    nickname: str = Field(default="", max_length=64, description="显示名；中文放这里，不放 username")
    captcha: str | None = Field(default=None, max_length=32, description="预留：本版本不校验")


class LoginBody(_Body):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)
    captcha: str | None = Field(default=None, max_length=32, description="预留：本版本不校验")
    remember_me: bool = Field(default=False, description="勾选后 refresh 有效期改为 30 天")


class RefreshBody(_Body):
    refresh_token: str = Field(min_length=1, description="上一次登录/刷新拿到的 refresh_token")


class LogoutBody(_Body):
    refresh_token: str | None = Field(
        default=None, description="不传也返回成功：登出是幂等的"
    )


# ---------------------------------------------------------------------------
# 信封
# ---------------------------------------------------------------------------


def _ok(message: str, data: Any) -> JSONResponse:
    return JSONResponse(status_code=200, content={"code": 200, "message": message, "data": data})


def _fail(exc: EduRAGError) -> JSONResponse:
    """业务异常 → 信封。`code` 取 HTTP 状态，机器码进 `data.error`。"""
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "code": exc.http_status,
            "message": exc.message,
            "data": {"error": exc.code},
        },
    )


async def _guard(impl: Callable[[], Awaitable[JSONResponse]]) -> JSONResponse:
    """把异常收进同一个信封。

    `EduRAGError` 原样翻（它自己带码与状态）；其它异常记日志后统一 500——
    认证接口不该把堆栈或驱动错误文本回给客户端，那里面可能带着 SQL 片段与库名。
    """
    try:
        return await impl()
    except EduRAGError as exc:
        # 401 是正常业务（用户密码打错），不打 error 级；未知码才值得看
        log.info("auth_rejected", code=exc.code, http_status=exc.http_status)
        return _fail(exc)
    except Exception as exc:  # noqa: BLE001 —— 兜底，见 docstring
        log.exception("auth_internal_error", error=type(exc).__name__)
        return _fail(InternalError("认证服务内部错误，请稍后重试"))


def _service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth", None)
    if service is None:
        # 与其它模块一致：服务没装载就是 503，不是 500——这是部署问题，不是代码出错
        raise InvalidRequest("认证服务未就绪", code="auth_not_ready", http_status=503)
    return service


# ---------------------------------------------------------------------------
# 读请求体：体积 → JSON → 字段（顺序固定，先拒的不花后者的钱）
# ---------------------------------------------------------------------------


def _body_meta(model: type[BaseModel], example: dict[str, Any]) -> dict[str, Any]:
    """把 Pydantic 模型翻成 `openapi_extra.requestBody`。

    为什么要手写这一层：本模块的路由签名里**没有请求体参数**（见 `_read` 的注释），
    FastAPI 因此不会自动生成 `requestBody`，Swagger UI 只会显示 "No parameters"。
    从 `model_json_schema()` 生成而不是手抄一份，是为了"改了模型不会忘了改文档"。

    `example` 是**整个请求体**的样例：Swagger 有它就以它为准，否则按类型现造
    （`string → "string"`），而现造出来的值十次有九次过不了校验，
    "点开文档、直接 Execute"必然失败，看起来像接口坏了。
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    return {"required": True, "content": {"application/json": {"schema": {**schema, "example": example}}}}


async def _read(request: Request, model: type[BaseModel]) -> Any:
    """读原始字节再交给模型校验。

    **不把模型写进路由签名**：那样 FastAPI 会先解析再进函数，非法字段被它自己的
    `RequestValidationError` 接走，返回的是 `{"detail": [...]}`——与本模块的
    `{code, message, data}` 信封不是同一个形状，前端得为它单开一条分支。
    这里自己读、自己校验，所有错误都是同一种形状。
    """
    limit = _service(request).settings.max_request_bytes
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit:
                raise InvalidRequest(f"请求体超过 {limit} 字节上限", code="request_too_large", http_status=413)
        except ValueError:
            pass  # 解不开的 Content-Length 交给下面的实际长度判

    raw = await request.body()
    if len(raw) > limit:
        raise InvalidRequest(f"请求体超过 {limit} 字节上限", code="request_too_large", http_status=413)

    try:
        data: Any = json.loads(raw or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise InvalidRequest("请求体不是合法 JSON", code="invalid_request") from None
    if not isinstance(data, dict):
        raise InvalidRequest("请求体必须是一个 JSON 对象", code="invalid_request")

    try:
        return model.model_validate(data)
    except ValidationError as exc:
        # 只回字段名，不回值：请求体里是口令与学生姓名，错误信息会进日志与前端控制台
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ())) or "-"
        raise InvalidRequest(f"请求体字段不合法（{field}）", code="invalid_request") from None


# ---------------------------------------------------------------------------
# 出参
# ---------------------------------------------------------------------------


def _session_data(session: AuthSession) -> dict[str, Any]:
    """登录 / 注册 / 刷新三处共用同一个 `data` 形状。

    三处给同一个形状，前端就只有一个"拿到令牌之后该做什么"的函数；
    给三种（比如注册少一个 `refresh_token`）会逼出三个分支，然后在某一天漏掉一个。
    """
    tokens = session.tokens
    return {
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "token_type": tokens.token_type,
        "expires_in": tokens.expires_in,
        "refresh_expires_in": tokens.refresh_expires_in,
        "user": session.user.public(),
    }


def _bearer(request: Request) -> str:
    """取 `Authorization: Bearer <access_token>` 里的令牌。

    格式不对就是 `invalid_token`（401），**不回落**到任何请求头身份：
    带了凭证却验不过，说明调用方以为自己是某个身份，此时按匿名放行
    等于把"凭证坏了"变成"看到一份别人的资料"（§7.8 同一口径）。
    """
    raw = (request.headers.get("authorization") or "").strip()
    scheme, _, token = raw.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise InvalidToken("缺少 Authorization: Bearer 令牌")
    return token.strip()


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@router.post("/register", openapi_extra={"requestBody": _body_meta(RegisterBody, {
    "username": "student001",
    "password": "your_password",
    "nickname": "小明",
})})
async def register(request: Request) -> JSONResponse:
    """注册并直接登录。

    角色在这里**一次性声明**（`STUDENT` / `TEACHER`，缺省 `STUDENT`），之后由库说了算：
    登录、刷新、`/me` 都读 `sys_user.role`，请求体再也改不动它。因此这不是"客户端传角色"
    ——传的是一个只在注册时有效的声明，服务端校验后才落库；后续每一个请求的身份都来自
    库或令牌 claim，与请求体无关。

    `ADMIN` 不能自助注册（400 `role_not_allowed`）：管理员的拒绝意图集合为空，
    放开的代价是一次 HTTP 请求拿到全部意图权限；提权只能走运维流程（改库 + 审计）。
    拼错的角色名同样是 400，但码是 `invalid_role`——"你写错了"与"这个角色不许自助注册"
    是两件事，前端要显示的提示也不一样。

    用户名已被占用时 409 `username_taken`——这一条**必须**说清楚，因为攻击者从这里
    能得到的只有"这个名字存在"，而注册成功本来就是公之于众的事；含糊其辞只会让
    正常用户反复重试。
    """

    async def impl() -> JSONResponse:
        body = await _read(request, RegisterBody)
        # 公共注册入口只能创建 STUDENT。TEACHER / ADMIN 由管理员审核后在用户表授予，
        # 防止任何人把注册请求里的 role 改成 TEACHER 来获得教案/备课权限。
        session = await _service(request).register(
            body.username, body.password, nickname=body.nickname
        )
        return _ok("注册成功", _session_data(session))

    return await _guard(impl)


@router.post("/login", openapi_extra={"requestBody": _body_meta(LoginBody, {
    "username": "student001",
    "password": "your_password",
    "remember_me": False,
})})
async def login(request: Request) -> JSONResponse:
    """账号口令登录。

    用户名不存在与口令错误返回**同一句话、同一个码**（`invalid_credentials`）：
    区分开就等于把登录接口变成一个用户名字典。
    """
    async def impl() -> JSONResponse:
        body = await _read(request, LoginBody)
        session = await _service(request).login(
            body.username, body.password, remember_me=body.remember_me
        )
        return _ok("登录成功", _session_data(session))

    return await _guard(impl)


@router.post("/refresh", openapi_extra={"requestBody": _body_meta(RefreshBody, {
    "refresh_token": "<登录返回的 refresh_token>",
})})
async def refresh(request: Request) -> JSONResponse:
    """用 refresh 换一对新令牌。**轮换**：旧的立刻作废，响应里给的是新的。

    客户端必须把 `data.refresh_token` 存回去。存旧的会在下一次刷新时拿到 401
    `invalid_token`——那不是 bug，是"被拷走的令牌只能用一次"的代价。
    """
    async def impl() -> JSONResponse:
        body = await _read(request, RefreshBody)
        session = await _service(request).refresh(body.refresh_token)
        return _ok("刷新成功", _session_data(session))

    return await _guard(impl)


@router.post("/logout", openapi_extra={"requestBody": _body_meta(LogoutBody, {
    "refresh_token": "<登录返回的 refresh_token>",
})})
async def logout(request: Request) -> JSONResponse:
    """登出：作废该用户的全部 refresh。**幂等，永远 200**。

    令牌过期、签名坏了、压根没传，都算"现在没有有效会话"，返回 `revoked: 0`。
    登出接口如果能 401，前端就得为"清本地存储"写一条异常分支，而用户按下登出键时
    想要的恰恰是"别再问我了"。

    ⚠ **access_token 不会立刻失效**：它无状态、只有 `AUTH_ACCESS_TTL_S`（默认 600 秒），
    靠 TTL 自然死亡（见 `auth/store.py` 模块注释）。这段时间里旧 access 调 `/me`
    仍然成功——这不是漏洞，是把"每次请求查一次库"换掉的代价，必须写进前端预期。
    """
    async def impl() -> JSONResponse:
        body = await _read(request, LogoutBody)
        revoked = await _service(request).logout(body.refresh_token)
        return _ok("已退出登录", {"revoked": revoked})

    return await _guard(impl)


@router.get("/me")
async def me(request: Request) -> JSONResponse:
    """当前登录用户。资料**从库里读**，不从令牌 claim 拼一份。

    令牌里只有 `sub / role / tenant / acl`，没有昵称头像，而且它是 10 分钟前的快照；
    用户刚改了昵称，这里该显示新的。
    """
    async def impl() -> JSONResponse:
        user = await _service(request).me(_bearer(request))
        return _ok("ok", {"user": user.public()})

    return await _guard(impl)
