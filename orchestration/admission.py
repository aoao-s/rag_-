"""阶段 1 准入层：所有入口（REST / SSE）共用的**唯一**一道门。

设计见 `docs/EduRAG-Pro-阶段1-请求上下文与入口设计.md`。一句话：

```
HTTP/SSE → orchestration/api.py → admit() → RequestContext → ChatPipeline.ask()
```

为什么必须只有一处：长度上限、学科词表、会话归属、身份来源，每多一处判就多一处
能忘、能改歪、能在两条入口上给出不同结论的地方。`main.py` 的校验中间件、
`query_processor` 的截断、各路由自己的 `if len(...)`——全部收在这里。

三条硬边界（违反即打回）：

1. **身份只能来自可信上下文**：JWT（`Authorization: Bearer`）或白名单环境下网关注入的
   请求头。请求体里的任何字段（含 `role`）**永不参与**身份判定——请求体是用户写的。
2. **带了凭证却验不过 → 401，绝不降级为匿名**。降级的后果不是"少答一条"，
   而是"一个以为自己是张老师的人，拿到了匿名视角下本不该给他的东西，
   且日志里看起来一切正常"（`base/errors.py::Unauthorized` 同一口径）。
3. **不 import `memory/`**：会话归属通过注入的 `owner_of` 回调进入，本层只依赖
   `base/` 与 `qa_system/res/qa_slots`（分层方向：编排层 → 理解/检索/记忆 → base）。

`request_id` / `trace_id` 也在这里定：两者是"把用户截图和日志对上"的唯一线索，
生成规则只能有一份，否则两条入口的 id 长得不一样，排障时对不上。
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from auth import tokens as auth_tokens
from base.config import Settings, is_trusted_env
from base.contracts import PermissionDecision
from base.errors import EduRAGError
from base.labels import INTENT_LABELS, ROLE_ANONYMOUS, ROLES, SUBJECTS, normalize_role
from base.logger import get_logger
from qa_system.res import qa_slots

log = get_logger(__name__)

# 一次请求里 ACL 标签的上限。超过就截断而不是拒——多出来的标签是"多给一份可见性"，
# 与"少给"不同级；但完全不限会让 Redis 键与 Milvus 表达式无界增长。
_ACL_MAX: Final[int] = 20
# 会话 id 口径：32 位小写十六进制（`uuid4().hex`）。它同时是 Redis 键的一段，
# 放开字符集会顺带把"键注入"变成可能（`:` 与换行）。
_SESSION_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{32}$")
# `X-Request-Id` 只接受这个字符集：它会原样进日志与 `X-Request-Id` 响应头，
# 换行符能把一行日志掰成两行（伪造现场），冒号会污染结构化字段的解析。
_REQUEST_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_TRACE_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{32}$")


# ---------------------------------------------------------------------------
# 业务权限（阶段 3）：角色 × 意图
# ---------------------------------------------------------------------------

# 角色 → 允许的意图。**表就是策略**：改权限只改这张表，不在下游写 if/else
# （与 `understanding/profiles.py` 的档位表同一原则，§7.7 硬规则 1）。
#
# 三处刻意的取舍：
# - 学生**可以**出题（「学生可以生成试卷测,测试题等」），但不许做教案与备课检索——
#   那两件事的产物是给老师用的，权限上属于"以老师身份作业"；
# - `ANONYMOUS` 比学生再少一项「考试出题」：它既没有身份也没有归属，
#   而生成类任务的产物要能追到人（§7.9 边界 3 同一口径：无主的产物不给）；
# - 词表外的角色一律按 `ANONYMOUS` 处理（fail-closed，"往少给倒"）。
ROLE_INTENTS: Final[dict[str, frozenset[str]]] = {
    "STUDENT": frozenset({"学习讲解", "学习规划", "考试出题", "闲聊", "超出范围"}),
    ROLE_ANONYMOUS: frozenset({"学习讲解", "学习规划", "闲聊", "超出范围"}),
    "TEACHER": frozenset(INTENT_LABELS),
    "ADMIN": frozenset(INTENT_LABELS),
}


def permissions_for(role: str | None, *, enabled: bool = True) -> PermissionDecision:
    """角色 → 权限判定。**纯函数，无 IO**，因此可以在每一层重复调用而不必担心不一致。

    `enabled=False`（`FEATURE_PERMISSION=false`）返回空拒绝集，等于整条门禁不存在——
    这是紧急回滚开关，与 `FEATURE_INTENT=false` 是同一角色。

    词表外角色按 `ANONYMOUS` 算并标 `source="unknown_role"`：既不因此 401
    （角色问题该由登录环节暴露，不该让一条问答请求失败），也不放行到比匿名更多的东西。
    """
    normalized = normalize_role(role) or ROLE_ANONYMOUS
    if not enabled:
        return PermissionDecision(role=normalized, denied=frozenset(), source="disabled")

    allowed = ROLE_INTENTS.get(normalized)
    source: str = "role"
    if allowed is None:
        allowed = ROLE_INTENTS[ROLE_ANONYMOUS]
        source = "unknown_role"
    denied = frozenset(label for label in INTENT_LABELS if label not in allowed)
    return PermissionDecision(role=normalized, denied=denied, source=source)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 请求上下文（节点之间的唯一载体）
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Deadline:
    """单请求时间预算。**单调时钟**（`perf_counter`），不随系统时间跳变。

    `child(cap)` 只收窄上限、**不动起点**：每个节点拿到的都是"整条请求还剩多少"，
    而不是"从我这里重新开始算 800 ms"——后者会让总预算随时长线性膨胀
    （三个各 800 ms 的节点 = 2.4 s，而 §6.1 给的总额是 8000 ms）。
    """

    started: float
    budget_ms: int

    @classmethod
    def start(cls, budget_ms: int) -> Deadline:
        return cls(started=time.perf_counter(), budget_ms=max(0, int(budget_ms)))

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started) * 1000)

    def remaining_ms(self) -> int:
        return max(0, self.budget_ms - self.elapsed_ms())

    def child(self, cap_ms: int | None = None) -> Deadline:
        if cap_ms is None:
            return self
        return Deadline(started=self.started, budget_ms=min(self.budget_ms, max(0, int(cap_ms))))

    def expired(self) -> bool:
        return self.remaining_ms() <= 0


@dataclass(frozen=True, slots=True)
class Identity:
    """调用方身份。`user_id` 为空 = 匿名（`role` 恒为 `ANONYMOUS`）。

    `source` 不是装饰：`jwt` 与 `header` 的信任级别完全不同，排障时要能一眼看出
    "这次的身份是验过签的，还是开发环境自报的"。
    """

    user_id: str = ""
    role: str = ROLE_ANONYMOUS
    tenant_id: str = ""
    acl_tags: tuple[str, ...] = ()
    source: Literal["jwt", "header", "anonymous"] = "anonymous"

    @property
    def anonymous(self) -> bool:
        return not self.user_id


@dataclass(frozen=True, slots=True)
class RequestContext:
    """准入层交给编排层的全部输入。字段顺序即文档 §3.2 的口径。

    `degraded` 是**有意留下的尾巴**：`owner_of` 查询失败时按 fail-open 放行，
    但"这次没校验过归属"必须让下游看得见（`session_owner_unchecked`）——
    静默 fail-open 与"校验通过"在日志里长得一模一样，是这类设计最常见的坑。
    """

    request_id: str
    trace_id: str
    user_id: str
    tenant_id: str
    role: str
    session_id: str | None
    acl_tags: tuple[str, ...]
    kb_version: str
    deadline: Deadline
    query: str
    subject: str | None = None
    selected_faq_id: int | None = None
    require_citation: bool | None = None
    # 业务权限结论：**在这里算出来**，编排层只读不重算（`permissions_for()` 是纯函数，
    # 但没有第二个地方该决定"这个角色能不能用这项功能"）。
    permission: PermissionDecision | None = None
    degraded: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 异常：同一类型承载 7 个对外错误码
# ---------------------------------------------------------------------------


class AdmissionError(EduRAGError):
    """准入拒绝。默认 400；鉴权失败由抛出点传 `http_status=401`。

    `extra` 进 `detail()`，**不**进 `as_dict()`：`subject_invalid` 要带
    `allowed`（前端据此渲染可选学科），而 `as_dict()` 是错误码的通用形状，
    往里塞可变字段会让所有调用方都要处理"这个键有时在有时不在"。
    """

    code = "invalid_request"
    http_status = 400

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        http_status: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, http_status=http_status)
        self.extra: dict[str, Any] = dict(extra or {})

    def detail(self) -> dict[str, Any]:
        """旧口径的 `detail` 字段：`{"error": <code>, ...extra}`。

        保留它是因为既有前端与测试都按 `detail.error` 判分支，而设计又要求
        400 的响应体是完整的 `ChatResponse` 形状。两者不冲突：**外层给形状，
        内层给分支依据**（见 `orchestration/api.py::_admission_response`）。
        """
        return {"error": self.code, **self.extra}


# ---------------------------------------------------------------------------
# 请求体模型
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    """`POST /v1/chat` 的请求体。

    三个刻意的选择：

    - **不设 `max_length`**：长度上限归 `admit()`（`CHAT_MAX_QUERY_CHARS`，默认 400）。
      在模型上再写一份 200，就会出现"入口按 200 拒、内部策略按 400 算"的两套上限，
      而且报错形状是 422 而不是准入层的 400（阶段 1 之前就是这么错的）。
    - **也不设 `min_length`**（2026-10-31 修）：空问句同样归 `admit()`。写了 `min_length=1`，
      `""` 就会在 pydantic 那层报 `invalid_request`，与 `"   "` 的 `empty_query` 分成两个码；
      更糟的是它发生在 `resolve_identity()` **之前**——坏 token + 空问句会先拿到 400，
      把 §3.4「401 先于一切 400」这条顺序破掉，等于让没凭证的调用方借参数错误探路。
      `GET /v1/chat/stream` 用 `model_construct` 绕过校验，就是为了不让第二套规则抢先判。
    - **`role` 字段存在但永不读取**：它在这里的唯一作用是**不报 422**。
      客户端（尤其老前端）会顺手带上 `role`，而"多一个字段就整条请求失败"对用户
      毫无提示价值。真正的口径是"身份只看 JWT/网关头"——所以它被无条件忽略，
      响应体里的 `role` 永远来自服务端解析出的身份。
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="用户原始问题")
    subject: str | None = Field(default=None, description="学科；不在词表内 → 400 subject_invalid")
    selected_faq_id: int | None = Field(default=None, description="FAQ 确认通道：用户点选的 faq_id")
    session_id: str | None = Field(default=None, description="会话 id（32 位十六进制）")
    require_citation: bool | None = Field(default=None, description="是否强制引用；空 = 按档位")
    role: str | None = Field(default=None, description="**永不读取**：身份只来自可信上下文")


# ---------------------------------------------------------------------------
# id：request_id / trace_id 的唯一出处
# ---------------------------------------------------------------------------


def make_request_id(request: Any) -> str:
    """中间件已生成的优先；否则鉴权过的 `X-Request-Id`；再否则新生成。

    **不合法的调用方头不会被采纳，而是重新生成并记一行日志**：将就着用它，
    等于让外部输入直接进日志字段（换行注入），而拒掉整条请求又太重——
    id 只是个线索，换个新的不损失任何信息。
    """
    existing = getattr(getattr(request, "state", None), "request_id", None)
    if isinstance(existing, str) and existing.strip():
        existing = existing.strip()
        # 中间件写进来的也要**过一遍格式**：`main.py` 的 trace 中间件直接把
        # `X-Request-Id` 存进 `request.state`，不校验就等于"校验发生在别的分支里"。
        # 一份 id 有两种待遇是最难查的那种不一致。
        if _REQUEST_ID_RE.match(existing):
            return existing
        log.info("request_id_rejected", reason="format", length=len(existing))
        return uuid.uuid4().hex

    candidate = _header(request, "X-Request-Id")
    if candidate and _REQUEST_ID_RE.match(candidate):
        return candidate
    if candidate:
        log.info("request_id_rejected", reason="format", length=len(candidate))
    return uuid.uuid4().hex


def make_trace_id(request: Any, *, request_id: str) -> str:
    """`X-Trace-Id` → W3C `traceparent` 的第 2 段 → 退回 `request_id`。

    退回而不是留空：空的 `trace_id` 在日志里与"这个字段忘了填"无法区分，
    而 `= request_id` 至少保证"本次请求的每一行都能被同一个键串起来"。
    """
    candidate = _header(request, "X-Trace-Id")
    if candidate and _REQUEST_ID_RE.match(candidate):
        return candidate

    parent = _header(request, "traceparent")
    if parent:
        parts = parent.split("-")
        if len(parts) >= 2 and _TRACE_ID_RE.match(parts[1]):
            return parts[1]
    return request_id


def _header(request: Any, name: str) -> str:
    headers = getattr(request, "headers", None)
    if headers is None:
        return ""
    return (headers.get(name) or "").strip()


# ---------------------------------------------------------------------------
# 身份解析：三层，由严到宽
# ---------------------------------------------------------------------------


def resolve_identity(request: Any, settings: Settings) -> Identity:
    """按 §3.3 的顺序解析身份：JWT → 开发期请求头 → 匿名。

    第二条只在 `jwt_secret` 为空**且**环境在白名单内时可用。两个条件缺一不可：
    配了 JWT 还认请求头，等于给验证过的身份系统留了一个"直接自报"的后门；
    而白名单外（`prod`/`staging`/任何没见过的取值）一律不认，因为预发环境
    连的常常就是生产数据。
    """
    auth = _header(request, "Authorization")
    if auth:
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AdmissionError(
                "Authorization 头格式应为 `Bearer <token>`",
                code="invalid_token",
                http_status=401,
            )
        return _identity_from_token(token.strip(), settings)

    if settings.jwt_secret or not is_trusted_env(settings):
        return _anonymous(settings)
    return _identity_from_headers(request, settings)


def header_identity(
    user_header: str | None,
    session_id: str | None,
    *,
    tenant_header: str | None = None,
    acl_header: str | None = None,
    role_header: str | None = None,
    default_tenant: str = "default",
    trust_headers: bool = True,
) -> tuple[str, str | None, str, tuple[str, ...]]:
    """**开发期**请求头 → `(user_id, session_id, tenant_id, acl_tags)`。

    这是阶段 1 之前 `orchestration/api.py::_identity` 的实现，签名与返回值一字未改，
    因为 `tests/test_permissions_contract.py` 直接 import 它来守"没有 user_id 就不签发
    session"这条口径。新代码走 `admit()`；它保留下来是为了让那条契约测试继续有意义，
    也为了让"身份从哪来"这件事在两处**只有一份实现**（`admit` 内部同样调它）。
    """
    user_id = (user_header or "").strip()
    tenant = (tenant_header or "").strip() if trust_headers else ""
    acl = _parse_acl(acl_header) if trust_headers else ()
    tenant = tenant or default_tenant
    if not user_id:
        return "", None, tenant, acl
    return user_id, session_id or uuid.uuid4().hex, tenant, acl


def _identity_from_headers(request: Any, settings: Settings) -> Identity:
    user_id, _, tenant, acl = header_identity(
        _header(request, "X-User-Id"),
        None,
        tenant_header=_header(request, "X-Tenant-Id"),
        acl_header=_header(request, "X-ACL-Tags"),
        default_tenant=settings.tenant_id,
        trust_headers=True,
    )
    if not user_id:
        # 匿名**但保留**头里给出的租户与 ACL：白名单环境下这两项是可信输入，
        # 与"有没有登录"是两件事。丢掉它们会让"未登录也按租户隔离"失效——
        # 所有匿名用户一起落到部署默认租户，反而是更大范围的可读。
        return Identity(
            user_id="",
            role=ROLE_ANONYMOUS,
            tenant_id=tenant,
            acl_tags=acl,
            source="anonymous",
        )

    role_header = _header(request, "X-User-Role")
    role = normalize_role(role_header)
    if role is None or role not in ROLES:
        # 空 = 没填 → 最小可授予角色；非空但不认识 → 401。
        # 把"写错的角色"悄悄降成 STUDENT，会让"本该是老师"的人拿着学生权限跑一周
        # 而没有任何人发现——错误必须响亮。
        if role_header:
            raise AdmissionError(
                "X-User-Role 不可识别",
                code="invalid_role",
                http_status=401,
                extra={"allowed": list(ROLES)},
            )
        role = "STUDENT"
    return Identity(user_id=user_id, role=role, tenant_id=tenant, acl_tags=acl, source="header")


def _identity_from_token(token: str, settings: Settings) -> Identity:
    payload = decode_token(token, secret=settings.jwt_secret, leeway_s=settings.jwt_leeway_s)

    sub = payload.get("sub")
    user_id = sub.strip() if isinstance(sub, str) else ""
    if not user_id:
        raise AdmissionError("令牌缺少 sub", code="invalid_token", http_status=401)

    raw_role = payload.get("role")
    role = normalize_role(raw_role if isinstance(raw_role, str) else None)
    if role is None or role not in ROLES:
        raise AdmissionError(
            "令牌里的角色不可识别",
            code="invalid_role",
            http_status=401,
            extra={"allowed": list(ROLES)},
        )

    tenant = payload.get("tenant")
    tenant_id = tenant.strip() if isinstance(tenant, str) and tenant.strip() else settings.tenant_id

    acl_raw = payload.get("acl")
    if acl_raw is None:
        acl: tuple[str, ...] = ()
    elif isinstance(acl_raw, list):
        acl = tuple(dict.fromkeys(str(t).strip() for t in acl_raw if str(t).strip()))[:_ACL_MAX]
    else:
        # **不猜**：把字符串当单标签、把 dict 当键，都会让"权限范围"变成实现细节。
        raise AdmissionError("令牌里的 acl 不是列表", code="invalid_token", http_status=401)

    # `kb_version` **绝不从 token 来**：它决定"在哪一版索引上检索"，
    # 由部署配置独裁，否则客户端能自己挑一个含旧答案的版本绕过更新。
    return Identity(user_id=user_id, role=role, tenant_id=tenant_id, acl_tags=acl, source="jwt")


def _anonymous(settings: Settings) -> Identity:
    return Identity(
        user_id="",
        role=ROLE_ANONYMOUS,
        tenant_id=settings.tenant_id,
        acl_tags=(),
        source="anonymous",
    )


def _parse_acl(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    tags = (part.strip() for part in raw.split(","))
    return tuple(dict.fromkeys(tag for tag in tags if tag))[:_ACL_MAX]


# ---------------------------------------------------------------------------
# JWT（HS256）
# ---------------------------------------------------------------------------


def decode_token(token: str, *, secret: str, leeway_s: int = 60) -> dict[str, Any]:
    """校验 HS256 JWT 并返回 payload。任何一步不过都抛 401，**不返回部分结果**。

    **实现只有一份**，在 `auth/tokens.py`：签发（`auth/service.py::_mint`）与验签
    必须用同一套口径，否则会出现"自己签的令牌自己验不过"这种只在某条路径上炸的
    问题（`alg` 白名单、`compare_digest`、数值排除 `bool` —— 这些细节各写一遍
    迟早分叉）。本函数保留名字与签名，是为了让准入层不依赖 `auth` 的异常词汇：
    `TokenError` 在这里翻成 `AdmissionError`，上层看到的仍然是 401 `invalid_token`。
    """
    try:
        return auth_tokens.decode_token(token, secret=secret, leeway_s=leeway_s)
    except auth_tokens.TokenError as exc:
        # `exc.message` 是给人看的细节（"签名不匹配"/"已过期"），码统一是 invalid_token：
        # 对外不必区分，攻击者也不需要知道是哪一步拦住了他。
        raise AdmissionError(exc.message, code="invalid_token", http_status=401) from None


# ---------------------------------------------------------------------------
# 准入判定
# ---------------------------------------------------------------------------


async def admit(
    payload: ChatRequest,
    request: Any,
    *,
    settings: Settings,
    owner_of: Callable[[str], Awaitable[str | None]] | None = None,
    request_id: str | None = None,
) -> RequestContext:
    """把一次请求判成 `RequestContext` 或一条 4xx。

    **顺序是契约**：先鉴权（401），再按下面的次序逐条判 400。三条理由：

    - 401 先于 400：一个没凭证的调用方不该借参数错误探出"哪些学科合法""多长算超限"。
    - 空 → 超长 → 无实义：从便宜到贵，且前者能解释后者（空串当然不超长）。
    - 学科 → 会话 → 会话归属：参数自身的问题先暴露，才轮到需要查外部状态的归属判断。

    | # | 规则 | 码 |
    |---|---|---|
    | 1 | 去空白后为空 | `empty_query` |
    | 2 | 超过 `CHAT_MAX_QUERY_CHARS` | `query_too_long` |
    | 3 | 一个字母数字都没有（`？？`、`!!!`） | `empty_query` |
    | 4 | `subject` 不在词表内 | `subject_invalid` |
    | 5 | `session_id` 不是 32 位十六进制 | `invalid_session` |
    | 6 | 会话属于别人 | `invalid_session` |
    | 7 | 带了 `session_id` 却没有身份 | `invalid_session` |

    规则 4 **只判"参数写错了"**，不判"参数与问句矛盾"——后者是 `subject_source="conflict"`，
    由 `qa_slots.resolve()` 产出、走全库检索 + FAQ 封顶 mid（阶段 1/2 对齐口径）。
    在这里就判冲突，等于让准入层依赖一个还没有问句上下文的判断。
    """
    rid = request_id or make_request_id(request)
    trace_id = make_trace_id(request, request_id=rid)

    identity = resolve_identity(request, settings)  # 401 优先于任何 400

    query = (payload.query or "").strip()
    if not query:
        raise AdmissionError("问句不能为空", code="empty_query")
    if len(query) > settings.chat_max_query_chars:
        raise AdmissionError(
            f"问句超过 {settings.chat_max_query_chars} 字上限",
            code="query_too_long",
            extra={"max_chars": settings.chat_max_query_chars, "length": len(query)},
        )
    if not any(char.isalnum() for char in query):
        raise AdmissionError("问句里没有可检索的内容", code="empty_query")

    subject = payload.subject
    if subject is not None and qa_slots.from_param(subject).subject_source == "invalid":
        raise AdmissionError(
            "学科不在支持范围内",
            code="subject_invalid",
            extra={"subject": subject, "allowed": list(SUBJECTS)},
        )

    session_id, degraded = await _resolve_session(
        payload.session_id, identity, owner_of, request_id=rid
    )

    # 身份一确定就**立刻**算权限（「先身份后 FAQ」）：结论随 `RequestContext` 一起
    # 交给编排层，FAQ / 检索 / 生成三条路都只看这一个结论，不存在"某条路忘了校验"。
    permission = permissions_for(identity.role, enabled=settings.feature_permission)

    ctx = RequestContext(
        request_id=rid,
        trace_id=trace_id,
        user_id=identity.user_id,
        tenant_id=identity.tenant_id,
        role=identity.role,
        session_id=session_id,
        acl_tags=identity.acl_tags,
        kb_version=settings.kb_version,
        deadline=Deadline.start(settings.total_budget_ms),
        query=query,
        subject=subject.strip() if isinstance(subject, str) and subject.strip() else None,
        selected_faq_id=payload.selected_faq_id,
        require_citation=payload.require_citation,
        permission=permission,
        degraded=degraded,
    )
    log.info(
        "admission_ok",
        request_id=rid,
        user_id=ctx.user_id or "-",
        role=ctx.role,
        identity_source=identity.source,
        session_id=session_id or "-",
        subject=ctx.subject or "-",
        q_len=len(query),
        denied=sorted(permission.denied) if permission else [],
        permission_source=permission.source if permission else "-",
        degraded=list(degraded),
    )
    return ctx


async def _resolve_session(
    raw: str | None,
    identity: Identity,
    owner_of: Callable[[str], Awaitable[str | None]] | None,
    *,
    request_id: str,
) -> tuple[str | None, tuple[str, ...]]:
    """会话 id 的发放与归属校验（规则 5/6/7）。

    规则 7（有 sid 没身份）**排在规则 6 之前**：匿名调用方带着一个 sid 时，
    归属校验必然要查一次外部状态，而结论已经由"没有身份"决定了。两条出口都是
    `400 invalid_session`，所以提前判不泄露任何信息。

    `owner_of` 抛异常时 **fail-open**：会话服务抖动不该让整条问答链挂掉
    （记忆与历史是增强项，不是必需项），但必须留下 `session_owner_unchecked`——
    否则"没校验"与"校验通过"在日志里长得一样。
    """
    sid = (raw or "").strip() or None

    if sid is None:
        if not identity.user_id:
            # 匿名不签发会话：没有归属的会话写进库就再也删不掉（§7.9 边界 3）。
            return None, ()
        return uuid.uuid4().hex, ()

    if not _SESSION_RE.match(sid):
        raise AdmissionError("会话标识格式不正确", code="invalid_session")
    if not identity.user_id:
        raise AdmissionError("会话标识需要登录身份", code="invalid_session")

    if owner_of is None:
        return sid, ()

    try:
        owner = await owner_of(sid)
    except Exception as exc:  # noqa: BLE001 —— 存储层异常类型不由本层定义
        log.warning("session_owner_unchecked", request_id=request_id, error=str(exc))
        return sid, ("session_owner_unchecked",)

    if owner and owner != identity.user_id:
        # 400 而不是 403：403 等于承认"这个会话存在，只是不是你的"，
        # 而"某个学员有一个会话"本身就是隐私（与 §7.9.3 的记忆 404 同一口径）。
        raise AdmissionError("会话不属于当前用户", code="invalid_session")
    return sid, ()
