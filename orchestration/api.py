"""`POST /v1/chat` 与 `GET /v1/chat/stream`（CLAUDE.md §7.8 第 3、5 项 / 阶段 1）。

协议约定：
- **一律 200**：兜底话术与追问是正常产品行为，不是错误；参数错误 400、鉴权失败 401、
  体积超限 413、未预期异常 500；
- **所有分支同一个响应体**：`ChatState.to_response()` 是唯一出口，`/v1/chat` 返回它；
  连未预期异常的 500 也返回同一形状（`status="error"` + `error_code="internal_error"`），
  这样前端只写一份渲染逻辑，不必为"异常"单开一条解析路径；
- 流式为 SSE，四类帧 `meta / delta / citations / done`，**`done` 是权威结果**：
  `done.source != "rag"` 时客户端应丢弃此前 delta，改显示 `done.answer`；
- `X-Request-Id` 由中间件回传，异常时响应体里也带 `request_id`——否则用户报障时
  我们只有一个"500"，没有任何能对上日志的线索。

**本模块只做"搬运"，不做判定**（阶段 1 §1）：身份、长度、学科、会话归属全部在
`orchestration/admission.py::admit()` 里判，两条入口调用的是同一个函数。这里剩下的
只有三件"框架层面"的事——读请求体、把异常翻成 HTTP、把 `RequestContext` 交给编排层。

三处刻意的选择：

1. **体积/格式两关放在路由里而不是 `main.py` 中间件里**：验收用的是裸 `FastAPI()` +
   `chat_router`，中间件不存在。判定写在中间件里等于"生产上有、测试里没有"，
   而这类差异永远只在验收时才被发现。`main.py` 再挂一层是**兜底**，不是唯一防线。
2. **400 的响应体是"完整 `ChatResponse` + `detail`"**：设计要求统一形状，
   既有前端与测试按 `detail.error` 判分支。两者不冲突——外层给形状，内层给依据。
3. **空问句是 400 而不是 422**：422 是 pydantic 的"字段不合法"，而"空问句"是产品口径的
   参数错误，和超长、学科拼错应该是同一类响应；混进 422 会让前端多一条分支。
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from base.config import Settings, is_trusted_env
from base.contracts import ChatResponse
from base.log_messages import degraded_reasons
from base.logger import get_logger
from orchestration.admission import (
    AdmissionError,
    ChatRequest,
    admit,
    header_identity,
    make_request_id,
)
from orchestration.chat import ChatPipeline

router = APIRouter(prefix="/v1", tags=["chat"])
log = get_logger(__name__)

# `tests/test_permissions_contract.py` 直接 import 这两个名字守"没有 user_id 就不签发
# session""生产不信请求头"两条口径。保留别名而不是把实现留在这里：判定只有一份
# （`admit()` 内部同样调 `header_identity`），测试守的才是线上真正跑的那份代码。
_identity = header_identity
_trust_headers = is_trusted_env


def _pipeline(request: Request) -> ChatPipeline:
    pipeline = getattr(request.app.state, "chat", None)
    if pipeline is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="chat not ready")
    return pipeline


# ---------------------------------------------------------------------------
# 拒绝：形状统一，HTTP 码由异常自己带
# ---------------------------------------------------------------------------


def _rejected(exc: AdmissionError, request_id: str) -> JSONResponse:
    """把一条准入拒绝翻成 HTTP 响应：**完整 `ChatResponse` 形状** + `detail`。

    `degraded` 里放错误码本身，`degraded_reasons` 走同一张文案表——
    `base/log_messages.degraded_reason()` 对没登记过的码**原样返回**，
    所以这里既不用为每个新错误码补表，也不会编造一个看起来像解释的说法。
    """
    body = ChatResponse(
        request_id=request_id,
        status="error",
        source="fallback",
        error_code=exc.code,
        degraded=[exc.code],
        degraded_reasons=degraded_reasons([exc.code]),
    ).as_dict()
    body["detail"] = exc.detail()
    return JSONResponse(status_code=exc.http_status, content=body)


def _internal_error(request_id: str, pipeline: ChatPipeline) -> JSONResponse:
    """未捕获异常的响应：**结构与成功时一模一样**，只是 `status="error"`。

    `request_id` 一定有值（没有中间件时自己生成一个），因为它此刻是唯一能把
    用户截图和日志对上的线索。HTTP 码用 500 而不是 200：这不是"知识库没有"
    那种产品行为，是代码出了我们没预料到的岔子，监控要能直接按 5xx 报警。
    """
    rid = request_id or uuid.uuid4().hex
    body = ChatResponse(
        request_id=rid,
        status="error",
        source="fallback",
        answer=pipeline.system_error(),
        error_code="internal_error",
        degraded=["internal_error"],
        degraded_reasons=degraded_reasons(["internal_error"]),
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=body.as_dict())


# ---------------------------------------------------------------------------
# 请求体：先看体积，再看 JSON，再看字段
# ---------------------------------------------------------------------------


async def _read_payload(request: Request, settings: Settings) -> ChatRequest:
    """读并校验请求体。三道关的顺序是"便宜 → 贵"，且**先拒的不能花后者的钱**。

    | 顺序 | 判据 | 码 / 状态 |
    |---|---|---|
    | 1 | `Content-Length` 或实际长度 > `MAX_REQUEST_BYTES` | 413 `request_too_large` |
    | 2 | 不是合法 JSON / 不是对象 | 400 `invalid_request` |
    | 3 | 字段不合法（缺 `query`、多出未知字段） | 400 `invalid_request` |

    第 1 关不能只信 `Content-Length`：它是客户端说的，`Transfer-Encoding: chunked`
    时压根没有。所以两个都比一遍，`max()` 取实际发生的那份。
    """
    limit = int(settings.max_request_bytes)
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit:
                raise AdmissionError(
                    f"请求体超过 {limit} 字节上限",
                    code="request_too_large",
                    http_status=413,
                    extra={"max_bytes": limit},
                )
        except ValueError:
            # 解不开的 `Content-Length` 交给下面按实际长度判，不在这里拒
            pass

    raw = await request.body()
    if len(raw) > limit:
        raise AdmissionError(
            f"请求体超过 {limit} 字节上限",
            code="request_too_large",
            http_status=413,
            extra={"max_bytes": limit},
        )

    try:
        data: Any = json.loads(raw or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AdmissionError(f"请求体不是合法 JSON：{exc}", code="invalid_request") from None
    if not isinstance(data, dict):
        raise AdmissionError("请求体必须是一个 JSON 对象", code="invalid_request")

    try:
        return ChatRequest.model_validate(data)
    except ValidationError as exc:
        # 摘要而不回显原文：请求体里可能有学生姓名/手机号，错误信息会进日志。
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ())) or "-"
        raise AdmissionError(
            f"请求体字段不合法（{field}）",
            code="invalid_request",
            extra={"field": field},
        ) from None


# ---------------------------------------------------------------------------
# POST /v1/chat
# ---------------------------------------------------------------------------


_CHAT_BODY_SCHEMA = {
    "required": True,
    "content": {
        "application/json": {
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["query"],
                # 整个请求体的样例：Swagger UI 有它就以它为准，没有就**按类型现造**
                # （string → "string"、integer → 0）。现造出来的 `"session_id": "string"`
                # 不是 32 位十六进制，`admit()` 规则 5 直接回 400 invalid_session——
                # 于是"点开文档、直接 Execute"必然失败，看起来像接口坏了。
                # 这里只留两个字段：可选字段一律不示范，带了反而更容易踩坑
                # （有 session_id 但没身份也是 400，见规则 7）。
                "example": {"query": "二次函数的顶点式怎么求", "subject": "数学"},
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户原始问题（空 → 400 empty_query；上限 CHAT_MAX_QUERY_CHARS，默认 400）",
                        "example": "二次函数的顶点式怎么求",
                    },
                    "subject": {
                        "type": ["string", "null"],
                        "description": "学科；不在词表内 → 400 subject_invalid",
                        "example": "数学",
                    },
                    "selected_faq_id": {
                        "type": ["integer", "null"],
                        "description": "FAQ 确认通道：mid 档追问被采纳后带上该 faq_id",
                    },
                    "session_id": {
                        "type": ["string", "null"],
                        "description": "会话 id（32 位十六进制）；配合 X-User-Id 使用",
                    },
                    "require_citation": {
                        "type": ["boolean", "null"],
                        "description": "是否强制引用；不传按意图档位",
                    },
                    "role": {
                        "type": ["string", "null"],
                        "description": "**永不读取**：身份只来自 JWT 或网关注入的上下文",
                    },
                },
            }
        }
    },
}


@router.post("/chat", openapi_extra={"requestBody": _CHAT_BODY_SCHEMA})
async def chat(request: Request) -> Any:
    """非流式问答。**注意签名里没有请求体参数**：模型校验必须先看体积，
    而体积只能在拿到原始字节之后才知道——让 FastAPI 先解析等于跳过了 413 那一关。

    代价是 FastAPI 不会自动生成 `requestBody`，Swagger UI 因此只显示 "No parameters"、
    没有输入框。所以这里用 `openapi_extra` **手写一份 schema**：文档里能填能试，
    而运行期仍由 `_read_payload()` 自己读原始字节（声明归声明，绑定归绑定）。
    ⚠ 改了 `ChatRequest` 的字段必须同步改上面那份 schema，否则文档与真相会各说一套。
    """
    pipeline = _pipeline(request)  # 503 优先于一切：服务没就绪时不必再判参数
    rid = make_request_id(request)

    try:
        payload = await _read_payload(request, pipeline.settings)
        ctx = await admit(
            payload,
            request,
            settings=pipeline.settings,
            owner_of=pipeline.session_owner,
            request_id=rid,
        )
        state = await pipeline.ask(
            ctx.query,
            subject=ctx.subject,
            selected_faq_id=ctx.selected_faq_id,
            request_id=ctx.request_id,
            require_citation=ctx.require_citation,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            tenant_id=ctx.tenant_id,
            acl_tags=ctx.acl_tags,
            role=ctx.role,
            trace_id=ctx.trace_id,
            kb_version=ctx.kb_version,
            deadline=ctx.deadline,
            # 业务权限（阶段 3）：准入层在 `admit()` 里算好的结论，**原样**交下去。
            # 编排层不再重算（`permissions_for()` 是纯函数，但没有第二个地方该决定
            # "这个角色能不能用这项功能"）；只在"没传结论"时才由 `_permission_for()` 补算
            permission=ctx.permission,
        )
    except AdmissionError as exc:
        log.warning("admission_rejected", request_id=rid, rule=exc.code, reason=str(exc))
        return _rejected(exc, rid)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        # 编排层已经把已知故障都收成了兜底话术，走到这里的是**没预料到**的异常
        # （空指针、类型错误、第三方库行为变化）。宁可回一个结构一致的 500，
        # 也不要让 FastAPI 抛一段 HTML/纯文本错误页——前端解析不了，用户看到的是乱码。
        log.exception("unhandled_error", error=str(exc), request_id=rid, stage="chat")
        return _internal_error(rid, pipeline)
    return state.to_response().as_dict()


# ---------------------------------------------------------------------------
# GET /v1/chat/stream
# ---------------------------------------------------------------------------


@router.get("/chat/stream")
async def chat_stream(
    request: Request,
    query: Annotated[str, Query(description="用户原始问题")] = "",
    subject: Annotated[str | None, Query(description="限定学科；不传则全库")] = None,
    selected_faq_id: Annotated[int | None, Query(description="mid 追问被采纳后的确认通道")] = None,
    require_citation: Annotated[
        bool | None, Query(description="是否要求答案必须有可验证引用；不传则按意图档位")
    ] = None,
    session_id: Annotated[str | None, Query(description="会话 id；不传且带了 X-User-Id 时自动生成")] = None,
) -> Any:
    """SSE 流式回答。`GET` 是因为 SSE 只能单向推送，用查询串传参即可（§7.8）。

    查询串天然没有"未知字段""体积"这两类问题，但它必须走**同一个 `admit()`**：
    长度上限、学科词表、会话归属在两条入口上给出不同结论，是最难查的那种不一致。
    空问句因此是 **400 `empty_query`**（曾经是 422），与 POST 一字不差。
    """
    pipeline = _pipeline(request)
    rid = make_request_id(request)

    # `model_construct` 跳过 pydantic 校验：查询串类型由 FastAPI 保证，
    # 而"空问句""超长"是准入层的口径，不能在这里被另一套规则抢先判掉。
    payload = ChatRequest.model_construct(
        query=query or "",
        subject=subject,
        selected_faq_id=selected_faq_id,
        session_id=session_id,
        require_citation=require_citation,
        role=None,
    )
    try:
        ctx = await admit(
            payload,
            request,
            settings=pipeline.settings,
            owner_of=pipeline.session_owner,
            request_id=rid,
        )
    except AdmissionError as exc:
        log.warning("admission_rejected", request_id=rid, rule=exc.code, reason=str(exc))
        return _rejected(exc, rid)

    async def frames() -> AsyncIterator[bytes]:
        async for frame in pipeline.ask_stream(
            ctx.query,
            subject=ctx.subject,
            selected_faq_id=ctx.selected_faq_id,
            request_id=ctx.request_id,
            require_citation=ctx.require_citation,
            user_id=ctx.user_id,
            session_id=ctx.session_id,
            tenant_id=ctx.tenant_id,
            acl_tags=ctx.acl_tags,
            role=ctx.role,
            trace_id=ctx.trace_id,
            kb_version=ctx.kb_version,
            deadline=ctx.deadline,
            permission=ctx.permission,
        ):
            yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode()

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # 反向代理（nginx）默认会把 SSE 缓冲成"一次性返回"，等于没流式
            "X-Accel-Buffering": "no",
        },
    )
