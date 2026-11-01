"""业务异常体系（CLAUDE.md §5.2）。

只分这几类，够了：调用方要判断的是"还能不能继续/该回什么话术"，不是错误来源的细节。
细节写进 `str(exc)`；`trace_id` 与节点名由抛出方补（`node=`），不要在异常里猜。

禁止裸 `except Exception: pass`——要么在这里挑一个类型抛出去，要么在调用侧记日志后降级。
"""

from __future__ import annotations


class EduRAGError(Exception):
    """所有业务异常的基类。`code` 进日志与响应体，`http_status` 只给 API 层参考。

    `retryable` 描述的是"**同一个请求再发一次有没有可能成功**"，与 `code` 正交：
    参数错误、鉴权失败、权限拒绝重试一万次也是同样的结果（实施手册 §11.5），
    网络抖动、连接中断、429 限流才值得再试一次。默认 False——**新增异常必须显式声明可重试**，
    免得一个没人想过的错误类型被重试循环悄悄放大成 N 次上游调用。
    """

    code = "internal_error"
    http_status = 500
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        node: str | None = None,
        trace_id: str | None = None,
        retryable: bool | None = None,
        code: str | None = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.node = node
        self.trace_id = trace_id
        # 类属性给默认值，实例属性给"这一次的特例"（例：同一个 `UpstreamUnavailable`，
        # HTTP 429 可重试、HTTP 401 不可重试）
        if retryable is not None:
            self.retryable = retryable
        # 同一个异常类型承载多个对外错误码（准入层 7 条规则各有一个码），
        # 与其为每个码写一个空子类，不如让抛出点把码传进来——响应体只需要码，不需要类型。
        # 不传则沿用类属性，老调用方零改动。
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status

    def as_dict(self) -> dict[str, str | None]:
        return {"error": self.code, "detail": self.message, "node": self.node, "request_id": self.trace_id}


class BudgetExceeded(EduRAGError):
    """剩余预算不够跑完当前节点。降级而不是等待——超时是产品行为，不是故障。"""

    code = "budget_exceeded"
    http_status = 504


class UpstreamUnavailable(EduRAGError):
    """外部依赖（LLM / Milvus / Redis）不可用或返回错误。

    默认 **可重试**：这类异常最常见的来源是超时与连接中断，同一个请求再发一次常常就好了。
    但"可重试"不等于"一定会重试"——调用方还要看剩余预算，且只对**幂等**调用生效
    （检索可重试、生成节点不重试，§5.3）。
    """

    code = "upstream_unavailable"
    http_status = 503
    retryable = True


class LLMNotConfigured(UpstreamUnavailable):
    """没配 LLM_API_KEY。它属于"上游不可用"，但单独一类便于把配置问题与网络问题分开排。

    不可重试：key 不对的时候再发 N 次还是 401，只会把用户预算烧在没有结果的重试上。
    """

    code = "llm_not_configured"
    retryable = False


class EmptyRetrieval(EduRAGError):
    """检索结果为空。通常不该抛（空结果不是异常），留给需要显式分支的调用方。"""

    code = "empty_retrieval"
    http_status = 404


class PolicyViolation(EduRAGError):
    """越界/敏感请求，或调用方参数违反契约（如非法学科）。"""

    code = "policy_violation"
    http_status = 400


class InternalError(EduRAGError):
    """兜底：未预期异常在节点边界统一包装，保证响应体里一定有 request_id。"""

    code = "internal_error"
    http_status = 500


# ---- 对外错误码（CLAUDE.md §7.2.7）：七类固定取值，进日志与响应体 -----------------
# 关键是 `retrieval_unavailable` 与 `no_evidence` 必须分开：
# 前者是"检索系统坏了"（修服务），后者是"知识库里确实没有"（补语料）。
# 合成一句话，运营就永远不知道该找运维还是找教研。


class InvalidRequest(EduRAGError):
    """参数不合法（空问句、越界学科、超长）。"""

    code = "invalid_request"
    http_status = 400


class Forbidden(EduRAGError):
    """越权：跨租户、ACL 不含该文档、或调用了未授权接口。"""

    code = "forbidden"
    http_status = 403


class Unauthorized(EduRAGError):
    """**没有可用身份**：token 无效/过期、token 里的角色不认识。

    与 `Forbidden` 的分野是"你是谁知道吗"（401，换凭证再来）与
    "知道你是谁，但这件事你不能做"（403）。混成 403 的后果是前端不会跳登录，
    用户只会看到一个"权限不足"并反复重试同一个已经过期的 token。

    **绝不降级为匿名**：带了 token 却验不过，说明调用方以为自己是某个身份——
    此时按匿名放行，等于把"凭证坏了"变成"看到一份别人的答案"（阶段 1 §3.3 三层身份）。
    """

    code = "unauthorized"
    http_status = 401


class RetrievalUnavailable(UpstreamUnavailable):
    """检索链路不可用（Milvus 未就绪 / 连接失败 / 超时）。**不是**"没有证据"。"""

    code = "retrieval_unavailable"


class NoEvidence(EduRAGError):
    """检索成功但证据不足以作答。产品分支，API 层按 200 + 兜底话术处理。"""

    code = "no_evidence"
    http_status = 200


class GenerationUnavailable(UpstreamUnavailable):
    """LLM 未配置或调用失败，无法生成。"""

    code = "generation_unavailable"


class CitationInvalid(EduRAGError):
    """引用无法对齐（越界 / 全部无支撑 / 整体无引用）。产品分支，按兜底处理。"""

    code = "citation_invalid"
    http_status = 200


class OutOfScope(EduRAGError):
    """问题不在教育领域内，拒答。**产品分支，200 而不是 400**（§7.8.5 同一条口径）。

    "用户问了天气"不是参数错误：请求合法、系统正常、我们主动不答。
    它也不能记成 `no_evidence`——那会把一句 `明天成都天气` 塞进"补语料"队列，
    教研据此去建语料，而正确答案是"这个问题我们本来就不答"。三类出口必须分开：

    | 出口 | `error_code` | 该谁处理 | HTTP |
    |---|---|---|---|
    | 知识库没有 | `no_evidence` | 补语料 | 200 |
    | 越界拒答 | `out_of_scope` | 产品口径（谁定的边界谁改） | 200 |
    | 系统坏了 | `retrieval_unavailable` 等 | 修服务 | 5xx |

    它**不**取代 `PolicyViolation`：后者是"参数非法/敏感内容"（400，客户端该改请求）。
    """

    code = "out_of_scope"
    http_status = 200
