"""LLM 网关（CLAUDE.md §7.8）：OpenAI 兼容 `/chat/completions` 的最小客户端。

为什么不用第三方 SDK：换供应商（DeepSeek / 通义 / vLLM / Ollama）只改 `LLM_BASE_URL` 与
`LLM_QA_MODEL` 两个环境变量，不动代码，也不会因为 SDK 版本把 `transformers` 之类的依赖搅乱。

三条规则：
1. **生成节点不重试**（§5.3）。真正被允许的只有两层：
   ① **网关级有限重试**——一个字都还没生成前，对**传输层故障**（超时 / 连接中断 /
   HTTP 408 / 429）重发同一次请求，最多 `LLM_MAX_RETRIES` 次（默认 1）；
   ② **换模型**——主模型失败后换 `LLM_FALLBACK_MODEL`，这是降级不是重试。
   两者都不重跑整条检索 + 生成链，也不在已经吐出内容之后再动。4xx（含 401/403）与 5xx
   不重试：前者重试一万次还是同样的结果，后者直接换模型更划算（§11.5）。
2. **超时 = min(LLM_TIMEOUT_S, 剩余预算)**。剩余预算不足直接抛 `BudgetExceeded`，不发起请求；
   重试前也要再看一眼剩余预算，不够就带着原始错误上去，不制造第二个错误类型。
3. **不打印 key、不把上游原始响应整段塞进异常**：错误信息只留状态码与截断后的片段。

异常映射：未配置 → `LLMNotConfigured`；连接/5xx/超时/响应不可解析 → `UpstreamUnavailable`；
预算不足 → `BudgetExceeded`。401/403 同样抛 `LLMNotConfigured`（**配置问题**，不是网络问题），
消息里保留"鉴权"二字并把排查方向写清楚——它决定"要不要换模型"，把两者混成一句
会出现"key 错了却换个模型再撞一次"的白跑。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, TypedDict

import httpx

from base.config import Settings
from base.errors import BudgetExceeded, LLMNotConfigured, UpstreamUnavailable
from base.logger import get_logger

log = get_logger(__name__)

# 重试前至少还要留这么多时间，否则"再试一次"必然超时，白白多一次计费请求
_MIN_RETRY_BUDGET_S = 0.5


class Msg(TypedDict):
    role: str  # system / user / assistant
    content: str


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    degraded: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    """编排层只依赖这个协议，测试里可以换成任何实现。"""

    @property
    def enabled(self) -> bool: ...

    async def complete(
        self, messages: list[Msg], *, budget_ms: int, model: str | None = None
    ) -> LLMResult: ...

    def stream(
        self, messages: list[Msg], *, budget_ms: int, model: str | None = None
    ) -> AsyncIterator[str]: ...


@dataclass(frozen=True, slots=True)
class LLMTarget:
    """一个生成端点：打到哪儿、用什么身份、哪个模型、请求体里额外带什么。

    为什么要有这一层：网关原先所有连接参数都直接读 `Settings`（`llm_endpoint` /
    `_headers()` / `llm_timeout_s` / `llm_max_tokens` / `enabled = bool(llm_api_key)`），
    于是"换一个模型"只能改 `model` 字段——端点、key、可用性仍然绑在云端那一份上。
    本地 Ollama 没有 key，`enabled` 若还取 `bool(llm_api_key)` 就永远是 False，
    本地路径一辈子走不到。把"端点的事实"收成一个结构后，云侧与本地侧各建一个，
    网关本身不再区分来源，也不需要在业务代码里写 `if 本地 ...`。

    `extra_body` 是**请求体附加字段**（不覆盖 `model/messages/temperature/max_tokens/stream`
    四个必填键）。当前只有一处用它：本地 qwen3.5 是思考型模型，不带
    `reasoning_effort: "none"` 就一个字的 `content` 都拿不到（见 `base/config.py` 同名注释）。
    """

    name: str
    endpoint: str
    model: str
    api_key: str = ""
    # 本地 Ollama 不需要 key：置 False 时不校验 key，也不发 Authorization 头
    requires_api_key: bool = True
    fallback_model: str = ""
    timeout_s: float = 60.0
    max_tokens: int = 1024
    temperature: float = 0.2
    extra_body: dict[str, Any] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        if not self.endpoint.strip() or not self.model.strip():
            return False
        return bool(self.api_key.strip()) or not self.requires_api_key

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key.strip():
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def payload(
        self, messages: list[Msg], *, model: str, stream: bool
    ) -> dict[str, Any]:
        """请求体。附加字段在前、必填键在后——**附加字段不许覆盖必填键**。"""
        return {
            **self.extra_body,
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMTarget:
        """主网关（云端）：与接入前的行为逐字段一致。"""
        return cls(
            name="cloud",
            endpoint=settings.llm_endpoint,
            model=settings.llm_qa_model,
            api_key=settings.llm_api_key,
            requires_api_key=True,
            fallback_model=settings.llm_fallback_model,
            timeout_s=float(settings.llm_timeout_s),
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

    @classmethod
    def local_from_settings(cls, settings: Settings) -> LLMTarget:
        """本地小模型（闲聊档）：不要求 key，不换模型（本地只有一个）。"""
        effort = settings.llm_local_reasoning_effort.strip()
        return cls(
            name="local",
            endpoint=settings.llm_local_endpoint,
            model=settings.llm_local_model,
            api_key=settings.llm_local_api_key,
            requires_api_key=False,
            fallback_model="",
            timeout_s=float(settings.llm_local_timeout_s),
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            extra_body={"reasoning_effort": effort} if effort else {},
        )


class LLMGateway:
    def __init__(
        self,
        settings: Settings,
        *,
        target: LLMTarget | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        # 不传 = 主网关（云端），保持既有构造点不用改
        self._target = target or LLMTarget.from_settings(settings)
        self._client = client
        self._owns_client = client is None

    @property
    def enabled(self) -> bool:
        return self._target.enabled

    @property
    def model(self) -> str:
        return self._target.model

    @property
    def target_name(self) -> str:
        """`cloud` / `local`：进日志用，回答"这一轮到底是谁生成的"。"""
        return self._target.name

    # ---- 对外 ----
    async def complete(
        self,
        messages: list[Msg],
        *,
        budget_ms: int,
        model: str | None = None,
    ) -> LLMResult:
        if not self.enabled:
            raise LLMNotConfigured(
                f"生成模型未配置（{self._target.name}），无法生成"
            )
        deadline = time.perf_counter() + self._budget_s(budget_ms)
        primary = model or self._target.model
        try:
            return await self._complete_with_retry(messages, model=primary, deadline=deadline)
        except UpstreamUnavailable as exc:
            fallback = self._target.fallback_model.strip()
            # key 错了换模型没有意义：两个模型打的是同一个网关、同一个 key，
            # 换过去只是把同一个 401 再撞一次并多花一次预算（§11.5）。
            if isinstance(exc, LLMNotConfigured) or not fallback or fallback == primary:
                raise
            # 只有"一个字都没生成"才换模型：已经吐了一半再换，用户会看到两段拼接的回答
            log.warning("llm_model_switch", primary=primary, fallback=fallback, error=str(exc))
            result = await self._complete_with_retry(messages, model=fallback, deadline=deadline)
            result.degraded.append("llm_fallback_model")
            return result

    async def stream(
        self,
        messages: list[Msg],
        *,
        budget_ms: int,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """逐段产出增量文本。流式不换模型：客户端已经看到首字后再换模型无法拼接。"""
        if not self.enabled:
            raise LLMNotConfigured(
                f"生成模型未配置（{self._target.name}），无法生成"
            )
        timeout_s = self._budget_s(budget_ms)
        target = model or self._target.model
        payload = self._target.payload(messages, model=target, stream=True)
        client = self._get_client()
        try:
            async with client.stream(
                "POST",
                self._target.endpoint,
                json=payload,
                headers=self._target.headers(),
                timeout=timeout_s,
            ) as resp:
                if resp.status_code >= 400:
                    # 只有失败时才读 body：流式响应提前 aread() 会把整个回答缓冲住，等于没流式
                    self._raise_for_status(resp.status_code, await resp.aread())
                async for line in resp.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk:
                        yield chunk
        except httpx.TimeoutException as exc:
            raise UpstreamUnavailable(f"LLM 流式超时（{timeout_s:.1f}s）", node="generate") from exc
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"LLM 流式连接失败：{exc}", node="generate") from exc

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    # ---- 内部 ----
    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._target.timeout_s)
        return self._client

    def _budget_s(self, budget_ms: int) -> float:
        """取小，并且不发起"预算已经不够"的请求。"""
        if budget_ms <= 0:
            raise BudgetExceeded("剩余预算已耗尽，无法调用生成模型", node="generate")
        return min(float(self._target.timeout_s), budget_ms / 1000.0)

    def _headers(self) -> dict[str, str]:
        return self._target.headers()

    async def _complete_with_retry(
        self, messages: list[Msg], *, model: str, deadline: float
    ) -> LLMResult:
        """同一次请求的有限重试——**只重试传输层故障**。

        重试的判据是异常自己带的 `retryable`，不是"捕获到任何错就再来一次"：
        401/403 与其它 4xx、5xx 都是不可重试的（见 `_raise_for_status`），
        盲目重试只会把一次鉴权失败放大成三次计费请求。
        """
        attempts = 1 + max(0, int(self._settings.llm_max_retries))
        for attempt in range(1, attempts + 1):
            try:
                return await self._complete_once(messages, model=model, deadline=deadline)
            except UpstreamUnavailable as exc:
                if attempt >= attempts or not exc.retryable:
                    raise
                if deadline - time.perf_counter() <= _MIN_RETRY_BUDGET_S:
                    # 预算不够就别试了：把**原始错误**往上抛，别制造一个此处特有的
                    # `BudgetExceeded`——上层只想知道"生成不可用"，不想分辨是超时还是没钱
                    raise
                log.warning(
                    "llm_retry", model=model, attempt=attempt, max_attempts=attempts,
                    error=str(exc),
                )
        raise AssertionError("unreachable")  # pragma: no cover

    async def _complete_once(
        self, messages: list[Msg], *, model: str, deadline: float
    ) -> LLMResult:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.2:
            raise BudgetExceeded("剩余预算不足，未发起生成请求", node="generate")

        payload = self._target.payload(messages, model=model, stream=False)
        started = time.perf_counter()
        try:
            resp = await self._get_client().post(
                self._target.endpoint,
                json=payload,
                headers=self._target.headers(),
                timeout=remaining,
            )
        except httpx.TimeoutException as exc:
            raise UpstreamUnavailable(f"LLM 超时（{remaining:.1f}s）", node="generate") from exc
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"LLM 连接失败：{exc}", node="generate") from exc

        self._raise_for_status(resp.status_code, resp.content)
        data = _load_json(resp)
        text = _extract_text(data)
        usage = data.get("usage") or {}
        return LLMResult(
            text=text,
            model=model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _redact(self, text: str) -> str:
        """把 key 从上游回显里抹掉。

        为什么需要这一步：错误体是**上游写的**，不是我们写的。OpenAI 兼容网关在
        key 无效时的标准回复就是 `Incorrect API key provided: sk-xxx`——把这句话
        原样塞进异常，再由 `log.warning("llm_retry", error=str(exc))` 落到文件日志，
        一条本该只进内存的密钥就永久留在了 `logs/app.log`（§5.2 双出口 = 两份落盘）。
        改了 key 也没用：旧 key 已经躺在日志里了。
        """
        key = (self._target.api_key or "").strip()
        return text.replace(key, "***") if key and key in text else text

    def _raise_for_status(self, status: int, body: bytes) -> None:
        """HTTP 状态码 → 异常，并显式标注**能不能重试**。

        这张表决定了"哪个旋钮会被转动"，不要随手改：

        | 状态 | 异常 | 可重试 | 理由 |
        |---|---|---|---|
        | 401 / 403 | `LLMNotConfigured` | 否 | key 的问题，换模型 / 重试都是同一个结果 |
        | 408 / 429 | `UpstreamUnavailable` | 是 | 明确"稍后再来"的语义 |
        | 其它 4xx | `UpstreamUnavailable` | 否 | 参数错了，重发一万次也一样 |
        | 5xx | `UpstreamUnavailable` | 否 | 交回 `complete()` 换模型，比原地重试划算 |

        5xx 之所以不标可重试：换模型是**换供应商**级别的一次机会，原地重试只是把同一个
        故障点再撞一次。两者都要花预算，留更可能有效的那一个。
        """
        if status < 400:
            return
        snippet = self._redact(body.decode("utf-8", "replace")[:200])
        if status in (401, 403):
            # 不要回显 key；只提示"检查 LLM_API_KEY"，因为最常见的原因是环境变量里的旧 key 覆盖了 .env
            raise LLMNotConfigured(
                f"LLM 鉴权失败 HTTP {status}：检查 LLM_API_KEY（注意 IDE/系统环境变量可能覆盖 .env）",
                node="generate",
                retryable=False,
            )
        if status in (408, 429):
            raise UpstreamUnavailable(
                f"LLM HTTP {status}（限流/超时，可重试）：{snippet}", node="generate", retryable=True
            )
        raise UpstreamUnavailable(f"LLM HTTP {status}：{snippet}", node="generate", retryable=False)


def _load_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError) as exc:
        raise UpstreamUnavailable(
            f"LLM 响应不是 JSON：{resp.text[:200]}", node="generate"
        ) from exc
    if not isinstance(data, dict):
        raise UpstreamUnavailable("LLM 响应结构异常：顶层不是对象", node="generate")
    return data


def _extract_text(data: dict[str, Any]) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise UpstreamUnavailable(f"LLM 响应缺少 choices[0].message.content：{str(data)[:200]}",
                                  node="generate") from exc
    if not isinstance(content, str) or not content.strip():
        raise UpstreamUnavailable("LLM 返回空答案", node="generate")
    return content.strip()


def _parse_sse_line(line: str) -> str | None:
    """解析 OpenAI 兼容的流式行。非 data 行与 [DONE] 都返回 None。"""
    line = line.strip()
    if not line.startswith("data:"):
        return None
    body = line[5:].strip()
    if not body or body == "[DONE]":
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    choices = data.get("choices") or []
    if not choices:
        return None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    return content if isinstance(content, str) and content else None
