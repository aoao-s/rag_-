"""LLM 网关单测：只测协议与异常映射，不打真实网络。"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from base.config import Settings
from base.errors import BudgetExceeded, LLMNotConfigured, UpstreamUnavailable
from rag_system.core.llm_gateway import LLMGateway

MESSAGES = [{"role": "user", "content": "顶点式怎么求"}]


def _settings(**overrides) -> Settings:
    base = {
        "llm_api_key": "sk-test",
        "llm_base_url": "https://llm.test/v1",
        "llm_qa_model": "primary",
        "llm_fallback_model": "",
        "llm_timeout_s": 5,
        "llm_max_tokens": 128,
        "llm_temperature": 0.0,
    }
    base.update(overrides)
    return Settings(**base)


def _gateway(handler, **overrides) -> tuple[LLMGateway, httpx.AsyncClient]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return LLMGateway(_settings(**overrides), client=client), client


def _ok(content: str = "顶点式可以通过配方法得到[1]。") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        },
    )


def test_complete_parses_text_and_usage() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return _ok()

    async def run() -> None:
        gateway, client = _gateway(handler)
        result = await gateway.complete(MESSAGES, budget_ms=2000)
        assert result.text == "顶点式可以通过配方法得到[1]。"
        assert result.model == "primary"
        assert (result.prompt_tokens, result.completion_tokens) == (120, 30)
        assert seen["stream"] is False
        await client.aclose()

    asyncio.run(run())


def test_not_configured_raises() -> None:
    gateway = LLMGateway(_settings(llm_api_key=""))

    async def run() -> None:
        with pytest.raises(LLMNotConfigured):
            await gateway.complete(MESSAGES, budget_ms=1000)

    asyncio.run(run())


def test_auth_error_is_upstream_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid api key"})

    async def run() -> None:
        gateway, client = _gateway(handler)
        with pytest.raises(UpstreamUnavailable) as exc:
            await gateway.complete(MESSAGES, budget_ms=1000)
        assert "鉴权" in str(exc.value)
        await client.aclose()

    asyncio.run(run())


def test_timeout_maps_to_upstream_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("boom", request=request)

    async def run() -> None:
        gateway, client = _gateway(handler)
        with pytest.raises(UpstreamUnavailable):
            await gateway.complete(MESSAGES, budget_ms=1000)
        await client.aclose()

    asyncio.run(run())


def test_switch_to_fallback_model_once() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload["model"])
        if payload["model"] == "primary":
            return httpx.Response(503, text="upstream busy")
        return _ok("兜底模型的答案[1]。")

    async def run() -> None:
        gateway, client = _gateway(handler, llm_fallback_model="backup")
        result = await gateway.complete(MESSAGES, budget_ms=2000)
        assert calls == ["primary", "backup"]
        assert result.model == "backup"
        assert "llm_fallback_model" in result.degraded
        await client.aclose()

    asyncio.run(run())


def test_no_fallback_configured_reraises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async def run() -> None:
        gateway, client = _gateway(handler)
        with pytest.raises(UpstreamUnavailable):
            await gateway.complete(MESSAGES, budget_ms=1000)
        await client.aclose()

    asyncio.run(run())


def test_zero_budget_raises_before_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - 不应被调用
        raise AssertionError("预算为 0 时不应发起请求")

    async def run() -> None:
        gateway, client = _gateway(handler)
        with pytest.raises(BudgetExceeded):
            await gateway.complete(MESSAGES, budget_ms=0)
        await client.aclose()

    asyncio.run(run())


def test_empty_answer_is_upstream_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _ok("   ")

    async def run() -> None:
        gateway, client = _gateway(handler)
        with pytest.raises(UpstreamUnavailable):
            await gateway.complete(MESSAGES, budget_ms=1000)
        await client.aclose()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# 有限重试（§11.5）：只重试**传输层故障**，且只重试"一个字都还没生成"的那次请求
# ---------------------------------------------------------------------------
def test_retryable_transport_error_retries_once() -> None:
    """超时/连接中断是可重试的：换一次同样的请求，很可能就通了。"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow upstream", request=request)
        return _ok()

    async def run() -> None:
        gateway, client = _gateway(handler)  # llm_max_retries 默认 1
        result = await gateway.complete(MESSAGES, budget_ms=2000)
        assert len(calls) == 2
        assert result.text == "顶点式可以通过配方法得到[1]。"
        # 重试次数是**运维信号**（日志里的 llm_retry），不是用户可感知的降级：
        # 最后拿到的就是正常答案，不该把 status 推到 degraded
        assert result.degraded == []
        await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        (400, "参数错了，重发一万次也一样"),
        (401, "key 错了，重试只是把同一个 401 再撞一次"),
        (503, "5xx 交回 complete() 换模型，比原地重试划算"),
    ],
)
def test_non_retryable_status_never_retries(status: int, reason: str) -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls) + 1)
        return httpx.Response(status, text="nope")

    async def run() -> None:
        gateway, client = _gateway(handler)
        with pytest.raises((UpstreamUnavailable, LLMNotConfigured)):
            await gateway.complete(MESSAGES, budget_ms=5000)
        assert len(calls) == 1, f"不该重试：{reason}"
        await client.aclose()

    asyncio.run(run())


def test_rate_limited_is_retryable() -> None:
    """429 是上游明说"稍后再来"，与 400/503 分开处理。"""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            return httpx.Response(429, text="rate limited")
        return _ok()

    async def run() -> None:
        gateway, client = _gateway(handler)
        result = await gateway.complete(MESSAGES, budget_ms=2000)
        assert len(calls) == 2 and result.model == "primary"
        await client.aclose()

    asyncio.run(run())


def test_fallback_model_also_failing_reraises() -> None:
    """主备都挂：把错误抛上去让编排层兜底，不要在这里吞成空答案。"""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(503, text="busy")

    async def run() -> None:
        gateway, client = _gateway(handler, llm_fallback_model="backup")
        with pytest.raises(UpstreamUnavailable):
            await gateway.complete(MESSAGES, budget_ms=2000)
        assert calls == ["primary", "backup"]  # 换模型只试一次，不在 backup 上再换
        await client.aclose()

    asyncio.run(run())


def test_api_key_never_leaks_into_error_or_log(caplog: pytest.LogCaptureFixture) -> None:
    """上游把 key 回显在错误体里（真实网关的常见行为），我们不许原样带出去。

    错误信息会进 `log.warning("llm_retry", error=...)`，而日志是**双出口**
    （stdout + `logs/app.log` 各一份）——漏一次就是两份永久留档。
    """
    caplog.set_level("DEBUG")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Incorrect API key provided: sk-test")

    async def run() -> None:
        gateway, client = _gateway(handler)
        with pytest.raises(UpstreamUnavailable) as exc:
            await gateway.complete(MESSAGES, budget_ms=1000)
        assert "sk-test" not in str(exc.value)
        assert "***" in str(exc.value)  # 位置还在，只是内容被抹掉：仍能看出上游说了什么
        assert "sk-test" not in caplog.text
        await client.aclose()

    asyncio.run(run())
