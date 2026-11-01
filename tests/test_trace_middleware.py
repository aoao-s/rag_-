"""HTTP 中间件的两个回归点（2026-10-29 从一次真实启动日志里发现）。

1. **`http_request` 必须带 path**。模板是 `HTTP {method} {path} -> {status}`，
   缺字段时 `render()` 会把 `{path}` 渲染成 `-`——线上日志因此变成
   `HTTP GET - -> 404`，正好丢掉排查 404 最需要的那一段。
2. **异常分支不能让 `finally` 读未绑定的 `response`**。原实现 `except` 里直接
   `return`，`finally` 仍执行且读 `response.status_code` → `UnboundLocalError`
   顶掉本该返回的 500 JSON，用户看到 uvicorn 的纯文本错误页，`request_id` 也没了。

测试直接复用 `main.trace_middleware`（Starlette 的装饰器会原样返回函数），
挂到一个只有两条路由的干净 app 上——不启动真实 lifespan，也就不装载模型。
"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

import main

REQUEST_ID_HEADER = "X-Request-Id"


class _Recorder:
    """替掉 `main.log`：只记下调用点传了什么，避免依赖 Formatter 的渲染时机。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def info(self, event: str, **fields) -> None:
        self.calls.append((event, fields))

    def error(self, event: str, **fields) -> None:
        self.calls.append((event, fields))

    def http_request(self) -> tuple[str, dict]:
        for event, fields in reversed(self.calls):
            if event == "http_request":
                return event, fields
        raise AssertionError(f"没有 http_request 日志：{self.calls}")


def _app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(main.trace_middleware)

    @app.get("/ok")
    async def ok() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/boom")
    async def boom() -> dict[str, bool]:
        raise RuntimeError("boom")

    return app


def _call(url: str) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get(url)

    return asyncio.run(run())


def test_404_logs_the_actual_path(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(main, "log", recorder)

    resp = _call("/definitely-not-here")

    assert resp.status_code == 404
    _, fields = recorder.http_request()
    assert fields["path"] == "/definitely-not-here"  # 不是 "-"
    assert fields["status"] == 404
    assert fields["method"] == "GET"
    assert resp.headers[REQUEST_ID_HEADER]


def test_unhandled_exception_is_json_500_with_request_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    monkeypatch.setattr(main, "log", recorder)

    resp = _call("/boom")

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "internal error"
    assert body["request_id"]
    # 响应头与响应体必须是同一个 id：报障截图能直接对上日志
    assert resp.headers[REQUEST_ID_HEADER] == body["request_id"]
    _, fields = recorder.http_request()
    assert (fields["path"], fields["status"]) == ("/boom", 500)
