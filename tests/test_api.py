"""HTTP 层单测：不起 MySQL，直接把内存仓库挂到 app.state。"""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from qa_system.api import router as faq_router
from qa_system.res.qa_matcher import MatchPolicy
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository
from qa_system.res.qa_search import FAQService


def _make_app(records: list[FAQRecord] | None = None) -> FastAPI:
    repo = InMemoryFAQRepository(
        records
        or [
            FAQRecord(1, "数学", "勾股定理的内容是什么？", "直角三角形两直角边平方和等于斜边平方。"),
            FAQRecord(2, "数学", "勾股定理怎么证明？", "面积法。"),
        ]
    )
    service = FAQService(repo, policy=MatchPolicy(threshold=0.3, margin=0.0))
    asyncio.run(service.load())

    app = FastAPI()
    app.include_router(faq_router)
    app.state.faq_service = service
    return app


def _call(app: FastAPI, method: str, url: str, **kwargs) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.request(method, url, **kwargs)

    return asyncio.run(run())


def test_search_hit() -> None:
    resp = _call(_make_app(), "GET", "/v1/faq/search", params={"q": "勾股定理的内容是什么？"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hit"] is True and body["faq_id"] == 1 and body["reason"] == "exact"


def test_search_miss_has_candidates_field() -> None:
    resp = _call(_make_app(), "GET", "/v1/faq/search", params={"q": "量子纠缠"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hit"] is False and isinstance(body["candidates"], list)


def test_search_requires_query() -> None:
    resp = _call(_make_app(), "GET", "/v1/faq/search", params={"q": ""})
    assert resp.status_code == 422


def test_stats_endpoint() -> None:
    app = _make_app()
    _call(app, "GET", "/v1/faq/search", params={"q": "勾股定理"})
    body = _call(app, "GET", "/v1/faq/stats").json()
    assert body["state"]["docs"] == 2
    assert body["stats"]["searches"] >= 1


def test_get_faq_not_found() -> None:
    resp = _call(_make_app(), "GET", "/v1/faq/999")
    assert resp.status_code == 404


def test_publish_then_searchable() -> None:
    app = _make_app([FAQRecord(7, "英语", "甲问题", "甲答案", status="draft")])
    assert _call(app, "GET", "/v1/faq/search", params={"q": "甲问题"}).json()["hit"] is False
    published = _call(app, "POST", "/v1/faq/7/publish")
    assert published.status_code == 200 and published.json()["docs"] == 1
    assert _call(app, "GET", "/v1/faq/search", params={"q": "甲问题"}).json()["hit"] is True


def test_rebuild_endpoint() -> None:
    resp = _call(_make_app(), "POST", "/v1/faq/rebuild")
    assert resp.status_code == 200 and resp.json()["docs"] == 2


@pytest.mark.parametrize("path", ["/v1/faq/stats"])
def test_admin_endpoints_open_in_dev_without_token(path: str) -> None:
    # FAQ_ADMIN_TOKEN 未配置时允许本机调试（生产必须配置，见 api/faq.py）
    assert _call(_make_app(), "GET", path).status_code == 200

