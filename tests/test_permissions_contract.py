"""权限契约单测（CLAUDE.md §7.2.7 DoD 第 3 条 / §5.2 隐私基线）。

这一层要守住三件事，任何一条松掉都是**跨租户读取**：

1. **安全三项只能由接入层写**：`tenant_id / acl_tags / kb_version` 由身份决定，
   为空时**构造即失败**，不许静默退化成"全租户可见"（§7.2.7 第 3 条）。
2. **放宽只放宽业务过滤**：`relaxed()` 可以丢 `subjects / doc_types / stages / grades`，
   安全三项逐字节不变——否则一次"救召回"的重试就能把租户过滤顺手关掉。
3. **请求头只在本地/测试生效**：`X-Tenant-Id` / `X-ACL-Tags` 是开发期身份来源，
   生产与预发必须由网关内部注入；默认拒绝，且用**白名单**判断。

第 3 条走完整 HTTP 链路验证（头 → `_identity` → `ChatState` → `QueryContext` →
`RetrievalFilters` → 检索器实参），只测 `_identity` 一个函数证明不了"头真的传到了检索层"。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from base.config import Settings
from base.contracts import (
    EvidenceDecision,
    FAQHit,
    QueryContext,
    RetrievalFilters,
    RetrievalResult,
    SlotResult,
)
from base.errors import PolicyViolation
from orchestration.api import _identity, _trust_headers
from orchestration.api import router as chat_router
from orchestration.chat import ChatPipeline
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import RagPipeline


# ---------------------------------------------------------------- 替身
def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = dict(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        chat_support_contact="400-000-0000",
        citation_min_support=0.30,
        total_budget_ms=8000,
        tenant_id="school-default",
        kb_version="v1",
        # **显式关掉 JWT 分支**：配了 `JWT_SECRET` 还认开发期请求头等于给验证过的
        # 身份系统留一个"直接自报"的后门（`resolve_identity()` 的两个条件之一）。
        # 不写这一行的后果是本文件的"请求头 → 检索过滤"用例**验的是开发机 `.env`
        # 有没有配密钥**，而不是代码——`.env` 一旦配上真实密钥，它们会集体变红。
        jwt_secret="",
    )
    base.update(overrides)
    return Settings(**base)


class _FAQ:
    """永远低置信：让链路必然走到检索，才能观察到检索器拿到的过滤条件。"""

    async def search(self, query: str, subject: str | None = None, *, slots: SlotResult | None = None,
                     selected_faq_id: int | None = None) -> FAQHit:
        return FAQHit(tier="low", reason="below_threshold", score=0.1)


class _CapturingRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls.append({"query": query, **kwargs})
        return RetrievalResult(query=query, evidences=[], raw_hits=0)


class _EmptyGate:
    """门不过，且动作不可修（`fallback`），保证只检索一次、不需要 LLM。"""

    def __call__(self, result: RetrievalResult, **kwargs: Any) -> EvidenceDecision:
        return EvidenceDecision(
            status="insufficient",
            reasons=("below_threshold",),
            retry_action="fallback",
            top_score=0.2,
            qcov=0.5,
        )


class _LLM:
    async def complete(self, messages: list[dict], *, budget_ms: int, model: str | None = None) -> LLMResult:
        raise AssertionError("本文件的用例不该走到生成（门不过且是 low 档）")


def _retriever_of(settings: Settings) -> tuple[ChatPipeline, _CapturingRetriever]:
    retriever = _CapturingRetriever()
    rag = {
        "ready": True,
        "retriever": retriever,
        "pipeline": RagPipeline(settings, retriever, gate=_EmptyGate(), max_repairs=0),
        "gate": _EmptyGate(),
        "embedder_name": "stub",
        "rerank_enabled": False,
    }
    pipeline = ChatPipeline(settings, faq_service=_FAQ(), rag=rag, llm=_LLM())
    return pipeline, retriever


def _post(settings: Settings, headers: dict[str, str] | None = None) -> tuple[dict, _CapturingRetriever]:
    pipeline, retriever = _retriever_of(settings)
    app = FastAPI()
    app.include_router(chat_router)
    app.state.chat = pipeline

    async def run() -> dict:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/v1/chat", json={"query": "二次函数的顶点式怎么求"}, headers=headers or {})
            assert resp.status_code == 200, resp.text
            return resp.json()

    return asyncio.run(run()), retriever


# ---------------------------------------------------------------- 1. 安全三项不可为空
def test_blank_tenant_is_rejected() -> None:
    """空租户在 Milvus 表达式里会退化成 `tenant_id == ""`，绝不能"查不到就算了"。"""
    with pytest.raises(PolicyViolation) as err:
        RetrievalFilters(tenant_id="", kb_version="v1")
    assert "tenant_id" in str(err.value)


def test_blank_kb_version_is_rejected() -> None:
    """索引版本进缓存 key 与标量过滤，为空等于把不同代语料的证据混在一起。"""
    with pytest.raises(PolicyViolation) as err:
        RetrievalFilters(tenant_id="school-a", kb_version="")
    assert "kb_version" in str(err.value)


def test_whitespace_only_counts_as_blank() -> None:
    with pytest.raises(PolicyViolation):
        RetrievalFilters(tenant_id="   ", kb_version="v1")


def test_context_without_tenant_cannot_build_filters() -> None:
    """接入层漏填租户时，**检索侧的入口**必须炸，而不是悄悄查全库。"""
    ctx = QueryContext(request_id="r1", tenant_id="")
    with pytest.raises(PolicyViolation):
        ctx.filters(subjects=("数学",))


def test_filters_take_all_three_from_context() -> None:
    ctx = QueryContext(
        request_id="r1",
        tenant_id="school-a",
        acl_tags=("grade9", "internal"),
        kb_version="v7",
        subject="数学",
    )
    f = ctx.filters(doc_types=("textbook",))
    assert (f.tenant_id, f.acl_tags, f.kb_version) == ("school-a", ("grade9", "internal"), "v7")
    assert f.subjects == ("数学",) and f.doc_types == ("textbook",)


# ---------------------------------------------------------------- 2. 放宽不碰安全三项
def test_relaxed_keeps_security_triple_byte_identical() -> None:
    f = RetrievalFilters(
        tenant_id="school-a",
        acl_tags=("grade9", "internal"),
        kb_version="v7",
        subjects=("数学",),
        doc_types=("textbook",),
        stages=("初中",),
        grades=("九年级",),
    )
    r = f.relaxed()
    # 业务过滤：全清
    assert (r.subjects, r.doc_types, r.stages, r.grades) == ((), (), (), ())
    # 安全三项：逐字段等于原值（不是"差不多"，是同一个值）
    assert r.tenant_id == f.tenant_id
    assert r.acl_tags == f.acl_tags
    assert r.kb_version == f.kb_version


def test_empty_acl_means_no_tags_not_all_visible() -> None:
    """空 ACL 是"没有标签"，不是"全部可见"——它不该被展开成通配或 None。"""
    f = RetrievalFilters(tenant_id="school-a", kb_version="v1")
    assert f.acl_tags == ()
    assert f.relaxed().acl_tags == ()
    assert f.as_dict()["acl_tags"] == []  # 响应里是空数组，不是 null


# ---------------------------------------------------------------- 3. 身份：头只在本地/测试生效
def test_identity_uses_headers_when_trusted() -> None:
    user_id, sid, tenant, acl = _identity(
        "u1", None, tenant_header="school-a", acl_header="grade9, internal", trust_headers=True
    )
    assert (user_id, tenant, acl) == ("u1", "school-a", ("grade9", "internal"))
    assert sid  # 有 user_id 才签发会话 id


def test_identity_ignores_headers_when_not_trusted() -> None:
    """生产：客户端自报的租户一律丢弃，只用部署配置里的默认租户。"""
    user_id, _sid, tenant, acl = _identity(
        "u1",
        None,
        tenant_header="school-b",
        acl_header="secret",
        default_tenant="school-default",
        trust_headers=False,
    )
    assert (tenant, acl) == ("school-default", ())


def test_identity_never_issues_session_without_user() -> None:
    """即使租户头合法，没有 user_id 也不给 session_id（无主会话写进库就删不掉）。"""
    user_id, sid, tenant, _acl = _identity("  ", None, tenant_header="school-a")
    assert (user_id, sid) == ("", None)
    assert tenant == "school-a"


def test_empty_acl_header_is_no_tags() -> None:
    _u, _s, _t, acl = _identity("u1", None, acl_header=" , ,", trust_headers=True)
    assert acl == ()


@pytest.mark.parametrize("env", ["dev", "development", "local", "test", "testing"])
def test_headers_are_trusted_in_local_style_envs(env: str) -> None:
    assert _trust_headers(_settings(app_env=env)) is True


@pytest.mark.parametrize("env", ["prod", "production", "staging", "STAGING", "uat", ""])
def test_headers_are_distrusted_everywhere_else(env: str) -> None:
    """白名单语义：预发与任何没见过的取值都**不**信任（黑名单会把新环境默默算成安全）。"""
    assert _trust_headers(_settings(app_env=env or "prod")) is False


# ---------------------------------------------------------------- 4. 完整链路：头 → 检索器实参
def test_headers_reach_retrieval_filters_in_dev() -> None:
    body, retriever = _post(
        _settings(app_env="dev"), {"X-Tenant-Id": "school-a", "X-Acl-Tags": "grade9,internal"}
    )
    assert body["source"] == "fallback"  # 门不过 + low 档
    assert retriever.calls, "低置信必须真的走检索"
    f: RetrievalFilters = retriever.calls[0]["filters"]
    assert f.tenant_id == "school-a"
    assert f.acl_tags == ("grade9", "internal")


def test_headers_are_dropped_in_production() -> None:
    """同一个头在生产被丢弃：落到检索器上的必须是部署配置，不是客户端说的。"""
    _body, retriever = _post(
        _settings(app_env="prod"), {"X-Tenant-Id": "school-b", "X-Acl-Tags": "secret"}
    )
    f: RetrievalFilters = retriever.calls[0]["filters"]
    assert f.tenant_id == "school-default"
    assert f.acl_tags == ()


def test_missing_headers_fall_back_to_deployment_tenant() -> None:
    _body, retriever = _post(_settings(app_env="dev"))
    f: RetrievalFilters = retriever.calls[0]["filters"]
    assert f.tenant_id == "school-default"
    assert f.acl_tags == ()


def test_kb_version_always_comes_from_settings_not_from_request() -> None:
    """索引版本同理：请求方连改的机会都不该有（它决定缓存 key 与证据的代际）。"""
    _body, retriever = _post(_settings(app_env="dev", kb_version="v9"))
    f: RetrievalFilters = retriever.calls[0]["filters"]
    assert f.kb_version == "v9"
