"""阶段 1 验收测试：5 条验收标准，逐条可执行。

验收标准原文（《EduRag-Pro项目详细执行计划.md》阶段 1）：

1. 非法请求不能进入后续流程。
2. 用户身份和角色只能来自可信认证上下文。
3. 高置信 FAQ 能够短路。
4. 中置信 FAQ 不直接返回答案。
5. 缓存命中不会跨租户、跨权限、跨知识库版本泄露内容。

**这份文件是"验收尺"，不是"实现说明"**：它按验收标准写成，失败的条目就是当前代码
与标准的差距。阶段 1 落地前跑它，红的那几条正是要做的活；落地后应当全绿。

不依赖 MySQL / Milvus / Redis / 真实 LLM——验收验的是"分支与边界选对了"，
检索质量由黄金集评估负责（CLAUDE.md §7.2.5）。
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from typing import Any

import httpx
from fastapi import FastAPI

from base.config import Settings
from base.contracts import EvidenceDecision, FAQCandidate, FAQHit, SlotResult
from orchestration.api import router as chat_router
from orchestration.chat import ChatPipeline
from qa_system.res.qa_cache import InMemoryFAQCache
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository
from qa_system.res.qa_matcher import MatchPolicy
from qa_system.res.qa_search import FAQService
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult

EVIDENCE_TEXT = (
    "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k，其中 h=-b/(2a)，k=(4ac-b²)/(4a)。"
)
FAQ_ANSWER = "学费是 12800 元。"


# --------------------------------------------------------------------- 装配
def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "app_env": "dev",
        "tenant_id": "default",
        "llm_api_key": "sk-test",
        "llm_qa_model": "primary",
        "chat_support_contact": "400-000-0000",
        "citation_min_support": 0.30,
        "total_budget_ms": 8000,
        # 关掉"一次大模型理解"的**四个**消费者：改写 + 意图 + 槽位抽取 + 检索策略。
        # 本文件数的是**生成**调用次数（验收 3），开着任何一个同一个 StubLLM
        # 都会多收一次理解调用。槽位抽取（`FEATURE_SLOT_LLM`，任务 2.3）与检索策略
        # （`FEATURE_PLANNER_LLM`，2026-11-01）默认都开，漏写任一行时验收 3 的
        # `llm.calls == 0` 会被一次 `understanding_bad_json` 打破
        "feature_intent_llm": False,
        "feature_query_rewrite": False,
        "feature_slot_llm": False,
        "feature_planner_llm": False,
        # **显式关掉 JWT 分支**：验收 2 的用例用 `X-User-Id` 观察"身份只来自可信上下文"，
        # 走的是开发期请求头。配了 `JWT_SECRET` 就自动切到 JWT-only，不写这一行
        # 这些用例验的是开发机 `.env` 有没有配密钥，而不是代码
        "jwt_secret": "",
    }
    base.update(overrides)
    return Settings(**base)


class StubFAQ:
    def __init__(self, hit: FAQHit) -> None:
        self.hit = hit
        self.calls = 0

    async def search(
        self,
        query: str,
        subject: str | None = None,
        *,
        slots: SlotResult | None = None,
        selected_faq_id: int | None = None,
    ) -> FAQHit:
        self.calls += 1
        return self.hit


class StubRetriever:
    def __init__(self, *, boom: bool = False) -> None:
        self.boom = boom
        self.calls = 0

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls += 1
        if self.boom:
            raise RuntimeError("milvus down")
        return _retrieval()


class FixedGate:
    """固定结论的替身证据门。"""

    def __init__(self, passed: bool) -> None:
        self.passed = passed

    def __call__(self, result: RetrievalResult, **kwargs: Any) -> EvidenceDecision:
        if self.passed:
            return EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)
        return EvidenceDecision(
            status="insufficient",
            reasons=("below_threshold",),
            retry_action="fallback",
            top_score=0.2,
            margin=0.3,
            n_docs=1,
            qcov=0.6,
        )


class StubLLM:
    """记录每次收到的 messages —— 验收 4 要断言"FAQ 答案没进 prompt"。"""

    enabled = True

    def __init__(self, text: str = "顶点式可以通过配方法得到[1]。") -> None:
        self.text = text
        self.calls = 0
        self.seen: list[list[dict]] = []

    async def complete(
        self, messages: list[dict], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        self.calls += 1
        self.seen.append(messages)
        return LLMResult(text=self.text, model="primary", prompt_tokens=100, completion_tokens=20)


def _evidence() -> Evidence:
    return Evidence(
        parent_id="doc-1:p3",
        text=EVIDENCE_TEXT,
        score=0.91,
        rerank_score=0.9,
        child_hits=[EVIDENCE_TEXT[:60]],
        title="人教版-数学-九年级上册",
        source="F:/docs/人教版-数学-九年级上册.pdf",
        subject="数学",
        doc_type="textbook",
        heading="22.1 二次函数",
        page=12,
    )


def _retrieval() -> RetrievalResult:
    return RetrievalResult(query="顶点式怎么求", evidences=[_evidence()], raw_hits=12, cost_ms=120)


def _rag(*, ready: bool = True, pass_gate: bool = True, retriever: StubRetriever | None = None) -> dict:
    retriever = retriever or StubRetriever()
    return {
        "ready": ready,
        "retriever": retriever,
        "pipeline": RagPipeline(_settings(), retriever, gate=FixedGate(pass_gate), max_repairs=0),
        "gate": FixedGate(pass_gate),
        "embedder_name": "stub",
        "rerank_enabled": False,
    }


def _app(
    hit: FAQHit,
    *,
    rag: dict | None = None,
    llm: StubLLM | None = None,
    settings: Settings | None = None,
) -> tuple[FastAPI, StubFAQ, StubLLM]:
    faq = StubFAQ(hit)
    llm = llm or StubLLM()
    pipeline = ChatPipeline(
        settings or _settings(),
        faq_service=faq,
        rag=rag if rag is not None else _rag(),
        llm=llm,
    )
    app = FastAPI()
    app.include_router(chat_router)
    app.state.chat = pipeline
    return app, faq, llm


def _post(
    app: FastAPI, payload: dict, headers: dict[str, str] | None = None
) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/chat", json=payload, headers=headers or {})

    return asyncio.run(run())


def _high_hit() -> FAQHit:
    return FAQHit(tier="high", faq_id=3, answer=FAQ_ANSWER, score=0.95, subject="数学")


def _mid_hit() -> FAQHit:
    return FAQHit(
        tier="mid",
        faq_id=7,
        answer=None,
        score=0.6,
        matched_question="二次函数的顶点式怎么写？",
        candidates=(FAQCandidate(7, "二次函数的顶点式怎么写？", 0.6),),
    )


def _low_hit() -> FAQHit:
    return FAQHit(tier="low", reason="below_threshold", score=0.1)


# ============================================================ 验收 1
class TestCriterion1InvalidRequestNeverReachesPipeline:
    """验收标准 1：非法请求不能进入后续流程。"""

    def test_blank_query_is_rejected_and_faq_never_called(self) -> None:
        app, faq, llm = _app(_high_hit())
        resp = _post(app, {"query": "   "})
        assert resp.status_code >= 400
        assert faq.calls == 0, "空问句进了后续流程"
        assert llm.calls == 0

    def test_over_long_query_is_rejected_and_never_reaches_faq(self) -> None:
        """标准判据本身：超长要挡住。响应该是 400 还是 422 由下面那条单独验。"""
        app, faq, _ = _app(_high_hit())
        resp = _post(app, {"query": "勾" * 401})
        assert resp.status_code >= 400
        assert faq.calls == 0

    def test_rejection_uses_the_unified_400_shape(self) -> None:
        """设计 §3.4：入口校验失败一律 400 + 统一错误体，框架层的 422 也要映射掉。

        422 体是 pydantic 的内部结构，与 `ChatResponse` 不是一套，前端得写两套解析；
        而且这条路径恰是最难被测到的（设计 §3.4「框架层的 422 也必须变成 400」）。
        """
        app, _, _ = _app(_high_hit())
        resp = _post(app, {"query": "勾" * 401})
        assert resp.status_code == 400, f"超长问句返回 {resp.status_code}，设计口径是 400"

    def test_query_at_design_limit_is_accepted(self) -> None:
        """边界：正好等于上限必须通过。当前实现按 200 拒绝，400 字是合法请求。"""
        app, faq, _ = _app(_high_hit())
        resp = _post(app, {"query": "勾" * 400})
        assert resp.status_code == 200, (
            "400 字问句在设计上限内，却被 "
            f"{resp.status_code} 拒绝——入口闸门仍按 200 字（ChatRequest.max_length=200），"
            "内部策略上限是 400"
        )
        assert faq.calls == 1

    def test_invalid_subject_is_400(self) -> None:
        app, faq, _ = _app(_high_hit())
        resp = _post(app, {"query": "学费多少", "subject": "天文学"})
        assert resp.status_code == 400
        assert resp.json()["detail"]["error"] == "subject_invalid"
        assert faq.calls == 0

    def test_unknown_request_field_is_rejected(self) -> None:
        """请求格式错误（把 query 拼成 qurey）必须报错，不能悄悄忽略。"""
        app, faq, _ = _app(_high_hit())
        resp = _post(app, {"query": "学费多少", "qurey": "拼错的字段"})
        assert resp.status_code == 400, (
            f"未知字段被静默忽略（实际 {resp.status_code}）：`extra=\"forbid\"` 未生效，"
            "客户端拼错参数名时拿到的是正常回答，问题不会被发现"
        )
        assert faq.calls == 0


# ============================================================ 验收 2
class TestCriterion2IdentityFromTrustedContext:
    """验收标准 2：用户身份和角色只能来自可信认证上下文。"""

    def test_prod_ignores_client_supplied_user_id(self) -> None:
        """非白名单环境（生产/预发）不认客户端自报的 `X-User-Id`。

        `_identity()` 只把 `tenant` / `acl` 交给 `trust_headers` 管，`user_id` 是**无条件**
        采信请求头的——而它一路流到 `memory.recall(user_id)` 与 `session.append(user_id)`。
        也就是说生产上改一个请求头 `X-User-Id: alice`，就能把 alice 的长期记忆召进
        prompt、把这一轮写进 alice 的会话历史。`memory/api.py::_user()` 已经按白名单
        收口（§7.9.3），编排入口这一处还没收。
        """
        app, _, _ = _app(_high_hit(), settings=_settings(app_env="prod"))
        body = _post(app, {"query": "学费多少"}, {"X-User-Id": "attacker"}).json()
        assert body["session_id"] is None, (
            "生产环境采信了客户端自报的 X-User-Id：该值会进入 memory.recall / session.append，"
            "冒充任意用户即可读到对方的长期记忆与会话历史"
        )

    def test_dev_honours_user_id_and_issues_session(self) -> None:
        app, _, _ = _app(_high_hit())
        body = _post(app, {"query": "学费多少"}, {"X-User-Id": "u1"}).json()
        assert body["session_id"]

    def test_role_comes_from_server_side_context(self) -> None:
        """没有认证上下文时 role 必须是显式的 ANONYMOUS，不能缺席、不能由请求体决定。"""
        app, _, _ = _app(_high_hit())
        body = _post(app, {"query": "学费多少", "role": "ADMIN"}).json()
        assert "role" in body, (
            "响应体没有 role 字段：角色尚未进入请求上下文，"
            "无法断言它来自可信来源（阶段 1 任务 1.2）"
        )
        assert body["role"] == "ANONYMOUS", f"未认证请求的角色是 {body['role']}，应为 ANONYMOUS"

    def test_foreign_session_id_is_rejected(self) -> None:
        """A 拿 B 的 session_id 不能读到 B 的历史（当前可直接越权）。"""
        app, _, _ = _app(_high_hit())
        owner = _post(app, {"query": "学费多少"}, {"X-User-Id": "alice"}).json()
        sid = owner["session_id"]
        stolen = _post(app, {"query": "学费多少", "session_id": sid}, {"X-User-Id": "bob"})
        assert stolen.status_code == 400, (
            f"用别人的 session_id 发问未被拒绝（{stolen.status_code}）："
            "session 无归属校验，可越权读取他人历史"
        )


# ============================================================ 验收 3
class TestCriterion3HighFaqShortCircuits:
    """验收标准 3：高置信 FAQ 能够短路。"""

    def test_high_hit_skips_retrieval_and_generation(self) -> None:
        retriever = StubRetriever()
        llm = StubLLM()
        app, _, _ = _app(_high_hit(), rag=_rag(retriever=retriever), llm=llm)
        body = _post(app, {"query": "学费多少"}).json()
        assert body["source"] == "faq"
        assert body["answer"] == FAQ_ANSWER
        assert body["faq_id"] == 3
        assert retriever.calls == 0, "高置信仍跑了检索"
        assert llm.calls == 0, "高置信仍调用了一次大模型"

    def test_high_hit_returns_no_citations_and_no_clarify(self) -> None:
        app, _, _ = _app(_high_hit())
        body = _post(app, {"query": "学费多少"}).json()
        assert body["citations"] == []
        assert body["clarify"] is None


# ============================================================ 验收 4
class TestCriterion4MidFaqNeverReturnsAnswer:
    """验收标准 4：中置信 FAQ 不直接返回答案。"""

    def test_mid_with_failed_gate_clarifies_instead_of_answering(self) -> None:
        app, _, _ = _app(_mid_hit(), rag=_rag(pass_gate=False))
        body = _post(app, {"query": "顶点式怎么写"}).json()
        assert body["source"] == "clarify"
        assert body["answer"] != FAQ_ANSWER
        assert body["faq_suggestion"] is not None
        assert "answer" not in body["faq_suggestion"], "中置信把 FAQ 答案回给了调用方"

    def test_mid_with_passed_gate_never_feeds_faq_answer_to_llm(self) -> None:
        """中置信候选答案严禁进 prompt 知识段（CLAUDE.md §5.6 第 9 条）。"""
        llm = StubLLM()
        app, _, _ = _app(_mid_hit(), rag=_rag(pass_gate=True), llm=llm)
        body = _post(app, {"query": "顶点式怎么写"}).json()
        assert body["source"] == "rag"
        assert "answer" not in (body["faq_suggestion"] or {})
        joined = "\n".join(str(m.get("content", "")) for turn in llm.seen for m in turn)
        assert FAQ_ANSWER not in joined
        assert all(c.get("doc_id") != "faq:7" for c in body["citations"]), "FAQ 被包成了引用证据"

    def test_real_matcher_never_emits_answer_for_mid(self) -> None:
        """档位由 qa_system 自己算：mid 的 `answer` 恒为 None（靠类型不给，不靠调用方记得）。"""
        service = _faq_service(high_margin=0.99)
        hit = asyncio.run(service.search("勾股定理 内容和证明"))
        assert hit.tier == "mid", f"构造的样本没落在 mid 档（实际 {hit.tier}）"
        assert hit.answer is None


# ============================================================ 验收 5
class TestCriterion5CacheNeverLeaks:
    """验收标准 5：缓存命中不会跨租户、跨权限、跨知识库版本泄露内容。"""

    def test_faq_cache_key_carries_tenant(self) -> None:
        """两个租户问同一句话必须落到两条缓存上。"""
        service = _faq_service()
        params = inspect.signature(service._cache_key).parameters
        assert "tenant" in params, (
            "FAQ 缓存键不含 tenant（edurag:{env}:faq:v{N}:{scope}:{sha1}）："
            "两个租户问同一句话会命中同一条缓存。今天不泄漏仅因 faq 表没有租户列，"
            "阶段 2 给 faq 加 tenant_id 后立刻变成跨租户答案泄漏（设计 §3.9.4）"
        )

    def test_faq_cache_key_never_contains_raw_query(self) -> None:
        service = _faq_service()
        key = service._cache_key(1, "我今年初三住在杭州手机号13800138000", "*")
        assert "13800138000" not in key and "杭州" not in key
        assert hashlib.sha1("我今年初三住在杭州手机号13800138000".encode()).hexdigest()[:20] in key

    def test_faq_cache_key_separates_retrieval_scope(self) -> None:
        service = _faq_service()
        assert service._cache_key(1, "勾股定理", "数学") != service._cache_key(1, "勾股定理", "*")

    def test_faq_cache_key_separates_corpus_version(self) -> None:
        """知识库版本维度：FAQ 层的版本是语料版本，变一版就整体换键。"""
        service = _faq_service()
        assert service._cache_key(1, "勾股定理", "*") != service._cache_key(2, "勾股定理", "*")

    def test_two_tenants_do_not_share_one_answer(self) -> None:
        """端到端口径：同一份共享缓存，两个租户的检索结论不能互相命中。"""
        cache = InMemoryFAQCache()
        repo = InMemoryFAQRepository(_faq_records())
        school_a = _faq_service(cache=cache, repo=repo, tenant="school_a")
        school_b = _faq_service(cache=cache, repo=repo, tenant="school_b")
        asyncio.run(school_a.search("勾股定理的内容是什么"))
        asyncio.run(school_b.search("勾股定理的内容是什么"))
        # 键相同 → 第二次直接吃第一次的缓存条目，`_data` 只有 1 条；键不同才有 2 条
        assert len(cache._data) == 2, (
            f"两个租户共用了一条缓存（缓存里只有 {len(cache._data)} 条）："
            "租户没能进入缓存键，跨租户内容可被复用"
        )


# --------------------------------------------------------------------- 工具
def _faq_records() -> list[FAQRecord]:
    return [
        FAQRecord(1, "数学", "勾股定理的内容是什么？", "直角三角形两直角边平方和等于斜边平方。"),
        FAQRecord(2, "数学", "勾股定理怎么证明？", "常用面积法（赵爽弦图）。"),
        FAQRecord(3, "英语", "怎么报名", "在官网填写报名表并缴费。"),
    ]


def _faq_service(
    *,
    high_margin: float = 0.22,
    cache: InMemoryFAQCache | None = None,
    repo: InMemoryFAQRepository | None = None,
    tenant: str = "",
) -> FAQService:
    """构造一个只依赖内存的 FAQService；`tenant` 走 FAQService 的租户参数（阶段 1 新增）。"""
    kwargs: dict[str, Any] = {}
    if "tenant" in inspect.signature(FAQService.__init__).parameters:
        kwargs["tenant_id"] = tenant
    service = FAQService(
        repo or InMemoryFAQRepository(_faq_records()),
        policy=MatchPolicy(threshold=0.05, margin=0.0, candidate_n=3, high_margin=high_margin),
        top_k=10,
        cache=cache,
        cache_ttl=60,
        cache_negative_ttl=5,
        **kwargs,
    )
    asyncio.run(service.load())
    return service
