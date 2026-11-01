"""第二轮（PR3-E / PR3-F）验收：可问答闭环的十三条路径 + 统一响应体契约。

依据实施手册 §7.5（十三条集成测试）与 §2.7（所有分支返回统一 `ChatResponse`），
补的是 `tests/test_chat_api.py` 没覆盖的那一半：

- **状态归一化**：同一份证据不足，`ambiguous` 要追问、`conflicting` 要并列、`insufficient`
  才是兜底——三者都是 `decision.passed == False`，出口却完全不同；
- **系统失败与知识库没有要分开**：`retrieval_unavailable` / `generation_unavailable` /
  `citation_invalid` 记 `status="error"`，`no_evidence` 记 `status="no_evidence"`；
- **引用受控修复**：剔掉无支撑句后还能答，是 `citation_repaired`（`status="degraded"`），
  剔完一句不剩才是拒答（`citation_invalid`）；
- **响应体形状唯一**：成功、追问、兜底、500 的字段集合必须完全一致。

全部用替身，不打 MySQL / Milvus / LLM：这一层验的是"分支选对了"，检索质量归黄金集。
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import httpx
from fastapi import FastAPI

from base.config import Settings
from base.contracts import (
    ChatResponse,
    EvidenceDecision,
    FAQCandidate,
    FAQHit,
    SlotResult,
)
from base.errors import UpstreamUnavailable
from orchestration.api import router as chat_router
from orchestration.chat import ChatPipeline
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult

EVIDENCE_TEXT = (
    "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k，其中 h=-b/(2a)，k=(4ac-b²)/(4a)。"
)

# 成功、追问、兜底、500 都必须带齐这 23 个键——前端只写一份渲染逻辑
RESPONSE_KEYS = {f.name for f in dataclasses.fields(ChatResponse)}


# ---------------------------------------------------------------- 替身
def _settings() -> Settings:
    return Settings(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        chat_support_contact="400-000-0000",
        citation_min_support=0.30,
        total_budget_ms=8000,
        # 关掉"一次大模型理解"的**四个**消费者（改写 + 意图 + 槽位抽取 + 检索策略）：
        # 本文件验的是 **生成分支**，开着任何一个都会让同一个 `StubLLM` 多收一次调用，
        # `llm.calls == 1` 这类断言就变成在数"理解 + 生成"。理解路径的门道在
        # `tests/test_chat_rewrite.py` 里单独验。
        feature_intent_llm=False,
        feature_query_rewrite=False,
        feature_slot_llm=False,
        feature_planner_llm=False,
    )


class StubFAQ:
    def __init__(self, hit: FAQHit | None = None, *, boom: bool = False) -> None:
        self.hit = hit if hit is not None else _low_hit()
        self.boom = boom
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
        if self.boom:
            raise RuntimeError("mysql gone")
        return self.hit


class BoomIntent:
    """意图分类器炸掉：只能退回 BASELINE，不许影响能不能答（§7.7 降级表）。"""

    def classify(self, query: str) -> Any:
        raise RuntimeError("onnx model not loaded")


class StubRetriever:
    def __init__(self, result: RetrievalResult | None = None, *, boom: bool = False) -> None:
        self.result = result or _retrieval()
        self.boom = boom
        self.calls = 0

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls += 1
        if self.boom:
            raise RuntimeError("milvus down")
        return self.result


class ScriptedGate:
    """按脚本依次出结论的门替身：用来验"修完必须重新过门"（§7.2.7 约束 2）。"""

    def __init__(self, *decisions: EvidenceDecision) -> None:
        self._decisions = list(decisions)
        self.calls = 0

    def __call__(self, result: RetrievalResult, **kwargs: Any) -> EvidenceDecision:
        self.calls += 1
        index = min(self.calls - 1, len(self._decisions) - 1)
        return self._decisions[index]


class StubLLM:
    def __init__(
        self,
        text: str = "顶点式可以通过配方法得到[1]。",
        *,
        enabled: bool = True,
        boom: bool = False,
    ) -> None:
        self.text = text
        self.enabled = enabled
        self.boom = boom
        self.calls = 0

    async def complete(
        self, messages: list[dict], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        self.calls += 1
        if self.boom:
            raise UpstreamUnavailable("gateway 502", node="generate")
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
    return RetrievalResult(
        query="顶点式怎么求",
        evidences=[_evidence()],
        raw_hits=12,
        cost_ms=120,
        # 分阶段耗时由 retriever 写；编排层要原样透传，不许自己编
        stages={
            "query_embed": 9,
            "recall": 40,
            "parent_expand": 6,
            "fusion": 0,
            "rerank": 30,
            "total": 85,
        },
    )


def _passed() -> EvidenceDecision:
    return EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)


def _insufficient(action: str = "retry_direct") -> EvidenceDecision:
    return EvidenceDecision(
        status="insufficient",
        reasons=("below_threshold",),
        retry_action=action,  # type: ignore[arg-type]
        top_score=0.2,
        margin=0.3,
        n_docs=1,
        qcov=0.6,
    )


def _rag(*decisions: EvidenceDecision, retriever: StubRetriever | None = None, repairs: int = 1) -> dict:
    """装配成 `main.Container.load_rag()` 的同一形状：编排层只认 `ready` 与 `pipeline`。"""
    retriever = retriever or StubRetriever()
    gate = ScriptedGate(*(decisions or (_passed(),)))
    return {
        "ready": True,
        "retriever": retriever,
        "pipeline": RagPipeline(_settings(), retriever, gate=gate, max_repairs=repairs),
        "gate": gate,
        "embedder_name": "stub",
        "rerank_enabled": False,
    }


def _low_hit() -> FAQHit:
    return FAQHit(tier="low", reason="below_threshold", score=0.1)


def _high_hit() -> FAQHit:
    return FAQHit(tier="high", faq_id=3, answer="学费是 12800 元。", score=0.95, subject="数学")


def _mid_hit() -> FAQHit:
    return FAQHit(
        tier="mid",
        faq_id=7,
        answer=None,
        score=0.6,
        matched_question="二次函数的顶点式怎么写？",
        candidates=(FAQCandidate(7, "二次函数的顶点式怎么写？", 0.6),),
    )


def _pipeline(
    *,
    hit: FAQHit | None = None,
    faq_boom: bool = False,
    rag: dict | None = None,
    llm: StubLLM | None = None,
    intent: Any = None,
) -> ChatPipeline:
    return ChatPipeline(
        _settings(),
        faq_service=StubFAQ(hit, boom=faq_boom),
        rag=rag if rag is not None else _rag(),
        llm=llm or StubLLM(),
        intent=intent,
    )


def _ask(pipeline: ChatPipeline, query: str = "顶点式怎么求", **kwargs: Any) -> Any:
    return asyncio.run(pipeline.ask(query, **kwargs))


# ---------------------------------------------------------------------------
# §7.5 十三条：每条对应一个"如果选错了，用户就会拿到错误答案"的分支
# ---------------------------------------------------------------------------
def test_01_faq_high_skips_retrieval_and_generation() -> None:
    """① FAQ 高置信：不检索、不生成（§7.1.9）。

    阶段 3 之后它**不再跳过理解**：时序是 会话 → 理解 → 权限 → FAQ → 检索，
    权限要读意图，术语层就必须先跑。所以意图分类器炸掉时，答案照样是那条 FAQ，
    但 `degraded` 里留 `intent_unavailable`、`status` 记 `degraded`——
    **这不是退化，是如实上报**："这次请求的权限闸门是在没有意图的情况下放行的"。
    把这条降级吞掉，等于让"权限按角色 × 意图判定"在故障时静默变成"不判定"。
    """
    retriever = StubRetriever()
    llm = StubLLM()
    intent = BoomIntent()
    state = _ask(
        _pipeline(hit=_high_hit(), rag=_rag(retriever=retriever), llm=llm, intent=intent)
    )
    body = state.to_response()
    assert body.source == "faq"  # FAQ 结论本身不受影响
    assert body.answer == "学费是 12800 元。" and body.faq_id == 3
    assert body.status == "degraded" and "intent_unavailable" in body.degraded
    assert retriever.calls == 0 and llm.calls == 0


def test_02_faq_miss_falls_through_to_rag() -> None:
    """② FAQ 未命中 → 检索 → 生成。"""
    state = _ask(_pipeline(hit=_low_hit(), rag=_rag(_passed())))
    body = state.to_response()
    assert body.status == "ok" and body.source == "rag"
    assert [c["n"] for c in body.citations] == [1]


def test_03_faq_failure_degrades_to_rag() -> None:
    """③ FAQ 挂了退回全量 RAG：照常回答，但必须留痕（否则命中率归零查不出原因）。"""
    state = _ask(_pipeline(faq_boom=True, rag=_rag(_passed())))
    body = state.to_response()
    assert "faq_unavailable" in body.degraded
    assert body.source == "rag" and body.answer
    # `degraded` 的判据是"读到的这句话是否变弱"，不是"这次有没有出岔子"：
    # 本该是人工审核过的逐字答案，现在变成了生成答案，所以确实弱了（state._DEGRADED_ANSWER）
    assert body.status == "degraded"


def test_04_understand_failure_uses_baseline_profile() -> None:
    """④ 理解层失败 → 用原 query + BASELINE 档位（不收窄、必检索、必引用）。"""
    state = _ask(_pipeline(rag=_rag(_passed()), intent=BoomIntent()))
    body = state.to_response()
    assert "intent_unavailable" in body.degraded
    assert body.profile == "baseline"
    assert body.answer  # 分类器挂了不影响能不能答


def test_05_sufficient_gate_generates_with_citations() -> None:
    """⑤ 门通过 → 生成 → 引用可对齐。"""
    llm = StubLLM()
    state = _ask(_pipeline(rag=_rag(_passed()), llm=llm))
    body = state.to_response()
    assert body.status == "ok" and llm.calls == 1
    assert body.citations[0]["file_name"] == "人教版-数学-九年级上册.pdf"
    assert body.usage == {"prompt_tokens": 100, "completion_tokens": 20}


def test_06_insufficient_repairs_once_then_regates() -> None:
    """⑥ 门不过 → **至多一次**修复 → **修完重新过门**（不是"改过过滤就当过了"）。"""
    gate_decisions = (_insufficient("retry_direct"), _passed())
    rag = _rag(*gate_decisions, repairs=1)
    state = _ask(_pipeline(rag=rag))
    body = state.to_response()
    assert rag["gate"].calls == 2  # 第二遍真的过了门
    assert body.status == "ok" and body.source == "rag"
    assert any(code.startswith("retry:") for code in body.degraded)


def test_07_still_insufficient_after_repair_falls_back() -> None:
    """⑦ 修完仍不足 → 兜底 + 转人工，`error_code=no_evidence`（知识库没有，不是服务坏了）。"""
    rag = _rag(_insufficient("retry_direct"), repairs=1)
    state = _ask(_pipeline(rag=rag))
    body = state.to_response()
    assert body.status == "no_evidence" and body.source == "fallback"
    assert body.error_code == "no_evidence"
    assert "retry_exhausted" in body.degraded
    assert "400-000-0000" in body.answer


def test_08_ambiguous_asks_instead_of_guessing() -> None:
    """⑧ ambiguous → 追问。按分数挑一条就是抽签（§7.2.7-B）。"""
    rag = _rag(
        EvidenceDecision(
            status="ambiguous",
            reasons=("small_margin",),
            retry_action="ask_disambiguate",
            top_score=0.8,
            margin=0.004,
            n_docs=2,
            qcov=0.5,
        )
    )
    state = _ask(_pipeline(rag=rag))
    body = state.to_response()
    assert body.status == "clarify" and body.source == "clarify"
    assert body.answer == body.clarify
    assert "能再具体一点吗" in body.clarify
    # 追问时把"咬得很紧"的那几条摆出来，用户才有得选
    assert body.citations


def test_09_conflicting_states_both_sides() -> None:
    """⑨ conflicting → 并列说明，不替用户挑一个（§7.2.7-C）。"""
    rag = _rag(
        EvidenceDecision(
            status="conflicting",
            reasons=("conflict:page_no",),
            retry_action="return_conflict",
            top_score=0.8,
            margin=0.1,
            n_docs=2,
            qcov=0.5,
        )
    )
    state = _ask(_pipeline(rag=rag))
    body = state.to_response()
    assert body.source == "fallback" and body.status == "no_evidence"
    assert "page_no" in body.answer
    assert body.citations  # 冲突时更需要来源，否则用户无法自行判断


def test_10_retrieval_failure_is_retrieval_unavailable() -> None:
    """⑩ 检索异常 → `retrieval_unavailable`（该修服务），不是 no_evidence（该补语料）。"""
    state = _ask(_pipeline(rag=_rag(retriever=StubRetriever(boom=True))))
    body = state.to_response()
    assert body.status == "error"
    assert body.error_code == "retrieval_unavailable"
    assert "retrieve_failed" in body.degraded


def test_11_generation_failure_is_generation_unavailable() -> None:
    """⑪ 生成失败 → `generation_unavailable`，话术带客服电话。"""
    state = _ask(_pipeline(rag=_rag(_passed()), llm=StubLLM(boom=True)))
    body = state.to_response()
    assert body.status == "error"
    assert body.error_code == "generation_unavailable"
    assert "400-000-0000" in body.answer


def test_12_citation_repair_keeps_the_supported_half() -> None:
    """⑫ 部分句子没有依据 → 剔掉那句再验一遍，能答就答（`citation_repaired`）。"""
    text = "顶点式可以通过配方法得到[1]。今天天气不错适合出门[1]。"
    state = _ask(_pipeline(rag=_rag(_passed()), llm=StubLLM(text)))
    body = state.to_response()
    assert body.source == "rag"
    assert "今天天气不错" not in body.answer  # 无支撑句被剔除，不是被改写
    assert "配方法" in body.answer
    assert "citation_repaired" in body.degraded
    assert body.status == "degraded"  # 答案还在，但确实比理想路径弱


def test_13_citation_total_failure_refuses_answer() -> None:
    """⑬ 剔完一句不剩 → `citation_invalid`，整段拒答（§7.8 硬约束 3）。"""
    state = _ask(_pipeline(rag=_rag(_passed()), llm=StubLLM("今天天气不错适合出门[1]。")))
    body = state.to_response()
    assert body.status == "error"
    assert body.error_code == "citation_invalid"
    assert body.citations == []


# ---------------------------------------------------------------------------
# 统一响应体（§2.7）：所有出口同一形状，500 也不例外
# ---------------------------------------------------------------------------
def test_all_branches_share_one_response_shape() -> None:
    states = [
        _ask(_pipeline(hit=_high_hit())),                                     # ok / faq
        _ask(_pipeline(rag=_rag(_passed()))),                                 # ok / rag
        _ask(_pipeline(rag=_rag(_insufficient()))),                           # no_evidence
        _ask(_pipeline(rag=_rag(retriever=StubRetriever(boom=True)))),        # error
        _ask(_pipeline(rag=_rag(_passed()), llm=StubLLM("顶点式可以这样求[9]。"))),  # citation_invalid
    ]
    for state in states:
        assert set(state.to_response().as_dict()) == RESPONSE_KEYS
    # 五个出口的状态确实分得开，不是全被压成同一个
    assert {s.to_response().status for s in states} == {"ok", "no_evidence", "error"}


def test_degraded_reasons_are_human_readable() -> None:
    """机器码进 `degraded`，中文说明进 `degraded_reasons`（文案唯一出处 log_messages）。"""
    body = _ask(_pipeline(rag=_rag(retriever=StubRetriever(boom=True)))).to_response()
    assert "retrieve_failed" in body.degraded
    assert "知识库检索异常，本次没有拿到证据" in body.degraded_reasons
    # 中英文不能混：每一条 provenances 都要能在文案表里找到
    assert all(reason and reason != "retrieve_failed" for reason in body.degraded_reasons)


def test_stages_carry_plan_and_evidence_gate() -> None:
    """§11.4 阶段遥测：`plan` 与 `evidence_gate` 必须出现在 stages 里，不能只有 recall/rerank。"""
    state = _ask(_pipeline(rag=_rag(_passed())))
    assert {"plan", "evidence_gate"} <= set(state.stages)
    # retriever 报的阶段原样透传；`fusion` 由 Milvus 内部完成，如实是 0
    assert state.stages["fusion"] == 0
    assert state.stages["rerank"] == 30


def test_generation_and_citation_latency_are_recorded() -> None:
    state = _ask(_pipeline(rag=_rag(_passed())))
    assert {"faq", "generate", "citation", "total"} <= set(state.latency_ms)


# ---------------------------------------------------------------------------
# HTTP：未预期异常也要回统一结构（而不是 FastAPI 的纯文本错误页）
# ---------------------------------------------------------------------------
def _app(pipeline: ChatPipeline) -> FastAPI:
    app = FastAPI()
    app.include_router(chat_router)
    app.state.chat = pipeline
    return app


def _call(app: FastAPI, payload: dict, *, headers: dict[str, str] | None = None) -> httpx.Response:
    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/v1/chat", json=payload, headers=headers or {})

    return asyncio.run(run())


def test_http_unexpected_exception_returns_unified_500(monkeypatch) -> None:
    pipeline = _pipeline(hit=_high_hit())

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise TypeError("'NoneType' object is not subscriptable")

    monkeypatch.setattr(pipeline, "ask", boom)
    resp = _call(_app(pipeline), {"query": "学费多少"})

    assert resp.status_code == 500
    body = resp.json()
    assert set(body) == RESPONSE_KEYS          # 与 200 同一个形状
    assert body["status"] == "error"
    assert body["error_code"] == "internal_error"
    assert body["request_id"]                   # 唯一能把用户报障对上日志的线索
    assert "400-000-0000" in body["answer"]     # 话术与编排内兜底同一出处
    assert body["degraded_reasons"] == ["服务内部异常，本次没有完成回答"]


def test_http_ok_and_500_bodies_are_key_compatible() -> None:
    ok_body = _call(_app(_pipeline(hit=_high_hit())), {"query": "学费多少"}).json()
    assert set(ok_body) == RESPONSE_KEYS
    assert ok_body["status"] == "ok"
