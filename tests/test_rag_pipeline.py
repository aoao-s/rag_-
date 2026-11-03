"""检索流水线（CLAUDE.md §7.2.7-B/C/D）：计划 → 召回 → 融合 → 证据门 → 至多一次修复。

这一层要证明的是**状态迁移**，不是检索质量：

- 门说"为什么不够"，pipeline 决定"要不要再试、怎么试"，且**至多一次**；
- 修完必须**重新过门**，不许"改了过滤就当过了"；
- 修复只能放宽业务过滤，`tenant_id / acl_tags / kb_version` 一次都不能变；
- 分阶段耗时是**实测**的，并行任务取最大值而不是相加。

真链路（Milvus + BGE-M3）的质量由黄金集评估负责，这里全部用替身。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from base.config import Settings
from base.contracts import Evidence, EvidenceDecision, QueryContext, RetrievalFilters, RetrievalResult
from base.errors import RetrievalUnavailable
from rag_system.core.planner import build_plan
from rag_system.core.rag_pipeline import PipelineOutcome, RagPipeline

QUERY = "二次函数的顶点式怎么求"
PASS_TEXT = "二次函数 y=ax²+bx+c 可通过配方法化为顶点式 y=a(x-h)²+k。"


def _settings(**kwargs: Any) -> Settings:
    base = dict(
        # Settings 使用大写 Field alias；populate_by_name 未开启，测试也必须按 alias 传值。
        RAG_EVIDENCE_THRESHOLD=0.5,
        RAG_EVIDENCE_MARGIN=0.03,
        RAG_EVIDENCE_MIN_DOCS=1,
        RAG_EVIDENCE_QCOV=0.0,
    )
    base.update(kwargs)
    return Settings(**base)


def _ctx(**kwargs: Any) -> QueryContext:
    base = dict(request_id="req-1", tenant_id="t1", acl_tags=("acl-a",), kb_version="v7")
    base.update(kwargs)
    return QueryContext(**base)


def _ev(parent_id: str, score: float, *, task_id: str = "") -> Evidence:
    return Evidence(
        parent_id=parent_id,
        text=PASS_TEXT,
        score=score,
        child_hits=["二次函数 y=ax²+bx+c 可通过配方法化为顶点式"],
        title="人教版-数学-九年级上册",
        source="F:/docs/math.pdf",
        subject="数学",
        doc_type="textbook",
        heading="22.1 二次函数",
        page=12,
        task_id=task_id,
    )


class ScriptedRetriever:
    """按调用次序返回预设结果；把每次收到的过滤条件记下来供断言。"""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls.append({"query": query, **kwargs})
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        evidences, stages = step if isinstance(step, tuple) else (step, {})
        return RetrievalResult(
            query=query,
            evidences=list(evidences),
            raw_hits=len(evidences),
            cost_ms=7,
            stages=dict(stages),
        )


def _run(pipeline: RagPipeline, ctx: QueryContext, **kwargs: Any) -> PipelineOutcome:
    return asyncio.run(pipeline.run(QUERY, ctx, **kwargs))


# ---- 状态迁移 ----
def test_relax_business_filter_repairs_and_rees_gate() -> None:
    """第一轮被业务过滤挡空 → 放宽一次 → **重新过门** → 通过。"""
    retriever = ScriptedRetriever([[], [_ev("docA:p1", 0.90), _ev("docB:p1", 0.50)]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx(), subjects=("数学",))

    assert outcome.passed and outcome.decision.status == "sufficient"
    assert len(retriever.calls) == 2          # 恰好一次修复
    assert "retry:relax_business_filter" in outcome.degraded
    assert outcome.attempts == 1

    first, second = retriever.calls
    assert first["filters"].subjects == ("数学",)  # 第一次按业务过滤查
    assert second["filters"].subjects == ()        # 第二次放宽
    assert second["filters"].doc_types == ()


def test_security_filters_survive_relaxation() -> None:
    """放宽只碰业务过滤；租户 / ACL / 索引版本一次都不能变——这是安全边界。"""
    retriever = ScriptedRetriever([[], [_ev("docA:p1", 0.9), _ev("docB:p1", 0.4)]])
    pipeline = RagPipeline(_settings(), retriever)

    _run(pipeline, _ctx(), subjects=("数学",))

    original = _ctx().filters(subjects=("数学",))
    for call in retriever.calls:
        f: RetrievalFilters = call["filters"]
        assert f.tenant_id == original.tenant_id == "t1"
        assert f.acl_tags == original.acl_tags == ("acl-a",)
        assert f.kb_version == original.kb_version == "v7"


def test_retry_original_query_is_skipped_when_original_was_already_retrieved() -> None:
    """专门覆盖理解层失败后回退原问句的路径：原问句已检索就不再重复。"""
    retriever = ScriptedRetriever([[]])

    def force_retry_original(result, **kwargs):
        return EvidenceDecision(
            status="insufficient",
            reasons=("below_threshold",),
            retry_action="retry_original_query",
        )

    pipeline = RagPipeline(_settings(), retriever, gate=force_retry_original)
    outcome = asyncio.run(
        pipeline.run(QUERY, _ctx(original_query=QUERY))
    )

    assert len(retriever.calls) == 1
    assert retriever.calls[0]["query"] == QUERY
    assert "retry_skipped_duplicate_query" in outcome.degraded
    assert outcome.decision.retry_action == "fallback"
    assert outcome.decision.retry_count == 0


def test_identical_retry_query_is_skipped_after_first_retrieval() -> None:
    """原始 DIRECT 问句已按相同过滤与 top_k 检索，retry_direct 不应再跑一遍。"""
    retriever = ScriptedRetriever([[]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx())

    assert len(retriever.calls) == 1
    assert "retry_skipped_duplicate_query" in outcome.degraded
    assert outcome.decision.retry_action == "fallback"


def test_repair_happens_at_most_once() -> None:
    """修一次还是不够 → 收敛成 fallback，绝不循环重试。"""
    retriever = ScriptedRetriever([[]])  # 永远空
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx(), subjects=("数学",))

    assert len(retriever.calls) == 2                       # 初次 + 一次修复
    assert outcome.decision.passed is False
    assert outcome.decision.retry_count == 1
    assert outcome.decision.retry_action == "fallback"     # 不许再让人以为"还能试"
    assert "retry_exhausted" in outcome.degraded


def test_non_repairable_action_is_passed_up_untouched() -> None:
    """歧义要追问，不是重试——动作原样交给编排层。"""
    retriever = ScriptedRetriever([[_ev("docA:p1", 0.60), _ev("docB:p1", 0.59)]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx())

    assert outcome.decision.status == "ambiguous"
    assert outcome.decision.retry_action == "ask_disambiguate"
    assert outcome.decision.retry_count == 0
    assert len(retriever.calls) == 1
    assert "retry_exhausted" not in outcome.degraded


def test_max_repairs_zero_does_not_claim_retry_exhausted() -> None:
    """没试过的修复不能报成"重试耗尽"——否则报告里的重试收益就是假数。"""
    retriever = ScriptedRetriever([[]])
    pipeline = RagPipeline(_settings(), retriever, max_repairs=0)

    outcome = _run(pipeline, _ctx(), subjects=("数学",))

    assert len(retriever.calls) == 1
    assert outcome.decision.retry_action == "relax_business_filter"
    assert "retry_exhausted" not in outcome.degraded


def test_budget_exhaustion_skips_repair() -> None:
    retriever = ScriptedRetriever([[]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx(budget_ms=0), subjects=("数学",))

    assert len(retriever.calls) == 1
    assert "budget_exceeded" in outcome.degraded


# ---- 策略 ----
def test_backtrack_without_history_asks_instead_of_searching() -> None:
    """指代且无历史：检索拿原句必然召回不到，追问比乱检便宜（省掉整条链路）。"""
    retriever = ScriptedRetriever([[_ev("docA:p1", 0.9)]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = asyncio.run(pipeline.run("它的公式是什么", _ctx()))

    assert retriever.calls == []
    assert outcome.decision.status == "ambiguous"
    assert outcome.decision.reasons == ("referential_without_context",)
    assert outcome.decision.retry_action == "ask_disambiguate"
    assert outcome.attempts == 0
    assert "backtrack_without_history" in outcome.degraded


def test_backtrack_with_history_does_search() -> None:
    retriever = ScriptedRetriever([[_ev("docA:p1", 0.9), _ev("docB:p1", 0.4)]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = asyncio.run(pipeline.run("它的公式是什么", _ctx(), history_present=True))

    assert len(retriever.calls) == 1
    assert outcome.plan is not None and outcome.plan.strategy.value == "direct"


def test_sub_query_runs_one_task_per_query() -> None:
    retriever = ScriptedRetriever([[_ev("docA:p1", 0.9, task_id="t1"), _ev("docB:p1", 0.4, task_id="t2")]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx(), sub_queries=("顶点式怎么求", "一般式怎么化顶点式"))

    assert outcome.plan is not None and outcome.plan.strategy.value == "sub_query"
    assert len(retriever.calls) == 2
    assert outcome.result.tasks == ("t1", "t2")


# ---- 失败语义 ----
def test_all_tasks_failing_is_retrieval_unavailable() -> None:
    """"检索系统坏了"与"知识库里没有"必须分得开：前者抛异常，后者是 insufficient。"""
    retriever = ScriptedRetriever([RuntimeError("milvus down")])
    pipeline = RagPipeline(_settings(), retriever)

    with pytest.raises(RetrievalUnavailable) as exc:
        _run(pipeline, _ctx())

    assert exc.value.code == "retrieval_unavailable"


def test_partial_task_failure_degrades_instead_of_aborting() -> None:
    """一路失败不影响其余路：降级继续，比整体失败更接近"部分可用"。"""

    class FlakyRetriever(ScriptedRetriever):
        async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
            if "一般式" in query:
                raise RuntimeError("sparse index timeout")
            return await super().search(query, **kwargs)

    retriever = FlakyRetriever([[_ev("docA:p1", 0.9, task_id="t1"), _ev("docB:p1", 0.4, task_id="t2")]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx(), sub_queries=("顶点式怎么求", "一般式怎么化顶点式"))

    assert "partial_recall_failed" in outcome.degraded
    assert outcome.passed is True


def test_evidence_builder_dedupes_same_parent_across_tasks() -> None:
    retriever = ScriptedRetriever(
        [
            [
                _ev("docA:p1", 0.9, task_id="t1"),
                _ev("docA:p1", 0.8, task_id="t2"),
                _ev("docB:p1", 0.4, task_id="t2"),
            ]
        ]
    )
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx(), sub_queries=("顶点式怎么求", "顶点式怎么写"))

    assert [e.parent_id for e in outcome.result.evidences] == ["docA:p1", "docB:p1"]


# ---- 分阶段耗时 ----
def test_stage_timings_are_copied_from_real_measurements() -> None:
    """stages 必须是实测值：没有的阶段不编，并行阶段取最大值而不是相加。"""
    stages = {"query_embed": 12, "recall": 30, "parent_expand": 8, "fusion": 0, "rerank": 25, "total": 75}
    retriever = ScriptedRetriever([([ _ev("docA:p1", 0.9)], stages)])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx())

    for key in ("query_embed", "parent_expand", "fusion", "rerank", "recall", "total"):
        assert key in outcome.stages
    assert outcome.stages["fusion"] == 0     # Milvus 内部完成，如实透传 0
    assert outcome.stages["rerank"] == 25
    assert outcome.stages["total"] >= 0


def test_retry_stage_is_recorded_only_when_a_repair_ran() -> None:
    retriever = ScriptedRetriever([[], [_ev("docA:p1", 0.9), _ev("docB:p1", 0.4)]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx(), subjects=("数学",))

    assert "retry" in outcome.stages


def test_outcome_carries_the_plan_that_actually_ran() -> None:
    retriever = ScriptedRetriever([[], [_ev("docA:p1", 0.9), _ev("docB:p1", 0.4)]])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = _run(pipeline, _ctx(), subjects=("数学",))

    assert outcome.plan is not None
    assert outcome.plan.reason.startswith("repair:")   # 记录的是真正执行的那份计划
    assert outcome.plan.tasks[0].filters.subjects == ()


# ---- 重排仲裁与检索时间预算（2026-10-29 真实故障的回归）--------------------
class RerankAwareRetriever:
    """`rerank is not False` 就一直挂住，`rerank=False` 立刻返回。

    复刻本机的真实形态：CPU 重排 40 条候选实测 227 s，关掉重排同一批候选几十毫秒。
    """

    def __init__(self, evidences: list[Evidence]) -> None:
        self.evidences = evidences
        self.calls: list[dict[str, Any]] = []

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.calls.append({"query": query, **kwargs})
        if kwargs.get("rerank") is not False:
            await asyncio.sleep(3600)  # 永远不回
        return RetrievalResult(
            query=query, evidences=list(self.evidences), raw_hits=len(self.evidences), cost_ms=5
        )


def test_plan_rerank_reaches_the_retriever_as_unset() -> None:
    """计划里不确定重排，就一个字都别替它挡——`None` 必须原样到达 retriever。

    `retriever.search()` 只在 `rerank is None` 时读 `RAG_RERANK_ENABLED`；
    中间任何一层把它填成 `True`，`auto`/`.env` 就全失效（§5.6 CPU 跑重排）。
    """
    retriever = ScriptedRetriever([[_ev("docA:p1", 0.9), _ev("docB:p1", 0.5)]])
    pipeline = RagPipeline(_settings(), retriever)

    _run(pipeline, _ctx())

    assert retriever.calls[0]["rerank"] is None


def test_recall_timeout_drops_rerank_and_retries_instead_of_hanging() -> None:
    """超预算不许干等：关掉重排重检一次，答案照答，只记 `retrieval_timeout`。"""
    retriever = RerankAwareRetriever([_ev("docA:p1", 0.9), _ev("docB:p1", 0.5)])
    pipeline = RagPipeline(_settings(), retriever)

    outcome = asyncio.run(pipeline.run(QUERY, _ctx(), budget_ms=300))

    assert [e.parent_id for e in outcome.result.evidences] == ["docA:p1", "docB:p1"]
    assert "retrieval_timeout" in outcome.degraded
    assert [c["rerank"] for c in retriever.calls] == [None, False]  # 先按配置，再强制关


def test_recall_timeout_with_rerank_already_off_is_retrieval_unavailable() -> None:
    """重排都关了还超时，就是检索真不可用——`retrieval_unavailable`，不是"知识库没有"。"""

    class AlwaysHang:
        def __init__(self) -> None:
            self.calls = 0

        async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
            self.calls += 1
            await asyncio.sleep(3600)
            raise AssertionError("超时之后不该再走到这里")

    retriever = AlwaysHang()
    pipeline = RagPipeline(_settings(), retriever)
    plan = build_plan(QUERY, _ctx(), rerank=False)

    with pytest.raises(RetrievalUnavailable):
        asyncio.run(pipeline.run(QUERY, _ctx(), plan=plan, budget_ms=200))

    assert retriever.calls == 1  # 没有第二次白跑


def test_duplicate_repair_query_is_skipped_without_losing_first_result() -> None:
    """第一次已经执行相同 query/filter/top_k，不能再用同一检索签名空转。"""

    class EmptyThenHang:
        def __init__(self) -> None:
            self.calls = 0

        async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
            self.calls += 1
            if self.calls == 1:
                return RetrievalResult(query=query, evidences=[], raw_hits=0, cost_ms=3)
            await asyncio.sleep(3600)
            raise AssertionError("重复检索不应进入第二次调用")

    retriever = EmptyThenHang()
    pipeline = RagPipeline(_settings(), retriever)
    outcome = asyncio.run(pipeline.run(QUERY, _ctx(), budget_ms=300))

    assert retriever.calls == 1
    assert outcome.decision.status != "sufficient"
    assert "retry_skipped_duplicate_query" in outcome.degraded
    assert "retry_failed" not in outcome.degraded
    assert outcome.decision.retry_count == 0
