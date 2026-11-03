from __future__ import annotations

import asyncio

from base.config import Settings
from base.contracts import (
    Evidence,
    QueryContext,
    QueryTask,
    RetrievalFilters,
    RetrievalPlan,
    RetrievalResult,
    Strategy,
    TaskCoverage,
    TaskEvidence,
)
from rag_system.core.evidence_gate import evaluate
from rag_system.core.prompts import build_messages
from rag_system.core.rag_pipeline import RagPipeline


def _filters() -> RetrievalFilters:
    return RetrievalFilters(tenant_id="tenant-a", kb_version="kb-v1")


def _evidence(parent: str, score: float, text: str = "这是一段与子问题相关的教材证据。") -> Evidence:
    return Evidence(
        parent_id=parent,
        text=text,
        score=score,
        title="数学教材",
        source="/docs/math.pdf",
        subject="数学",
        doc_type="textbook",
        heading="函数",
    )


def _task(task_id: str, query: str, evidence: list[Evidence], *, failed: bool = False) -> TaskEvidence:
    return TaskEvidence(
        task_id=task_id,
        query=query,
        filters=_filters(),
        evidences=evidence,
        failed=failed,
    )


def test_gate_uses_task_quality_not_only_task_hits() -> None:
    """两路都召回了，但第二路分数不足，整体只能部分覆盖。"""
    result = RetrievalResult(
        query="比较一次函数和二次函数",
        evidences=[_evidence("doc-a:p1", 0.93), _evidence("doc-b:p1", 0.22)],
        tasks=("t1", "t2"),
        task_hits=("t1", "t2"),
        task_results=(
            _task("t1", "一次函数的性质", [_evidence("doc-a:p1", 0.93)]),
            _task("t2", "二次函数的性质", [_evidence("doc-b:p1", 0.22)]),
        ),
    )

    decision = evaluate(
        result,
        threshold=0.5,
        margin=0.01,
        min_docs=1,
        single_doc_bonus=0.1,
        min_qcov=0.0,
    )

    assert decision.status == "partial"
    assert decision.passed is True
    assert decision.missing_tasks == ("t2",)
    assert decision.coverage_rate == 0.5
    assert [item.covered for item in decision.task_coverage] == [True, False]


def test_gate_requires_all_tasks_to_pass_for_full_coverage() -> None:
    result = RetrievalResult(
        query="比较一次函数和二次函数",
        evidences=[_evidence("doc-a:p1", 0.93), _evidence("doc-b:p1", 0.88)],
        tasks=("t1", "t2"),
        task_hits=("t1", "t2"),
        task_results=(
            _task("t1", "一次函数的性质", [_evidence("doc-a:p1", 0.93)]),
            _task("t2", "二次函数的性质", [_evidence("doc-b:p1", 0.88)]),
        ),
    )

    decision = evaluate(
        result, threshold=0.5, margin=0.01, min_docs=1,
        single_doc_bonus=0.1, min_qcov=0.0,
    )

    assert decision.status == "sufficient"
    assert decision.passed is True
    assert decision.missing_tasks == ()
    assert decision.coverage_rate == 1.0


class _TaskRetriever:
    async def search(self, query: str, **kwargs) -> RetrievalResult:
        if query == "一次函数的性质":
            evidence = [_evidence("doc-a:p1", 0.93)]
        else:
            # 有命中，但相关性不足，验证“有命中不等于有可用证据”。
            evidence = [_evidence("doc-b:p1", 0.22, "这段资料实际讲的是平行四边形面积。")]
        return RetrievalResult(query=query, evidences=evidence, raw_hits=1)


def test_pipeline_partial_coverage_only_keeps_supported_evidence() -> None:
    settings = Settings(
        RAG_EVIDENCE_THRESHOLD=0.5,
        RAG_EVIDENCE_MARGIN=0.01,
        RAG_EVIDENCE_MIN_DOCS=1,
        RAG_EVIDENCE_QCOV=0.0,
        RAG_RERANK_ENABLED="false",
    )
    pipeline = RagPipeline(settings, _TaskRetriever())
    filters = _filters()
    plan = RetrievalPlan(
        strategy=Strategy.SUB_QUERY,
        tasks=(
            QueryTask("t1", "一次函数的性质", Strategy.SUB_QUERY, filters=filters),
            QueryTask("t2", "二次函数的性质", Strategy.SUB_QUERY, filters=filters),
        ),
        final_n=6,
        rerank=False,
    )
    ctx = QueryContext(
        request_id="coverage-test",
        tenant_id="tenant-a",
        kb_version="kb-v1",
        original_query="比较一次函数和二次函数",
    )

    outcome = asyncio.run(
        pipeline.run("比较一次函数和二次函数", ctx, plan=plan)
    )

    assert outcome.decision.status == "partial"
    assert outcome.passed is True
    assert outcome.decision.missing_tasks == ("t2",)
    assert len(outcome.result.evidences) == 1
    assert outcome.result.evidences[0].parent_id == "doc-a:p1"
    assert len(outcome.result.task_results) == 2


def test_prompt_explicitly_marks_uncovered_subquestions() -> None:
    coverage = (
        TaskCoverage("t1", "一次函数的性质", "sufficient", covered=True),
        TaskCoverage("t2", "二次函数的性质", "insufficient", reasons=("below_threshold",)),
    )
    messages = build_messages(
        "比较一次函数和二次函数",
        [_evidence("doc-a:p1", 0.93)],
        task_coverage=coverage,
    )
    joined = "\n".join(message["content"] for message in messages)
    assert "部分覆盖回答" in joined
    assert "二次函数的性质" in joined
    assert "知识库暂无该部分资料" in joined
