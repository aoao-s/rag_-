"""证据门判据表 + 过滤放宽的安全边界（CLAUDE.md §7.2.7-B 第 6 步 / §7.2.7-C）。

门的职责只有一条：**说清楚为什么不够**（`status` + `reasons` + `retry_action`）。
要不要再试、怎么试，是 `RagPipeline` 的事——这张表就是它的输入契约，
所以每条判据都要有一个确定性的落点，顺序也不能乱（阈值 → 候选数 → 冲突 → 歧义 → 覆盖度）。
"""

from __future__ import annotations

import pytest

from base.contracts import Evidence, RetrievalFilters, RetrievalResult
from rag_system.core.evidence_gate import evaluate, query_coverage

QUERY = "二次函数的顶点式怎么求"


def _ev(parent_id: str, score: float, *, text: str = "二次函数可通过配方法化为顶点式。", task_id: str = "") -> Evidence:
    return Evidence(parent_id=parent_id, text=text, score=score, task_id=task_id)


def _result(evidences: list[Evidence], *, tasks: tuple[str, ...] = ()) -> RetrievalResult:
    return RetrievalResult(query=QUERY, evidences=evidences, raw_hits=len(evidences), tasks=tasks)


def _filters(**kwargs) -> RetrievalFilters:
    base = dict(tenant_id="t1", acl_tags=("a1",), kb_version="v1")
    base.update(kwargs)
    return RetrievalFilters(**base)


# ---- 判据表 ----
@pytest.mark.parametrize(
    ("label", "evidences", "kwargs", "status", "reason", "action"),
    [
        ("没有候选且只收窄了学科", [], {"filters": _filters(subjects=("数学",))}, "insufficient", "subject_filter_empty", "relax_business_filter"),
        ("没有候选且只收窄了 doc_type", [], {"filters": _filters(doc_types=("exam",))}, "insufficient", "doc_type_too_narrow", "relax_business_filter"),
        ("没有候选且学科+doc_type 同时收窄", [], {"filters": _filters(subjects=("数学",), doc_types=("exam",))}, "insufficient", "no_candidates_with_filter", "relax_business_filter"),
        ("没有候选且只按学段过滤", [], {"filters": _filters(stages=("初中",))}, "insufficient", "no_candidates_with_filter", "relax_business_filter"),
        ("没有候选且没过滤", [], {}, "insufficient", "no_candidates", "retry_direct"),
        ("分数不够", [_ev("docA:p1", 0.40), _ev("docB:p1", 0.20)], {"threshold": 0.5}, "insufficient", "below_threshold", "retry_original_query"),
        ("孤证且偏弱", [_ev("docA:p1", 0.55)], {"threshold": 0.5, "single_doc_bonus": 0.10}, "insufficient", "single_weak_evidence", "expand_neighborhood"),
        ("跨文档分差不足", [_ev("docA:p1", 0.60), _ev("docB:p1", 0.59)], {"threshold": 0.5, "margin": 0.03}, "ambiguous", "ambiguous", "ask_disambiguate"),
        ("问句实词覆盖不足", [_ev("docA:p1", 0.60, text="勾股定理的证明方法。"), _ev("docB:p1", 0.20, text="勾股定理的证明方法。")], {"threshold": 0.5, "min_qcov": 0.5}, "insufficient", "low_query_coverage", "rewrite_query"),
        ("冲突值", [_ev("docA:p1", 0.90), _ev("docB:p1", 0.20)], {"threshold": 0.5, "conflicts": ("conflicting_values:π",)}, "conflicting", "conflicting_values:π", "return_conflict"),
        ("通过", [_ev("docA:p1", 0.90), _ev("docB:p1", 0.40)], {"threshold": 0.5}, "sufficient", "ok", "none"),
    ],
)
def test_gate_decision_table(label, evidences, kwargs, status, reason, action) -> None:
    decision = evaluate(_result(evidences), **kwargs)
    assert decision.status == status, label
    assert decision.reasons[0] == reason, label
    assert decision.retry_action == action, label
    assert decision.passed is (status == "sufficient"), label


def test_business_filter_is_suspected_before_the_knowledge_base() -> None:
    """空结果时先怀疑"过滤太窄"：放宽一次能救回来的答案，不该被记成"知识库没有"。"""
    restricted = evaluate(_result([]), filters=_filters(subjects=("数学",)))
    unrestricted = evaluate(_result([]), filters=_filters())

    assert restricted.retry_action == "relax_business_filter"
    assert unrestricted.retry_action == "retry_direct"
    assert restricted.reasons != unrestricted.reasons


def test_single_evidence_bonus_can_pass() -> None:
    """孤证不是死罪：分数明显高于阈值时放行（此时没有第二篇文档可比 margin）。"""
    decision = evaluate(_result([_ev("docA:p1", 0.95)]), threshold=0.5, single_doc_bonus=0.10)
    assert decision.status == "sufficient"


def test_threshold_is_checked_before_single_evidence() -> None:
    """顺序写死：分数不够就说分数不够，别再叠一个"孤证"，否则 `reasons` 会误导修复动作。"""
    decision = evaluate(_result([_ev("docA:p1", 0.30)]), threshold=0.5)
    assert decision.reasons == ("below_threshold",)


def test_missing_tasks_name_the_route_that_found_nothing() -> None:
    """SUB_QUERY 时要说清哪一路没找到，否则"召回不够"没法定位到子问句。"""
    result = _result([_ev("docA:p1", 0.30, task_id="t1")], tasks=("t1", "t2"))
    decision = evaluate(result, threshold=0.5)
    assert decision.status == "insufficient"
    assert decision.missing_tasks == ("t2",)


def test_missing_route_blocks_sufficient_even_when_top1_is_strong() -> None:
    """覆盖度看**所有必要任务**，不只看整体 top1（阶段 4.2）。

    旧契约是"过了门就把缺口清空"（当时 SUB_QUERY 只有一路，`missing_tasks` 仅作诊断）。
    真拆多路之后这条不再成立：三个子问题只答了一个、另两路一条证据都没有，
    按整体 top1 判 sufficient 等于拿一路的结果冒充整句的答案。
    现在缺路就是证据不全，交给 pipeline 用 `retry_direct`（拿整句原话再检一次）修一次，
    修不好才由编排层追问 / 兜底。
    """
    result = _result([_ev("docA:p1", 0.90, task_id="t1")], tasks=("t1", "t2"))
    decision = evaluate(result, threshold=0.5)
    assert decision.status == "insufficient"
    assert decision.reasons == ("missing_task_evidence",)
    assert decision.missing_tasks == ("t2",)
    assert decision.retry_action == "retry_direct"


# ---- 覆盖率 ----
def test_query_coverage_is_full_when_the_evidence_contains_the_question_terms() -> None:
    assert query_coverage("顶点式", "顶点式可由配方法得到") == 1.0


def test_query_coverage_is_zero_when_nothing_overlaps() -> None:
    assert query_coverage("勾股定理", "顶点式可由配方法得到") == 0.0


def test_partial_overlap_lands_between() -> None:
    """部分命中要落在 (0, 1) 之间——判据 4 的价值就在这里，不能退化成二值。"""
    assert 0.0 < query_coverage("二次函数 顶点式", "顶点式可由配方法得到") < 1.0


def test_query_coverage_of_pure_function_words_is_one() -> None:
    """判不了就不挡人：整句都是虚词时返回 1.0，而不是 0（0 会被当成"证据不相关"）。"""
    assert query_coverage("怎么办", "任意证据文本") == 1.0


# ---- 契约细节 ----
def test_with_retry_is_a_copy_not_a_mutation() -> None:
    original = evaluate(_result([]), filters=_filters(subjects=("数学",)))
    updated = original.with_retry("fallback", count=1)
    assert original.retry_action == "relax_business_filter"
    assert original.retry_count == 0
    assert updated.retry_action == "fallback"
    assert updated.retry_count == 1


def test_as_dict_exposes_both_single_and_multi_reason() -> None:
    body = evaluate(_result([]), filters=_filters(subjects=("数学",))).as_dict()
    assert body["reason"] == "subject_filter_empty"           # 旧响应体形状
    assert body["reasons"] == ["subject_filter_empty"]        # 判据细节
    assert body["passed"] is False


# ---- 过滤放宽：安全边界 ----
def test_relaxed_drops_only_business_filters() -> None:
    f = _filters(subjects=("数学",), doc_types=("textbook",))
    r = f.relaxed()
    assert r.subjects == ()
    assert r.doc_types == ()
    # 安全三项一次都不能动
    assert (r.tenant_id, r.acl_tags, r.kb_version) == ("t1", ("a1",), "v1")


def test_relaxed_is_idempotent() -> None:
    f = _filters(subjects=("数学",))
    assert f.relaxed().relaxed() == f.relaxed()


def test_business_restricted_reflects_only_subjects_and_doc_types() -> None:
    assert _filters().business_restricted is False
    # 只有安全三项时不算"业务过滤"——放宽它没有任何意义，也不能放宽
    assert _filters(tenant_id="t9", acl_tags=("x",), kb_version="v9").business_restricted is False
    assert _filters(subjects=("数学",)).business_restricted is True
    assert _filters(doc_types=("textbook",)).business_restricted is True


# ---- 学段 / 年级过滤（2026-10-29 新语料 `阶段/学科/年级/`）----
def test_relaxed_also_drops_stage_and_grade() -> None:
    """学段/年级属于业务过滤：放宽可以清，安全三项一次都不能动。"""
    f = _filters(stages=("初中",), grades=("九年级",))
    r = f.relaxed()
    assert r.stages == () and r.grades == ()
    assert (r.tenant_id, r.acl_tags, r.kb_version) == ("t1", ("a1",), "v1")


def test_business_restricted_covers_stage_and_grade() -> None:
    assert _filters(stages=("高中",)).business_restricted is True
    assert _filters(grades=("高一", "高二")).business_restricted is True


def test_build_expr_includes_stage_and_grade() -> None:
    from rag_system.store.milvus_store import ChunkStore

    expr = ChunkStore._build_expr(None, None, _filters(stages=("初中",), grades=("七年级", "八年级")))
    assert 'stage == "初中"' in expr  # 单值走 `==`
    assert 'grade in ["七年级", "八年级"]' in expr  # 多值走 `in`
    assert 'tenant_id == "t1"' in expr and 'kb_version == "v1"' in expr

    # 空值时不能凭空多出一个空 `in []`（Milvus 会直接报表达式错误）
    bare = ChunkStore._build_expr(None, None, _filters())
    assert "stage" not in bare and "grade" not in bare
