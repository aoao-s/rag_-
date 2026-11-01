"""策略规划（CLAUDE.md §7.2.7-B 第 5 步）：规则优先，档位只给先验。

规划只有 300 ms 预算，**不让 LLM 参与**：同一问句两次规划出不同策略，排障时无从下手。
所以每条规则都要有确定性的单测，尤其是"没有生成模型时不假装 HyDE"与
"意图只能收窄 doc_type、学科只从槽位来"这两条边界。
"""

from __future__ import annotations

from base.contracts import IntentProfile, QueryContext, Strategy
from rag_system.core.planner import build_plan, looks_multi, looks_referential


def _ctx(**kwargs) -> QueryContext:
    base = dict(request_id="r1", top_k=40, final_n=6)
    base.update(kwargs)
    return QueryContext(**base)


def _profile(**kwargs) -> IntentProfile:
    base = dict(
        name="kb_stepwise",
        retrieval=True,
        doc_types=(),
        strategy="direct",
        require_citation=True,
        allow_general=False,
        prompt_id="kb_v1",
        cacheable=True,
    )
    base.update(kwargs)
    return IntentProfile(**base)


# ---- 触发器本身 ----
def test_referential_and_multi_detectors() -> None:
    assert looks_referential("它的公式是什么")
    assert looks_referential("上面说的那一步")
    assert not looks_referential("二次函数的顶点式怎么求")
    assert looks_multi("勾股定理和余弦定理的区别")
    assert not looks_multi("二次函数的顶点式怎么求")


# ---- 四条策略 ----
def test_default_is_direct() -> None:
    plan = build_plan("二次函数的顶点式怎么求", _ctx())
    assert plan.strategy is Strategy.DIRECT
    assert plan.reason == "default"
    assert len(plan.tasks) == 1


def test_referential_without_history_becomes_backtrack() -> None:
    plan = build_plan("它的公式是什么", _ctx())
    assert plan.strategy is Strategy.BACKTRACK
    assert plan.tasks[0].note == "referential_without_history"
    assert plan.reason == "referential_without_history"


def test_referential_with_history_falls_through_to_direct() -> None:
    """有历史时理解层已经改写过；还在走回溯说明改写没生效，但不能因此重复追问。"""
    plan = build_plan("它的公式是什么", _ctx(), history_present=True)
    assert plan.strategy is Strategy.DIRECT


def test_sub_queries_are_capped_at_three() -> None:
    plan = build_plan(
        "顶点式",
        _ctx(),
        sub_queries=("顶点式怎么求", "一般式怎么化顶点式", "对称轴怎么求", "最值怎么求"),
    )
    assert plan.strategy is Strategy.SUB_QUERY
    assert len(plan.tasks) == 3
    assert [t.task_id for t in plan.tasks] == ["t1", "t2", "t3"]


def test_multi_connector_marks_strategy_even_with_single_task() -> None:
    """拆句需要理解层，规划层只标策略——不在这里拿正则硬拆，拆错比不拆更糟。"""
    plan = build_plan("勾股定理和余弦定理的区别", _ctx())
    assert plan.strategy is Strategy.SUB_QUERY
    assert len(plan.tasks) == 1
    assert plan.reason == "multi_intent"


def test_short_abstract_uses_hyde_only_when_a_model_is_available() -> None:
    with_model = build_plan("什么是熵", _ctx(), hyde_available=True)
    assert with_model.strategy is Strategy.HYDE

    without = build_plan("什么是熵", _ctx(), hyde_available=False)
    assert without.strategy is Strategy.DIRECT
    assert without.reason == "hyde_unavailable"
    assert without.tasks[0].note == "hyde_unavailable"


def test_long_abstract_question_is_not_hyde() -> None:
    plan = build_plan("什么是二次函数的顶点式怎么求", _ctx(), hyde_available=True)
    assert plan.strategy is Strategy.DIRECT


def test_profile_prior_sub_query_is_honoured() -> None:
    plan = build_plan("备课要点", _ctx(), profile=_profile(strategy="sub_query"))
    assert plan.strategy is Strategy.SUB_QUERY
    assert plan.reason == "profile_prior"


def test_profile_cannot_force_hyde() -> None:
    """HYDE / BACKTRACK 必须由硬规则触发；档位先验只认 DIRECT / SUB_QUERY。"""
    plan = build_plan("二次函数的顶点式怎么求", _ctx(), profile=_profile(strategy="hyde"))
    assert plan.strategy is Strategy.DIRECT


# ---- 过滤条件：意图只能收窄 doc_type ----
def test_profile_only_narrows_doc_type() -> None:
    plan = build_plan(
        "二次函数的顶点式怎么求",
        _ctx(subject="物理"),
        profile=_profile(doc_types=("textbook", "lesson_plan")),
        subjects=("数学",),
    )
    filters = plan.tasks[0].filters
    assert filters.doc_types == ("textbook", "lesson_plan")
    # 学科只从 SlotResult / subjects 来，档位与 ctx.subject 都不能抢这个位置
    assert filters.subjects == ("数学",)


def test_subject_falls_back_to_context_when_slot_missing() -> None:
    plan = build_plan("顶点式", _ctx(subject="数学"))
    assert plan.tasks[0].filters.subjects == ("数学",)


def test_security_filters_come_from_context_not_profile() -> None:
    plan = build_plan(
        "顶点式",
        _ctx(tenant_id="t9", acl_tags=("a1",), kb_version="v3"),
        profile=_profile(doc_types=("textbook",)),
    )
    f = plan.tasks[0].filters
    assert (f.tenant_id, f.acl_tags, f.kb_version) == ("t9", ("a1",), "v3")


def test_final_n_and_rerank_are_carried_through() -> None:
    plan = build_plan("顶点式", _ctx(final_n=6), final_n=4, rerank=False)
    assert plan.final_n == 4
    assert plan.rerank is False


def test_rerank_default_is_unset_not_true() -> None:
    """不传 rerank = **不表态**，由 `RAG_RERANK_ENABLED` 仲裁。

    2026-10-29 真实故障：这里原先默认 `True`，而 `retriever.search()` 只在
    `rerank is None` 时才看配置，于是 `RAG_RERANK_ENABLED=auto`（本机无 CUDA
    → 本该跳过）被彻底短路，40 条候选在 CPU 上重排 227 s，一次检索挂了 232 s。
    默认值写成 `True` 还会让 `/v1/rag/stats` 的 `rerank_enabled: false` 与日志
    里的 `reranked: True` 各说一套——两个数都"对"，坏的是仲裁链。
    """
    assert build_plan("顶点式", _ctx()).rerank is None
