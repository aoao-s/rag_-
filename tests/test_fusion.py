"""多路结果融合（CLAUDE.md §7.2.7-B）：同分空间合并，阈值才仍然可解释。

融合不做 RRF：证据门的 0.74 / 0.01 / 0.20 是在 **Milvus 融合分**空间标定的，
换成 `1/(k+rank)` 之后"分数够不够"就没法比了。这几条用例把"同分空间"钉住。
"""

from __future__ import annotations

from base.contracts import Evidence, RetrievalResult
from rag_system.core.fusion import fuse


def _ev(parent_id: str, score: float, *, task_id: str = "", child: str = "") -> Evidence:
    return Evidence(
        parent_id=parent_id,
        text=f"{parent_id} 的正文",
        score=score,
        child_hits=[child] if child else [],
        title=f"title-{parent_id}",
        source="F:/docs/a.pdf",
        page=3,
        task_id=task_id,
    )


def _res(evidences: list[Evidence]) -> RetrievalResult:
    return RetrievalResult(query="q", evidences=evidences, raw_hits=len(evidences))


def test_empty_input_is_empty() -> None:
    assert fuse([]) == []


def test_single_result_passes_through_untouched() -> None:
    """单路不做任何加工：分数要原样保留（它已经和阈值同空间）。"""
    ev = _ev("docA:p1", 0.82)
    out = fuse([_res([ev])])
    assert [e.parent_id for e in out] == ["docA:p1"]
    assert out[0].score == 0.82


def test_scores_are_scaled_by_task_weight_not_ranked() -> None:
    """多路时按 分数 × 权重 融合，**不是按名次**——这是"阈值仍可解释"的前提。"""
    a = _res([_ev("docA:p1", 0.90, task_id="t1")])
    b = _res([_ev("docB:p1", 0.90, task_id="t2")])
    out = fuse([a, b], weights=[1.0, 0.5])
    assert {e.parent_id: e.score for e in out} == {"docA:p1": 0.90, "docB:p1": 0.45}


def test_same_parent_is_deduped_and_child_hits_merged() -> None:
    a = _res([_ev("docA:p1", 0.60, task_id="t1", child="子块甲")])
    b = _res([_ev("docA:p1", 0.90, task_id="t2", child="子块乙")])
    out = fuse([a, b])
    assert len(out) == 1
    # 分数取最高，命中片段两路都留下（互相印证体现在 child_hits 变多）
    assert out[0].score == 0.90
    assert set(out[0].child_hits) == {"子块甲", "子块乙"}


def test_higher_score_keeps_its_metadata_but_not_the_merged_children() -> None:
    """主证据换成高分那条，但它不能吃掉另一路合进来的命中片段（否则丢一次印证）。"""
    low = _ev("docA:p1", 0.30, task_id="t1", child="低分的片段")
    low.heading = "第一节"
    high = _ev("docA:p1", 0.80, task_id="t2", child="高分的片段")
    high.heading = "第二节"
    out = fuse([_res([low]), _res([high])])
    assert out[0].heading == "第二节"          # 元数据跟着高分那条走
    assert set(out[0].child_hits) == {"低分的片段", "高分的片段"}


def test_tie_break_prefers_multi_route_support() -> None:
    """同分时"两路都找到"排在"一路偶然高分"前面。"""
    a = _res([_ev("docA:p1", 0.70, task_id="t1"), _ev("docB:p1", 0.70, task_id="t1")])
    b = _res([_ev("docB:p1", 0.70, task_id="t2")])
    out = fuse([a, b])
    assert [e.parent_id for e in out] == ["docB:p1", "docA:p1"]


def test_weights_length_mismatch_falls_back_to_uniform() -> None:
    """权重与路数对不上时按 1.0 处理——宁可退化成不加权，也不要错配权重。"""
    a = _res([_ev("docA:p1", 0.50, task_id="t1")])
    b = _res([_ev("docB:p1", 0.50, task_id="t2")])
    out = fuse([a, b], weights=[2.0])
    assert {e.parent_id: e.score for e in out} == {"docA:p1": 0.50, "docB:p1": 0.50}
