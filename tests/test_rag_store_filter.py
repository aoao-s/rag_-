"""过滤表达式必须**真的到达 Milvus 请求**，而不只是拼得对。

2026-10-29 真实故障：`GET /v1/rag/search?q=新加坡是一个什么样的国家&subject=地理`
返回的六条证据里有一条地理、四条道德与法治、一条历史。整条通路（规划 → `QueryContext.filters()`
→ `QueryTask.filters` → `RagPipeline._spawn` → `Retriever.search`）逐层查都没有问题，
`_build_expr()` 也能拼出 `subject == "地理"`——问题出在最后一跳：`MilvusClient.hybrid_search`
**没有 `filter` 参数**，`filter=` 落进 `**kwargs` 后被静默丢弃，连
`subject == "不存在的学科"` 都照样返回结果。

所以这个文件断言的不是字符串内容，而是**请求对象上有没有 expr**：
`AnnSearchRequest.expr` 是唯一生效的入口，dense 与 sparse 必须各持同一份。
只断言 `_build_expr()` 的输出（`test_evidence_gate_table.py` 原先只做到这一步）
正是这个 bug 能上线的覆盖缺口。
"""

from __future__ import annotations

from typing import Any
import sys
from types import SimpleNamespace

import pytest

from base.config import Settings
from base.contracts import RetrievalFilters
from rag_system.store.milvus_store import DOC_TYPE_PARTITIONS, ChunkStore

SUBJECT_EXPR = 'subject == "地理"'


class _FakeAnnSearchRequest:
    def __init__(self, *, data, anns_field, param, limit, expr=None):
        self.data = data
        self.anns_field = anns_field
        self.param = param
        self.limit = limit
        self.expr = expr


class _FakeWeightedRanker:
    def __init__(self, *weights):
        self.weights = weights


@pytest.fixture(autouse=True)
def _stub_pymilvus(monkeypatch):
    """请求构造测试不依赖本机是否安装 pymilvus，也不连接真实 Milvus。"""
    monkeypatch.setitem(
        sys.modules,
        "pymilvus",
        SimpleNamespace(
            AnnSearchRequest=_FakeAnnSearchRequest,
            WeightedRanker=_FakeWeightedRanker,
        ),
    )


class _FakeClient:
    """拦住 `hybrid_search`，把真正发出去的请求原样记下来。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def hybrid_search(self, **kwargs: Any) -> list[list[Any]]:
        self.calls.append(kwargs)
        return [[]]  # 没有命中，本文件只关心请求形状


def _store() -> tuple[ChunkStore, _FakeClient]:
    # 不走 `__init__`：那里会真的连 Milvus。这里只要一个能读 `_client` / `_partitions` 的壳。
    store = object.__new__(ChunkStore)
    client = _FakeClient()
    store.settings = Settings(milvus_collection="edurag_chunk")  # type: ignore[attr-defined]
    store._client = client  # type: ignore[attr-defined]
    store._partitions = list(DOC_TYPE_PARTITIONS)  # type: ignore[attr-defined]
    return store, client


def _filters(**kwargs: Any) -> RetrievalFilters:
    base: dict[str, Any] = dict(tenant_id="t1", acl_tags=("acl-a",), kb_version="v1")
    base.update(kwargs)
    return RetrievalFilters(**base)


def _exprs(call: dict[str, Any]) -> list[str | None]:
    return [getattr(req, "expr", None) for req in call["reqs"]]


# ---- 回归：表达式必须落在每路请求上 ----------------------------------------


def test_subject_filter_reaches_both_ann_requests() -> None:
    store, client = _store()

    store.search([0.1, 0.2], {1: 0.5}, subject="地理")

    (call,) = client.calls
    exprs = _exprs(call)
    assert len(exprs) == 2                      # dense + sparse
    assert all(e and SUBJECT_EXPR in e for e in exprs)


def test_filter_kwarg_is_not_used_because_it_is_silently_dropped() -> None:
    """`hybrid_search(filter=...)` 在 pymilvus 3.x 会被 `**kwargs` 吞掉。

    这条断言是防线本身：只要有人把过滤改回顶层 kwarg，测试立刻红——
    它不会报错、不会告警，只会让租户 / ACL / 学科过滤集体失效。
    """
    store, client = _store()

    store.search([0.1, 0.2], {1: 0.5}, filters=_filters(subjects=("地理",)))

    (call,) = client.calls
    assert "filter" not in call
    assert all(SUBJECT_EXPR in (e or "") for e in _exprs(call))


def test_dense_and_sparse_share_the_same_expr() -> None:
    """两路给不同（或只给一路）expr 时，另一路仍全库召回，0.3 的权重足以带进脏块。"""
    store, client = _store()

    store.search([0.1], {1: 0.5}, filters=_filters(subjects=("地理",), doc_types=("textbook",)))

    (call,) = client.calls
    dense_expr, sparse_expr = _exprs(call)
    assert dense_expr == sparse_expr
    assert dense_expr is not None
    assert 'doc_type in ["textbook"]' in dense_expr


# ---- 安全三项与业务项都必须进表达式 ----------------------------------------


def test_security_filters_are_in_the_expr_too() -> None:
    """学科只是最显眼的那一项；租户 / ACL / 版本走的是同一根管道，一起失效才是真事故。"""
    store, client = _store()

    store.search([0.1], {1: 0.5}, filters=_filters(subjects=("地理",)))

    (call,) = client.calls
    expr = _exprs(call)[0] or ""
    assert 'tenant_id == "t1"' in expr
    assert 'kb_version == "v1"' in expr
    assert "array_contains_any" in expr
    assert SUBJECT_EXPR in expr


def test_grade_and_stage_reach_both_ann_requests() -> None:
    """年级/学段是 2026-10-30 新接上的业务过滤，和学科走同一根管道，同样要两路都带。

    单值走 `==`、多值走 `in [...]`（`_enum_conds`）：这里断言的是实际拼法，
    不是"看起来应该长这样"——两者不一致时这条测试就是唯一能发现的地方。
    """
    store, client = _store()

    store.search([0.1], {1: 0.5}, filters=_filters(stages=("初中",), grades=("八年级",)))

    (call,) = client.calls
    exprs = _exprs(call)
    assert len(exprs) == 2
    assert all(e and 'stage == "初中"' in e and 'grade == "八年级"' in e for e in exprs)


def test_relaxed_drops_business_filters_but_keeps_the_security_trio() -> None:
    """空结果时的自动放宽只能清业务项：学科/年级放宽是修 bug，租户放宽是事故。"""
    store, client = _store()
    filters = _filters(subjects=("地理",), doc_types=("textbook",),
                       stages=("初中",), grades=("八年级",))

    store.search([0.1], {1: 0.5}, filters=filters.relaxed())

    (call,) = client.calls
    expr = _exprs(call)[0] or ""
    assert 'subject ==' not in expr
    assert "doc_type" not in expr
    assert "stage ==" not in expr
    assert "grade ==" not in expr
    assert 'tenant_id == "t1"' in expr
    assert 'kb_version == "v1"' in expr
    assert "array_contains_any" in expr


def test_no_filters_means_no_expr_not_an_empty_string() -> None:
    """不表态时 expr 要是 `None`：空串在部分 pymilvus 版本上会被当成非法表达式。"""
    store, client = _store()

    store.search([0.1], {1: 0.5})

    (call,) = client.calls
    assert _exprs(call) == [None, None]


# ---- `filters` 优先，便捷入口同样有效 --------------------------------------


def test_legacy_subject_entry_point_still_filters() -> None:
    """诊断接口与脚本走 `subject=` / `doc_types=`；它们同样必须真的过滤。"""
    store, client = _store()

    store.search([0.1], {1: 0.5}, subject="地理", doc_types=("textbook",))

    (call,) = client.calls
    expr = _exprs(call)[0] or ""
    assert SUBJECT_EXPR in expr
    assert 'doc_type in ["textbook"]' in expr


def test_filters_win_over_legacy_kwargs() -> None:
    """同时给时以 `filters` 为准，绝不合并——两处过滤谁生效说不清就没法排障。"""
    store, client = _store()

    store.search(
        [0.1],
        {1: 0.5},
        subject="历史",                              # 应被忽略
        filters=_filters(subjects=("地理",)),
    )

    (call,) = client.calls
    expr = _exprs(call)[0] or ""
    assert SUBJECT_EXPR in expr
    assert 'subject == "历史"' not in expr


def test_all_dense_sparse_security_and_business_filters_are_identical() -> None:
    """Dense/Sparse 两路都必须带齐 Tenant、ACL、KB Version 与业务过滤。"""
    store, client = _store()
    filters = _filters(
        subjects=("地理", "历史"),
        doc_types=("textbook", "lesson_plan"),
        stages=("初中",),
        grades=("八年级", "九年级"),
    )

    store.search([0.1], {1: 0.5}, filters=filters)

    (call,) = client.calls
    dense_expr, sparse_expr = _exprs(call)
    assert dense_expr == sparse_expr
    assert dense_expr is not None
    for expected in (
        'tenant_id == "t1"',
        'kb_version == "v1"',
        "array_contains_any",
        'subject in ["地理", "历史"]',
        'doc_type in ["textbook", "lesson_plan"]',
        'stage == "初中"',
        'grade in ["八年级", "九年级"]',
    ):
        assert expected in dense_expr


def test_empty_acl_tags_only_retrieve_public_documents() -> None:
    """用户没有 ACL 标签时，只允许无标签公共资料，不能省略 ACL 过滤。"""
    store, client = _store()

    store.search([0.1], {1: 0.5}, filters=_filters(acl_tags=()))

    (call,) = client.calls
    exprs = _exprs(call)
    assert len(exprs) == 2
    assert exprs[0] == exprs[1]
    assert "array_length(acl_tags) == 0" in (exprs[0] or "")
    assert "array_contains_any" not in (exprs[0] or "")
