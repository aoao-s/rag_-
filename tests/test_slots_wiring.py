"""槽位接线验收：问句里的学科 / 年级 / 学段是不是真的变成了检索过滤条件。

盯的仍然不是"抽得准不准"（那是 `tests/test_slots.py` 的事），而是**抽出来的东西有没有
抵达执行点**——观测点是 `RecordingRetriever.search(..., filters=...)` 收到的那个
`RetrievalFilters` 对象。方法论同 `tests/test_rag_store_filter.py`：断言字段到了请求对象，
不是断言它被赋给了某个中间变量。§7.2.7-G 的教训就是"字符串看着对、到不了请求"。

`feature_intent=False` 让档位恒为 `BASELINE`（不收窄 `doc_type`），这样本文件里的失败
只可能来自槽位本身，而不是意图把过滤条件搅混。`feature_intent_llm` / `feature_query_rewrite`
一并关掉：本文件不验理解层，也不该为一堆无关断言付一次 LLM 调用的语义成本。

多学科那条用的是 `"物理和数学的区别"`——两个**不同**学科。不要用
`"初二和初三的数学"`：那只有一个学科（数学），年级出现两次不构成 `multi`。
"""

from __future__ import annotations

import asyncio
from typing import Any

from base.config import Settings
from base.contracts import EvidenceDecision, FAQHit, RetrievalFilters, SlotResult
from orchestration.chat import ChatPipeline
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult

EVIDENCE_TEXT = "勾股定理：直角三角形两直角边的平方和等于斜边的平方。"
ANSWER = "勾股定理的内容是两条直角边的平方和等于斜边的平方[1]。"


# ---------------------------------------------------------------- 替身
def _settings(**overrides: Any) -> Settings:
    base = dict(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        chat_support_contact="400-000-0000",
        citation_min_support=0.30,
        total_budget_ms=8000,
        # 档位恒为 BASELINE：本文件只验槽位，不收窄 doc_type
        feature_intent=False,
        feature_intent_llm=False,
        feature_query_rewrite=False,
        feature_grade_filter=True,
    )
    base.update(overrides)
    return Settings(**base)


class StubFAQ:
    """记下 FAQ 收到的 `SlotResult`——槽位下推的第一站就是它。"""

    def __init__(self) -> None:
        self.slots: list[SlotResult | None] = []

    async def search(
        self,
        query: str,
        subject: str | None = None,
        *,
        slots: SlotResult | None = None,
        selected_faq_id: int | None = None,
    ) -> FAQHit:
        self.slots.append(slots)
        return FAQHit(tier="low", reason="below_threshold", score=0.1)


class RecordingRetriever:
    """记下**真正送去检索的过滤条件**——整份接线验收的核心观测点。"""

    def __init__(self) -> None:
        self.filters: list[RetrievalFilters] = []
        self.subjects: list[str | None] = []

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.filters.append(kwargs["filters"])
        self.subjects.append(kwargs.get("subject"))
        return _retrieval(query)


class StubLLM:
    def __init__(self, text: str = ANSWER) -> None:
        self.text = text

    @property
    def enabled(self) -> bool:
        return True

    async def complete(
        self, messages: list[dict], *, budget_ms: int, model: str | None = None
    ) -> Any:
        from rag_system.core.llm_gateway import LLMResult

        return LLMResult(text=self.text, model="stub", prompt_tokens=10, completion_tokens=5)


def _retrieval(query: str = "勾股定理怎么证明") -> RetrievalResult:
    return RetrievalResult(
        query=query,
        evidences=[
            Evidence(
                parent_id="doc-1:p3",
                text=EVIDENCE_TEXT,
                score=0.91,
                rerank_score=0.9,
                child_hits=[EVIDENCE_TEXT[:20]],
                title="初中数学教材",
                source="F:/docs/教材.pdf",
                subject="数学",
                doc_type="textbook",
                heading="第十七章 勾股定理",
                page=0,
            )
        ],
        raw_hits=8,
        cost_ms=15,
    )


def _passed() -> EvidenceDecision:
    return EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)


def _pipeline(**settings_overrides: Any) -> tuple[ChatPipeline, RecordingRetriever, StubFAQ]:
    retriever = RecordingRetriever()
    faq = StubFAQ()
    settings = _settings(**settings_overrides)
    pipeline = ChatPipeline(
        settings,
        faq_service=faq,
        rag={
            "ready": True,
            "retriever": retriever,
            "pipeline": RagPipeline(
                settings, retriever, gate=lambda result, **kw: _passed(), max_repairs=0
            ),
            "gate": None,
            "embedder_name": "stub",
            "rerank_enabled": False,
        },
        llm=StubLLM(),
    )
    return pipeline, retriever, faq


def _ask(pipeline: ChatPipeline, query: str, **kwargs: Any) -> Any:
    return asyncio.run(pipeline.ask(query, **kwargs))


# ---------------------------------------------------------------- ① 学科下推
def test_explicit_subject_reaches_the_retriever_filters() -> None:
    """用户点名了学科，检索就必须收窄到该学科——这是本轮接线的第一目的。"""
    pipeline, retriever, _faq = _pipeline()
    state = _ask(pipeline, "数学勾股定理怎么证明")
    assert state.slots.subject_source == "explicit"
    assert state.slots.subject == "数学"
    assert retriever.filters[0].subjects == ("数学",)


def test_slot_result_also_reaches_the_faq_service() -> None:
    """FAQ 是槽位的第一站：它按学科选索引范围，拿不到槽位就等于按全库找。"""
    pipeline, _retriever, faq = _pipeline()
    _ask(pipeline, "数学勾股定理怎么证明")
    assert faq.slots and faq.slots[0] is not None
    assert faq.slots[0].subject == "数学"
    assert faq.slots[0].subject_source == "explicit"


def test_no_subject_in_query_does_not_become_a_filter() -> None:
    """没提学科 ≠ 学科不明：这条不能退化成"过滤一个空学科"或猜一个学科。"""
    pipeline, retriever, _faq = _pipeline()
    state = _ask(pipeline, "勾股定理怎么证明")
    assert state.slots.subject_source == "unknown"
    assert retriever.filters[0].subjects == ()
    # 全库检索：`subject` 也不该被塞进单值通道（否则 `_build_expr` 会加一条等值条件）
    assert retriever.subjects == [None]


def test_multi_subject_query_pushes_all_subjects_down() -> None:
    """两个学科都要下推。只传 `subject`（=第一个）会让另一门的证据永远进不了候选。"""
    pipeline, retriever, _faq = _pipeline()
    state = _ask(pipeline, "物理和数学的区别")
    assert state.slots.subject_source == "multi"
    assert state.slots.subjects == ("物理", "数学")
    assert retriever.filters[0].subjects == ("物理", "数学")
    # 多学科走 `subject in [...]` 这一路，单值通道必须留空，否则会多一条等值条件
    assert retriever.subjects == [None]


def test_invalid_subject_param_is_not_overwritten_by_the_query() -> None:
    """参数写错要原样暴露（HTTP 已由 `test_chat_api` 判 400），不许被问句里的学科悄悄修好。"""
    pipeline, retriever, faq = _pipeline()
    state = _ask(pipeline, "数学勾股定理怎么证明", subject="天文学")
    assert state.slots.subject_source == "invalid"
    assert state.slots.subject is None
    # 参数错 → FAQ 侧走全库兜底，但必须留痕；检索侧不能因为问句里有"数学"就收窄
    assert faq.slots and faq.slots[0] is not None
    assert faq.slots[0].subject_source == "invalid"
    assert retriever.filters[0].subjects == ()


# ---------------------------------------------------------------- ② 年级 / 学段下推
def test_grade_and_stage_reach_the_retriever_filters() -> None:
    """年级要成对下推：只给年级不给学段，教材里的 `初中/高中` 分区照样能串。"""
    pipeline, retriever, _faq = _pipeline()
    state = _ask(pipeline, "八年级数学的勾股定理怎么证明")
    assert state.slots.grade == "八年级"
    assert retriever.filters[0].grades == ("八年级",)
    assert retriever.filters[0].stages == ("初中",)
    # 学科与年级是两件事，必须同时在场
    assert retriever.filters[0].subjects == ("数学",)


def test_grade_filter_can_be_rolled_back() -> None:
    """年级误抽（"八年级的题现在高一还考吗"）的一键回滚：关掉只丢年级，学科照旧。"""
    pipeline, retriever, _faq = _pipeline(feature_grade_filter=False)
    _ask(pipeline, "八年级数学的勾股定理怎么证明")
    assert retriever.filters[0].grades == ()
    assert retriever.filters[0].stages == ()
    assert retriever.filters[0].subjects == ("数学",)


# ---------------------------------------------------------------- ③ 安全边界
def test_business_slot_filters_do_not_shadow_the_security_trio() -> None:
    """学科/年级是业务过滤，租户/ACL/索引版本是安全过滤，必须在**同一个**请求对象上同时生效。

    这正是 §7.2.7-G 的教训：过滤条件只在"到达请求对象"时才算数，分成两个对象传就会出现
    "学科过滤生效了、租户过滤没生效"这种谁也看不出来的状态。
    """
    pipeline, retriever, _faq = _pipeline()
    _ask(pipeline, "八年级数学的勾股定理怎么证明")
    filters = retriever.filters[0]
    assert filters.tenant_id == "default"
    assert filters.kb_version == "v1"
    assert filters.acl_tags == ()
    # 三项业务过滤同时在场，证明它们没互相覆盖
    assert filters.business_restricted is True
    assert (filters.subjects, filters.grades, filters.stages) == (("数学",), ("八年级",), ("初中",))


def test_relaxing_a_dead_end_query_keeps_the_security_trio() -> None:
    """业务过滤可以放宽（§7.2.7-E），安全三件套一次都不许动。"""
    pipeline, retriever, _faq = _pipeline()
    _ask(pipeline, "八年级数学的勾股定理怎么证明")
    relaxed = retriever.filters[0].relaxed()
    assert relaxed.subjects == () and relaxed.grades == () and relaxed.stages == ()
    assert relaxed.tenant_id == retriever.filters[0].tenant_id
    assert relaxed.acl_tags == retriever.filters[0].acl_tags
    assert relaxed.kb_version == retriever.filters[0].kb_version
