"""槽位提取验收（任务 2.3）：模型槽位的校验 / 合并 / 收尾 / 追问，以及编排层的接线。

分两段：

1. **纯函数**（`understanding/slots.py`）：校验只丢不改、规则优先模型补空、
   `kind` 只来自意图、澄清只在"缺了就没法执行"时问；
2. **接线**（`orchestration/chat.py::_apply_slots` / `_slot_clarify`）：
   模型给的过滤类字段到不了检索、开关关掉时一次都不问、追问发生在检索之前。

安全口径（§7.1.8 的不对称）是这一份文件的主线：**模型写错学科是危险失效**，
所以 `ModelSlots` 在**类型上**就没有 `subject` / `grade` / `textbook_version` / `semester`——
这条约束用 `hasattr` 断言，而不是靠"我们记得在校验里丢掉它"。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import Any

from base.config import Settings
from base.contracts import (
    EvidenceDecision,
    FAQHit,
    IntentResult,
    PlanSlots,
    SlotResult,
    TaskSlots,
)
from orchestration.chat import ChatPipeline
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult
from understanding import slots as slot_fusion
from understanding.slots import ModelSlots

EVIDENCE_TEXT = "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k。"
_UNDERSTAND_MARK = "问题理解"
ANSWER = "顶点式可以通过配方法得到[1]。"
NORMALIZED = "二次函数的顶点式怎么求?"
QUERY = "二次函数的顶点式怎么求？"


# ================================================================ ① 校验
def test_parse_model_treats_absent_slots_as_normal() -> None:
    """没给槽位（`None`）是常态，不是降级。"""
    model, codes = slot_fusion.parse_model(None)
    assert codes == ()
    assert model.empty is True


def test_parse_model_rejects_non_dict_payload() -> None:
    model, codes = slot_fusion.parse_model(["chapter"])
    assert codes == ("slot_model_invalid",)
    assert model.empty is True


def test_model_slots_has_no_filter_fields_at_all() -> None:
    """类型级保证：过滤类槽位在 `ModelSlots` 上压根不存在，下游不可能误用。"""
    for name in ("subject", "subjects", "grade", "textbook_version", "semester"):
        assert not hasattr(ModelSlots(), name)


def test_parse_model_drops_filter_fields_but_keeps_the_rest() -> None:
    model, codes = slot_fusion.parse_model(
        {
            "subject": "数学",
            "grade": "八年级",
            "textbook_version": "人教版",
            "semester": "上册",
            "chapter": "一次函数",
        }
    )
    # 四个字段只记一次标记：日志里要能看出"模型在猜学科"，但不该刷四行
    assert codes == ("slot_model_filter_dropped",)
    assert model.chapter == "一次函数"


def test_parse_model_ignores_placeholder_filter_values() -> None:
    """`subject: "无"` 是"没给"，不是"模型越权"——不该报 `filter_dropped`。"""
    model, codes = slot_fusion.parse_model({"subject": "无", "chapter": "一次函数"})
    assert codes == ()
    assert model.chapter == "一次函数"


def test_parse_model_rejects_out_of_vocab_values_instead_of_guessing() -> None:
    model, codes = slot_fusion.parse_model({"difficulty": "有点难"})
    assert model.difficulty is None
    assert "slot_model_value_rejected" in codes


def test_parse_model_normalizes_vocab_values() -> None:
    model, codes = slot_fusion.parse_model({"difficulty": "基础", "exam_type": "期末"})
    assert model.difficulty == "easy"
    assert model.exam_type == "期末考试"
    assert codes == ()


def test_parse_model_treats_placeholders_as_absent() -> None:
    model, codes = slot_fusion.parse_model(
        {"chapter": "无", "knowledge_points": ["未知", "函数概念"]}
    )
    assert model.chapter is None
    assert model.knowledge_points == ("函数概念",)
    assert codes == ()


def test_parse_model_accepts_delimited_string_for_multi_value_slots() -> None:
    model, _ = slot_fusion.parse_model(
        {"knowledge_points": "函数概念、函数图像、函数概念"}
    )
    # 单串也收；去重保持出现顺序
    assert model.knowledge_points == ("函数概念", "函数图像")


def test_parse_model_clamps_task_numbers_and_keeps_tri_state_bool() -> None:
    model, _ = slot_fusion.parse_model(
        {
            "task": {
                "exam": {
                    "question_count": "1000",
                    "total_score": 120,
                    "difficulty_mix": "3：5：2",
                    "need_analysis": "不需要",
                }
            }
        }
    )
    exam = model.task.exam
    assert exam.question_count == 100  # 越界钳制，不是拒绝
    assert exam.total_score == 120
    assert exam.difficulty_mix == "3:5:2"  # 全角冒号归一
    assert exam.need_analysis is False


def test_parse_model_leaves_unstated_need_analysis_as_none() -> None:
    """三态：`None`（用户没表态）≠ `False`（用户说不要解析）。"""
    model, _ = slot_fusion.parse_model({"task": {"exam": {"question_count": 10}}})
    exam = model.task.exam
    assert exam.question_count == 10
    assert exam.need_analysis is None


def test_parse_model_never_reads_task_kind() -> None:
    """`kind` 是意图的结论，模型写了也不收。"""
    model, _ = slot_fusion.parse_model({"task": {"kind": "exam"}})
    assert model.task.kind is None


# ================================================================ ② 合并
def test_merge_rule_value_wins_over_model() -> None:
    rule = SlotResult(
        subject="数学",
        grade="八年级",
        chapter="第17章",
        knowledge_points=("勾股定理",),
        difficulty="easy",
    )
    model = ModelSlots(
        chapter="一次函数",
        knowledge_points=("函数概念", "函数图像"),
        difficulty="hard",
    )
    merged = slot_fusion.merge(rule, model)

    assert merged.subject == "数学" and merged.grade == "八年级"  # 过滤类根本不在模型侧
    assert merged.chapter == "第17章"
    # 多值槽位整体取舍，**不取并集**——并集之后分不清哪几个是用户写的
    assert merged.knowledge_points == ("勾股定理",)
    assert merged.difficulty == "easy"


def test_merge_fills_only_the_gaps() -> None:
    rule = SlotResult(subject="数学")
    model = ModelSlots(
        chapter="一次函数", knowledge_points=("函数概念",), difficulty="medium"
    )
    merged = slot_fusion.merge(rule, model)

    assert merged.subject == "数学"
    assert merged.chapter == "一次函数"
    assert merged.knowledge_points == ("函数概念",)
    assert merged.difficulty == "medium"


def test_merge_with_empty_model_returns_the_same_object() -> None:
    rule = SlotResult(subject="数学")
    assert slot_fusion.merge(rule, ModelSlots()) is rule


def test_merge_never_lets_the_model_write_the_task_kind() -> None:
    rule = SlotResult(task=TaskSlots(kind="lesson"))
    model = ModelSlots(task=TaskSlots(kind="exam"))
    assert slot_fusion.merge(rule, model).task.kind == "lesson"


def test_merge_fills_task_numbers_from_the_model() -> None:
    rule = SlotResult(task=TaskSlots())
    model = ModelSlots(task=TaskSlots(plan=PlanSlots(weekly_hours=6.5)))
    assert slot_fusion.merge(rule, model).task.plan.weekly_hours == 6.5


# ================================================================ ③ 收尾
def test_finalize_maps_intent_to_task_kind() -> None:
    slots = SlotResult()
    for label, kind in (
        ("教案制作", "lesson"),
        ("考试出题", "exam"),
        ("备课检索", "prep"),
        ("学习规划", "plan"),
    ):
        assert slot_fusion.finalize(slots, label).task.kind == kind
    # 讲解 / 闲聊 / 超出范围没有对应任务，`None` 是正确结论，不是"没跑"
    for label in ("学习讲解", "闲聊", "超出范围", None):
        assert slot_fusion.finalize(slots, label).task.kind is None


def test_finalize_projects_exam_type_into_plan_target_exam() -> None:
    done = slot_fusion.finalize(SlotResult(exam_type="中考"), "学习规划")
    assert done.task.kind == "plan"
    assert done.task.plan.target_exam == "中考"


def test_finalize_does_not_overwrite_an_explicit_target_exam() -> None:
    slots = SlotResult(
        exam_type="中考", task=TaskSlots(plan=PlanSlots(target_exam="高考"))
    )
    assert slot_fusion.finalize(slots, "学习规划").task.plan.target_exam == "高考"


def test_finalize_projects_coordinates_into_prep_slots() -> None:
    """备课检索的三个字段就是顶层槽位的投影：下游读一个位置就够了。"""
    slots = SlotResult(
        subject="数学", grade="八年级", knowledge_points=("勾股定理",)
    )
    done = slot_fusion.finalize(slots, "备课检索")
    assert done.task.prep.subject == "数学"
    assert done.task.prep.grade == "八年级"
    assert done.task.prep.knowledge_points == ("勾股定理",)


def test_finalize_is_idempotent() -> None:
    once = slot_fusion.finalize(SlotResult(subject="数学", exam_type="中考"), "学习规划")
    assert slot_fusion.finalize(once, "学习规划") is once


# ================================================================ ④ 澄清
def test_clarification_asks_only_when_the_task_cannot_run() -> None:
    lesson = SlotResult(task=TaskSlots(kind="lesson"))
    assert slot_fusion.clarification(lesson) == "要制作哪个章节或知识点的教案？"
    assert slot_fusion.clarification(replace(lesson, chapter="一次函数")) is None
    assert slot_fusion.clarification(replace(lesson, knowledge_points=("函数概念",))) is None

    exam = SlotResult(task=TaskSlots(kind="exam"))
    assert slot_fusion.clarification(exam) == "这套卷子考哪个学科或哪个知识点？"
    assert slot_fusion.clarification(replace(exam, subject="数学")) is None

    prep = SlotResult(task=TaskSlots(kind="prep"))
    assert slot_fusion.clarification(prep) == "想找哪个学科、哪个章节或知识点的备课资源？"
    assert slot_fusion.clarification(replace(prep, chapter="一次函数")) is None

    plan = SlotResult(task=TaskSlots(kind="plan"))
    assert slot_fusion.clarification(plan) == "这份学习规划针对哪个学科，目标考试是什么？"
    assert slot_fusion.clarification(replace(plan, subject="数学")) is None


def test_clarification_reads_the_plan_from_the_task_slots() -> None:
    """`weak_points` / `target_exam` 只活在 `task.plan` 里；顶层的 `exam_type` 也认。

    这条守的是一个真 bug：早先这里读的是 `slots.plan`，而 `SlotResult` 上没有这个字段，
    于是"学习规划 + 没提学科 + 开了澄清开关"会直接 `AttributeError`——不是答错，是 500。
    """
    bare = SlotResult(task=TaskSlots(kind="plan"))
    assert slot_fusion.clarification(bare) == "这份学习规划针对哪个学科，目标考试是什么？"
    # 顶层 exam_type 也算"说清了给谁规划"，不依赖 finalize 是否跑过
    assert slot_fusion.clarification(replace(bare, exam_type="中考")) is None
    assert slot_fusion.clarification(
        replace(bare, task=TaskSlots(kind="plan", plan=PlanSlots(target_exam="高考")))
    ) is None
    assert slot_fusion.clarification(
        replace(bare, task=TaskSlots(kind="plan", plan=PlanSlots(weak_points=("函数",))))
    ) is None


def test_clarification_silent_for_intents_without_a_task() -> None:
    assert slot_fusion.clarification(SlotResult()) is None


# ================================================================ ⑤ 接线
def _settings(**overrides: Any) -> Settings:
    base = dict(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        llm_router_model="router",
        chat_support_contact="400-000-0000",
        citation_min_support=0.30,
        total_budget_ms=8000,
        feature_intent=True,
        feature_intent_llm=True,
        feature_query_rewrite=True,
        feature_grade_filter=True,
        feature_slot_llm=True,
        feature_slot_clarify=False,
        understanding_timeout_ms=800,
        intent_bert_timeout_ms=200,
    )
    base.update(overrides)
    return Settings(**base)


class StubLLM:
    """按 system prompt 把"理解"与"生成"两次调用分开计数。"""

    def __init__(self, *, payload: dict[str, Any] | None = None) -> None:
        self.payload = json.dumps(
            payload
            if payload is not None
            else {"query": QUERY, "intent": "备课检索", "confidence": 0.9},
            ensure_ascii=False,
        )
        self.enabled = True
        self.understanding_calls = 0
        self.generation_calls = 0

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        if _UNDERSTAND_MARK in messages[0].get("content", ""):
            self.understanding_calls += 1
            text = self.payload
        else:
            self.generation_calls += 1
            text = ANSWER
        return LLMResult(text=text, model="stub", prompt_tokens=10, completion_tokens=5)


class StubBert:
    """弃权（`None`）：让大模型/规则接手，这里测的是槽位不是意图级联。"""

    async def classify(self, query: str, *, timeout_ms: int = 200) -> IntentResult | None:
        return None


class StubFAQ:
    async def search(
        self,
        query: str,
        subject: str | None = None,
        *,
        slots: SlotResult | None = None,
        selected_faq_id: int | None = None,
    ) -> FAQHit:
        return FAQHit(tier="low", reason="below_threshold", score=0.1)


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        self.queries.append(query)
        return _retrieval(query)


class StubIntent:
    def classify(self, query: str) -> IntentResult:
        return IntentResult(label="学习讲解", confidence=0.8, method="rule", profile="")


def _retrieval(query: str = QUERY) -> RetrievalResult:
    return RetrievalResult(
        query=query,
        evidences=[
            Evidence(
                parent_id="doc-1:p3",
                text=EVIDENCE_TEXT,
                score=0.91,
                rerank_score=0.9,
                child_hits=[EVIDENCE_TEXT[:40]],
                title="初中数学教材",
                source="F:/docs/教材.pdf",
                subject="数学",
                doc_type="textbook",
                heading="22.1 二次函数",
                page=12,
            )
        ],
        raw_hits=10,
        cost_ms=20,
    )


def _passed() -> EvidenceDecision:
    return EvidenceDecision(
        status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6
    )


def _pipeline(
    payload: dict[str, Any] | None = None, **settings_overrides: Any
) -> tuple[ChatPipeline, StubLLM, RecordingRetriever]:
    llm = StubLLM(payload=payload)
    retriever = RecordingRetriever()
    settings = _settings(**settings_overrides)
    pipeline = ChatPipeline(
        settings,
        faq_service=StubFAQ(),
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
        llm=llm,
        intent=StubIntent(),
        bert=StubBert(),
    )
    return pipeline, llm, retriever


def _ask(pipeline: ChatPipeline, query: str = QUERY) -> Any:
    return asyncio.run(pipeline.ask(query))


def test_wiring_no_model_slots_leaves_rule_slots_untouched() -> None:
    """`model_slots=None` 时行为与加这条链路之前一致，且不产生任何槽位降级码。"""
    pipeline, _llm, _retriever = _pipeline()
    state = _ask(pipeline)

    assert state.slots.chapter is None
    assert state.slots.task.kind == "prep"  # 意图"备课检索"的结论
    assert not [d for d in state.degraded if d.startswith("slot_")]


def test_wiring_model_slots_fill_the_gaps() -> None:
    pipeline, _llm, _retriever = _pipeline(
        {"query": QUERY, "intent": "备课检索", "confidence": 0.9, "slots": {"chapter": "一次函数"}}
    )
    state = _ask(pipeline)

    assert state.slots.chapter == "一次函数"
    assert state.slots.task.kind == "prep"
    # 备课检索的三个投影字段都用最终的顶层值（规则没抽到学科 → None，不是模型的猜测）
    assert state.slots.task.prep.subject is None
    assert state.answer  # 请求照常答完


def test_wiring_rule_slots_win_over_the_model() -> None:
    """规则抽到的章节不许被模型改写：规则认的是用户自己写的字。"""
    pipeline, _llm, _retriever = _pipeline(
        {"query": QUERY, "intent": "教案制作", "confidence": 0.9,
         "slots": {"chapter": "一次函数"}}
    )
    state = _ask(pipeline, "第十七章勾股定理的教案怎么写？")

    assert state.slots.chapter == "第十七章"


def test_wiring_model_subject_never_reaches_the_retrieval_scope() -> None:
    """模型给的学科被丢弃并计数，检索范围仍旧是规则结论（这里的结论是"没抽到"）。"""
    pipeline, _llm, retriever = _pipeline(
        {"query": QUERY, "intent": "备课检索", "confidence": 0.9,
         "slots": {"subject": "化学", "chapter": "一次函数"}}
    )
    state = _ask(pipeline)

    assert "slot_model_filter_dropped" in state.degraded
    assert state.slots.subject is None
    assert state.slots.subject_source == "unknown"
    assert state.slots.chapter == "一次函数"  # 只丢过滤类，其余照收
    assert retriever.queries  # 正常检索，没有被这次丢弃拦下


def test_wiring_out_of_vocab_value_is_counted() -> None:
    pipeline, _llm, _retriever = _pipeline(
        {"query": QUERY, "intent": "备课检索", "confidence": 0.9,
         "slots": {"difficulty": "有点难"}}
    )
    state = _ask(pipeline)

    assert "slot_model_value_rejected" in state.degraded
    assert state.slots.difficulty is None


def test_wiring_model_cannot_write_the_task_kind() -> None:
    """`kind` 是意图的结论：模型给 `exam`、意图是教案制作，最终仍是 `lesson`。"""
    pipeline, _llm, _retriever = _pipeline(
        {"query": QUERY, "intent": "教案制作", "confidence": 0.9,
         "slots": {"chapter": "一次函数", "task": {"kind": "exam"}}}
    )
    state = _ask(pipeline)

    assert state.slots.task.kind == "lesson"


def test_wiring_clarify_off_never_blocks_the_request() -> None:
    """默认关闭：缺课题也照常执行——拦不拦要拿真实流量判断，不在这里替产品拍板。"""
    pipeline, _llm, retriever = _pipeline(
        {"query": QUERY, "intent": "教案制作", "confidence": 0.9},
        feature_slot_clarify=False,
    )
    state = _ask(pipeline)

    assert state.source != "clarify"
    assert state.clarify is None
    assert "slot_clarify" not in state.degraded
    assert retriever.queries  # 真的去检索了


def test_wiring_clarify_on_asks_before_retrieving() -> None:
    pipeline, _llm, retriever = _pipeline(
        {"query": QUERY, "intent": "教案制作", "confidence": 0.9},
        feature_slot_clarify=True,
    )
    state = _ask(pipeline)

    assert state.source == "clarify"
    assert state.clarify == "要制作哪个章节或知识点的教案？"
    assert state.answer == state.clarify
    assert "slot_clarify" in state.degraded
    # 追问发生在检索之前：没执行就不该先花掉一次检索预算
    assert retriever.queries == []


def test_wiring_clarify_on_asks_for_the_missing_plan_subject() -> None:
    """学习规划这条分支曾经是 500：`clarification()` 读了不存在的 `slots.plan`。"""
    pipeline, _llm, retriever = _pipeline(
        {"query": QUERY, "intent": "学习规划", "confidence": 0.9},
        feature_slot_clarify=True,
    )
    state = _ask(pipeline)

    assert state.source == "clarify"
    assert state.clarify == "这份学习规划针对哪个学科，目标考试是什么？"
    assert retriever.queries == []


def test_wiring_clarify_on_stays_silent_when_the_slot_is_present() -> None:
    pipeline, _llm, retriever = _pipeline(
        {"query": QUERY, "intent": "教案制作", "confidence": 0.9,
         "slots": {"chapter": "一次函数"}},
        feature_slot_clarify=True,
    )
    state = _ask(pipeline)

    assert state.source != "clarify"
    assert "slot_clarify" not in state.degraded
    assert retriever.queries


def test_wiring_clarify_on_stays_silent_for_intents_without_a_task() -> None:
    """学习讲解没有任务，哪怕缺一堆槽位也不许追问。"""
    pipeline, _llm, _retriever = _pipeline(
        {"query": QUERY, "intent": "学习讲解", "confidence": 0.9},
        feature_slot_clarify=True,
    )
    state = _ask(pipeline)

    assert state.source != "clarify"
    assert state.slots.task.kind is None


def test_wiring_slot_llm_off_does_not_ask_for_slots() -> None:
    """`FEATURE_SLOT_LLM=false`：模型照旧给 `slots` 也不采纳（提示词里根本没要）。"""
    pipeline, llm, _retriever = _pipeline(
        {"query": QUERY, "intent": "备课检索", "confidence": 0.9,
         "slots": {"chapter": "一次函数"}},
        feature_slot_llm=False,
    )
    state = _ask(pipeline)

    assert llm.understanding_calls == 1  # 改写与意图还在同一次调用里
    assert state.model_slots is None
    assert state.slots.chapter is None
