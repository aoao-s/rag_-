"""阶段 2 验收：Query Understanding（问题理解）。

验收标准（与任务书逐字一致）：

- 七类意图都有足够的训练和测试样本。
- 槽位能正确处理多值、缺失和冲突。
- 复合问题能够输出最多 3 个有效子问题。
- 指代问题能够使用真实会话历史。
- 理解失败能够安全降级。

**这份文件是"验收尺"，不是"实现说明"**：它按验收标准写成，失败或 xfail 的条目
就是当前代码与标准的差距。与 `tests/test_acceptance_stage1.py` 同一体例——
不依赖 MySQL / Milvus / Redis / 真实 LLM，只依赖仓库里真实的数据文件
（`datasets/06_intent/*.jsonl`）与真实的 `SessionStore`（Redis 缺省时走进程内回退，
那正是生产里"Redis 挂了"的同一条路径）。

两处刻意保持诚实，不为了"全绿"改断言：

1. **槽位"冲突"两端都验**：产出端是 `qa_slots.resolve()`（参数是合法学科、而问句点名的
   学科集合不含它 → `subject_source="conflict"`，学科清空走全库），消费端是 `FAQService`
   （拿到 `conflict` 按 §7.1.8 全库检索并封顶 mid，不直接作答）。2026-10-31 补上产出端之前
   这一条是 `xfail(strict=True)`——生产环境里"参数与问句矛盾"只会静默采信参数，用错的学科
   做硬过滤，正确证据被挡在门外且不留痕迹。
2. **"真实会话历史"用的是 `SessionStore` 本体**，不是替身。替身只能证明"接线接上了"，
   证明不了"窗口里真的读得到上一轮"。
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from base.config import Settings
from base.contracts import (
    EvidenceDecision,
    FAQHit,
    IntentResult,
    MemoryContext,
    SlotResult,
)
from base.labels import INTENT_LABELS
from memory.session import SessionStore
from orchestration.chat import ChatPipeline
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository
from qa_system.res.qa_matcher import MatchPolicy
from qa_system.res.qa_search import FAQService
from qa_system.res.qa_slots import extract, from_param, resolve
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult
from understanding import slots as uslots
from understanding.compound_query import MAX_SUB_QUERIES, split

DATA_DIR = Path(__file__).resolve().parents[1] / "datasets" / "06_intent"

EVIDENCE_TEXT = "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k。"
ANSWER = "顶点式可以通过配方法得到[1]。"
_UNDERSTAND_MARK = "问题理解"

# 任务 2.4 规格里的原例
SPEC = "比较一次函数和二次函数的区别，并分别给我出三道练习题。"
SPEC_SUBS = (
    "一次函数和二次函数的区别",
    "给我出三道一次函数练习题",
    "给我出三道二次函数练习题",
)


# ---------------------------------------------------------------- 替身
def _settings(**overrides: Any) -> Settings:
    base = dict(
        llm_api_key="sk-test",
        llm_qa_model="primary",
        llm_router_model="router",
        chat_support_contact="400-000-0000",
        citation_min_support=0.30,
        total_budget_ms=8000,
        feature_intent=True,
        # 本文件验的是验收标准本身，不是某次模型的运气：理解层默认走规则，
        # 需要验降级的用例自己把开关打开
        feature_intent_llm=False,
        feature_query_rewrite=False,
        feature_slot_llm=False,
        feature_planner_llm=False,
        feature_grade_filter=True,
        understanding_timeout_ms=800,
    )
    base.update(overrides)
    return Settings(**base)


class StubLLM:
    def __init__(
        self,
        *,
        payload: str | None = None,
        answer: str = ANSWER,
        enabled: bool = True,
    ) -> None:
        self.payload = (
            payload
            if payload is not None
            else json.dumps(
                {"query": "二次函数的顶点式怎么求？", "intent": "学习讲解", "confidence": 0.9},
                ensure_ascii=False,
            )
        )
        self.answer = answer
        self.enabled = enabled
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
            text = self.answer
        return LLMResult(text=text, model="stub", prompt_tokens=10, completion_tokens=5)


class ExplodingLLM(StubLLM):
    """理解这一次调用直接抛异常；生成照常，用来验"理解挂了答案还在"。"""

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        if _UNDERSTAND_MARK in messages[0].get("content", ""):
            self.understanding_calls += 1
            raise RuntimeError("upstream exploded")
        return await super().complete(messages, budget_ms=budget_ms, model=model)


class SlowLLM(StubLLM):
    """理解慢到必然超时（配合 `understanding_timeout_ms=1`）。"""

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        if _UNDERSTAND_MARK in messages[0].get("content", ""):
            await asyncio.sleep(0.05)
        return await super().complete(messages, budget_ms=budget_ms, model=model)


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
    def __init__(self) -> None:
        self.calls = 0

    def classify(self, query: str) -> IntentResult:
        self.calls += 1
        return IntentResult(label="学习讲解", confidence=0.8, method="rule", profile="kb_stepwise")


class _RealSessionMemory:
    """把真实的 `SessionStore` 挂进编排层；召回/回写退化成空实现。"""

    def __init__(self, store: SessionStore) -> None:
        self.session = store

    async def recall(
        self, user_id: str, query: str, *, sid: str | None = None, budget_ms: int | None = None
    ) -> MemoryContext:
        return MemoryContext()

    async def remember(
        self, user_id: str, sid: str, query: str, *, turn_index: int = 0
    ) -> list[Any]:
        return []


@dataclass
class _Turn:
    role: str
    content: str


def _retrieval(query: str = "") -> RetrievalResult:
    return RetrievalResult(
        query=query or "二次函数的顶点式怎么求",
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
    return EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)


def _pipeline(
    *,
    llm: StubLLM | None = None,
    retriever: RecordingRetriever | None = None,
    memory: Any = None,
    **settings_overrides: Any,
) -> tuple[ChatPipeline, StubLLM, RecordingRetriever]:
    llm = llm or StubLLM()
    retriever = retriever or RecordingRetriever()
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
        memory=memory,
    )
    return pipeline, llm, retriever


def _ask(pipeline: ChatPipeline, query: str, **kwargs: Any) -> Any:
    async def _run() -> Any:
        state = await pipeline.ask(query, **kwargs)
        for _ in range(10):  # `_persist_later` 是 fire-and-forget，等它落定再关循环
            await asyncio.sleep(0)
        return state

    return asyncio.run(_run())


# ================================================================ ① 七类意图样本
def _rows(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / name
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_every_intent_has_enough_train_and_test_samples() -> None:
    """七类意图在训练集与测试集里都成规模，且**标签集与 `INTENT_LABELS` 逐字相等**。

    最后这一条是重点：标签的唯一出处是 `base/labels.py`（§3.2 / §7.7）。数据里多一个
    "招生咨询"或少一个"闲聊"，都说明有人绕过词表改了数据——那会让意图 F1 与档位表
    对不上号，而且只在跑评估时才暴露。
    """
    train, test = _rows("intent_train.jsonl"), _rows("intent_test.jsonl")
    assert set(INTENT_LABELS) == set(r["label"] for r in train) == set(r["label"] for r in test)

    train_counts = Counter(r["label"] for r in train)
    test_counts = Counter(r["label"] for r in test)
    assert set(train_counts) == set(INTENT_LABELS)
    assert set(test_counts) == set(INTENT_LABELS)
    # 每类 300 条训练 / ≥70 条测试：低于这个量，单类的置信区间宽到没法判 F1
    assert min(train_counts.values()) >= 300, train_counts
    assert min(test_counts.values()) >= 70, test_counts
    assert len(train) >= 2100 and len(test) >= 490


def test_intent_train_and_test_do_not_overlap() -> None:
    """训练集与测试集不许出现同一句话：同源只能证明"没过拟合到个别句子"。"""
    train = {r["query"] for r in _rows("intent_train.jsonl")}
    test = {r["query"] for r in _rows("intent_test.jsonl")}
    assert train & test == set()


# ================================================================ ② 槽位：多值 / 缺失 / 冲突
def test_multi_subject_query_is_reported_as_multi_not_silently_narrowed() -> None:
    """多值：两门学科都记下来，不用第一个静默覆盖掉第二个。

    `subject` 取第一个只是给"只认单值"的老调用方一个可用的值；真正的范围由
    `subjects` + `subject_source="multi"` 表达（§7.1.8 的多学科行）。
    """
    slots = extract("数学和物理的题目怎么讲")
    assert slots.subject_source == "multi"
    assert slots.subjects == ("数学", "物理")
    assert slots.subject == "数学"
    assert slots.subject_confidence == 1.0


def test_missing_subject_is_unknown_and_never_guessed() -> None:
    """缺失（抽不到）：`unknown` + 学科为空。**不猜**——猜错就是"只查另一门学科"。"""
    slots = extract("这道题怎么做")
    assert slots.subject is None
    assert slots.subjects == ()
    assert slots.subject_source == "unknown"
    assert slots.subject_confidence == 0.0


def test_missing_param_is_not_provided_while_bogus_param_is_invalid() -> None:
    """两种"没有学科"必须分开：上游没传（`not_provided`）vs 传了个不存在的（`invalid`）。

    排障时这是两条完全不同的线索；混成一个，监控里就分不清"客户端没传"
    和"客户端传错了"。
    """
    assert from_param(None).subject_source == "not_provided"
    assert from_param("   ").subject_source == "not_provided"
    bogus = from_param("天文学")
    assert bogus.subject_source == "invalid"
    assert bogus.subject is None


def test_missing_free_text_slots_stay_empty_instead_of_echoing_placeholders() -> None:
    """模型给的"无 / 未知 / 不确定"是占位符，不是值：落成 None/空元组。

    真把它们当成章节名带进检索，等于拿一句"无"去召回。
    """
    parsed, degraded = uslots.parse_model(
        {"chapter": "无", "knowledge_points": ["未知"], "difficulty": "不确定", "question_count": "很多"}
    )
    assert parsed.chapter is None
    assert parsed.knowledge_points == ()
    assert parsed.difficulty is None
    assert parsed.task.exam.question_count is None
    assert degraded == ()


def test_model_supplied_filter_fields_are_dropped_with_a_degraded_mark() -> None:
    """过滤类字段只能来自规则（§7.7）：模型即使返回 `subject/grade`，也必须被丢弃。

    这不是"模型不可信"，而是失效方向不对称——规则抽不到是"退回全库"（安全），
    模型抽错是"只查另一门学科"（危险）。所以模型在这一层没有发言权，且要留痕。
    """
    parsed, degraded = uslots.parse_model(
        {"subject": "化学", "grade": "高三", "textbook_version": "人教A版", "chapter": "电离平衡"}
    )
    assert "slot_model_filter_dropped" in degraded
    # 章节名不是过滤字段，照常收下
    assert parsed.chapter == "电离平衡"
    # `ModelSlots` 在**类型上**就没有 subject/grade：模型改不了过滤条件
    assert not hasattr(parsed, "subject")
    assert not hasattr(parsed, "grade")


def _faq_service() -> FAQService:
    repo = InMemoryFAQRepository(
        [
            FAQRecord(1, "数学", "勾股定理的内容是什么？", "直角三角形两直角边平方和等于斜边平方。"),
            FAQRecord(2, "数学", "勾股定理怎么证明？", "常用面积法（赵爽弦图）。"),
            FAQRecord(3, "英语", "怎么报名", "在官网填写报名表并缴费。"),
            FAQRecord(4, "物理", "牛顿第二定律是什么", "F=ma。"),
        ]
    )
    service = FAQService(
        repo,
        policy=MatchPolicy(threshold=0.05, margin=0.0, candidate_n=3),
        top_k=10,
    )
    asyncio.run(service.load())
    return service


def test_conflicting_subject_source_forces_full_scan_and_never_answers_directly() -> None:
    """冲突（消费端）：学科结论自相矛盾时，既不能硬按某一门过滤，也不能直接作答。

    两条都在这里钉住：`subject_state="full"`（全库检索，不拿矛盾的学科去收窄）
    + `tier="mid"`（封顶，`answer is None`，交给 RAG 与追问）。对照组是同一条问句
    在没有冲突时是一档 `high` 带答案——差别只来自这一个字段。
    """
    service = _faq_service()

    normal = asyncio.run(service.search("勾股定理的证明方法"))
    assert normal.tier == "high"
    assert normal.answer is not None

    conflicted = asyncio.run(
        service.search("勾股定理的证明方法", slots=SlotResult(subject_source="conflict"))
    )
    assert conflicted.subject_state == "full"
    assert "subject_conflict_full_scan" in conflicted.degraded
    assert conflicted.tier == "mid"
    assert conflicted.answer is None
    assert conflicted.reason == "subject_conflict_handoff"


def test_conflicting_param_and_query_subject_is_flagged_as_conflict() -> None:
    """冲突（产出端）：参数说化学、问句说数学，必须标成 `conflict` 而不是静默采信参数。

    "静默采信参数"的后果不是报错，而是**用错的学科做了硬过滤**——正确证据被挡在门外
    且不留任何痕迹（错误过滤的典型形态，§7.1.8 要求优先统计它）。

    结论必须是"不选边"：`subject` 清空、`subjects` 也清空（留着第一个会被下游当成
    "就查这一门"），问句里其余的槽位照旧带出来。
    """
    slots = resolve("数学的二次函数怎么求", param="化学")
    assert slots.subject_source == "conflict"
    assert slots.subject is None
    assert slots.subjects == ()
    assert slots.scope == ()  # 下游据此走全库，不拿矛盾的学科去收窄


def test_conflict_reaches_the_faq_as_a_full_scan_handoff() -> None:
    """产出端与消费端串起来：`resolve()` 的结论进 `FAQService` 后必须是全库 + mid。

    分成两段各自正确还不够——中间那段（编排层把 `SlotResult` 原样递过去）断了的话，
    两个单测都会绿，线上仍然是"按错的学科硬过滤、还直接给答案"。
    """
    service = _faq_service()
    slots = resolve("数学的勾股定理怎么证明", param="化学")
    assert slots.subject_source == "conflict"

    hit = asyncio.run(service.search("数学的勾股定理怎么证明", slots=slots))
    assert hit.subject_state == "full"
    assert "subject_conflict_full_scan" in hit.degraded
    assert hit.tier == "mid"
    assert hit.answer is None
    assert hit.reason == "subject_conflict_handoff"


# ================================================================ ③ 复合问题：最多 3 个有效子问题
def test_complex_question_yields_at_most_three_verified_sub_queries() -> None:
    """规格原例拆出 3 路，且每一路都**有来源**（原问句里每个字都出现过）。

    "有效"在这里是可验证的：`_verified()` 用字符多重集校验，编出来的子问题会被丢弃
    而不是被修好。一条编造的子问题不是"少召回一路"，而是**真的在花检索预算**。
    """
    result = split(SPEC)
    assert result.is_complex is True
    assert len(result.sub_queries) == 3 <= MAX_SUB_QUERIES
    assert result.sub_queries == SPEC_SUBS

    allowed = Counter(SPEC)
    for sub in result.sub_queries:
        assert not (Counter(sub) - allowed), f"子问题引入了原问句里没有的字：{sub!r}"
    assert result.dropped == ()


def test_four_demands_are_capped_at_three() -> None:
    """四件诉求也只放三路：上限是硬约束，不是"尽量"。多出来的宁可不拆。"""
    result = split("什么是勾股定理，什么是二次函数，什么是平行线，什么是浮力")
    assert result.is_complex is True
    assert len(result.sub_queries) == MAX_SUB_QUERIES == 3
    assert result.sub_queries == ("什么是勾股定理", "什么是二次函数", "什么是平行线")


def test_sub_queries_reach_the_retriever_as_separate_searches() -> None:
    """拆出来的三路真的变成三次检索——只看 `plan.strategy` 分不出"拆了"和"只是标了策略"。

    纯规则下 `looks_multi()` 也会把整句标成 `sub_query`，那时候只有一路检索。
    所以断言的是 `retriever.search` 的实参，不是策略名。
    """
    pipeline, _llm, retriever = _pipeline()
    state = _ask(pipeline, SPEC)

    assert state.understanding is not None
    assert state.understanding.is_complex is True
    assert state.understanding.sub_queries == SPEC_SUBS
    assert retriever.queries == list(SPEC_SUBS)


# ================================================================ ④ 指代：真实会话历史
def test_follow_up_uses_the_real_session_store_as_history() -> None:
    """「那它的图像有什么特点」的学科只能来自上一轮——而上一轮是从**真实** `SessionStore` 读的。

    用 `SessionStore` 本体（`redis=None` 时走进程内回退，与生产里 Redis 挂掉同一条路径），
    不是替身：替身能证明"接线接上了"，证明不了"窗口里真的读得到上一轮"。

    `subject_source="context"` 是结论而不是过程：它告诉下游"这个学科是猜的、置信 0.6"，
    按 §7.1.8 路由表这一档**不许直接作答**，只能交 RAG。
    """
    settings = _settings()
    store = SessionStore(settings, redis=None, chat_log=None)
    pipeline, _llm, _retriever = _pipeline(memory=_RealSessionMemory(store))

    async def _run() -> Any:
        await store.append("s1", "user", "二次函数是数学里很重要的内容", user_id="u1")
        await store.append("s1", "assistant", "好的，二次函数是初中数学的重要章节。", user_id="u1")
        state = await pipeline.ask("那它的图像有什么特点？", session_id="s1", user_id="u1")
        for _ in range(10):
            await asyncio.sleep(0)
        return state

    state = asyncio.run(_run())

    assert state.slots.subject == "数学"
    assert state.slots.subject_source == "context"
    assert state.slots.subject_confidence == 0.6
    assert state.rewrite_method == "rule"
    assert "coreference_unresolved" not in state.degraded
    assert state.understanding is not None
    assert state.understanding.is_follow_up is True


def test_self_contained_question_does_not_borrow_the_history_subject() -> None:
    """同一份历史，换一句**自成一句**的问句：不许拿上一轮的学科去收窄这一轮。

    这是指代链路最容易犯的错误过滤——用一句不相干的话决定这一轮查哪一门。
    """
    settings = _settings()
    store = SessionStore(settings, redis=None, chat_log=None)
    pipeline, _llm, _retriever = _pipeline(memory=_RealSessionMemory(store))

    async def _run() -> Any:
        await store.append("s1", "user", "二次函数是数学里很重要的内容", user_id="u1")
        state = await pipeline.ask("浮力大小怎么算", session_id="s1", user_id="u1")
        for _ in range(10):
            await asyncio.sleep(0)
        return state

    state = asyncio.run(_run())

    assert state.slots.subject is None
    assert state.slots.subject_source == "unknown"
    assert state.understanding is not None
    assert state.understanding.is_follow_up is False


# ================================================================ ⑤ 理解失败：安全降级
def test_understanding_exception_still_answers_with_the_original_query() -> None:
    """理解调用抛异常：记 `understanding_failed`，检索用原话，**用户照样拿到答案**。

    理解层是"增强"不是"前置闸门"——它挂了不该变成 5xx。这是 §6.1 预算表里
    "改写退归一化句、意图退规则分类器"那条降级口径在异常路径上的形态。
    """
    pipeline, _llm, retriever = _pipeline(llm=ExplodingLLM(), feature_intent_llm=True)
    state = _ask(pipeline, "二次函数的顶点式怎么求")

    assert "understanding_failed" in state.degraded
    assert state.understanding is not None
    assert state.understanding.success is False
    assert state.answer == ANSWER
    assert retriever.queries == ["二次函数的顶点式怎么求"]


def test_understanding_timeout_still_answers_with_the_original_query() -> None:
    """理解超时：记 `understanding_timeout`，不拖住检索与生成。"""
    pipeline, _llm, retriever = _pipeline(
        llm=SlowLLM(), feature_intent_llm=True, understanding_timeout_ms=1
    )
    state = _ask(pipeline, "二次函数的顶点式怎么求")

    assert "understanding_timeout" in state.degraded
    assert state.understanding is not None
    assert state.understanding.success is False
    assert state.answer == ANSWER
    assert retriever.queries == ["二次函数的顶点式怎么求"]


def test_understanding_bad_json_still_answers_with_the_original_query() -> None:
    """模型没给出可解析的结果：整体判失败，但要回退到**原问题**而不是另造一句。"""
    pipeline, _llm, retriever = _pipeline(
        llm=StubLLM(payload="我不确定你想问什么"), feature_intent_llm=True
    )
    state = _ask(pipeline, "二次函数的顶点式怎么求")

    assert "understanding_bad_json" in state.degraded
    assert state.understanding is not None
    assert state.understanding.success is False
    assert state.understanding.rewritten == ""
    assert state.answer == ANSWER
    assert retriever.queries == ["二次函数的顶点式怎么求"]


def test_a_rejected_field_alone_does_not_fail_the_whole_understanding() -> None:
    """单个字段被拒（意图词表外）只丢那个字段：`success` 仍为 True。

    合并的是网络往返，不是成败——`intent` 坏了不该顺手把 `query` 也丢掉。
    """
    payload = json.dumps(
        {"query": "二次函数的顶点式怎么求？", "intent": "数学题", "confidence": 0.9},
        ensure_ascii=False,
    )
    pipeline, _llm, _retriever = _pipeline(llm=StubLLM(payload=payload), feature_intent_llm=True)
    state = _ask(pipeline, "二次函数的顶点式怎么求？")

    assert "intent_rejected" in state.degraded
    assert state.understanding is not None
    assert state.understanding.success is True
    assert state.answer == ANSWER


def test_all_understanding_switches_off_means_zero_llm_calls_and_a_normal_answer() -> None:
    """三个开关全关：理解层**一次大模型调用都不发**，请求照常走完。

    这是"A/B 基线"的可观测形态（§7.8.4）：档位恒为 `BASELINE`，检索用原话。
    """
    pipeline, llm, retriever = _pipeline()
    state = _ask(pipeline, "二次函数的顶点式怎么求")

    assert llm.understanding_calls == 0
    assert state.answer == ANSWER
    assert retriever.queries == ["二次函数的顶点式怎么求"]
    assert state.understanding is not None
    assert state.understanding.success is True
    assert state.understanding.degraded == ()
