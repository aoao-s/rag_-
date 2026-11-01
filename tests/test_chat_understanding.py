"""任务 2.4 / 2.5 的接线验收：复合问句的拆分、指代的落地，以及统一理解结果。

盯的是**字段有没有抵达执行点**（方法同 `tests/test_rag_store_filter.py` /
`tests/test_chat_rewrite.py`），不是"函数被调过"：

- 拆出来的子问题有没有真的变成三路检索（`retriever.search` 被调了几次、传了什么）；
- 多轮里的"它"能不能靠会话历史落到一个学科上（`SlotResult.subject_source == "context"`）；
- 指代解不开时是不是**记一笔降级**而不是编一个主语（任务 2.5 第 4 条）；
- `UnderstandingResult` 的五个字段与五条规则是不是自洽（原话保留 / 改写不覆盖 /
  子问题有来源 / 不猜指代 / 超时回退）。

三处刻意用**确定性配置**（关掉大模型意图与改写、关掉模型槽位），原因不是"怕慢"：
任务 2.2 / 2.3 的级联已经有各自的用例，这里再让同一段大模型跑一遍，
断言失败时分不清是"接线断了"还是"模型今天心情不好"。真正要验的是**接线**。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from base.config import Settings
from base.contracts import (
    EvidenceDecision,
    FAQHit,
    IntentResult,
    MemoryContext,
    SlotResult,
    UnderstandingResult,
)
from orchestration.chat import ChatPipeline
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult

EVIDENCE_TEXT = "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k。"
ANSWER = "顶点式可以通过配方法得到[1]。"
# 理解与生成共用同一个网关，按 system prompt 分流计数
_UNDERSTAND_MARK = "问题理解"

# 任务 2.4 规格里的原例
SPEC = "比较一次函数和二次函数的区别，并分别给我出三道练习题。"
SPEC_SUBS = (
    "一次函数和二次函数的区别",
    "给我出三道一次函数练习题",
    "给我出三道二次函数练习题",
)

# 上一轮已经确认过的学科：第二轮只说"它"，学科只能从历史来
HISTORY = [
    ("user", "二次函数是数学里很重要的内容"),
    ("assistant", "好的，二次函数是初中数学的重要章节。"),
]
FOLLOW_UP = "那它的图像有什么特点？"
# 归一化只做 NFKC：全角 `？` 与 `，` 都折成半角，这句话因此和原话逐字不等
# （于是 `UnderstandingResult.rewritten` 非空、`rewritten` 与 `query` 是两个不同的字段）
FOLLOW_UP_NORMALIZED = "二次函数是数学里很重要的内容,那它的图像有什么特点?"


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
        # 本文件验的是 2.4 / 2.5 的接线，不是 2.2 / 2.3 的模型级联：
        # 默认走规则分类器与规则槽位，需要时由用例自己打开。
        # 检索策略（`FEATURE_PLANNER_LLM`）同理——本文件验的是拆句结果有没有变成多路检索，
        # 不是模型选得准不准，开着它只会让 `understanding_bad_json` 混进 degraded 断言
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
        self.payload = payload if payload is not None else json.dumps(
            {"query": "二次函数的顶点式怎么求？", "intent": "学习讲解", "confidence": 0.92},
            ensure_ascii=False,
        )
        self.answer = answer
        # 编排层用 `getattr(llm, "enabled", False)` 判"生成可用"，替身必须显式给出
        self.enabled = enabled
        self.understanding_calls = 0
        self.generation_calls = 0

    @property
    def calls(self) -> int:
        return self.understanding_calls + self.generation_calls

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
    """记下**真正送去检索的那几串字**。复合问句的验收全靠它：拆没拆、拆了几路。"""

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


@dataclass
class _Turn:
    role: str
    content: str


class StubSession:
    def __init__(self, turns: list[tuple[str, str]] | None = None) -> None:
        self.turns = [_Turn(role, content) for role, content in (turns or [])]
        self.appended: list[tuple[str, str]] = []

    async def window(self, sid: str) -> list[_Turn]:
        return list(self.turns)

    async def summary(self, sid: str) -> str:
        return ""

    async def append(self, sid: str, role: str, content: str, *, user_id: str = "") -> None:
        self.appended.append((role, content))


class StubMemory:
    """只实现编排层真正会碰的四个方法（`session.window/append`、`recall`、`remember`）。"""

    def __init__(self, turns: list[tuple[str, str]] | None = None) -> None:
        self.session = StubSession(turns)
        self.recalled: list[str] = []
        self.reminded: list[str] = []

    async def recall(
        self, user_id: str, query: str, *, sid: str | None = None, budget_ms: int | None = None
    ) -> MemoryContext:
        self.recalled.append(query)
        return MemoryContext()

    async def remember(
        self, user_id: str, sid: str, query: str, *, turn_index: int = 0
    ) -> list[Any]:
        self.reminded.append(query)
        return []


def _retrieval(query: str = "二次函数的顶点式怎么求？") -> RetrievalResult:
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
    return EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)


def _pipeline(
    *,
    llm: StubLLM | None = None,
    retriever: RecordingRetriever | None = None,
    memory: StubMemory | None = None,
    llm_local: Any = None,
    **settings_overrides: Any,
) -> tuple[ChatPipeline, StubLLM, RecordingRetriever]:
    llm = llm or StubLLM()
    retriever = retriever or RecordingRetriever()
    pipeline = ChatPipeline(
        _settings(**settings_overrides),
        faq_service=StubFAQ(),
        rag={
            "ready": True,
            "retriever": retriever,
            "pipeline": RagPipeline(
                _settings(**settings_overrides),
                retriever,
                gate=lambda result, **kw: _passed(),
                max_repairs=0,
            ),
            "gate": None,
            "embedder_name": "stub",
            "rerank_enabled": False,
        },
        llm=llm,
        llm_local=llm_local,
        intent=StubIntent(),
        memory=memory,
    )
    return pipeline, llm, retriever


def _ask(pipeline: ChatPipeline, query: str, **kwargs: Any) -> Any:
    """跑一轮并**等后台回写落定**：`_persist_later` 是 fire-and-forget 的 task，
    直接 `asyncio.run` 会在它跑起来之前就关掉事件循环。"""

    async def _run() -> Any:
        state = await pipeline.ask(query, **kwargs)
        for _ in range(10):
            await asyncio.sleep(0)
        return state

    return asyncio.run(_run())


# ---------------------------------------------------------------- ① 复合问句抵达检索
def test_compound_query_becomes_three_retrieval_calls() -> None:
    """规格原例：三路子问题，就应该有三次 `retriever.search`。

    这里刻意**不断言 `plan.strategy`**：纯规则下 `looks_multi()` 也会把整句标成
    `sub_query`，那时候只有一路检索。真正区分"拆了"和"只是标了策略"的是任务数。
    """
    pipeline, llm, retriever = _pipeline()
    state = _ask(pipeline, SPEC)

    assert llm.understanding_calls == 0  # 三个开关都关着：理解层零次大模型调用
    assert state.understanding is not None
    assert state.understanding.is_complex is True
    assert state.understanding.sub_queries == SPEC_SUBS
    assert retriever.queries == list(SPEC_SUBS)


def test_compound_query_reports_no_understanding_degradation() -> None:
    """拆分成功不产生任何理解层降级码，也不该顺手问用户一句（`ambiguity` 为空）。"""
    pipeline, _llm, _retriever = _pipeline()
    state = _ask(pipeline, SPEC)

    assert state.understanding is not None
    assert state.understanding.degraded == ()
    assert state.understanding.success is True
    assert state.understanding.ambiguity is None


def test_compound_query_keeps_intent_confidence_and_method() -> None:
    pipeline, _llm, _retriever = _pipeline()
    state = _ask(pipeline, SPEC)

    understanding = state.understanding
    assert understanding is not None
    assert understanding.intent is not None
    assert understanding.intent.label == "学习讲解"
    assert understanding.method == "rule"
    assert understanding.confidence == 0.8


class _LocalLLM:
    """本地小模型替身：拆句只用得上 `enabled` + `complete`。

    记 `calls` 是为了区分"走了模型"与"走了规则"——只看 `state.compound.source`
    在模型恰好拆得和规则一样时分不出来（那正是正常的成功路径）。
    """

    def __init__(self, text: str) -> None:
        self.enabled = True
        self.text = text
        self.calls: list[list[dict[str, str]]] = []

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        self.calls.append(messages)
        return LLMResult(
            text=self.text, model=model or "local", prompt_tokens=1, completion_tokens=1,
            latency_ms=1,
        )


def test_local_model_split_reaches_the_retriever() -> None:
    """配了本地端点就用它拆：`source == "model"`，子问题照旧变成三路检索。

    断言落在 `retriever.queries` 而不是只落在 `state.compound`：拆分器自己拆对，
    和"拆对的结果真的驱动了检索"是两件事（后者才是这条链路的交付物）。
    """
    local = _LocalLLM(json.dumps({"sub_queries": list(SPEC_SUBS)}, ensure_ascii=False))
    pipeline, _llm, retriever = _pipeline(llm_local=local)
    state = _ask(pipeline, SPEC)

    assert state.compound is not None
    assert state.compound.source == "model"
    assert state.compound.sub_queries == SPEC_SUBS
    assert retriever.queries == list(SPEC_SUBS)
    assert len(local.calls) == 1


def test_compound_switch_off_never_touches_the_local_model() -> None:
    """`FEATURE_COMPOUND_LLM=false` 是回滚开关：一次本地调用都不发，结果与纯规则逐字一致。"""
    local = _LocalLLM(json.dumps({"sub_queries": list(SPEC_SUBS)}, ensure_ascii=False))
    pipeline, _llm, _retriever = _pipeline(llm_local=local, feature_compound_llm=False)
    state = _ask(pipeline, SPEC)

    assert local.calls == []
    assert state.compound is not None
    assert state.compound.source == "distribute"
    assert state.compound.sub_queries == SPEC_SUBS


# ---------------------------------------------------------------- ② 指代落到会话历史
def test_follow_up_resolves_subject_from_history() -> None:
    """「那它的图像有什么特点」：问句里没有学科，学科只能来自上一轮。

    `subject_source == "context"` 是**结论**而不是过程：它告诉下游"这个学科是猜的、
    置信 0.6"，按 §7.1.8 路由表，这一档不允许直接作答，只能交 RAG。
    """
    pipeline, _llm, _retriever = _pipeline(memory=StubMemory(HISTORY))
    state = _ask(pipeline, FOLLOW_UP, session_id="s1", user_id="u1")

    assert state.slots.subject == "数学"
    assert state.slots.subject_source == "context"
    assert state.slots.subject_confidence == 0.6
    # 规则改写落地（把上一轮问句接在前面），于是"指代已解开"，不再记降级
    assert state.rewrite_method == "rule"
    assert state.rewritten == FOLLOW_UP_NORMALIZED
    assert "coreference_unresolved" not in state.degraded
    assert state.understanding is not None
    assert state.understanding.is_follow_up is True


def test_question_without_reference_never_borrows_the_history_subject() -> None:
    """同样的历史，换一个**自成一句**的问句：不许拿上一轮的学科去收窄检索范围。

    这是最容易犯的错误过滤——用一句不相干的话决定这一轮查哪一门。
    """
    pipeline, _llm, _retriever = _pipeline(memory=StubMemory(HISTORY))
    state = _ask(pipeline, "浮力大小怎么算", session_id="s1", user_id="u1")

    assert state.slots.subject is None
    assert state.slots.subject_source == "unknown"
    assert state.understanding is not None
    assert state.understanding.is_follow_up is False
    assert "coreference_unresolved" not in state.degraded


def test_unresolved_reference_is_recorded_not_guessed() -> None:
    """历史为空时"它"解不开：记一笔降级，**不编主语**（任务 2.5 第 4 条）。

    `success` 仍是 True：单个字段退回是"降级可用"，不是"理解不可用"。
    只有超时 / 异常 / JSON 坏才算整体失败。
    """
    pipeline, _llm, _retriever = _pipeline(memory=StubMemory())
    state = _ask(pipeline, FOLLOW_UP, session_id="s1", user_id="u1")

    assert "coreference_unresolved" in state.degraded
    assert state.slots.subject is None
    assert state.slots.subject_source == "unknown"
    assert state.understanding is not None
    assert "coreference_unresolved" in state.understanding.degraded
    assert state.understanding.success is True
    # 没有历史就谈不上"追问上一轮"
    assert state.understanding.is_follow_up is False


# ---------------------------------------------------------------- ③ 原话与改写各就各位
def test_original_query_is_preserved_when_rewrite_changes_it() -> None:
    """改写只进 `rewritten`，`query` 一个字都不动（任务 2.5 第 1、2 条）。

    FAQ 的精确哈希与 `request_log` 都靠 `query`；被改写串覆盖过一次，
    就再也对不回用户说的那句话。
    """
    pipeline, _llm, _retriever = _pipeline(memory=StubMemory(HISTORY))
    state = _ask(pipeline, FOLLOW_UP, session_id="s1", user_id="u1")

    assert state.query == FOLLOW_UP
    assert state.understanding is not None
    assert state.understanding.query == FOLLOW_UP
    assert state.understanding.rewritten == FOLLOW_UP_NORMALIZED
    # 检索用的是改写句，不是原话
    assert state.understanding.retrieval_query == FOLLOW_UP_NORMALIZED


def test_unchanged_query_leaves_rewritten_empty() -> None:
    """没改写过就留空：响应体里摆两份一样的问句，"改写生效没有"就看不出来了。"""
    pipeline, _llm, retriever = _pipeline()
    state = _ask(pipeline, "什么是勾股定理")

    assert state.understanding is not None
    assert state.understanding.rewritten == ""
    assert state.understanding.retrieval_query == "什么是勾股定理"
    assert state.understanding.is_complex is False
    assert retriever.queries == ["什么是勾股定理"]


def test_retrieval_query_falls_back_rewritten_then_normalized_then_original() -> None:
    """回退顺序是**结构**，不是分支：三个坑位空一个就退一格（任务 2.5 第 5 条）。"""
    assert UnderstandingResult(query="原话", normalized="归一", rewritten="改写").retrieval_query == "改写"
    assert UnderstandingResult(query="原话", normalized="归一", rewritten="").retrieval_query == "归一"
    assert UnderstandingResult(query="原话").retrieval_query == "原话"


# ---------------------------------------------------------------- ④ success 的边界
def test_bad_json_marks_understanding_failed() -> None:
    """整体性失败（模型没给出可解析的结果）→ `success=False`，检索退回原问题（第 5 条）。

    这里刻意用一句**归一化前后逐字相同**的问句（不带全角标点），
    否则"退回到原问题"会被归一化顺手改掉一个字符，看不出回退到底回退到了哪。
    """
    pipeline, _llm, retriever = _pipeline(
        llm=StubLLM(payload="我不确定你想问什么"),
        feature_intent_llm=True,
    )
    state = _ask(pipeline, "二次函数的顶点式怎么求")

    assert "understanding_bad_json" in state.degraded
    assert state.understanding is not None
    assert state.understanding.success is False
    # 回退不是"另走一条路"：`rewritten` 为空，`retrieval_query` 自动落回原问题
    assert state.understanding.rewritten == ""
    assert state.understanding.retrieval_query == "二次函数的顶点式怎么求"
    assert retriever.queries == ["二次函数的顶点式怎么求"]


def test_rejected_intent_alone_does_not_fail_the_whole_understanding() -> None:
    """词表外的意图只丢意图（退回规则分类器），不把整个理解层判死。

    这是"合并的是网络往返，不是成败"那条口径的可观测形态：一次调用里
    `intent` 坏了不该顺手把 `query` 也丢掉。
    """
    payload = json.dumps(
        {"query": "二次函数的顶点式怎么求？", "intent": "数学题", "confidence": 0.9},
        ensure_ascii=False,
    )
    pipeline, _llm, _retriever = _pipeline(
        llm=StubLLM(payload=payload),
        feature_intent_llm=True,
    )
    state = _ask(pipeline, "二次函数的顶点式怎么求？")

    assert "intent_rejected" in state.degraded
    assert state.understanding is not None
    assert state.understanding.success is True
    assert "intent_rejected" in state.understanding.degraded
    # 意图退回到规则分类器
    assert state.intent is not None and state.understanding.method == "rule"


# ---------------------------------------------------------------- ⑤ 回写接线
def test_conversation_is_persisted_after_the_answer() -> None:
    """会话回写真的挂上了：一轮问答应落两条（user + assistant），且用的是**原话**。"""
    memory = StubMemory()
    pipeline, _llm, _retriever = _pipeline(memory=memory)
    state = _ask(pipeline, "什么是勾股定理", session_id="s1", user_id="u1")

    assert state.answer == ANSWER
    assert memory.session.appended == [
        ("user", "什么是勾股定理"),
        ("assistant", ANSWER),
    ]
    assert memory.reminded == ["什么是勾股定理"]
