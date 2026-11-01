"""中置信 FAQ 候选进生成 prompt（CLAUDE.md §7.1.9 边界 2，2026-10-31 修订）。

这条改动的**全部风险**都集中在"它会不会被当成证据"：

- 编号撞车 → 模型写出 `[1]`，`citation.align()` 把它对齐到知识片段，
  一条没过精度闸门的 FAQ 答案就变成了"带引用的事实"；
- 与 `【资料】` 混排 → 同一件事的另一种翻车方式，只是更难看出来；
- 低置信也塞进去 → 连阈值都没过的匹配成了回答的骨架。

所以用例分三层：`render_faq_candidates()` 的纯函数边界、`build_messages()` 的装配位置、
编排层三档的接线（只有 mid 且开关开着才给，且永远不进 `citations`）。
"""

from __future__ import annotations

import asyncio
from typing import Any

from base.config import Settings
from base.contracts import (
    EvidenceDecision,
    FAQCandidate,
    FAQHit,
    SlotResult,
)
from orchestration.chat import ChatPipeline
from rag_system.core.llm_gateway import LLMResult
from rag_system.core.prompts import (
    FAQ_CANDIDATE_MAX_CHARS,
    FAQ_CANDIDATE_NOTICE,
    FAQ_CANDIDATE_RULES,
    build_messages,
    render_faq_candidates,
)
from rag_system.core.rag_pipeline import RagPipeline
from rag_system.core.retriever import Evidence, RetrievalResult

EVIDENCE_TEXT = (
    "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k，其中 h=-b/(2a)，k=(4ac-b²)/(4a)。"
)


def _evidence() -> Evidence:
    return Evidence(
        parent_id="doc-1:p3",
        text=EVIDENCE_TEXT,
        score=0.91,
        rerank_score=0.9,
        child_hits=[EVIDENCE_TEXT[:60]],
        title="人教版-数学-九年级上册",
        source="F:/docs/人教版-数学-九年级上册.pdf",
        subject="数学",
        doc_type="textbook",
        heading="22.1 二次函数",
        page=12,
    )


def _cand(faq_id: int = 7, answer: str = "顶点式是 y=a(x-h)²+k。", score: float = 0.6) -> FAQCandidate:
    return FAQCandidate(faq_id=faq_id, question="二次函数的顶点式怎么写？", score=score, answer=answer)


def _all_text(messages: list[dict]) -> str:
    return "\n".join(str(m["content"]) for m in messages)


# ---------------------------------------------------------------------------
# 第一层：渲染本身
# ---------------------------------------------------------------------------
def test_render_uses_verbal_numbering_not_brackets() -> None:
    """「候选1：」而不是 `[1]`——方括号编号就是 citation.align() 的输入格式。"""
    out = render_faq_candidates([_cand(), _cand(8, "顶点式靠配方法得到。")])
    assert "候选1：" in out and "候选2：" in out
    assert "[1]" not in out and "[2]" not in out


def test_render_carries_notice_and_four_rules_verbatim() -> None:
    """提示词里的免责声明与四条要求逐字来自需求，被"优化"掉就等于没有约束。"""
    out = render_faq_candidates([_cand()])
    assert FAQ_CANDIDATE_NOTICE in out
    assert FAQ_CANDIDATE_RULES in out
    for rule in ("优先依据 RAG 检索得到的原始证据回答", "以可信的知识库证据为准", "不得直接将 FAQ 候选当作事实"):
        assert rule in out


def test_render_skips_candidates_without_answer() -> None:
    """`answer=""` 的含义是"这条没有答案"，渲染成空「答案：」只会浪费预算。"""
    out = render_faq_candidates([_cand(7, ""), _cand(8, "有答案的那条。")])
    assert "有答案的那条。" in out
    assert out.count("问题：") == 1  # 空答案那条整条跳过，不是跳过答案那一行
    assert "候选1：" in out and "候选2：" not in out


def test_render_returns_empty_when_nothing_usable() -> None:
    """全部没答案 → 空串，`build_messages` 据此不加这条 system 消息。"""
    assert render_faq_candidates([]) == ""
    assert render_faq_candidates([_cand(7, "   ")]) == ""


def test_render_truncates_long_answers() -> None:
    """上限定在 500 字：不设限时一条 10000 字的 FAQ 能把知识段挤出预算。"""
    long = "长" * (FAQ_CANDIDATE_MAX_CHARS + 200)
    out = render_faq_candidates([_cand(7, long)])
    assert out.count("长") == FAQ_CANDIDATE_MAX_CHARS
    # 截断点在「答案：」那一行，不是整块结尾（块尾部是四条要求）
    assert f"答案：{'长' * FAQ_CANDIDATE_MAX_CHARS}…" in out


def test_render_respects_custom_cap() -> None:
    out = render_faq_candidates([_cand(7, "abcdef")], max_answer_chars=3)
    assert "答案：abc…" in out


# ---------------------------------------------------------------------------
# 第二层：装配位置
# ---------------------------------------------------------------------------
def test_candidates_live_in_their_own_system_message() -> None:
    """不进【资料】、不在 user 消息里：物理隔开才谈得上"不是证据"。"""
    messages = build_messages("顶点式怎么求", [_evidence()], faq_candidates=[_cand()])
    faq_msgs = [m for m in messages if FAQ_CANDIDATE_NOTICE in m["content"]]
    assert len(faq_msgs) == 1 and faq_msgs[0]["role"] == "system"
    user = next(m for m in messages if m["role"] == "user")
    assert FAQ_CANDIDATE_NOTICE not in user["content"]
    assert "【资料】" in user["content"] and "候选1：" not in user["content"]


def test_candidates_do_not_shift_evidence_numbering() -> None:
    """知识的 `[1][2]` 编号与候选完全无关：候选插在前面也不能让编号改号。"""
    messages = build_messages(
        "顶点式怎么求", [_evidence(), _evidence()], faq_candidates=[_cand()]
    )
    user = next(m for m in messages if m["role"] == "user")
    assert "[1]" in user["content"] and "[2]" in user["content"]
    assert "[3]" not in user["content"]


def test_candidates_follow_memory_and_precede_history() -> None:
    """顺序固定为 记忆 → FAQ 候选 → 历史 → 资料 → 问题。"""
    messages = build_messages(
        "顶点式怎么求",
        [_evidence()],
        memory="【学员画像】九年级",
        history=[("user", "上一轮问的是函数")],
        faq_candidates=[_cand()],
    )
    roles = [m["role"] for m in messages]
    assert roles == ["system", "system", "system", "user", "user"]
    assert "学员画像" in messages[1]["content"]
    assert FAQ_CANDIDATE_NOTICE in messages[2]["content"]
    assert "上一轮问的是函数" in messages[3]["content"]


def test_general_branch_ignores_candidates() -> None:
    """没有证据走通用模板时**不给候选**：那个模板禁止给具体事实，塞进去只会自相矛盾。"""
    messages = build_messages("今天天气怎么样", [], faq_candidates=[_cand()])
    assert FAQ_CANDIDATE_NOTICE not in _all_text(messages)
    assert len(messages) == 2 and messages[-1]["role"] == "user"


def test_no_candidates_means_no_block() -> None:
    messages = build_messages("顶点式怎么求", [_evidence()])
    assert FAQ_CANDIDATE_NOTICE not in _all_text(messages)


# ---------------------------------------------------------------------------
# 第三层：编排层接线（三档 + 开关）
# ---------------------------------------------------------------------------
class _RecordingLLM:
    def __init__(self, text: str = "顶点式可以通过配方法得到[1]。") -> None:
        self.text = text
        self.enabled = True
        self.calls = 0
        self.messages: list[dict] | None = None

    async def complete(self, messages: list[dict], *, budget_ms: int, model: str | None = None) -> LLMResult:
        self.calls += 1
        self.messages = messages
        return LLMResult(text=self.text, model="primary", prompt_tokens=100, completion_tokens=20)


class _StubFAQ:
    def __init__(self, hit: FAQHit) -> None:
        self.hit = hit
        self.calls = 0

    async def search(
        self,
        query: str,
        subject: str | None = None,
        *,
        slots: SlotResult | None = None,
        selected_faq_id: int | None = None,
    ) -> FAQHit:
        self.calls += 1
        return self.hit


class _StubRetriever:
    async def search(self, query: str, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            evidences=[_evidence()],
            raw_hits=12,
            cost_ms=120,
            stages={"recall": 40, "parent_expand": 6, "fusion": 0, "total": 85},
        )


class _PassGate:
    def __call__(self, result: RetrievalResult, **kwargs: Any) -> EvidenceDecision:
        return EvidenceDecision(status="sufficient", top_score=0.91, margin=0.3, n_docs=1, qcov=0.6)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "llm_api_key": "sk-test",
        "llm_qa_model": "primary",
        "citation_min_support": 0.30,
        "total_budget_ms": 8000,
        # 关掉"一次大模型理解"的四个消费者（改写 / 意图 / 槽位 / 检索策略）：
        # 本文件数的是生成那一次调用
        "feature_intent_llm": False,
        "feature_query_rewrite": False,
        "feature_slot_llm": False,
        "feature_planner_llm": False,
    }
    base.update(overrides)
    return Settings(**base)


def _mid_hit(*candidates: FAQCandidate) -> FAQHit:
    return FAQHit(
        tier="mid",
        faq_id=7,
        answer=None,  # mid 的 answer 恒为 None——这正是本次修订**没有**改变的那一条
        score=0.6,
        matched_question="二次函数的顶点式怎么写？",
        candidates=candidates or (_cand(),),
    )


def _ask(hit: FAQHit, llm: _RecordingLLM, **overrides: Any) -> Any:
    retriever = _StubRetriever()
    pipeline = ChatPipeline(
        _settings(**overrides),
        faq_service=_StubFAQ(hit),
        rag={
            "ready": True,
            "retriever": retriever,
            "pipeline": RagPipeline(_settings(**overrides), retriever, gate=_PassGate(), max_repairs=0),
            "gate": _PassGate(),
            "embedder_name": "stub",
            "rerank_enabled": False,
        },
        llm=llm,
    )
    return asyncio.run(pipeline.ask("二次函数的顶点式怎么写"))


def test_mid_hit_puts_candidates_into_the_prompt() -> None:
    llm = _RecordingLLM()
    state = _ask(_mid_hit(), llm)
    assert llm.calls == 1
    assert llm.messages is not None
    assert FAQ_CANDIDATE_NOTICE in _all_text(llm.messages)
    assert "顶点式是 y=a(x-h)²+k。" in _all_text(llm.messages)
    assert state.source == "rag"


def test_mid_hit_still_returns_no_faq_answer() -> None:
    """候选进了 prompt，但 `FAQHit.answer` 仍是 None，响应里的 FAQ 卡片也仍不含答案。"""
    state = _ask(_mid_hit(), _RecordingLLM())
    body = state.to_response()
    assert state.faq is not None and state.faq.answer is None
    assert body.faq_suggestion is not None
    assert "answer" not in body.faq_suggestion
    assert all("answer" not in c for c in body.faq_suggestion["candidates"])


def test_candidates_never_become_citations() -> None:
    """`citations` 只来自【资料】：编号 1 必须指向知识片段，不是那条 FAQ。"""
    state = _ask(_mid_hit(), _RecordingLLM())
    citations = state.to_response().citations
    assert [c["n"] for c in citations] == [1]
    assert citations[0]["file_name"] == "人教版-数学-九年级上册.pdf"


def test_low_hit_never_reaches_the_prompt() -> None:
    """低置信连阈值都没过，把它塞进 prompt 等于用没把握的匹配污染一次证据不足的回答。"""
    llm = _RecordingLLM()
    low = FAQHit(
        tier="low",
        reason="below_threshold",
        score=0.1,
        candidates=(_cand(),),
    )
    _ask(low, llm)
    assert llm.messages is not None
    assert FAQ_CANDIDATE_NOTICE not in _all_text(llm.messages)


def test_high_hit_never_calls_the_model() -> None:
    """高置信逐字作答，压根不生成——候选块连渲染的机会都没有。"""
    llm = _RecordingLLM()
    high = FAQHit(
        tier="high",
        faq_id=3,
        answer="顶点式是 y=a(x-h)²+k。",
        score=0.95,
        subject="数学",
        candidates=(_cand(),),
    )
    state = _ask(high, llm)
    assert llm.calls == 0 and llm.messages is None
    assert state.to_response().source == "faq"


def test_flag_off_restores_old_behaviour() -> None:
    """`FEATURE_FAQ_CANDIDATES=false` → 回到"mid 的 FAQ 原文一个字都不进生成器"。"""
    llm = _RecordingLLM()
    _ask(_mid_hit(), llm, feature_faq_candidates=False)
    assert llm.messages is not None
    assert FAQ_CANDIDATE_NOTICE not in _all_text(llm.messages)


def test_candidates_without_answers_are_dropped_whole() -> None:
    """候选里一条答案都没有（例如脚本/单测构造的三参数版本）→ 整块不加。"""
    llm = _RecordingLLM()
    _ask(_mid_hit(FAQCandidate(7, "二次函数的顶点式怎么写？", 0.6)), llm)
    assert llm.messages is not None
    assert FAQ_CANDIDATE_NOTICE not in _all_text(llm.messages)


def test_at_most_three_candidates() -> None:
    """只取 top3：再多就是拿没把握的匹配挤知识段。"""
    llm = _RecordingLLM()
    cands = tuple(_cand(i, f"第 {i} 条的答案。") for i in range(1, 6))
    _ask(_mid_hit(*cands), llm)
    assert llm.messages is not None
    text = _all_text(llm.messages)
    assert "第 1 条的答案。" in text and "第 3 条的答案。" in text
    assert "第 4 条的答案。" not in text
