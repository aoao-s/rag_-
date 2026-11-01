"""理解层一次大模型调用的单测（CLAUDE.md §3.2 / §7.7，任务 2.1）。

`analyze()` 的契约只有两条，但两条都容易被后续改动悄悄破坏：

1. **永不抛异常**——它挂在检索之前，抛一次就等于这条请求全挂。超时 / 上游失败 /
   JSON 坏 / 标签不在词表，一律转成 `degraded` 标记，让调用方各自退回；
2. **两项各自降级**——改写不合格不影响意图，意图不合格不影响改写。合并成一次
   网络往返是为了省一次调用，不是为了把两者的成败绑在一起。

校验函数（`_valid_rewrite` / `_valid_intent`）单独测：它们看着是"防御性代码"，
实际是这份实现的判断力所在——模型给什么都往下游塞，等于没有理解层。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from base.labels import CODE2INTENT, INTENT_FALLBACK, INTENT_LABELS
from understanding.llm_understanding import (
    analyze,
    _valid_intent,
    _valid_rewrite,
)

GOOD = '{"query": "二次函数的顶点式怎么求？", "intent": "学习讲解", "confidence": 0.92}'
ORIGINAL = "二次函数顶点式咋求"


# ---------------------------------------------------------------- 替身
class StubLLM:
    def __init__(
        self,
        text: str = GOOD,
        *,
        enabled: bool = True,
        delay: float = 0.0,
        exc: Exception | None = None,
    ) -> None:
        self.text = text
        self.enabled = enabled
        self.delay = delay
        self.exc = exc
        self.calls = 0
        self.budgets: list[int] = []
        self.models: list[str | None] = []
        self.messages: list[list[dict[str, str]]] = []

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> Any:
        self.calls += 1
        self.messages.append(messages)
        self.budgets.append(budget_ms)
        self.models.append(model)
        if self.exc is not None:
            raise self.exc
        if self.delay:
            await asyncio.sleep(self.delay)
        return SimpleNamespace(text=self.text)


def _run(
    llm: StubLLM | None,
    *,
    query: str = ORIGINAL,
    budget_ms: int = 800,
    **kwargs: Any,
) -> Any:
    return asyncio.run(analyze(query, llm, budget_ms=budget_ms, **kwargs))


# ---------------------------------------------------------------- ① 不调用的情况
def test_missing_llm_is_a_configuration_state_not_a_degradation() -> None:
    """没配 key 不该出现在每一份 `degraded` 里——那是启动期就该告警的事。"""
    disabled = StubLLM(enabled=False)
    for llm in (None, disabled):
        result = _run(llm)
        assert result.rewritten is None and result.intent is None
        assert result.method == "none"
        assert result.degraded == ()
    assert disabled.calls == 0


def test_empty_query_and_switched_off_features_send_no_call() -> None:
    stub = StubLLM()
    assert _run(stub, query="   ").degraded == ()
    assert _run(stub, want_rewrite=False, want_intent=False).degraded == ()
    assert stub.calls == 0


def test_budget_and_router_model_reach_the_gateway() -> None:
    stub = StubLLM()
    _run(stub, budget_ms=700, model="router")
    assert stub.budgets == [700]
    assert stub.models == ["router"]


def test_system_prompt_lists_exactly_the_taxonomy_the_validator_accepts() -> None:
    """提示词与校验表必须是同一套标签：给模型的 7 类少了谁，
    它就会用别的说法回答，而校验一拒，整条意图线索就只剩规则兜底。"""
    stub = StubLLM()
    _run(stub)
    system = stub.messages[0][0]["content"]
    for label in INTENT_LABELS:
        assert label in system, f"提示词里缺 {label}"


# ---------------------------------------------------------------- ② 三种失败降级
def test_timeout_degrades_without_raising() -> None:
    result = _run(StubLLM(delay=0.5), budget_ms=50)
    assert result.degraded == ("understanding_timeout",)
    assert result.rewritten is None and result.intent is None


def test_upstream_error_degrades_without_raising() -> None:
    result = _run(StubLLM(exc=RuntimeError("429 rate limited")))
    assert result.degraded == ("understanding_failed",)
    assert result.intent is None


@pytest.mark.parametrize(
    "text",
    ["我不确定你想问什么", "{不是 json}", "", '["学习讲解"]'],
    ids=["prose", "broken-json", "empty", "json-array"],
)
def test_unparsable_output_degrades_to_bad_json(text: str) -> None:
    result = _run(StubLLM(text))
    assert result.degraded == ("understanding_bad_json",)


def test_code_fence_around_the_json_is_tolerated() -> None:
    """模型把 JSON 包在 ```json 里是常态，那不是"坏输出"。"""
    result = _run(StubLLM(f"```json\n{GOOD}\n```"))
    assert result.degraded == ()
    assert result.method == "llm"
    assert result.intent is not None and result.intent.label == "学习讲解"


# ---------------------------------------------------------------- ③ 两项独立降级
def test_success_sets_both_fields() -> None:
    result = _run(StubLLM())
    assert result.method == "llm"
    assert result.degraded == ()
    assert result.rewritten == "二次函数的顶点式怎么求？"
    assert result.intent is not None
    assert (result.intent.label, result.intent.method) == ("学习讲解", "model")
    assert result.intent.profile == "kb_stepwise"


def test_bad_rewrite_does_not_cost_the_intent() -> None:
    """`rewrite_rejected` 与 `intent_rejected` 是两条线：一项失败不许连坐另一项。"""
    payload = '{"query": "抱歉我无法回答", "intent": "考试出题", "confidence": 0.9}'
    result = _run(StubLLM(payload))
    assert result.degraded == ("rewrite_rejected",)
    assert result.rewritten is None
    assert result.intent is not None and result.intent.label == "考试出题"


def test_bad_intent_does_not_cost_the_rewrite() -> None:
    payload = '{"query": "二次函数的顶点式怎么求？", "intent": "数学题", "confidence": 0.9}'
    result = _run(StubLLM(payload))
    assert result.degraded == ("intent_rejected",)
    assert result.intent is None
    assert result.rewritten == "二次函数的顶点式怎么求？"


def test_switched_off_item_is_not_reported_as_rejected() -> None:
    """关掉改写时，`query` 字段写得再糟也不该冒出一个 `rewrite_rejected`——
    那会让"模型改写质量"的统计里混进一批我们从没要求它写的东西。"""
    payload = '{"query": "抱歉我无法回答", "intent": "学习规划", "confidence": 0.9}'
    result = _run(StubLLM(payload), want_rewrite=False)
    assert result.degraded == ()
    assert result.rewritten is None
    assert result.intent is not None and result.intent.label == "学习规划"


# ---------------------------------------------------------------- ④ 改写校验
@pytest.mark.parametrize(
    "candidate, original, expected",
    [
        # 空 → 拒
        ("", ORIGINAL, (None, True)),
        # 与原句相同 → **不是失败**（本来就规范），这一条区分"没改"与"改坏了"
        (ORIGINAL, ORIGINAL, (None, False)),
        # 超长：超过 120 字已经是答题，不是改写
        ("二次函数" * 40, ORIGINAL, (None, True)),
        # 换行：检索问句里出现换行说明模型在写结构化内容
        ("二次函数的顶点式\n怎么求", ORIGINAL, (None, True)),
        # 引用标记：`[1]` 说明它把检索问句当答案在写
        ("顶点式怎么求[1]", ORIGINAL, (None, True)),
        # 拒答开头：模型把"我不确定"写进了 query 字段
        ("抱歉，无法确定", ORIGINAL, (None, True)),
        # 丢限定词：8 字以上原句被砍到不足四成，等于静默改变检索范围
        ("勾股定理", "八年级数学的勾股定理怎么证明", (None, True)),
        # 短原句不受四成比例约束（1 字原句的四成没有意义）
        ("熵的定义", "熵", ("熵的定义", False)),
        # 正常改写
        ("二次函数的顶点式怎么求？", ORIGINAL, ("二次函数的顶点式怎么求？", False)),
    ],
    ids=[
        "empty",
        "identical",
        "too-long",
        "newline",
        "citation-mark",
        "refusal-opener",
        "qualifier-dropped",
        "short-original",
        "ok",
    ],
)
def test_rewrite_validation(
    candidate: str, original: str, expected: tuple[str | None, bool]
) -> None:
    assert _valid_rewrite(candidate, original) == expected


# ---------------------------------------------------------------- ⑤ 意图校验
@pytest.mark.parametrize("label", INTENT_LABELS)
def test_every_canonical_label_is_accepted_verbatim(label: str) -> None:
    intent = _valid_intent(label, 0.9)
    assert intent is not None
    assert (intent.label, intent.method) == (label, "model")


@pytest.mark.parametrize("code, label", sorted(CODE2INTENT.items()))
def test_english_codes_are_accepted(code: str, label: str) -> None:
    intent = _valid_intent(code, 0.8)
    assert intent is not None and intent.label == label


@pytest.mark.parametrize(
    "label",
    [
        # 退役的 6 类口径：库里还有存量，但**不许**被归一化着接受（评估会失真）
        "知识点查询",
        "题目讲解",
        "教案素材",
        "学习方法",
        "备课辅助",
        # 词表外
        "数学题",
        "",
        "  ",
        None,
        123,
    ],
    ids=["retired-kp", "retired-exercise", "retired-material", "retired-method",
         "retired-prep", "off-vocabulary", "empty", "blank", "none", "not-a-str"],
)
def test_illegal_labels_are_rejected_not_normalized(label: object) -> None:
    assert _valid_intent(label, 0.95) is None


def test_explicit_unknown_is_accepted_as_unknown() -> None:
    """模型自己说"不确定"是可以的：那正是 `unknown` 该走的路，不是词表外标签。"""
    intent = _valid_intent(INTENT_FALLBACK, 0.9)
    assert intent is not None
    assert intent.label == INTENT_FALLBACK
    assert intent.profile == "generic_clarify"


@pytest.mark.parametrize("confidence", [0.0, 0.3, 0.59, None, "很高", "0.3"])
def test_low_confidence_falls_to_unknown_instead_of_guessing(confidence: object) -> None:
    """§3.2 第 2 条：<0.6 不猜。落 unknown 走 generic_clarify，不硬路由到专业档位。

    `None` / 非数字按 0 处理：拿不到置信度时，"当成没给"比"当成很有把握"安全。
    """
    intent = _valid_intent("学习讲解", confidence)
    assert intent is not None
    assert intent.label == INTENT_FALLBACK
    assert intent.profile == "generic_clarify"
    assert intent.confidence < 0.6


def test_confidence_is_clamped_to_unit_interval() -> None:
    assert _valid_intent("学习讲解", 1.7).confidence == 1.0
    assert _valid_intent("学习讲解", -3).confidence == 0.0
    assert _valid_intent("学习讲解", "0.9").confidence == pytest.approx(0.9)
