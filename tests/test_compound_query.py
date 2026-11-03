"""复合问句拆分的验收（任务 2.4）：**来源可验，拆不动就不拆**。

这个模块唯一不能让步的规则是「子问题必须有明确来源」——拆出来的每一路子问题都会
各自变成一路检索召回（`planner` 按 `len(subs) >= 2` 走 `SUB_QUERY`，`rag_pipeline._spawn()`
一个 task 一次 `retriever.search()`）。所以一条"看起来合理但原问句里没有"的子问题是
**真的在花检索预算，而且事后无法追溯它是谁编的**。

实现上这条规则是结构性保证而不是承诺：`_verified()` 用**字符多重集**校验，
子问题里的每个字都必须已经在原问句里出现过（次数不超过）。本文件逐条钉住它。

本文件只测拆分器本身（纯函数、无 IO）。"拆出来的子问题有没有真的抵达检索器"
在 `tests/test_chat_understanding.py` 里用替身断言——那里才是接线验收。
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from understanding.compound_query import (
    MAX_SUB_QUERIES,
    _verified,
    looks_compound,
    split,
    split_with_model,
)

# 规格里的原例（任务 2.4）。它的三个子问题是整个模块的黄金样本：
# 一路"区别" + 两路"各出一份练习题"，正是"两件不相干的事各自有自己的召回"。
SPEC = "比较一次函数和二次函数的区别，并分别给我出三道练习题。"
SPEC_SUBS = (
    "一次函数和二次函数的区别",
    "给我出三道一次函数练习题",
    "给我出三道二次函数练习题",
)


# ---------------------------------------------------------------- ① 黄金样本
def test_spec_example_splits_into_three_verifiable_sub_queries() -> None:
    result = split(SPEC)
    assert result.is_complex is True
    # `source` 记的是**哪条规则**拆的：docstring 里的"区分/分别"走 distribute，
    # 排障时能指着一条规则说"就是它"，而不是只看到一个 bool
    assert result.source == "distribute"
    assert result.sub_queries == SPEC_SUBS
    assert result.dropped == ()


def test_every_sub_query_is_a_character_sub_multiset_of_the_original() -> None:
    """来源校验的可观测形态：子问题里不许出现原问句没有的字，且次数不超过。"""
    result = split(SPEC)
    allowed = Counter(SPEC)
    for sub in result.sub_queries:
        assert not (Counter(sub) - allowed), f"子问题引入了原问句里没有的字：{sub!r}"


# ---------------------------------------------------------------- ② 不拆的情形
@pytest.mark.parametrize(
    "query",
    [
        # 称呼 + 逗号，但只有一件诉求：按逗号硬切会被切出"老师"这一路
        "老师，二次函数的顶点式怎么求",
        # 单问句
        "什么是勾股定理",
        # 有"和"但只说了一件事（比较），没有第二件诉求
        "比较一次函数和二次函数的区别",
        # 后半句太短，当一个检索问句只会拉低召回
        "短问，还有呢",
        # 太短：整体低于 _MIN_SUB_LEN(6)
        "在吗",
    ],
)
def test_plain_questions_are_not_split(query: str) -> None:
    result = split(query)
    assert result.is_complex is False
    assert result.sub_queries == ()
    assert result.source == "none"


def test_short_query_returns_default_result_before_any_rule_runs() -> None:
    """短问句在入口就返回，连规则都不跑——空 `dropped` 是这条路径的既定形态。"""
    result = split("在吗")
    assert result == type(result)()


# ---------------------------------------------------------------- ③ 并列诉求（connective）
def test_connective_rule_keeps_both_clauses() -> None:
    result = split("二次函数的顶点式怎么求，还有它的对称轴是什么")
    assert result.is_complex is True
    assert result.source == "connective"
    assert result.sub_queries == ("二次函数的顶点式怎么求", "它的对称轴是什么")


def test_connective_rule_keeps_the_pronoun_and_does_not_resolve_it() -> None:
    """拆分**不解决指代**，只是把分句各自成路。

    「它的对称轴是什么」这一路里"它"原样留着——拆分器没有会话历史，
    它也无权猜测主语（任务 2.5 第 4 条）。指代由 `_load_session()` 的会话拼接解决，
    两条职责刻意分开：这里硬猜一个主语，等于把"裸着出去"的问题藏进检索结果里。
    """
    subs = split("二次函数的顶点式怎么求，还有它的对称轴是什么").sub_queries
    assert "它的对称轴是什么" in subs
    assert "它" in subs[-1]


def test_sub_query_count_is_capped() -> None:
    """四件诉求也只放三路：并行召回的路数有上限，多出来的宁可不拆。"""
    result = split("什么是勾股定理，什么是二次函数，什么是平行线，什么是浮力")
    assert result.is_complex is True
    assert len(result.sub_queries) == MAX_SUB_QUERIES == 3
    assert result.sub_queries == ("什么是勾股定理", "什么是二次函数", "什么是平行线")


# ---------------------------------------------------------------- ④ 来源校验单元
def test_verified_drops_candidates_carrying_foreign_characters() -> None:
    """「三角函数」的"三"在原句子里没有 → 它有一半是编出来的，丢掉而不是修。"""
    kept, dropped = _verified(
        "一次函数与二次函数的区别",
        ["一次函数与二次函数的区别", "二次函数", "三角函数"],
    )
    # 与原问句逐字相同的那条被静默跳过：它不是子问题，是原问题本身
    assert kept == ["二次函数"]
    assert dropped == ["三角函数"]


def test_verified_drops_candidates_below_the_length_floor() -> None:
    kept, dropped = _verified("一次函数的图像", ["一次函数", "函数"])
    assert kept == ["一次函数"]
    assert dropped == ["函数"]


def test_verified_deduplicates() -> None:
    kept, dropped = _verified("什么是勾股定理的应用", ["勾股定理的应用", "勾股定理的应用"])
    assert kept == ["勾股定理的应用"]
    assert dropped == []


# ---------------------------------------------------------------- ⑤ 诊断面
def test_failed_split_reports_no_dropped_candidates() -> None:
    """拆不动时 `dropped` 必须是空的。

    `dropped` 只描述"这条规则产出了候选、但被来源校验挡掉了"。拆不出候选是另一回事，
    两者混在一起，报告里"因为编造被丢弃 N 条"就会把"这条问句本来就不该拆"也算进去。
    """
    result = split("什么是勾股定理")
    assert result.is_complex is False
    assert result.dropped == ()


def test_split_is_deterministic() -> None:
    """同一句话两次拆分给同一结果：这是它能进线上路径的前提（可复现、可单测）。"""
    assert split(SPEC) == split(SPEC)


# ---------------------------------------------------------------- ⑥ 模型拆分（任务 2.4）
# 2026-11-01 起拆分先问本地小模型，模型那条路的三道闸（预筛 / 超时 / 来源校验）
# 各自对应一个失败码。这里逐个钉住：**任何一条失败都退回规则，永不抛异常**，
# 也永不因为"模型没说话"就把已经拆对的规则结果弄丢。
class _StubLLM:
    """只实现拆句用到的那一点：`enabled` + `complete`。"""

    def __init__(
        self,
        *,
        text: str = "",
        error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self.enabled = True
        self._text = text
        self._error = error
        self._delay = delay
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> Any:
        self.calls.append({"messages": messages, "budget_ms": budget_ms, "model": model})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(text=self._text)


def _split_with_model(query: str, llm: Any, *, budget_ms: int = 400) -> Any:
    return asyncio.run(split_with_model(query, llm, budget_ms=budget_ms, model="stub-model"))


def test_no_local_model_falls_back_to_rules_without_degrading() -> None:
    """没配 Ollama 是**部署形态**，不是这次请求出了岔子——不许记进 degraded。

    记了的话，开发机（永远没有本地端点）每条复合问句都会带一个降级码，
    线上告警里的降级率会被这份"本地噪声"永久污染。
    """
    result = _split_with_model(SPEC, None)
    assert result.source == "distribute"
    assert result.sub_queries == SPEC_SUBS
    assert result.degraded == ()


def test_simple_question_never_calls_the_model() -> None:
    """单诉求句在结构预筛就被拦住：一次本地调用都不发。"""
    llm = _StubLLM(text='{"sub_queries": []}')
    _split_with_model("什么是勾股定理", llm)
    assert llm.calls == []


def test_model_split_is_verified_and_marked_as_source_model() -> None:
    llm = _StubLLM(text=json.dumps({"sub_queries": list(SPEC_SUBS)}, ensure_ascii=False))
    result = _split_with_model(SPEC, llm)
    assert result.source == "model"
    assert result.sub_queries == SPEC_SUBS
    assert result.degraded == ()
    # 预算与模型名一路传到网关：上层配了 400 ms / 换模型，不能到不了真正发请求的地方
    assert llm.calls[0]["budget_ms"] == 400
    assert llm.calls[0]["model"] == "stub-model"


def test_model_inventing_words_is_rejected_and_rules_take_over() -> None:
    """模型把"区别"改写成"异同"——原句里没有这两个字，当场丢弃并退回规则。

    这条是整条链路的红线：**宁可退回规则，也不用一句改写过的话去检索**。
    """
    invented = {"sub_queries": ["一次函数和二次函数的异同", "给我出三道练习题"]}
    llm = _StubLLM(text=json.dumps(invented, ensure_ascii=False))
    result = _split_with_model(SPEC, llm)
    assert result.source == "distribute"  # 规则拆的那一份
    assert result.sub_queries == SPEC_SUBS
    assert result.degraded == ("compound_rejected",)


def test_timeout_falls_back_to_rules() -> None:
    llm = _StubLLM(text="{}", delay=0.3)
    result = _split_with_model(SPEC, llm, budget_ms=20)
    assert result.source == "distribute"
    assert result.degraded == ("compound_timeout",)


def test_gateway_error_falls_back_to_rules() -> None:
    llm = _StubLLM(error=RuntimeError("ollama down"))
    result = _split_with_model(SPEC, llm)
    assert result.source == "distribute"
    assert result.degraded == ("compound_failed",)


def test_bad_json_falls_back_to_rules() -> None:
    llm = _StubLLM(text="好的，我把它拆成两件事")  # 没给 JSON
    result = _split_with_model(SPEC, llm)
    assert result.source == "distribute"
    assert result.degraded == ("compound_bad_json",)


def test_model_saying_not_compound_does_not_erase_the_rule_split() -> None:
    """模型回空列表 ≠ 它赢了：规则已经拆对了就不能跟着丢。

    0.8b 偶尔把三诉求句当一件事。信它一次，用户就少拿到两件事的证据；
    而退回规则最坏也只是"多查了两路"，两种代价不对称，所以这里选规则。
    不记降级码：模型没编词，只是判断保守。
    """
    llm = _StubLLM(text=json.dumps({"sub_queries": []}))
    result = _split_with_model(SPEC, llm)
    assert result.sub_queries == SPEC_SUBS
    assert result.degraded == ()

def test_single_comparison_question_never_calls_model() -> None:
    """单纯比较两个知识点仍然只有一个诉求，不应该调用模型。"""
    llm = _StubLLM(text='{"sub_queries": []}')

    query = "比较一次函数和二次函数的区别"

    assert looks_compound(query) is False

    result = _split_with_model(query, llm)

    assert llm.calls == []
    assert result.is_complex is False
    assert result.sub_queries == ()
    assert result.degraded == ()


def test_single_value_question_with_and_never_calls_model() -> None:
    """出现“和”不代表一定有两个诉求。"""
    llm = _StubLLM(text='{"sub_queries": []}')

    query = "求一次函数和二次函数的值"

    result = _split_with_model(query, llm)

    assert llm.calls == []
    assert result.is_complex is False


def test_two_independent_clauses_can_call_model() -> None:
    """两个独立诉求不能被预筛误杀。"""
    llm = _StubLLM(
        text=json.dumps(
            {
                "sub_queries": [
                    "二次函数的顶点式怎么求",
                    "它的对称轴是什么",
                ]
            },
            ensure_ascii=False,
        )
    )

    query = "二次函数的顶点式怎么求，还有它的对称轴是什么"

    result = _split_with_model(query, llm)

    assert len(llm.calls) == 1
    assert result.source == "model"
    assert result.is_complex is True
    assert len(result.sub_queries) == 2


def test_model_returns_one_valid_candidate_marks_rejected() -> None:
    """模型只给一条有效子问题，不能当成成功拆分。"""
    llm = _StubLLM(
        text=json.dumps(
            {"sub_queries": ["一次函数和二次函数的区别"]},
            ensure_ascii=False,
        )
    )

    result = _split_with_model(SPEC, llm)

    assert result.source == "distribute"
    assert result.sub_queries == SPEC_SUBS
    assert result.degraded == ("compound_rejected",)


def test_model_returns_invalid_candidate_type_marks_bad_json() -> None:
    """JSON 能解析不代表结构正确，非字符串候选应该判为坏 JSON。"""
    llm = _StubLLM(
        text=json.dumps(
            {
                "sub_queries": [
                    "一次函数和二次函数的区别",
                    123,
                ]
            },
            ensure_ascii=False,
        )
    )

    result = _split_with_model(SPEC, llm)

    assert result.source == "distribute"
    assert result.sub_queries == SPEC_SUBS
    assert result.degraded == ("compound_bad_json",)


def test_zero_budget_does_not_call_model() -> None:
    """预算耗尽时直接规则回退，不发送模型请求。"""
    llm = _StubLLM(
        text=json.dumps(
            {"sub_queries": list(SPEC_SUBS)},
            ensure_ascii=False,
        )
    )

    result = asyncio.run(
        split_with_model(
            SPEC,
            llm,
            budget_ms=0,
            model="stub-model",
        )
    )

    assert llm.calls == []
    assert result.source == "distribute"
    assert result.sub_queries == SPEC_SUBS
    assert result.degraded == ("compound_timeout",)
