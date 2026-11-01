"""会话改写单测（CLAUDE.md §3.2 第 4 条 / §7.7）。

三条边界各一条用例，因为它们是这个模块唯一可能"改错东西"的地方：
1. 指代句要接上上一轮的问句；
2. 非指代、已经够长的问句**不许**被拼接稀释；
3. 没有历史 / 上一轮是寒暄时原样返回（改写失败等于没改）。
"""

from __future__ import annotations

from dataclasses import dataclass

from understanding.rewrite import needs_rewrite, rewrite


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


def _history(*pairs: tuple[str, str]) -> list[Turn]:
    return [Turn(role=r, content=c) for r, c in pairs]


def test_referential_query_borrows_previous_question() -> None:
    # "它" 离开上文读不懂：不接上一轮，检索只会召回一堆与"它"无关的块
    text, method = rewrite(
        "它的公式是什么", _history(("user", "勾股定理是什么"), ("assistant", "直角三角形的…"))
    )
    assert method == "rule"
    assert text.startswith("勾股定理是什么") and "它的公式是什么" in text


def test_assistant_turn_is_never_used_as_context() -> None:
    # 助手答的是知识库内容，接进检索 query 等于用上一轮答案污染这一轮召回
    text, method = rewrite("这个怎么用", _history(("assistant", "勾股定理的公式是 a²+b²=c²")))
    assert (text, method) == ("这个怎么用", "none")


def test_complete_question_is_left_alone() -> None:
    # 长度够、无指代 → 不触发（拼接只会稀释关键词）
    assert needs_rewrite("二次函数顶点式怎么写") is False
    text, method = rewrite("二次函数顶点式怎么写", _history(("user", "勾股定理是什么")))
    assert (text, method) == ("二次函数顶点式怎么写", "none")


def test_short_query_gets_context_even_without_pronoun() -> None:
    # §3.2：长度 < 8 字同样触发——"怎么证"这种省略句也需要上文
    text, method = rewrite("怎么证明", _history(("user", "勾股定理是什么")))
    assert method == "rule" and "勾股定理是什么" in text


def test_only_first_sentence_of_previous_turn_is_used() -> None:
    # 上一轮常常是"什么是X？顺便问下Y"，整段接进来会跑题
    text, method = rewrite("它呢", _history(("user", "什么是勾股定理？顺便问下三角函数")))
    assert method == "rule"
    assert "三角函数" not in text and "勾股定理" in text


def test_smalltalk_previous_turn_is_skipped() -> None:
    # "谢谢"这类上一轮接进来只有噪音，宁可原文查一次
    assert rewrite("它呢", _history(("user", "谢谢")))[1] == "none"


def test_no_history_is_a_no_op() -> None:
    assert rewrite("它是什么", []) == ("它是什么", "none")
    assert rewrite("", _history(("user", "勾股定理是什么"))) == ("", "none")


def test_rewritten_query_is_capped() -> None:
    # 检索 query 太长（>120 字）BM25 与向量都会失去焦点，必须截断
    long_q = "勾股定理" * 40
    text, method = rewrite("它的证明呢", _history(("user", long_q)))
    assert method == "rule" and len(text) <= 120
