"""查询预处理（CLAUDE.md §7.2.7-B 第 2 步）：归一化只影响向量质量，非法入参挡在检索之前。

与 FAQ 的归一化**故意不共用**：那边服务于"归一化串精确相等"，动一个标点就改缓存 key。
这里若也去删词，两边会互相牵制，谁都改不动。
"""

from __future__ import annotations

import pytest

from base.errors import InvalidRequest
from rag_system.core.query_processor import MAX_QUERY_CHARS, normalize, process


# ---- normalize ----
def test_nfkc_turns_fullwidth_into_ascii() -> None:
    """NFKC 会把全角字母/数字/标点压成半角；**同时把 `²` 压成 `2`**（见下条）。"""
    assert normalize("ｙ＝ａｘ＋ｂ") == "y=ax+b"


def test_nfkc_collapses_superscript_digits() -> None:
    """`x²` → `x2` 是 NFKC 的既定行为，检索侧接受它：
    BGE-M3 对 `x2` 与 `x²` 的表示差异远小于两者与整句的差异，不值得为它绕开 NFKC。
    FAQ 侧同理（§7.1.2 已记录），但那边靠精确哈希，影响面不同。"""
    assert normalize("x²") == "x2"


def test_whitespace_and_newlines_collapse() -> None:
    assert normalize("  二次函数\n\n  顶点式\t怎么求  ") == "二次函数 顶点式 怎么求"


def test_repeated_punctuation_collapses_to_one() -> None:
    assert normalize("真的吗？？？") == "真的吗?"
    assert normalize("等等。。。。。") == "等等。"


def test_single_punctuation_is_untouched() -> None:
    # 全角标点经 NFKC 变半角，但"只有一个"这件事必须保持
    assert normalize("勾股定理，怎么证明？") == "勾股定理,怎么证明?"


def test_normalize_never_drops_words() -> None:
    """检索归一化不删词——"请问一下"这种口语留给改写层，删掉了改写就拿不到原话。"""
    text = "请问一下二次函数的顶点式怎么求呢"
    assert normalize(text) == text


def test_math_operators_are_preserved() -> None:
    """`-` / `+` 是实词级别信息，删了 `x²-6x+9` 与 `x²+6x+9` 就撞成同一句。"""
    assert normalize("因式分解 x²-6x+9") == "因式分解 x2-6x+9"


def test_normalize_is_idempotent() -> None:
    once = normalize("　二次函数\n\n顶点式？？")
    assert normalize(once) == once


# ---- process ----
def test_empty_query_is_invalid_request() -> None:
    for bad in ("", "   ", "\n\t "):
        with pytest.raises(InvalidRequest) as exc:
            process(bad)
        assert exc.value.code == "invalid_request"


def test_normal_processing_reports_no_degradation() -> None:
    out = process("  二次函数的顶点式怎么求 ")
    # `original` 是**去首尾空白后**的原话：检索不需要那些空白，保留它们只会让日志多出噪声
    assert out.original == "二次函数的顶点式怎么求"
    assert out.normalized == "二次函数的顶点式怎么求"
    assert out.degraded == ()
    assert out.changed is False


def test_changed_flag_tracks_real_normalization() -> None:
    """`changed` 是给日志用的"这次归一化动过没"，不是降级信号。"""
    assert process("ｙ＝ａｘ²＋ｂ").changed is True


def test_too_long_query_is_truncated_not_rejected() -> None:
    """超长多半是粘贴了整段材料：截断后仍可检索，报错等于把整段输入判死。"""
    long_text = "二次函数" * 200
    out = process(long_text)
    assert out.degraded == ("query_truncated",)
    assert len(out.normalized) <= MAX_QUERY_CHARS


def test_exactly_at_limit_is_not_truncated() -> None:
    text = "a" * MAX_QUERY_CHARS
    out = process(text)
    assert out.degraded == ()
    assert out.normalized == text
