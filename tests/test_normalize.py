"""归一化与分词的单测：这一层错一点，FAQ 命中率就掉一大截。"""

from __future__ import annotations

import pytest

from qa_system.res.qa_normalize import Normalizer, normalize_numbers, strip_fillers, strip_punct


@pytest.fixture()
def norm() -> Normalizer:
    return Normalizer(aliases={"学费多少钱": "学费多少"}, extra_words=("勾股定理",))


class TestNormalize:
    def test_polite_prefix_and_particle_removed(self, norm: Normalizer) -> None:
        # 句首礼貌语 + 句尾"？"都要去掉，之后命中别名（学费多少钱 → 学费多少）
        assert norm.normalize("请问一下，学费多少钱？") == "学费多少"
        assert norm.normalize("请问这个课程学多久呢") == "这个课程学多久"

    def test_punctuation_and_space_removed(self, norm: Normalizer) -> None:
        assert norm.normalize(" 勾股 定理 ！？ ") == "勾股定理"

    def test_fullwidth_halfwidth(self, norm: Normalizer) -> None:
        assert norm.normalize("ＢＥＲＴ模型") == norm.normalize("bert模型")

    def test_chinese_number_normalized(self, norm: Normalizer) -> None:
        assert norm.normalize("学三个月") == "学3个月"

    def test_alias_applied_to_canonical(self, norm: Normalizer) -> None:
        assert norm.normalize("学费多少钱") == "学费多少"

    def test_alias_skipped_when_disabled(self, norm: Normalizer) -> None:
        assert norm.normalize("学费多少钱", apply_alias=False) == "学费多少钱"

    def test_empty(self, norm: Normalizer) -> None:
        assert norm.normalize("") == ""
        assert norm.normalize("？！。") == ""

    def test_no_over_strip(self) -> None:
        # "一下午" 不能被句尾语气词规则削成 "午"
        assert strip_fillers("一下午") == "一下午"

    def test_strip_punct_keeps_words(self) -> None:
        assert strip_punct("数学（必修）") == "数学必修"

    def test_math_operators_kept_between_operands(self) -> None:
        # 只差符号的题目必须归一成不同串，否则精确哈希撞车、后者永远命中不到自己（§7.1.2）
        assert strip_punct("因式分解 x²-6x+9") == "因式分解x²-6x+9"
        assert strip_punct("2/3-1/6") == "2/3-1/6"
        assert strip_punct("(a+b)-c") == "a+b-c"
        assert strip_punct("x>3") == "x>3"

    def test_math_operator_canonicalized(self) -> None:
        assert strip_punct("2×3") == strip_punct("2*3")
        assert strip_punct("6÷2") == strip_punct("6/2")

    def test_math_operator_dash_separator_still_removed(self) -> None:
        # 两侧不是 ASCII 操作数时仍是分隔符，不能被当成算子保留（"数学-七年级" → "数学七年级"）
        assert strip_punct("数学-七年级") == "数学七年级"
        assert strip_punct("学费+住宿费") == "学费住宿费"

    def test_unary_minus_kept(self) -> None:
        assert strip_punct("-5的绝对值") == "-5的绝对值"
        assert strip_punct("|-5|") == "-5"

    def test_normalize_numbers_only_with_unit(self) -> None:
        assert normalize_numbers("一二三") == "一二三"
        assert normalize_numbers("三门课") == "3门课"


class TestTokens:
    def test_tokens_non_empty_and_no_stopwords(self, norm: Normalizer) -> None:
        tokens = norm.tokens("勾股定理的内容是什么")
        assert tokens
        assert "的" not in tokens and "是" not in tokens

    def test_tokens_empty_for_punct(self, norm: Normalizer) -> None:
        assert norm.tokens("？？") == []

    def test_extra_words_loaded_after_preload(self, norm: Normalizer) -> None:
        norm.preload()
        assert "勾股定理" in norm.tokens("勾股定理")



