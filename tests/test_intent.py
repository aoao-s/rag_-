"""理解层单测（CLAUDE.md §7.7；阶段 2 任务 2.1 的 7 类业务意图）。

测三层，因为三层责任不同：

- **分类层**（`IntentClassifier`）：只出标签与置信度，不许碰学科、不许抛异常、低置信落 unknown。
- **档位层**（`profiles.py`）：标签 → 档位的**数据表**；非法标签一律 generic_clarify。
- **路由安全**（`resolve_profile`）：两个方向相反的守卫——`闲聊` 不许因疑问形态跳过检索，
  `超出范围` 不许因漏掉场景词而拒答；`FEATURE_INTENT=false` 必须精确等价于 `BASELINE`。

铁律：`IntentResult` 没有 `subject` 字段（§7.7 红线 1）——学科只由 `SlotResult` 给。

**旧 6 类标签（知识点查询 / 题目讲解 / 教案素材 / 学习方法 / 备课辅助）在本文件里
只作为"必须被拒"的反例出现**：它们是滚动发布期间的存量写入，任何一条被路由到专业档位
都意味着旧缓存能绕过新口径的分类结果。
"""

from __future__ import annotations

from dataclasses import fields

from base.contracts import IntentResult
from base.labels import INTENT_FALLBACK, INTENT_LABELS
from understanding.intent import IntentClassifier, classify
from understanding.profiles import (
    BASELINE,
    GENERIC_CLARIFY,
    NO_RETRIEVAL,
    OUT_OF_SCOPE,
    PROFILES,
    looks_like_question,
    profile_for,
    resolve_profile,
)

_CLF = IntentClassifier()

# 已退役的 6 类口径。它们仍在库里（`request_log.intent` 与共享缓存条目），
# 所以"不认它"这件事必须被测试钉住，而不是靠"反正代码里没写"。
_RETIRED_LABELS = ("知识点查询", "题目讲解", "教案素材", "学习方法", "备课辅助")


# ---------------------------------------------------------------- 分类层
def test_each_intent_has_a_representative_query() -> None:
    # 七类各一条典型问法。这些是"线上必须认得出"的锚点，改词表时先跑它们
    cases = {
        "教案制作": "帮我设计一份《氧气》的教案",
        "考试出题": "出一套二次函数的专项训练题，带答案解析",
        "备课检索": "有没有适合初二物理的课件可以参考",
        "学习讲解": "什么是勾股定理",
        "学习规划": "还有三个月就中考了，怎么提分",
        "闲聊": "你好呀",
        "超出范围": "明天成都天气怎么样",
    }
    for label, query in cases.items():
        result = _CLF.classify(query)
        assert result.label == label, f"{query} → {result.label}（期望 {label}）"


def test_all_seven_labels_plus_unknown_are_reachable() -> None:
    # 7+1 档路由全覆盖：每个标签都必须在 PROFILES 里有一行
    assert set(PROFILES) == set(INTENT_LABELS) | {INTENT_FALLBACK}
    assert len(PROFILES) == 8


def test_result_has_no_subject_field() -> None:
    # §7.7 红线 1：意图不碰学科。字段不存在，就没人能误用它做过滤
    assert "subject" not in {f.name for f in fields(IntentResult)}


def test_unknown_instead_of_guessing() -> None:
    # 长句、无关键词、也不像学业问题 → unknown，不许硬塞进某个专业档位
    result = _CLF.classify("那个东西大概齐呗反正我也说不清楚你想咋办就咋办吧")
    assert result.label == INTENT_FALLBACK
    assert result.method == "fallback"
    assert result.profile == GENERIC_CLARIFY.name
    assert result.confidence < 0.6


def test_short_utterance_is_chitchat_only_without_question_shape() -> None:
    # 短句兜底是"寒暄"的代理信号，但它不许吞掉短问句：
    # "怎么弄" 一旦被当成寒暄就会跳过检索，所以带疑问形态的一律落 unknown
    assert _CLF.classify("好的收到").label == "闲聊"
    assert _CLF.classify("怎么弄").label == INTENT_FALLBACK


def test_empty_query_is_unknown_not_exception() -> None:
    for query in ("", "   "):
        result = _CLF.classify(query)
        assert result.label == INTENT_FALLBACK and result.method == "fallback"


def test_academic_hint_alone_is_unknown() -> None:
    # 有学业语义但没命中任何意图词：走 unknown（全库检索 + 追问），不是闲聊
    result = _CLF.classify("那个单元的课文大概讲了什么内容呢你能告诉我一些细节吗")
    assert result.label == INTENT_FALLBACK


def test_confident_classification_reports_rule_method() -> None:
    result = _CLF.classify("什么是勾股定理")
    assert result.method == "rule"
    assert 0.55 <= result.confidence <= 0.99
    assert result.profile == "kb_stepwise"


def test_generate_vs_lookup_is_decided_by_the_verb() -> None:
    # 任务 2.1 里最容易混的一对：同一个名词（讲义），动词不同、类别不同。
    # 这条断言的意义是**钉住判据**——将来谁把"帮我出一份"当成出题信号，这里立刻红
    assert _CLF.classify("帮我设计一份《氧气》的讲义").label == "教案制作"
    assert _CLF.classify("帮我找一份《氧气》的讲义").label == "备课检索"


# ---------------------------------------------------------------- 档位层
def test_profile_table_values() -> None:
    # 改行为 = 改这张表，所以表里每一行的关键字段都钉死
    assert PROFILES["教案制作"].name == "material_generate"
    assert PROFILES["考试出题"].name == "exam_assemble"
    # 出题档位不搜教材：教材里没有成套题，搜进来只会稀释证据
    assert PROFILES["考试出题"].doc_types == ("exam", "question_bank")
    assert PROFILES["备课检索"].name == "doc_lookup"
    # 学习讲解合并了旧的"知识点查询 + 题目讲解"，doc_types 取并集
    assert PROFILES["学习讲解"].name == "kb_stepwise"
    assert PROFILES["学习讲解"].doc_types == ("question_bank", "exam", "textbook", "lesson_plan")
    assert PROFILES["学习规划"].name == "general_advisory"
    # 方法类证据散落，收窄只会把答案挡在外面
    assert PROFILES["学习规划"].doc_types == ()
    assert PROFILES["学习规划"].allow_general is True
    # 只有这两类不检索
    assert {p.name for p in PROFILES.values() if not p.retrieval} == {
        "no_retrieval",
        "out_of_scope",
    }


def test_out_of_scope_is_the_only_refusing_profile() -> None:
    # `allow_general` 是"闲聊"与"超出范围"之间**唯一**的字段差别，拒答分支只读它、
    # 不读标签字符串。所以它一旦被改错，拒答就会静默变成寒暄
    refusing = [p.name for p in PROFILES.values() if not p.retrieval and not p.allow_general]
    assert refusing == ["out_of_scope"]
    assert OUT_OF_SCOPE.prompt_id != NO_RETRIEVAL.prompt_id


def test_only_cacheable_profiles_are_shared_cache_eligible() -> None:
    # §7.7 作用点 7 + §7.9 边界：强依赖记忆/学情、以及"没听懂"的档位不许写共享缓存
    uncacheable = {p.name for p in PROFILES.values() if not p.cacheable}
    assert uncacheable == {"no_retrieval", "out_of_scope", "general_advisory", "generic_clarify"}


def test_illegal_label_never_routes_to_professional_profile() -> None:
    # 退役的 6 类标签 + 空串 + 随机串，一个都不许落到专业档位
    for label in (*_RETIRED_LABELS, "", "招生咨询", "知识点"):
        assert profile_for(label) is GENERIC_CLARIFY, f"{label} 不该被路由到专业档位"


def test_generic_clarify_keeps_citation_gate() -> None:
    # 对 §7.7 表格的有意偏离（见 profiles.py 注释）：最不确定的档位最该守住引用闸门
    assert GENERIC_CLARIFY.require_citation is True
    assert GENERIC_CLARIFY.allow_general is False


# ---------------------------------------------------------------- 路由安全
def test_no_retrieval_guarded_by_question_shape() -> None:
    # 带疑问形态 → 不许跳过检索（宁可多查一轮）
    assert looks_like_question("英语中如何提出建议？") is True
    profile, degraded = resolve_profile("英语中如何提出建议？", "闲聊")
    assert profile is GENERIC_CLARIFY
    assert degraded == ("intent_no_retrieval_guarded",)

    # 纯寒暄没有疑问形态 → 允许跳过检索
    profile, degraded = resolve_profile("你好呀", "闲聊")
    assert profile is NO_RETRIEVAL and degraded == ()


def test_out_of_scope_guarded_by_educational_signal() -> None:
    # 方向相反的守卫：准备拒答时，只要问句里带教学场景词就不许拒。
    # 误拒答的代价是"用户拿不到本来就有的答案"，比多查一轮严重得多
    profile, degraded = resolve_profile("这个单元的作业怎么设计才合理", "超出范围")
    assert profile is GENERIC_CLARIFY
    assert degraded == ("intent_out_of_scope_guarded",)

    # 真的跟教育无关 → 照常拒答
    profile, degraded = resolve_profile("明天成都天气怎么样", "超出范围")
    assert profile is OUT_OF_SCOPE and degraded == ()


def test_resolve_profile_ignores_label_when_feature_off() -> None:
    # 回滚开关：关掉意图后任何标签都落到同一个 BASELINE，且不记降级标记
    for label in [*INTENT_LABELS, INTENT_FALLBACK, *_RETIRED_LABELS]:
        profile, degraded = resolve_profile("随便问点什么", label, feature_intent=False)
        assert profile is BASELINE and degraded == ()


def test_baseline_is_full_scope_direct() -> None:
    # A/B 的对照组口径：不限 doc_type、必须检索、必须引用
    assert BASELINE.doc_types == ()
    assert BASELINE.retrieval is True
    assert BASELINE.strategy == "DIRECT"
    assert BASELINE.require_citation is True


def test_module_level_classify_matches_instance() -> None:
    assert classify("什么是勾股定理") == _CLF.classify("什么是勾股定理")
