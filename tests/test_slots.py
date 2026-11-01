"""槽位抽取与检索路由单测（CLAUDE.md §7.1.8）。

覆盖三类最容易"静默降级"的场景：学科拼错、学科合法但无数据、精确命中绕过学科过滤。
"""

from __future__ import annotations

import asyncio

from base.contracts import SlotResult
from base.labels import normalize_subject
from qa_system.res import qa_slots
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository
from qa_system.res.qa_matcher import MatchPolicy, decide, decide_multi
from qa_system.res.qa_search import FAQService


# --------------------------------------------------------------------------- 词表
def test_politics_alias_maps_to_current_name() -> None:
    # 语料写"道德与法治"，旧标签表只有"政治"——这 97 条以前全被归一成 None
    assert normalize_subject("政治") == "道德与法治"
    assert normalize_subject("道法") == "道德与法治"
    assert normalize_subject("道德与法治") == "道德与法治"
    assert normalize_subject("政治学") is None


# --------------------------------------------------------------------------- 参数
def test_from_param_semantics() -> None:
    assert qa_slots.from_param(None).subject_source == "not_provided"
    assert qa_slots.from_param("  ").subject_source == "not_provided"

    explicit = qa_slots.from_param("数学学科")
    assert explicit.subject == "数学" and explicit.subject_source == "explicit"
    assert explicit.subject_confidence == 1.0

    invalid = qa_slots.from_param("物理课")
    assert invalid.subject is None and invalid.subject_source == "invalid"


# --------------------------------------------------------------------------- 问句
def test_extract_single_and_multi() -> None:
    one = qa_slots.extract("数学题怎么做")
    assert one.subject == "数学" and one.subject_source == "explicit"

    many = qa_slots.extract("物理和化学有什么区别")
    assert many.subject_source == "multi"
    assert many.subjects == ("物理", "化学")
    assert many.subject == "物理"


def test_extract_unknown_and_grade() -> None:
    assert qa_slots.extract("学费多少钱").subject_source == "unknown"

    # 年级出口是 `labels.GRADES` 的规范名（初二 → 八年级）：Milvus 的 grade 字段、
    # `Atom.grade`、引用卡片全用这一套写法，抽出来的槽位也必须用，否则按年级任何比较都零命中
    graded = qa_slots.extract("初二数学怎么学")
    assert graded.subject == "数学" and graded.grade == "八年级"

    seven = qa_slots.extract("七年级的地理难吗")
    assert seven.subject == "地理" and seven.grade == "七年级"

    # 已经是规范名的原样保留，不再二次改写
    assert qa_slots.extract("九年级数学函数").grade == "九年级"
    assert qa_slots.extract("高2物理").grade == "高二"


def test_resolve_priority() -> None:
    # 参数写错不能被问句悄悄覆盖，否则用户永远发现不了自己传错了
    assert qa_slots.resolve("数学题", param="物理课").subject_source == "invalid"
    # 参数为空则看问句
    assert qa_slots.resolve("数学题", param=None).subject == "数学"
    # 问句没有学科时用历史提示，并明确标成 context（不直接答）
    context = qa_slots.resolve("怎么复习", history_hint="数学")
    assert context.subject == "数学" and context.subject_source == "context"


# --------------------------------------------------------------------------- 冲突
def test_param_and_query_disagree_becomes_conflict_not_silent_param() -> None:
    """参数说化学、问句说数学：两边都是用户写的字，选谁都是替他猜。

    静默采信参数的失败形状不是报错，而是**用错的学科做了硬过滤**——正确证据被挡在
    索引外面，而且事后从响应体里完全看不出来（`explicit / 化学` 看起来像一次正常的
    参数指定）。所以结论必须是"不选边"：学科清空、检索退回全库。
    """
    slots = qa_slots.resolve("数学的二次函数怎么求", param="化学")
    assert slots.subject_source == "conflict"
    assert slots.subject is None
    assert slots.subjects == ()
    assert slots.scope == ()  # 空 = 全库，`_context()` 据此不加 subject 过滤
    assert slots.subject_confidence == 0.0


def test_conflict_keeps_the_non_subject_slots_from_the_query() -> None:
    """冲突只清学科。年级、章节、任务参数照旧来自问句——它们与"查哪一门"无关。"""
    slots = qa_slots.resolve("初二数学的二次函数，出5道题", param="化学")
    assert slots.subject_source == "conflict"
    assert slots.subject is None
    assert slots.grade == "八年级"
    assert slots.task.exam.question_count == 5


def test_conflict_covers_multi_subject_queries() -> None:
    """问句点名多门而参数不在这几门里，同样算冲突（不能只比 `subject` 第一个）。"""
    slots = qa_slots.resolve("物理和化学有什么区别", param="历史")
    assert slots.subject_source == "conflict"
    assert slots.scope == ()


def test_param_inside_a_multi_subject_query_is_not_a_conflict() -> None:
    """"数学和物理的题" + `param=数学`：用户只是没把话说全，参数照旧生效。

    这不是矛盾而是收窄，且是用户明确要求的；把它判成冲突等于把他指定的一门丢掉，
    两边都查一遍——比他要的更费预算，也更不精确。
    """
    slots = qa_slots.resolve("数学和物理的题目怎么讲", param="数学")
    assert slots.subject_source == "explicit"
    assert slots.subject == "数学"


def test_history_hint_never_conflicts_with_the_query() -> None:
    """上一轮聊数学、这一轮问化学：主题切换是正常对话，不是冲突。

    这一轮说的算这一轮的——拿历史去否决用户当前写下的学科，正是"指代借用"
    那条错误过滤的另一种翻身方式。
    """
    slots = qa_slots.resolve("化学方程式怎么配平", history_hint="数学")
    assert slots.subject_source == "explicit"
    assert slots.subject == "化学"


def test_conflict_does_not_mask_an_invalid_param() -> None:
    """参数不在词表时仍是 `invalid`：它要被 400 暴露，不能被问句或冲突态盖住。"""
    assert qa_slots.resolve("数学的二次函数", param="物理课").subject_source == "invalid"


# --------------------------------------------------------------------------- 路由
def _service() -> FAQService:
    repo = InMemoryFAQRepository(
        [
            FAQRecord(1, "数学", "勾股定理的内容是什么？", "两直角边平方和等于斜边平方。"),
            FAQRecord(2, "数学", "勾股定理怎么证明？", "面积法。"),
            FAQRecord(3, "物理", "牛顿第二定律是什么", "F=ma。"),
            FAQRecord(4, None, "认识自己有哪些方法", "自我观察与他人评价。"),
        ]
    )
    # threshold=0 让"路由"成为唯一变量，不被分数门槛干扰
    service = FAQService(repo, policy=MatchPolicy(threshold=0.0, margin=0.0), top_k=10)
    asyncio.run(service.load())
    return service


def test_filtered_scope_hits_only_that_subject() -> None:
    hit = asyncio.run(
        _service().search("勾股定理的内容", slots=SlotResult(subject="数学", subject_source="explicit"))
    )
    assert hit.hit and hit.subject == "数学"
    assert hit.subject_state == "filtered"
    assert hit.degraded == ()


def test_empty_subject_does_not_fall_back_to_full() -> None:
    hit = asyncio.run(
        _service().search("勾股定理的内容", slots=SlotResult(subject="化学", subject_source="explicit"))
    )
    assert not hit.hit and hit.reason == "subject_empty"
    assert hit.subject_state == "empty"


def test_invalid_subject_falls_back_with_degraded_flag() -> None:
    service = _service()
    hit = asyncio.run(service.search("勾股定理的内容", slots=SlotResult(subject_source="invalid")))
    assert hit.subject_state == "full"
    assert "subject_invalid_fallback" in hit.degraded
    assert service.stats()["routes"]["full"] >= 1


def test_exact_match_respects_subject_scope() -> None:
    # exact 是全局哈希表：不校验范围的话，"限定物理"也会直接吐出数学答案
    hit = asyncio.run(
        _service().search(
            "勾股定理的内容是什么？", slots=SlotResult(subject="物理", subject_source="explicit")
        )
    )
    assert not hit.hit


def test_inferred_subject_never_answers_directly() -> None:
    hit = asyncio.run(
        _service().search("勾股定理的内容", slots=SlotResult(subject="数学", subject_source="inferred"))
    )
    assert not hit.hit and hit.reason == "subject_inferred_handoff"
    assert hit.candidates  # 仍把候选交给 RAG


def test_multi_subject_uses_multi_scope() -> None:
    hit = asyncio.run(
        _service().search(
            "勾股定理和牛顿第二定律",
            slots=SlotResult(subject="数学", subjects=("数学", "物理"), subject_source="multi"),
        )
    )
    assert hit.subject_state == "multi"


# --------------------------------------------------------------------------- 判定
def test_single_candidate_never_hits() -> None:
    result = decide([(1, 40.0)], MatchPolicy(threshold=0.0, margin=0.0))
    assert not result.hit and result.reason == "single_candidate"


def test_relative_margin_mode() -> None:
    scored = [(1, 10.0), (2, 9.0)]  # 相对分差 0.10
    loose = decide(scored, MatchPolicy(threshold=0.0, margin=0.05, margin_mode="relative"))
    strict = decide(scored, MatchPolicy(threshold=0.0, margin=0.15, margin_mode="relative"))
    assert loose.hit and loose.margin_relative == 0.1
    assert not strict.hit and strict.reason == "ambiguous"


def test_multi_requires_cross_subject_delta() -> None:
    per_subject = {"数学": [(1, 10.0), (2, 1.0)], "物理": [(3, 9.9)]}
    ok = decide_multi(per_subject, MatchPolicy(threshold=0.0, margin=0.05), cross_delta=0.0)
    assert ok.hit and ok.faq_id == 1
    blocked = decide_multi(per_subject, MatchPolicy(threshold=0.0, margin=0.05), cross_delta=0.5)
    assert not blocked.hit and blocked.reason == "cross_subject_ambiguous"
