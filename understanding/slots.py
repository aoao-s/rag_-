"""模型版槽位：校验 + 「规则优先，模型补空」（阶段 2 任务 2.3）。

三步，顺序不能换：

1. `parse_model()` —— 把大模型给的 `slots` 对象**逐字段校验**成 `ModelSlots`。
   词表外的值一律丢弃并计数，**不做"就近归一"**：槽位会被拼进提示词与检索过滤，
   一个词表外的值不是"多一种写法"，而是"永远零命中"（§7.1.8 的 97 条 unlabeled 就是这么来的）。
2. `merge()` —— 规则优先、模型补空。规则抽到的一律不动，模型只填 None / 空元组。
3. `finalize()` —— 拿到意图结论之后补 `TaskSlots.kind` 与 `PrepSlots` 投影，
   并给出"必要信息缺失"时的澄清问句。

**过滤类槽位（学科 / 年级 / 教材版本 / 学期）模型给的为什么一律丢弃**：
§7.1.8 的安全不对称——规则抽不到只是"全库多查一轮"（安全失效），
模型抽错却是"正确答案被挡在索引外面"（危险失效）。两者代价差一个数量级，
所以这一组只认 `qa_system/res/qa_slots.py` 的规则与词典，模型的同名字段丢弃并计数。

本模块**不 import `qa_system` / `rag_system`**（§5.1 分层）：它是 `understanding/` 的一部分，
要的是"规则已经算好了、我只做合并"，而不是复用规则实现的内部函数。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any

from base.contracts import (
    ExamSlots,
    LessonSlots,
    PlanSlots,
    PrepSlots,
    SlotResult,
    TaskKind,
    TaskSlots,
)
from base.labels import (
    normalize_difficulty,
    normalize_exam_type,
    normalize_lesson_type,
    normalize_output_format,
    normalize_question_type,
    normalize_resource_type,
)

# 意图 → 任务种类。**唯一出处**：加/改意图时只改 `base/labels.py` 与这张表。
# 讲解 / 闲聊 / 超出范围没有对应任务，`kind` 保持 None。
_TASK_KIND_BY_INTENT: dict[str, TaskKind] = {
    "教案制作": "lesson",
    "考试出题": "exam",
    "备课检索": "prep",
    "学习规划": "plan",
}

# 模型**不得**写入的字段（见模块开头）。命中即丢弃，字段名进日志便于发现"模型在猜学科"。
_FILTER_FIELDS: tuple[str, ...] = ("subject", "subjects", "grade", "textbook_version", "semester")

# 模型常用的"我没有这个值"写法：它们是占位符不是取值，按缺省处理。
_PLACEHOLDERS: frozenset[str] = frozenset({"无", "未知", "不确定", "none", "null", "n/a", "na", "-"})

_LIST_SPLIT = re.compile(r"[、，,；;/|]+")
_MIX_RE = re.compile(r"^(\d{1,2})\s*[:：]\s*(\d{1,2})\s*[:：]\s*(\d{1,2})$")


# ---------------------------------------------------------------- 基础清洗
def _clean_text(raw: Any, *, max_len: int) -> str | None:
    """取一段自由文本槽位；占位符、非字符串、超长一律当"没有"。"""
    if not isinstance(raw, str):
        return None
    value = raw.strip().strip("'\"“”‘’")
    if not value or value.lower() in _PLACEHOLDERS or len(value) > max_len:
        return None
    return value


def _as_list(raw: Any, *, max_len: int, limit: int) -> tuple[str, ...]:
    """多值槽位：接受 list / tuple / 顿号分隔的字符串，清洗后去重并截断。

    模型忽而给 `["A","B"]`、忽而给 `"A、B"` 是常态。两种都收，但**收进来的元素
    必须重新过长度与占位符校验**——否则它给个空串，下游就会拿空串去匹配。
    """
    if isinstance(raw, str):
        parts: list[Any] = _LIST_SPLIT.split(raw)
    elif isinstance(raw, (list, tuple)):
        parts = list(raw)
    else:
        return ()
    items: list[str] = []
    for part in parts:
        value = _clean_text(part, max_len=max_len)
        if value is None or len(value) < 2:
            continue
        if value not in items:
            items.append(value)
    return tuple(items[:limit])


def _as_int(raw: Any, *, low: int, high: int) -> int | None:
    """整数槽位：模型常把数字写成字符串（`"10"`），也常溢出（`"1000道"`）。"""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        value = raw
    elif isinstance(raw, float):
        value = int(raw)
    elif isinstance(raw, str) and raw.strip().isdigit():
        value = int(raw.strip())
    else:
        return None
    return max(low, min(value, high))


def _as_float(raw: Any, *, low: float, high: float) -> float | None:
    if isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return max(low, min(value, high))


def _as_bool(raw: Any) -> bool | None:
    """三态：`None` 表示用户没表态，**不等于 False**（出题模板据此决定默认）。"""
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in ("true", "yes", "1", "需要", "要", "是"):
            return True
        if value in ("false", "no", "0", "不需要", "不用", "否"):
            return False
    return None


def _pick(rule_value: Any, model_value: Any) -> Any:
    """规则优先、模型补空：`None` / `()` / `""` 才算"规则没抽到"。"""
    return rule_value if rule_value not in (None, (), "") else model_value


# ---------------------------------------------------------------- ① 校验
@dataclass(frozen=True, slots=True)
class ModelSlots:
    """模型给的槽位中**通过校验**的部分。

    注意这里**没有** subject / grade / textbook_version / semester：
    它们在 `parse_model()` 就被丢掉了，不是忘了写——类型上不给，
    下游就不可能误用模型的学科去过滤。
    """

    chapter: str | None = None
    knowledge_points: tuple[str, ...] = ()
    difficulty: str | None = None
    exam_type: str | None = None
    task: TaskSlots = field(default_factory=TaskSlots)

    @property
    def empty(self) -> bool:
        """模型什么都没给出——这时候连"模型给了槽位"这句日志都不该写。"""
        return (
            self.chapter is None
            and not self.knowledge_points
            and self.difficulty is None
            and self.exam_type is None
            and self.task == TaskSlots()
        )


def _parse_task(raw: Any) -> TaskSlots:
    """任务专属槽位。`kind` **不从模型读**——那是意图的结论（见 `finalize()`）。"""
    if not isinstance(raw, dict):
        return TaskSlots()
    lesson = raw.get("lesson") if isinstance(raw.get("lesson"), dict) else {}
    exam = raw.get("exam") if isinstance(raw.get("exam"), dict) else {}
    prep = raw.get("prep") if isinstance(raw.get("prep"), dict) else {}
    plan = raw.get("plan") if isinstance(raw.get("plan"), dict) else {}

    return TaskSlots(
        lesson=LessonSlots(
            duration_min=_as_int(lesson.get("duration_min"), low=5, high=300),
            objectives=_as_list(lesson.get("objectives"), max_len=60, limit=5),
            lesson_type=normalize_lesson_type(_clean_text(lesson.get("lesson_type"), max_len=10)),
            output_format=normalize_output_format(
                _clean_text(lesson.get("output_format"), max_len=12)
            ),
        ),
        exam=ExamSlots(
            question_count=_as_int(exam.get("question_count"), low=1, high=100),
            question_types=tuple(
                keep
                for keep in (
                    normalize_question_type(item)
                    for item in _as_list(exam.get("question_types"), max_len=10, limit=5)
                )
                if keep
            ),
            difficulty_mix=_mix(exam.get("difficulty_mix")),
            total_score=_as_int(exam.get("total_score"), low=1, high=300),
            need_analysis=_as_bool(exam.get("need_analysis")),
        ),
        prep=PrepSlots(
            resource_types=tuple(
                keep
                for keep in (
                    normalize_resource_type(item)
                    for item in _as_list(prep.get("resource_types"), max_len=10, limit=5)
                )
                if keep
            ),
        ),
        plan=PlanSlots(
            target_exam=normalize_exam_type(_clean_text(plan.get("target_exam"), max_len=12)),
            exam_date=_clean_text(plan.get("exam_date"), max_len=20),
            current_level=_clean_text(plan.get("current_level"), max_len=30),
            weak_points=_as_list(plan.get("weak_points"), max_len=20, limit=5),
            weekly_hours=_as_float(plan.get("weekly_hours"), low=0.0, high=80.0),
        ),
    )


def _mix(raw: Any) -> str | None:
    """难度分布（`"3:5:2"`）：只认三段比例，全角冒号归一成半角。"""
    if not isinstance(raw, str):
        return None
    match = _MIX_RE.match(raw.strip())
    return ":".join(match.groups()) if match else None


def parse_model(raw: Any) -> tuple[ModelSlots, tuple[str, ...]]:
    """校验模型槽位 → `(可用部分, 降级标记)`。

    降级标记会并进 `SlotResult.degraded`，与 `rewrite_rejected` / `intent_rejected` 同一口径：
    **模型给的东西不可用时，链路的其余部分照常**，不能因为槽位不合法就把意图与改写一起丢掉。
    """
    if raw is None:
        return ModelSlots(), ()
    if not isinstance(raw, dict):
        return ModelSlots(), ("slot_model_invalid",)

    degraded: list[str] = []
    dropped = [name for name in _FILTER_FIELDS if _has_value(raw.get(name))]
    if dropped:
        # 不抛错也不静默：这条日志是"模型在猜学科"的唯一证据
        degraded.append("slot_model_filter_dropped")

    slots = ModelSlots(
        chapter=_clean_text(raw.get("chapter"), max_len=30),
        knowledge_points=_as_list(raw.get("knowledge_points"), max_len=20, limit=5),
        difficulty=normalize_difficulty(_clean_text(raw.get("difficulty"), max_len=10)),
        exam_type=normalize_exam_type(_clean_text(raw.get("exam_type"), max_len=12)),
        task=_parse_task(raw.get("task")),
    )
    # 词表外的值是被"吞掉"而不是报错：difficulty="有点难" 就是没有难度，不是 hard
    if _has_value(raw.get("difficulty")) and slots.difficulty is None:
        degraded.append("slot_model_value_rejected")
    if _has_value(raw.get("exam_type")) and slots.exam_type is None:
        degraded.append("slot_model_value_rejected")
    return slots, tuple(dict.fromkeys(degraded))


def _has_value(raw: Any) -> bool:
    """模型"给了这个字段"吗（空串/空列表算没给，避免把占位当信号）。"""
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(raw.strip()) and raw.strip().lower() not in _PLACEHOLDERS
    if isinstance(raw, (list, tuple, dict)):
        return bool(raw)
    return True


# ---------------------------------------------------------------- ② 合并
def merge(rule: SlotResult, model: ModelSlots) -> SlotResult:
    """规则优先、模型补空。

    **多值槽位整体取舍而不是取并集**：规则抽出 `知识点：A、B` 时，模型的 C 可能是它自己
    发散出来的；拼成 `A、B、C` 之后就再也分不清"哪几个是用户写的"。
    """
    if model.empty:
        return rule

    task = rule.task
    merged_task = TaskSlots(
        kind=task.kind,  # 永远来自 `finalize()`，模型不许写
        lesson=LessonSlots(
            duration_min=_pick(task.lesson.duration_min, model.task.lesson.duration_min),
            objectives=_pick(task.lesson.objectives, model.task.lesson.objectives),
            lesson_type=_pick(task.lesson.lesson_type, model.task.lesson.lesson_type),
            output_format=_pick(task.lesson.output_format, model.task.lesson.output_format),
        ),
        exam=ExamSlots(
            question_count=_pick(task.exam.question_count, model.task.exam.question_count),
            question_types=_pick(task.exam.question_types, model.task.exam.question_types),
            difficulty_mix=_pick(task.exam.difficulty_mix, model.task.exam.difficulty_mix),
            total_score=_pick(task.exam.total_score, model.task.exam.total_score),
            need_analysis=_pick(task.exam.need_analysis, model.task.exam.need_analysis),
        ),
        # 三个投影字段由 `finalize()` 统一填，这里只接 model 的 resource_types
        prep=PrepSlots(
            resource_types=_pick(task.prep.resource_types, model.task.prep.resource_types),
            grade=task.prep.grade,
            subject=task.prep.subject,
            knowledge_points=task.prep.knowledge_points,
        ),
        plan=PlanSlots(
            target_exam=_pick(task.plan.target_exam, model.task.plan.target_exam),
            exam_date=_pick(task.plan.exam_date, model.task.plan.exam_date),
            current_level=_pick(task.plan.current_level, model.task.plan.current_level),
            weak_points=_pick(task.plan.weak_points, model.task.plan.weak_points),
            weekly_hours=_pick(task.plan.weekly_hours, model.task.plan.weekly_hours),
        ),
    )
    return replace(
        rule,
        chapter=_pick(rule.chapter, model.chapter),
        knowledge_points=_pick(rule.knowledge_points, model.knowledge_points),
        difficulty=_pick(rule.difficulty, model.difficulty),
        exam_type=_pick(rule.exam_type, model.exam_type),
        task=merged_task,
    )


# ---------------------------------------------------------------- ③ 收尾与澄清
def finalize(slots: SlotResult, intent_label: str | None) -> SlotResult:
    """补 `TaskSlots.kind` 与 `PrepSlots` 投影。**幂等**：没有变化就不造新对象。

    必须在 `merge()` 之后调用：投影读的是顶层的最终值（规则 + 模型补空后的结果），
    先 finalize 再 merge 会让投影停在"规则版本"上，与 `SlotResult` 顶层自相矛盾。
    """
    kind = _TASK_KIND_BY_INTENT.get(intent_label or "")
    task = slots.task

    plan = task.plan
    if kind == "plan" and plan.target_exam is None and slots.exam_type:
        # "我明年中考" 抽出来的 exam_type 就是学习规划要的目标考试——同一条事实，
        # 不投影的话下游得同时读两个位置才能拼出"目标考试"
        plan = replace(plan, target_exam=slots.exam_type)

    prep = task.prep
    if kind == "prep":
        prep = replace(
            prep,
            grade=slots.grade,
            subject=slots.subject,
            knowledge_points=slots.knowledge_points,
        )

    if kind == task.kind and prep == task.prep and plan == task.plan:
        return slots
    return replace(slots, task=replace(task, kind=kind, prep=prep, plan=plan))


# 澄清问句：一个任务一句，**只在真正缺了就没法干活**时用（任务 2.3「不要要求填满所有槽位」）
_CLARIFY: dict[TaskKind, str] = {
    "lesson": "要制作哪个章节或知识点的教案？",
    "exam": "这套卷子考哪个学科或哪个知识点？",
    "prep": "想找哪个学科、哪个章节或知识点的备课资源？",
    "plan": "这份学习规划针对哪个学科，目标考试是什么？",
}


def clarification(slots: SlotResult) -> str | None:
    """必要信息缺失时给出澄清问句，否则 `None`（不追问是常态，追问才是例外）。

    判定只看"**没有它这个任务就没法执行**"：

    - 教案制作 / 备课检索：要有课题（章节或知识点），否则不知道写什么；
    - 考试出题：要知道考哪门（学科或知识点），题量、题型都有默认值，缺了不算阻塞；
    - 学习规划：要知道对谁规划（学科 / 薄弱点 / 目标考试三选一即可）。

    其余意图（学习讲解 / 闲聊 / 超出范围）没有任务，一律不追问。

    读的是 `slots.task.plan` 而不是 `slots.plan`：`weak_points` / `target_exam` 只活在
    条目的任务槽位里，顶层没有这两个字段。顶层的 `exam_type` 也算数（`finalize()` 会把它
    投影成 `plan.target_exam`，但本函数不该依赖调用顺序）——用户都说了"中考"，还问
    "目标考试是什么"就是明知故问。
    """
    kind = slots.task.kind
    if kind is None:
        return None
    plan = slots.task.plan
    if kind == "lesson":
        enough = bool(slots.chapter or slots.knowledge_points)
    elif kind == "exam":
        enough = bool(slots.subject or slots.knowledge_points)
    elif kind == "prep":
        enough = bool(slots.subject or slots.knowledge_points or slots.chapter)
    elif kind == "plan":
        enough = bool(
            slots.subject or plan.weak_points or plan.target_exam or slots.exam_type
        )
    else:  # pragma: no cover - Literal 已穷举，留一条兜底不让未来加档位时静默放行
        enough = True
    return None if enough else _CLARIFY[kind]
