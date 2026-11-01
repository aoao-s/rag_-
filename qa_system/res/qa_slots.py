"""槽位规则抽取：把"用户说了什么"变成 `SlotResult`（CLAUDE.md §7.1.8 / 阶段 2 任务 2.3）。

边界：
- **纯函数、无 IO、不调模型**，任何时刻都能在请求路径上跑（微秒级）；
- **过滤类槽位只从这里出**：学科、年级、教材版本、学期一旦由模型"猜"出来，
  失败的形状不是"多查一轮"而是"正确答案被挡在索引外面"（§7.1.8 的安全不对称）。
  模型在 `understanding/slots.py` 里给的同名字段一律丢弃并计数；
- 只产出 `explicit / multi / unknown / invalid / not_provided / conflict` 六种来源。
  `conflict` 是**这里自己算的**（参数与问句点名的学科互相矛盾，见 `resolve()`），
  `inferred / context` 属于模型与编排层，它们算好后**通过参数**把 `SlotResult` 传进来，
  qa_system 不 import understanding；
- 词表只认 `base/labels.py`，不在这里另立一份。

任务的**数值型**参数（几道题、多少分钟、总分、考试日期）也走规则：
它们有稳定的字面形态，规则抽得到就抽，抽不到留给模型补，模型补不到就保持 None。
**"缺"不是错**——任务 2.3 明确要求不把槽位填满（§「不要要求每个请求都填满所有槽位」）。
"""

from __future__ import annotations

import re
from typing import Any

from base.contracts import ExamSlots, LessonSlots, PlanSlots, PrepSlots, SlotResult, TaskSlots
from base.labels import (
    DIFFICULTY_ALIASES,
    EXAM_TYPE_ALIASES,
    EXAM_TYPES,
    LESSON_TYPES,
    OUTPUT_FORMAT_ALIASES,
    OUTPUT_FORMATS,
    QUESTION_TYPES,
    RESOURCE_TYPES,
    SEMESTER_ALIASES,
    SEMESTERS,
    SUBJECT_ALIASES,
    SUBJECTS,
    TEXTBOOK_VERSION_ALIASES,
    TEXTBOOK_VERSIONS,
    normalize_grade,
    normalize_subject,
    surfaces,
)

# 表面词 → canonical。全部**按词面长度降序**（`surfaces()` 保证），配合"互不重叠"的
# 取词逻辑实现最长优先——否则 `期中考试` 里的 `中考` 也会被算成一次命中。
_SURFACES = surfaces(SUBJECTS, SUBJECT_ALIASES)
_TEXTBOOK_SURFACES = surfaces(TEXTBOOK_VERSIONS, TEXTBOOK_VERSION_ALIASES)
_SEMESTER_SURFACES = surfaces(SEMESTERS, SEMESTER_ALIASES)
_DIFFICULTY_SURFACES = surfaces(tuple(DIFFICULTY_ALIASES.values()), DIFFICULTY_ALIASES)
_EXAM_SURFACES = surfaces(EXAM_TYPES, EXAM_TYPE_ALIASES)
_QUESTION_SURFACES = surfaces(QUESTION_TYPES, {})
_LESSON_SURFACES = surfaces(LESSON_TYPES, {})
_RESOURCE_SURFACES = surfaces(RESOURCE_TYPES, {})
_FORMAT_SURFACES = surfaces(OUTPUT_FORMATS, OUTPUT_FORMAT_ALIASES)

# 年级：只做记录与过滤下推（`QueryContext.grades` → Milvus `grade`）
_GRADE_RE = re.compile(r"([初高])([一二三123])|([一二三四五六七八九1-9])年级")
_GRADE_DIGIT = {"1": "一", "2": "二", "3": "三"}

# 章节：只认"第N章 / 第N节"这种显式写法。
# "二次函数这一章"这类指代规则判不了（会被误当成任意名词），交给模型补。
_CHAPTER_RE = re.compile(r"第\s*([一二三四五六七八九十百零两0-9]{1,4})\s*([章节])")

# 知识点 / 教学目标：只认显式列举（`知识点：A、B`），且必须有分隔符或"是/为/包括"。
# 这是刻意的窄口径——"这节课的重点是什么"也含"重点"，宽口径会把"什么"抽成知识点。
_ENUM_MARKERS = re.compile(r"(?:知识点|考点|重点|难点)\s*(?:[:：]|是|为|包括|包含)")
_OBJ_MARKERS = re.compile(r"教学目标\s*(?:[:：]|是|为|包括|包含)")
_ENUM_SPLIT = re.compile(r"[、，,；;]+")
_ENUM_STOP = ("什么", "哪些", "哪几", "怎么", "如何", "多少", "哪些内容")

# 任务数值参数
_COUNT_RE = re.compile(r"(\d{1,3})\s*(?:道|个)?\s*(?:小题|题)")
_COUNT_DAO_RE = re.compile(r"(\d{1,3})\s*道")
_MINUTES_RE = re.compile(r"(\d{1,3})\s*(?:分钟|min)")
_SCORE_RE = re.compile(
    r"(?:总分|满分|一共|共计)\s*[^0-9]{0,4}(\d{1,3})\s*分|(\d{1,3})\s*分\s*(?:的)?\s*(?:卷|试卷|试题)"
)
_DIFFICULTY_MIX_RE = re.compile(
    r"(?:难度分布|难度比例|难度)\s*[^0-9]{0,6}(\d{1,2})\s*[:：]\s*(\d{1,2})\s*[:：]\s*(\d{1,2})"
)
_EXAM_DATE_RE = re.compile(
    r"(\d{4}\s*[年\-/]\s*\d{1,2}\s*[月\-/]\s*\d{1,2}\s*日?)|(\d{1,2}\s*月\s*\d{1,2}\s*[日号])"
)
_WEEKLY_RE = re.compile(r"每周\s*[^0-9]{0,4}(\d{1,2}(?:\.\d)?)\s*(?:个)?\s*(?:小时|钟头|h|H)")
_ANALYSIS_OFF_RE = re.compile(r"(?:不需要|不用|不要|无需)\s*(?:答案和)?\s*(?:解析|详解)")
_ANALYSIS_ON_RE = re.compile(r"(?:需要|要|带|含|附)\s*(?:答案和|答案与|答案)?\s*(?:解析|详解)")


# ---------------------------------------------------------------- 取词
def _positions(text: str, term: str) -> list[int]:
    spots: list[int] = []
    start = text.find(term)
    while start >= 0:
        spots.append(start)
        start = text.find(term, start + 1)
    return spots


def _claim_terms(text: str, table: tuple[tuple[str, str], ...]) -> list[tuple[str, int]]:
    """最长优先、**互不重叠**地取词，返回按位置升序的 `(规范名, 位置)`。

    互不重叠不是洁癖：`期中考试` 里就含 `中考`，两个词面都算命中时，
    "这次是期中考还是中考"取决于遍历顺序——同一句话在两个进程里能得出不同结论。
    """
    taken = [False] * len(text)
    hits: list[tuple[str, int]] = []
    for term, canonical in table:  # table 已按词面长度降序
        for pos in _positions(text, term):
            end = pos + len(term)
            if any(taken[pos:end]):
                continue
            taken[pos:end] = [True] * len(term)
            hits.append((canonical, pos))
    hits.sort(key=lambda item: item[1])
    return hits


def _first_term(text: str, table: tuple[tuple[str, str], ...]) -> str | None:
    """最先出现的那一个（用于"只能有一个值"的槽位，如教材版本、考试类型）。"""
    hits = _claim_terms(text, table)
    return hits[0][0] if hits else None


def _all_terms(text: str, table: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    """全部命中、去重、保持出现顺序（用于题型/资源类型这类多值槽位）。"""
    ordered: list[str] = []
    for canonical, _ in _claim_terms(text, table):
        if canonical not in ordered:
            ordered.append(canonical)
    return tuple(ordered)


# ---------------------------------------------------------------- 拆句
def _enumerated(text: str, marker: re.Pattern[str], *, limit: int, max_len: int) -> tuple[str, ...]:
    """取"标记词 + 列举串"的第一个匹配，按中文顿号/逗号/分号切开。"""
    match = marker.search(text)
    if match is None:
        return ()
    tail = text[match.end() :]
    # 列举串只吃到句末标点：再往后的内容已经不是这个槽位了
    tail = re.split(r"[。？！?!；;\n]", tail, maxsplit=1)[0]
    items: list[str] = []
    for chunk in _ENUM_SPLIT.split(tail):
        item = chunk.strip(" 的了")
        if not item or len(item) > max_len or len(item) < 2:
            continue
        if item.startswith(_ENUM_STOP):
            continue
        if item not in items:
            items.append(item)
    return tuple(items[:limit])


# ---------------------------------------------------------------- 基础槽位
def _extract_grade(text: str) -> str | None:
    """抽年级并**归一成 `labels.GRADES` 的规范名**（"初三" → "九年级"）。

    归一不是洁癖：Milvus 的 `grade` 字段、记忆的 `Atom.grade`、引用卡片都用规范名。
    这里返回"初三"，同一件事就会在三个地方有三个写法——而按年级做任何比较
    （画像渲染、过滤下推）都会零命中，且**不会报错**。
    """
    match = _GRADE_RE.search(text)
    if match is None:
        return None
    if match.group(1):
        raw = f"{match.group(1)}{_GRADE_DIGIT.get(match.group(2), match.group(2))}"
    else:
        digit = _GRADE_DIGIT.get(match.group(3), match.group(3))
        raw = f"{digit}年级"
    return normalize_grade(raw)


def _extract_chapter(text: str) -> str | None:
    match = _CHAPTER_RE.search(text)
    if match is None:
        return None
    return f"第{match.group(1)}{match.group(2)}"


def _extract_difficulty(text: str) -> str | None:
    """难度只在**难度语境**里认：`难度中等` / `中等难度的卷子`。

    直接按词面认会把"基础知识有哪些"标成 easy——"基础"在教育问句里是普通词，
    不是难度标记（同一条理由让 `DIFFICULTY_ALIASES` 不收单字）。
    """
    for term, canonical in _DIFFICULTY_SURFACES:
        for pos in _positions(text, term):
            before = text[max(0, pos - 4) : pos]
            after = text[pos + len(term) : pos + len(term) + 3]
            if before.endswith(("难度", "难度为", "难度是", "难度：", "难度:")) or after.startswith(
                ("难度", "的题", "题目", "题")
            ):
                return canonical
    return None


def _extract_shared(text: str) -> dict[str, Any]:
    """学科之外的基础槽位。学科单独抽，因为它是唯一有 `multi` 语义的一个。"""
    return {
        "grade": _extract_grade(text),
        "textbook_version": _first_term(text, _TEXTBOOK_SURFACES),
        "semester": _first_term(text, _SEMESTER_SURFACES),
        "chapter": _extract_chapter(text),
        "knowledge_points": _enumerated(text, _ENUM_MARKERS, limit=5, max_len=20),
        "difficulty": _extract_difficulty(text),
        "exam_type": _first_term(text, _EXAM_SURFACES),
    }


# ---------------------------------------------------------------- 任务数值槽位
def _extract_count(text: str) -> int | None:
    match = _COUNT_RE.search(text) or _COUNT_DAO_RE.search(text)
    if match is None:
        return None
    return _clamp(int(match.group(1)), 1, 100)


def _extract_minutes(text: str) -> int | None:
    match = _MINUTES_RE.search(text)
    return None if match is None else _clamp(int(match.group(1)), 5, 300)


def _extract_score(text: str) -> int | None:
    match = _SCORE_RE.search(text)
    if match is None:
        return None
    return _clamp(int(match.group(1) or match.group(2)), 1, 300)


def _extract_difficulty_mix(text: str) -> str | None:
    match = _DIFFICULTY_MIX_RE.search(text)
    if match is None:
        return None
    return ":".join(match.group(i) for i in (1, 2, 3))


def _extract_exam_date(text: str) -> str | None:
    """认得出就保留原话（不做"下个月"这类相对日期的解析——解析错比不解析糟）。"""
    match = _EXAM_DATE_RE.search(text)
    if match is None:
        return None
    return re.sub(r"\s+", "", match.group(1) or match.group(2))


def _extract_weekly_hours(text: str) -> float | None:
    match = _WEEKLY_RE.search(text)
    if match is None:
        return None
    return min(max(float(match.group(1)), 0.0), 80.0)


def _extract_need_analysis(text: str) -> bool | None:
    """否定式必须先判：`不需要解析` 里也含 `需要...解析`。"""
    if _ANALYSIS_OFF_RE.search(text):
        return False
    if _ANALYSIS_ON_RE.search(text):
        return True
    return None


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _task_from_rules(text: str) -> TaskSlots:
    """规则能填的任务字段。**不设 `kind`**——那是意图的结论，见 `understanding/slots.py`。

    规则填的是"字面形态稳定"的那几个（数量、时长、总分、日期、题型词）。
    教学目标 / 章节名 / 难点这类自由文本留给模型，规则只认显式列举。
    """
    return TaskSlots(
        lesson=LessonSlots(
            duration_min=_extract_minutes(text),
            objectives=_enumerated(text, _OBJ_MARKERS, limit=5, max_len=40),
            lesson_type=_first_term(text, _LESSON_SURFACES),
            output_format=_first_term(text.lower(), _FORMAT_SURFACES),
        ),
        exam=ExamSlots(
            question_count=_extract_count(text),
            question_types=_all_terms(text, _QUESTION_SURFACES),
            difficulty_mix=_extract_difficulty_mix(text),
            total_score=_extract_score(text),
            need_analysis=_extract_need_analysis(text),
        ),
        prep=PrepSlots(resource_types=_all_terms(text, _RESOURCE_SURFACES)),
        plan=PlanSlots(
            exam_date=_extract_exam_date(text),
            weekly_hours=_extract_weekly_hours(text),
        ),
    )


# ---------------------------------------------------------------- 对外入口
def from_param(subject: str | None) -> SlotResult:
    """公开 API 的 `subject` 参数 → SlotResult。

    传空 = `not_provided`（用户真的没限定学科，全库检索不是降级）；
    传了但不在词表内 = `invalid`（调用方必须能看出这是参数错误，不是"没数据"）。
    """
    if subject is None or not subject.strip():
        return SlotResult()
    canonical = normalize_subject(subject)
    if canonical is None:
        return SlotResult(subject_source="invalid")
    return SlotResult(subject=canonical, subject_source="explicit", subject_confidence=1.0)


def extract(query: str) -> SlotResult:
    """从问句里抽槽位：学科一个 → `explicit`，多个 → `multi`，抽不到 → `unknown`。

    规则版把"问句里点名了学科"当作显式信号（置信度 1.0）：这是用户自己写下的字，
    比任何模型推断都硬。抽不到就老实说 unknown，**不要**猜成最可能的学科——
    猜错会把正确答案挡在索引外面，那是最难排查的失败模式。
    """
    text = (query or "").strip()
    if not text:
        return SlotResult(subject_source="unknown")

    shared = _extract_shared(text)
    task = _task_from_rules(text)

    found: dict[str, int] = {}
    for canonical, pos in _claim_terms(text, _SURFACES):
        found.setdefault(canonical, pos)
    if not found:
        return SlotResult(subject_source="unknown", task=task, **shared)

    ordered = tuple(name for name, _ in sorted(found.items(), key=lambda kv: kv[1]))
    if len(ordered) == 1:
        return SlotResult(
            subject=ordered[0],
            subject_source="explicit",
            subject_confidence=1.0,
            task=task,
            **shared,
        )
    return SlotResult(
        subject=ordered[0],
        subjects=ordered,
        subject_source="multi",
        subject_confidence=1.0,
        task=task,
        **shared,
    )


def _query_fields(from_query: SlotResult) -> dict[str, Any]:
    """问句里抽出来的**学科之外**的槽位，供 `resolve()` 重新组合。

    学科之外的字段无论学科从哪来（参数 / 问句 / 历史 / 冲突）都取自问句：参数里只有学科，
    而"八年级、人教版、一次函数"这些只可能写在问句里，不能因为学科来自参数就整包丢掉。
    这里集中一份，是为了让下面三条分支（冲突 / 参数 / 历史）不会各自漏一个字段——
    漏了不报错，只是那个槽位静默消失。
    """
    return {
        "grade": from_query.grade,
        "textbook_version": from_query.textbook_version,
        "semester": from_query.semester,
        "chapter": from_query.chapter,
        "knowledge_points": from_query.knowledge_points,
        "difficulty": from_query.difficulty,
        "exam_type": from_query.exam_type,
        "task": from_query.task,
    }


def resolve(query: str, *, param: str | None = None, history_hint: str | None = None) -> SlotResult:
    """一次请求的槽位结论：参数优先，其次问句，最后会话历史。

    学科之外的槽位**无论学科从哪来都取自问句**：参数里只有学科，
    而"八年级、人教版、一次函数"这些只可能写在问句里，不能因为学科来自参数就整包丢掉。

    `history_hint` 只由编排层传入（上一轮已确认的学科）；qa_system 自己不去读会话，
    否则就成了"检索层偷偷访问记忆"，破坏 §7.0 的边界。

    **冲突只有一种形态**：`param` 是合法学科，而问句里点名的学科集合（一个或多个）
    **不包含**它 —— 例如 `param="化学"` 配问句"数学的二次函数怎么求"。两边都是用户
    自己写的字，选谁都是在替他猜，所以结论是 `subject_source="conflict"` + `subject=None`：
    检索退回全库（不过滤），FAQ 侧按 §7.1.8 路由表封顶到 mid、绝不直接作答。

    四种情况**不算**冲突，刻意分开：

    - `param` 在问句的学科集合里（"数学和物理的题" + `param="数学"`）→ 用户只是没把话说全，
      参数照旧生效，收窄到那一门是他明确要求的；
    - 问句没提任何学科 → 参数就是唯一信号，无从矛盾；
    - `param` 不在词表（`invalid`）→ 原样暴露参数错误，**不能让问句悄悄把它盖掉**，
      否则用户永远发现不了自己传错了；
    - 历史提示与问句不一致 → 主题切换是正常对话，这一轮说的算这一轮的。拿历史去否决
      用户这一步写下的学科，是指代借用那条错误过滤的另一种翻身方式。
    """
    param_slots = from_param(param)
    if param_slots.subject_source == "invalid":
        # 参数写错必须原样暴露，不能被问句里的学科悄悄覆盖——否则用户永远发现不了自己传错了
        return param_slots

    from_query = extract(query)
    fields = _query_fields(from_query)

    if param_slots.subject:
        asked = from_query.scope
        if asked and param_slots.subject not in asked:
            # 参数与问句互相矛盾：谁覆盖谁都是猜。`subjects` 也清空——留着第一个会被下游
            # 当成"就查这一门"，与 `conflict` 的全库语义正好相反
            return SlotResult(subject_source="conflict", **fields)
        return SlotResult(
            subject=param_slots.subject,
            subject_source="explicit",
            subject_confidence=1.0,
            **fields,
        )
    if from_query.subject:
        return from_query
    if history_hint:
        canonical = normalize_subject(history_hint)
        if canonical:
            return SlotResult(
                subject=canonical,
                subject_source="context",
                subject_confidence=0.6,
                **fields,
            )
    return from_query
