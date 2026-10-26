"""槽位规则抽取：把"用户说了什么学科"变成 `SlotResult`（CLAUDE.md §7.1.8）。

边界：
- **纯函数、无 IO、不调模型**，任何时刻都能在请求路径上跑（微秒级）；
- 只产出 `explicit / multi / unknown / invalid / not_provided` 五种来源，
  `inferred / context / conflict` 属于模型与编排层（`understanding/slots.py`，PR5），
  它们算好后**通过参数**把 `SlotResult` 传进来，qa_system 不 import understanding；
- 学科词表只认 `base/labels.py` 的 `SUBJECTS` + `SUBJECT_ALIASES`，不在这里另立一份。

为什么要单独一个文件：`_select_indexes` 以前把"没传学科""拼错学科""学科无数据"一律
当成全库检索，调用方看不出差别。抽取与路由分开后，"为什么查了全库"变成可断言的数据。
"""

from __future__ import annotations

import re

from base.contracts import SlotResult
from base.labels import SUBJECT_ALIASES, SUBJECTS, normalize_subject

# 表面词 → canonical 学科。长词优先匹配，避免"思想政治"只命中"政治"后重复计数。
_SURFACES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        {**{name: name for name in SUBJECTS}, **SUBJECT_ALIASES}.items(),
        key=lambda kv: (-len(kv[0]), kv[0]),
    )
)

# 年级：只做记录与评估，**不参与过滤**（FAQ 表没有年级字段，见 §7.1.8 暂缓项）
_GRADE_RE = re.compile(r"([初高])([一二三123])|([一二三四五六七八九1-9])年级")
_GRADE_DIGIT = {"1": "一", "2": "二", "3": "三"}


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
    """从问句里抽学科：一个 → `explicit`，多个 → `multi`，抽不到 → `unknown`。

    规则版把"问句里点名了学科"当作显式信号（置信度 1.0）：这是用户自己写下的字，
    比任何模型推断都硬。抽不到就老实说 unknown，**不要**猜成最可能的学科——
    猜错会把正确答案挡在索引外面，那是最难排查的失败模式。
    """
    text = (query or "").strip()
    if not text:
        return SlotResult(subject_source="unknown")

    found: dict[str, int] = {}
    for term, canonical in _SURFACES:
        pos = text.find(term)
        if pos >= 0 and (canonical not in found or pos < found[canonical]):
            found[canonical] = pos

    grade = _extract_grade(text)
    if not found:
        return SlotResult(grade=grade, subject_source="unknown")

    ordered = tuple(name for name, _ in sorted(found.items(), key=lambda kv: kv[1]))
    if len(ordered) == 1:
        return SlotResult(
            subject=ordered[0],
            grade=grade,
            subject_source="explicit",
            subject_confidence=1.0,
        )
    return SlotResult(
        subject=ordered[0],
        subjects=ordered,
        grade=grade,
        subject_source="multi",
        subject_confidence=1.0,
    )


def resolve(query: str, *, param: str | None = None, history_hint: str | None = None) -> SlotResult:
    """一次请求的槽位结论：参数优先，其次问句，最后会话历史。

    `history_hint` 只由编排层传入（上一轮已确认的学科）；qa_system 自己不去读会话，
    否则就成了"检索层偷偷访问记忆"，破坏 §7.0 的边界。
    """
    slots = from_param(param)
    if slots.subject_source == "invalid":
        # 参数写错必须原样暴露，不能被问句里的学科悄悄覆盖——否则用户永远发现不了自己传错了
        return slots
    if slots.subject:
        return slots

    from_query = extract(query)
    if from_query.subject:
        return from_query
    if history_hint:
        canonical = normalize_subject(history_hint)
        if canonical:
            return SlotResult(
                subject=canonical,
                grade=from_query.grade,
                subject_source="context",
                subject_confidence=0.6,
            )
    return SlotResult(grade=from_query.grade, subject_source="unknown")


def _extract_grade(text: str) -> str | None:
    match = _GRADE_RE.search(text)
    if match is None:
        return None
    if match.group(1):
        return f"{match.group(1)}{_GRADE_DIGIT.get(match.group(2), match.group(2))}"
    digit = _GRADE_DIGIT.get(match.group(3), match.group(3))
    return f"{digit}年级"
