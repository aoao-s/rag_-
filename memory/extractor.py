"""记忆抽取（CLAUDE.md §7.9 写入六步的前两步：门控 + 抽取）。

**规则版，不调模型**，理由与 `memory/summary.py` 同一条：本机没有可用 LLM key，
而记忆模块不能因为缺 key 就整条链路跑不起来。规则版的取舍写在下面两段，
轮到有 key 时替换的是 `extract()` 的实现，不是调用方。

**门控（先丢再看）**：只抽**用户自己说的事实/偏好/决定/承诺/进度**。
判据是"第一人称 + 陈述语气"，因为知识库问答里绝大多数轮次是"什么是X"这类
不带任何用户信息的问句——把它们抽成记忆是记忆系统最常见的污染源。
寒暄（"谢谢""好的"）同样丢弃：它们没有可复用的事实。

**为什么不抽助手的回答**：答案来自知识库，不是用户属性。把"勾股定理是 a²+b²=c²"
记成用户记忆，下一轮就会以"学员画像"的身份注入 prompt，这违反 §7.3 的
"记忆绝不进入事实性回答"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from base.labels import normalize_grade, normalize_subject

# 五类事实（§7.3.3 的 ENUM），顺序即优先级：一句话同时命中多条时取靠前的
# （"我初三数学不好" 是 constraint 而不是 identity，因为约束才是后续要照顾的）
FACT_PRIORITY = ("constraint", "decision", "progress", "preference", "identity")

# (fact_type, fact_key, 触发词)。`fact_key` 是槽位名，同 key 才谈得上冲突与去重
_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # 身份：年级 / 身份 / 学校
    ("identity", "grade", ("我今年上", "我是初", "我是高", "我读初", "我读高", "我在读初", "我在读高", "我上初", "我上高")),
    ("identity", "role", ("我是学生", "我是家长", "我是老师", "我是teacher", "我是家长")),
    # 约束：薄弱学科 / 时间 / 基础
    ("constraint", "subject_weak", ("不好", "不太好", "不行", "比较差", "较差", "薄弱", "跟不上", "听不懂", "偏科")),
    ("constraint", "time_budget", ("没时间", "时间不多", "只有周末", "每天半小时", "每天一小时")),
    ("constraint", "base_level", ("零基础", "基础差", "没有基础", "从没学过")),
    # 决定：选科 / 报名 / 目标
    ("decision", "goal_exam", ("我要考", "目标", "想考", "打算考")),
    ("decision", "enroll", ("我要报名", "我决定报", "就选这个", "我选")),
    # 进度：学到哪了
    ("progress", "progress", ("我已经学完", "我学到", "我在学", "我刚学完", "我复习到")),
    # 偏好：学习方式
    ("preference", "study_style", ("我喜欢", "我偏好", "我更想", "我习惯", "我倾向于")),
)

_PRONOUN = ("我",)
# 分句按句末标点**与逗号**切：`我上初三，数学不太好` 是一个句子里并列的两条事实，
# 不按逗号切就只能留优先级高的那条，另一条白白丢掉
_SENT_SPLIT = re.compile(r"[。！？!?；;，,\n]+")

# 学科词：这里的**词面**比 `base/labels.SUBJECTS` 宽（含"几何""方程"这类内容词，
# 它们不是学科名，但说明这句话在讲数学）。出口一律过 `normalize_subject`，
# 存进库的只会是 `SUBJECTS` 里的规范名——词表宽，落点窄。
# 出现在约束句里就当作薄弱学科的证据；不在此表里的词返回 None，不猜。
_SUBJECT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("数学", ("数学", "几何", "代数", "函数", "方程")),
    ("语文", ("语文", "作文", "阅读", "古诗")),
    ("英语", ("英语", "单词", "语法", "听力")),
    ("物理", ("物理", "力学", "电学")),
    ("化学", ("化学", "方程式", "元素")),
    ("生物", ("生物", "细胞", "遗传")),
    ("历史", ("历史", "朝代")),
    ("地理", ("地理", "气候", "地图")),
    ("道德与法治", ("政治", "道法", "思想品德")),
)

_GRADE_RE = re.compile(r"(初[一二三]|高[一二三]|[七八九]年级|[一二三]年级|小学|初中|高中)")


@dataclass(frozen=True, slots=True)
class Candidate:
    """抽出来、还没落库的一条候选记忆。"""

    fact_type: str
    fact_key: str
    content: str
    confidence: float
    source_span: str
    # 结构化的学科/年级，取值域是 `base/labels` 的 SUBJECTS / GRADES；空串 = 这句话没提到。
    # 为什么不是从 `fact_key` 里再拆一次：`subject_weak:数学` 只覆盖"薄弱"一种情况，
    # 而"我喜欢先看例题"同样属于数学语境。字段是**这句话的属性**，不是槽位名的附属物。
    subject: str = ""
    grade: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_type": self.fact_type,
            "fact_key": self.fact_key,
            "content": self.content,
            "confidence": self.confidence,
            "source_span": self.source_span,
            "subject": self.subject,
            "grade": self.grade,
        }


def gated(query: str) -> bool:
    """门控：含有第一人称陈述、且命中任一规则，才值得抽。

    只判断"要不要抽"，不判断"抽什么"——两步分开才能单独测门控的漏放/误放。
    """
    text = (query or "").strip()
    if not text or not any(p in text for p in _PRONOUN):
        return False
    return any(k in text for _, _, keys in _RULES for k in keys)


def extract(query: str) -> list[Candidate]:
    """从**用户这一轮的原话**里抽候选，最多 3 条（再多人也记不住，且会挤占 300 token 预算）。"""
    if not gated(query):
        return []

    out: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    # 一年级说一次就够，且它是**说话人的属性**而不是某句话的属性：
    # `我上初三，数学不太好` 的两条事实都该带"九年级"，只在第一句取会让第二条丢年级。
    # 学科反过来按句子取——`subject_weak:数学` 那条才该带学科，年级事实不该被标成数学。
    turn_grade = _grade_of(query) or ""
    # 第一人称是"谁的事实"的唯一凭据。分句后 `老师，我是初二家长` 的第二段没有"我"，
    # 但主语是从上一段承接下来的，不能因此丢掉（`我上初三，数学不太好` 同理）。
    # 只在**同一轮内**继承：跨轮继承会把别人的话记成这个用户的属性。
    carry = False
    for sent in (s.strip() for s in _SENT_SPLIT.split(query)):
        if not sent:
            continue
        has_pronoun = any(p in sent for p in _PRONOUN)
        if not has_pronoun and not carry:
            continue
        carry = carry or has_pronoun
        subject = _subject_of(sent) or ""
        grade = _grade_of(sent) or turn_grade
        for fact_type in FACT_PRIORITY:
            hit = _match(sent, fact_type)
            if hit is None:
                continue
            key, confidence = hit
            if (fact_type, key) in seen:
                continue
            seen.add((fact_type, key))
            out.append(
                Candidate(
                    fact_type=fact_type,
                    fact_key=key,
                    content=_normalize(sent),
                    confidence=confidence,
                    source_span=sent[:200],
                    subject=subject,
                    grade=grade,
                )
            )
            break  # 一句话只归一类，避免"我数学不好我要报名"产出两条重复事实
        if len(out) >= 3:
            break
    return out


def _match(sent: str, fact_type: str) -> tuple[str, float] | None:
    for ftype, key, triggers in _RULES:
        if ftype != fact_type:
            continue
        for trigger in triggers:
            if trigger not in sent:
                continue
            # 约束句里带学科 → 收敛到该学科，allow 更细的 key（同 key 才谈得上冲突）
            if key == "subject_weak":
                subject = _subject_of(sent)
                if subject is None:
                    continue  # 说"我数学不好"能抽；只说"我不太好"不抽，宁缺毋滥
                return f"{key}:{subject}", 0.85
            if key == "grade":
                # 年级是**单槽位**：`fact_key="grade"` 不带值，否则"我上初三"与
                # "我上高一"会各占一行、谁也 supersede 不了谁，画像里同时留着两个年级
                match = _GRADE_RE.search(sent)
                if match:
                    return key, 0.9
                continue
            return key, 0.8
    return None


def _subject_of(sent: str) -> str | None:
    for subject, words in _SUBJECT_HINTS:
        if any(w in sent for w in words):
            # 词表里的名字本来就都是规范名，仍过一遍 normalize_subject：
            # 将来往 `_SUBJECT_HINTS` 里加"生物学"这类写法时，不用记得再改这里
            return normalize_subject(subject)
    return None


def _grade_of(sent: str) -> str | None:
    """`'我上初三'` 的 grade 是 `'九年级'`，与 Milvus 的 `grade` 字段同一套写法。

    认得出学段但认不出年级时返回 None（"我是初中生"）：`GRADES` 里没有"初中"这个值，
    硬塞进去会让后续任何按年级的等值比较永远不成立。那句话仍会作为 identity 事实
    进画像，只是不带结构化年级——**知道多少记多少**。
    """
    match = _GRADE_RE.search(sent)
    return normalize_grade(match.group(1)) if match else None


def _normalize(sent: str) -> str:
    return " ".join(sent.split())[:200]
