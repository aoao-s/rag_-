"""跨模块契约：三模块之间只允许通过这些数据类通信。

规则（见 CLAUDE.md §7.0）：
- qa_system / rag_system / 记忆模块 平级，互不 import；
- 任何跨模块返回值必须在本文档中定义，禁止用 dict 传递业务字段。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from base.errors import PolicyViolation

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:  # Python 3.10（edu_rag 环境）没有 StrEnum，这里补一个等价实现

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)


class Strategy(StrEnum):
    """检索策略。faq 是短路分支，不进入多路检索。"""

    FAQ = "faq"
    DIRECT = "direct"
    HYDE = "hyde"
    SUB_QUERY = "sub_query"
    BACKTRACK = "backtrack"


# 规划层对**这一次请求要不要检索**的决定（阶段 4 任务 4.1）。
#
# 为什么不塞进 `Strategy`：`Strategy` 描述的是"检索该怎么做"（哪条召回路径），
# 而 `no_retrieval` / `clarify` 说的是"根本别检索"。把"不检索"写成一种检索策略，
# 会让下游每一个读 `Strategy` 的地方都得先排除它——`planner` 的两条早退分支
# 迟早被漏掉一处，于是"该追问的请求跑了一遍全库检索"。
#
# 两者是**正交的两层**：`decision` 先回答做不做，`strategy` 只在 `decision == "retrieve"`
# 时有意义。取值刻意用字符串而不是枚举：`no_retrieval` / `clarify` 是编排层决策，
# 不进 `Strategy` 这个对外表示"召回路径"的枚举。
PlanDecision = Literal["retrieve", "no_retrieval", "clarify"]


# 槽位来源：上游"怎么得到这个学科"的唯一枚举（CLAUDE.md §7.1.8）。
# not_provided（没人做槽位提取）与 unknown（做了但抽不出）必须分开——
# 排障时是两条完全不同的线索。
SubjectSource = Literal[
    "explicit",
    "inferred",
    "context",
    "multi",
    "conflict",
    "invalid",
    "empty",
    "unknown",
    "not_provided",
]

# 本次实际检索范围：filtered=单学科索引、multi=多学科合并、full=全库、empty=学科无数据。
# invalid/conflict/unknown 属于"为什么落到这个范围"，记在 FAQHit.degraded 里。
SubjectState = Literal["filtered", "multi", "full", "empty"]

# FAQ 置信档（CLAUDE.md §7.1.9）：high=唯一允许直接作答；mid=只给候选与追问；
# low=交棒 RAG。档位只在 qa_system 里算，编排层只读它决定动作，不自己再比阈值。
Tier = Literal["high", "mid", "low"]


# 任务槽位按意图二选一（阶段 2 任务 2.3）：
# `kind` 由意图决定，四个子结构各自只装本任务用得到的字段。
# **不为四个任务做一张"大宽表"**：宽表里任何一次改动都要全链路回归，
# 而四个任务之间真正共享的只有学科/年级/知识点——那三个字段在 SlotResult 顶层。
TaskKind = Literal["lesson", "exam", "prep", "plan"]


@dataclass(frozen=True, slots=True)
class LessonSlots:
    """教案制作：教学时长、教学目标、课程类型、输出格式。"""

    duration_min: int | None = None
    objectives: tuple[str, ...] = ()
    lesson_type: str | None = None
    output_format: str | None = None


@dataclass(frozen=True, slots=True)
class ExamSlots:
    """考试出题：题目数量、题型、难度分布、总分、是否需要解析。

    `difficulty_mix` 是分布描述（如 "3:5:2"），不是单值——
    一整套卷子的难度本身就是一个分布，塞进 `SlotResult.difficulty` 会丢掉这个信息。
    `need_analysis=None` 表示用户没表态，**不等于 False**：默认要不要解析是出题模板的事。
    """

    question_count: int | None = None
    question_types: tuple[str, ...] = ()
    difficulty_mix: str | None = None
    total_score: int | None = None
    need_analysis: bool | None = None


@dataclass(frozen=True, slots=True)
class PrepSlots:
    """备课：资源类型、目标年级、学科、知识点。

    后三个是 `SlotResult` 顶层字段的投影，由 `understanding/slots.py::finalize()`
    统一填进来——**只有一份真相**，模型不许另写一份。
    """

    resource_types: tuple[str, ...] = ()
    grade: str | None = None
    subject: str | None = None
    knowledge_points: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PlanSlots:
    """学习规划：目标考试、考试日期、当前水平、薄弱知识点、可用学习时间。"""

    target_exam: str | None = None
    exam_date: str | None = None  # ISO 尽量规范，认不出就保留原话（不猜日期）
    current_level: str | None = None  # 自由文本（"90/150""中等偏上"），不做数值解析
    weak_points: tuple[str, ...] = ()
    weekly_hours: float | None = None


@dataclass(frozen=True, slots=True)
class TaskSlots:
    """任务槽位容器。`kind=None` 表示这次请求没有对应的任务（讲解 / 闲聊 / 超出范围）。"""

    kind: TaskKind | None = None
    lesson: LessonSlots = field(default_factory=LessonSlots)
    exam: ExamSlots = field(default_factory=ExamSlots)
    prep: PrepSlots = field(default_factory=PrepSlots)
    plan: PlanSlots = field(default_factory=PlanSlots)


@dataclass(frozen=True, slots=True)
class SlotResult:
    """槽位抽取结果。qa_system / rag_system / 记忆模块共享，谁都不许另立字段。"""

    subject: str | None = None
    subjects: tuple[str, ...] = ()  # 多学科候选；非空时 subject 取第一个
    grade: str | None = None  # 只记日志，暂不参与过滤（FAQ 表无年级字段）
    subject_source: SubjectSource = "not_provided"
    subject_confidence: float = 0.0  # 规则=1.0；推断=0.6；未知/非法=0
    # ---- 阶段 2 任务 2.3 的基础槽位 ----
    # 前四个里 subject/grade 已经是过滤项（下推 Milvus），教材版本与学期目前只用于
    # 提示词与日志；**这四个一律只由规则/词典给出**，模型不得覆写（见 understanding/slots.py）。
    textbook_version: str | None = None
    semester: str | None = None
    chapter: str | None = None
    knowledge_points: tuple[str, ...] = ()
    difficulty: str | None = None  # easy / medium / hard
    exam_type: str | None = None
    task: TaskSlots = field(default_factory=TaskSlots)
    # 抽取过程中自行降级/忽略的来源（如模型给了学科被丢弃），与 FAQHit.degraded 同口径
    degraded: tuple[str, ...] = ()

    @property
    def scope(self) -> tuple[str, ...]:
        """检索要用到的学科集合（已去重、保持顺序）。空元组表示全库。"""
        if self.subjects:
            return tuple(dict.fromkeys(self.subjects))
        return (self.subject,) if self.subject else ()

    def as_dict(self) -> dict[str, Any]:
        """对外形状（GET /v1/chat 的诊断字段、日志、记忆坐标共用一份）。

        缺省值是 **None / 空列表**而不是省略键：调用方按固定键读，少一个键就得写 `.get()`，
        写着写着就分不清"没有这个槽位"和"这次没抽"。
        """
        return {
            "subject": self.subject,
            "subjects": list(self.subjects),
            "grade": self.grade,
            "textbook_version": self.textbook_version,
            "semester": self.semester,
            "chapter": self.chapter,
            "knowledge_point": list(self.knowledge_points),
            "difficulty": self.difficulty,
            "exam_type": self.exam_type,
            "task_kind": self.task.kind,
        }


@dataclass(frozen=True, slots=True)
class FetchFilter:
    """检索前置过滤条件，全部下推到 Milvus expr / SQL WHERE，禁止查后过滤。"""

    tenant_id: str = "default"
    kb_version: str = "v1"
    subject: str | None = None
    acl_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FAQCandidate:
    """中低置信命中时交给下游的候选问答对。

    两个用途，**都不是"答案"**：

    - `question` 做 query 扩展（未命中时扩充检索词）；
    - 中置信档把 `question + answer` 塞进生成 prompt 当**辅助上下文**（§7.1.9 边界 2，
      2026-10-31 修订）。它**不是证据**：不编号、不进 `【资料】`、不进 citations，
      `FAQHit.answer` 在中置信档仍然是 None——"不直接返回答案"这条没有变。

    `answer` 有默认值是为了保持三参数构造（脚本/单测）继续可用；缺省即"这条候选没有答案"，
    渲染时整条跳过，不会出现空的「答案：」行。
    """

    faq_id: int
    question: str
    score: float
    answer: str = ""


@dataclass(frozen=True, slots=True)
class FAQHit:
    """qa_system 对外的唯一返回结构。

    三档（§7.1.9）：`tier` 是唯一结论字段，`hit` 由它派生（`hit == (tier == "high")`），
    不提供第三种组合——否则调用方要同时判断两个可能互相矛盾的布尔。
    """

    # 高置信：可直接作答（answer 非空）；中置信：answer 恒为 None，只给候选与追问；
    # 低置信：交棒 RAG。见 CLAUDE.md §7.1.9。
    # 中置信的 `candidates` 会带 `answer` 进**生成 prompt** 当辅助上下文（不是证据，不进引用），
    # 但本字段（`answer`）在中置信档永远是 None——两者不是一回事，别合并。
    tier: Tier = "low"
    faq_id: int | None = None
    answer: str | None = None
    score: float = 0.0
    subject: str | None = None
    matched_question: str | None = None
    alias_hit: bool = False
    candidates: tuple[FAQCandidate, ...] = ()
    reason: str = ""
    latency_ms: float = 0.0
    # 命中来自哪一层：cache（Redis/进程内）、index（BM25/精确哈希）、none。
    # 调用方（以及排障的人）需要它来区分"服务慢"与"缓存没生效"。
    source: str = "index"
    cache: str = ""
    # 本次实际检索范围（§7.1.8 路由表）；"学科非法/冲突/主动推断"这些由来记在 degraded，
    # 调用方不必再从一堆字段里猜"刚才到底查了全库还是某个学科"。
    subject_state: SubjectState = "full"
    degraded: tuple[str, ...] = ()

    @property
    def hit(self) -> bool:
        """是否允许直接作答。派生而非独立字段：两处各写一遍必然对不上。"""
        return self.tier == "high"

    def as_dict(self) -> dict[str, object]:
        return {
            "hit": self.hit,
            "tier": self.tier,
            "faq_id": self.faq_id,
            "answer": self.answer,
            "score": round(self.score, 4),
            "subject": self.subject,
            "matched_question": self.matched_question,
            "alias_hit": self.alias_hit,
            "reason": self.reason,
            "latency_ms": round(self.latency_ms, 2),
            "source": self.source,
            "cache": self.cache,
            "subject_state": self.subject_state,
            "degraded": list(self.degraded),
            "candidates": [
                {"faq_id": c.faq_id, "question": c.question, "score": round(c.score, 4)}
                for c in self.candidates
            ],
        }


@dataclass(frozen=True, slots=True)
class Chunk:
    """父子块检索结果（rag_system）。"""

    child_id: str
    text: str
    parent_id: str
    parent_content: str
    doc_id: str
    file_name: str
    page_no: int = 0
    section: str = ""
    subject: str | None = None
    score_dense: float = 0.0
    score_sparse: float = 0.0
    score_rerank: float = 0.0


# ---- 检索链路契约（CLAUDE.md §7.2.7-A）---------------------------------------
# 一次请求从进来到出去，只在下面这些类型之间流动。禁止再用裸 dict 传业务字段：
# 三模块（理解/检索/生成）与编排层全靠它们对齐，字段漂移是"线上日志与行为对不上"的主因。

# 证据门结论（§7.2.7-B 第 6 步）：sufficient 才能生成；其余三档各有下游动作。
# 注意它**不包含"重试几次"**——重试决策属于 pipeline/orchestrator，门只负责说"为什么不够"。
GateStatus = Literal["sufficient", "insufficient", "ambiguous", "conflicting"]

# 门给出的建议动作（§7.2.7-C 的映射表的机读形态）。真正执行与否由 pipeline 决定，
# 且**至多执行一次**：反复重试会把预算吃光，还会把"检索系统坏了"伪装成"资料里没有"。
RetryAction = Literal[
    "none",                 # sufficient，直接生成
    "retry_direct",         # no_candidates：检查 query 处理后按 DIRECT 重检一次
    "retry_original_query", # below_threshold：用原问句重检，**不是**把阈值调低
    "rewrite_query",        # low_query_coverage：改写问句或以子查询重检
    "expand_neighborhood",  # single_weak_evidence：扩同文档邻域，禁止把同一父块拆成多条"独立证据"
    "ask_disambiguate",     # ambiguous：追问用户澄清
    "return_conflict",      # conflicting：并列返回 / 转人工，禁止静默挑高分那条
    "relax_business_filter",# doc_type_too_narrow / subject_filter_empty：放宽业务过滤一次
    "fallback",             # 无可用动作，走兜底话术 + 转人工
]


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """检索前置过滤。

    **安全三项（tenant_id / acl_tags / kb_version）永远前置且不可放宽**；
    只有业务项（subjects / doc_types）允许 `relaxed()` 掉一次——写在这里而不是散在调用侧，
    是因为"哪些能放、哪些不能放"必须只有一份定义（§7.2.7 DoD 第 5 条）。
    """

    tenant_id: str = "default"
    acl_tags: tuple[str, ...] = ()
    kb_version: str = "v1"
    subjects: tuple[str, ...] = ()
    doc_types: tuple[str, ...] = ()
    # 学段 / 年级：与 subjects / doc_types 同属**业务过滤**，
    # `relaxed()` 时一并清掉（安全三项仍然原样保留）。
    stages: tuple[str, ...] = ()
    grades: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """安全三项为空时**直接抛**，而不是静默当成"全租户可见"。

        空 `tenant_id` / `kb_version` 进 Milvus 表达式会退化成 `tenant_id == ""`，
        查不到东西还算好的——真正危险的是在"不传就不加过滤"的实现里变成**跨租户检索**。
        与其在每个调用侧记得校验，不如让构造本身失败：漏传的地方立刻炸在单测里，
        而不是等到某天有人发现别的学校能看到自己的备课材料。

        用 `PolicyViolation`（400）而不是 `ValueError`：这是"调用方参数违反契约"，
        与 §7.2.7 的错误码体系同一口径，API 层能直接映射成响应体。
        """
        for name, value in (("tenant_id", self.tenant_id), ("kb_version", self.kb_version)):
            if not str(value or "").strip():
                raise PolicyViolation(
                    f"检索过滤缺少安全字段 {name}（tenant_id / kb_version 由接入层决定，不可为空）",
                    node="retrieval_filters",
                )

    @property
    def business_restricted(self) -> bool:
        return bool(self.subjects or self.doc_types or self.stages or self.grades)

    def relaxed(self) -> "RetrievalFilters":
        """放宽业务过滤，**安全三项原样保留**。"""
        return RetrievalFilters(
            tenant_id=self.tenant_id,
            acl_tags=self.acl_tags,
            kb_version=self.kb_version,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "acl_tags": list(self.acl_tags),
            "kb_version": self.kb_version,
            "subjects": list(self.subjects),
            "doc_types": list(self.doc_types),
            "stages": list(self.stages),
            "grades": list(self.grades),
        }


@dataclass(frozen=True, slots=True)
class QueryContext:
    """一次请求的不可变上下文：身份、权限、预算。

    它由接入层构造，之后**任何节点都不得修改**（frozen）。检索侧的过滤条件只从
    `filters()` 出，节点自己拼过滤条件的口子一律不开——那是权限泄漏最常见的来源。
    """

    request_id: str
    trace_id: str = ""
    tenant_id: str = "default"
    user_id: str = ""
    session_id: str | None = None
    acl_tags: tuple[str, ...] = ()
    kb_version: str = "v1"
    subject: str | None = None
    top_k: int = 40
    final_n: int = 6
    require_citation: bool = True
    budget_ms: int = 8000
    # 显式传入的学科是"业务过滤"，由 SlotResult 决定，不在这里改写（§7.1.8）
    # 用户**原话**（未经会话改写）。`below_threshold` 的修复动作是"对原 query 再召回"
    # （实施手册 §5.3），而 `run()` 收到的 query 可能已经是改写后的——不单独留一份，
    # 那个修复动作就退化成"拿改写句再检一次"，与手册语义不符且收益接近 0。
    original_query: str = ""
    # 学段 / 年级的业务过滤（§7.2.7-E）。跟 `subject` 同一层，同样只由 `SlotResult` 决定，
    # 意图不得覆盖；`stages` 由 `grades` 经 `labels.GRADE_STAGE` 推出，不单独抽。
    # 追加在末尾而不是插在 `subject` 之后，是为了不挪动既有位置参数的序。
    stages: tuple[str, ...] = ()
    grades: tuple[str, ...] = ()

    def filters(
        self,
        *,
        subjects: tuple[str, ...] = (),
        doc_types: tuple[str, ...] = (),
        stages: tuple[str, ...] = (),
        grades: tuple[str, ...] = (),
    ) -> RetrievalFilters:
        """生成检索过滤条件。安全三项只能来自这里。"""
        picked = subjects or ((self.subject,) if self.subject else ())
        return RetrievalFilters(
            tenant_id=self.tenant_id,
            acl_tags=self.acl_tags,
            kb_version=self.kb_version,
            subjects=tuple(picked),
            doc_types=tuple(doc_types),
            stages=tuple(stages or self.stages),
            grades=tuple(grades or self.grades),
        )

    def log_fields(self) -> dict[str, object]:
        """日志字段。**不放原始问句**——问句由调用侧按 `LOG_QUERY_TEXT` 决定。"""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "kb_version": self.kb_version,
            "acl_n": len(self.acl_tags),
        }


@dataclass(frozen=True, slots=True)
class UnderstandingResult:
    """理解层的**唯一产物**（任务 2.5）：归一化、改写、槽位、意图、档位、复杂度、随访。

    编排层拿到它之后就不该再直接调 `understanding` 里的函数——否则同一请求会被分类两次，
    日志与行为必然分叉。

    五条规则落在这几个字段上，改之前先看一遍：

    1. **原始问题始终保留**：`query` 是第一字段，任何路径都不写它。FAQ 的精确命中与
       `request_log` 都靠它，被改写串覆盖过一次就再也对不回用户说的话。
    2. **改写问题不能覆盖原始问题**：改写只能进 `rewritten`，检索用 `retrieval_query`
       取值（改写 → 归一化 → 原话），三个坑位都空才是原话。
    3. **子问题必须有明确来源**：`sub_queries` 由 `understanding.compound_query` 拆出，
       出口处做字符多重集校验——原问句里没有的字拼不进来。这里**不接收外部传入的子问题**。
    4. **未解决的指代关系不能凭空猜测**：指代没解开就记 `coreference_unresolved`（进
       `degraded`）并把 `is_follow_up` 标上，而不是往 `rewritten` 里塞一个猜出来的主语。
    5. **理解模型超时应回退到原问题**：超时/异常时 `success=False`，但 `rewritten` 为空，
       `retrieval_query` 自动落回 `normalized` 或 `query`——回退是**字段为空**的结果，不是分支。
    """

    query: str                      # 用户原话，永不改写（FAQ 精确命中依赖它）
    normalized: str = ""            # 归一化后的检索用问句
    rewritten: str = ""             # 会话改写结果；等于 normalized 表示没改
    slots: SlotResult = field(default_factory=SlotResult)
    intent: IntentResult | None = None
    profile: IntentProfile | None = None
    method: str = "rule"
    degraded: tuple[str, ...] = ()
    # ---- 任务 2.5 追加（全部带默认值，按位置构造的老调用方不受影响）----
    # 理解层整体是否跑通。注意它**不**因为"没配 key"而 False：那是配置状态，
    # 不是本次请求降级；只有超时/异常/JSON 坏才是 False。
    success: bool = True
    confidence: float = 0.0         # 意图置信度；没有意图结论时为 0
    is_complex: bool = False        # 是否是复合问句（拆出了 ≥2 个子问题）
    sub_queries: tuple[str, ...] = ()   # 子问题，来源由 compound_query 的字符多重集校验保证
    is_follow_up: bool = False      # 是否是依赖上文的追问（含指代词 + 有历史）
    ambiguity: str | None = None    # 无法确定时给用户的追问话术；有值就不该硬答

    @property
    def retrieval_query(self) -> str:
        """真正拿去检索的问句：改写优先，其次归一化，最后原话。"""
        return self.rewritten or self.normalized or self.query


# 一路检索任务用的**是什么问句**（实施手册 §2.3 / §4.2 / §8.2）：
# `original` 用户原话、`rewrite` 会话改写结果、`hyde` 假设性答案、`sub` 拆出来的子问题。
# 它只用于诊断与守门，不参与打分——但**必须记下来**，否则"检索变差了"永远分不清
# 是用户改了问法、还是改写/假设文把检索带偏了。硬规则：`hyde` 文本只用于检索召回，
# **永远不能进 evidence、不能进引用**（见 `RagPipeline` 的证据来源校验）。
QueryRole = Literal["original", "rewrite", "hyde", "sub"]


@dataclass(frozen=True, slots=True)
class QueryTask:
    """检索计划里的一路任务。多路 = 多个 task 并行召回后融合（SUB_QUERY / BACKTRACK）。"""

    task_id: str
    query: str
    strategy: Strategy = Strategy.DIRECT
    weight: float = 1.0
    top_k: int = 40
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    note: str = ""
    # 末尾追加：`rag_pipeline._single()` 等按位置构造的地方不会被打乱
    query_role: QueryRole = "original"


@dataclass(frozen=True, slots=True)
class RetrievalPlan:
    """检索计划：策略 + 任务列表 + 收口参数。

    `reason` 记录**为什么选这个策略**（规则命中 / 档位先验 / 兜底），排障时它比策略名更有用。

    `decision` 是阶段 4 加的**上一层**开关：先回答"要不要检索"，`strategy` 才回答
    "检索该怎么做"。三条取值各自的含义写死如下（`RagPipeline.run()` 按它分派）：

    | decision | 谁触发 | pipeline 做什么 | 下游结果 |
    |---|---|---|---|
    | `retrieve` | 默认 | 正常多路召回 + 过门 + 至多一次修复 | 四态结论 |
    | `no_retrieval` | `needs_retrieval=False`（闲聊 / 业务短路） | **一路都不发** | `insufficient / retrieval_not_needed` |
    | `clarify` | 指代解不开 / 必要槽位缺失 | **一路都不发** | `ambiguous` + `ask_disambiguate` |

    两个非检索分支都**不消耗检索预算、不碰 Milvus**，所以它们省下的是整条检索链
    （§6.1 的 350 ms），而不只是"少一路"。`clarification_reason` 只在 `clarify` 时有值，
    它是给追问话术用的机器码（`referential_without_context` / `unresolved_reference`），
    不是给人读的句子——话术在编排层，规划层不写文案。
    """

    strategy: Strategy = Strategy.DIRECT
    tasks: tuple[QueryTask, ...] = ()
    final_n: int = 6
    # `None` = 由 `RAG_RERANK_ENABLED`（`auto` 即 CUDA 探测）仲裁，**不是关**。
    # 这里绝不能写死 True：CPU 上重排 40 条候选实测 227 s，而 `reranker` 只认
    # `None` 才会回落到配置——写成 True 等于让 `.env` 与 `auto` 彻底失效。
    rerank: bool | None = None
    reason: str = ""
    # 末尾追加（`tests/` 里有按位置构造 `RetrievalPlan` 的地方）。
    # 默认 `retrieve`：所有既有调用方（修复计划、诊断接口、单测）语义不变。
    decision: PlanDecision = "retrieve"
    clarification_reason: str = ""

    @property
    def needs_retrieval(self) -> bool:
        """要不要真的去查库。`decision` 的派生字段，**不单独存**（两份会漂）。"""
        return self.decision == "retrieve"

    def as_dict(self) -> dict[str, Any]:
        """响应体 `/v1/chat` 的 `plan` 字段（用户可见的计划模块）。

        与 `/v1/rag/search` 的 `_plan_body()` 是**两个受众的两份形状**：调试接口
        会额外摊开安全三项（验收要看"每一路有没有漏 tenant_id / ACL"），这里不给。
        两份都从同一个 `RetrievalPlan` 投影，不各自去猜计划长什么样。

        每路任务都要露出来，`task_id` / `query` / `query_role` 三个最关键：
        "这一路用的是原话、改写句还是拆出来的子问题"决定了报告里能不能归因，
        而"计划里有几路"是用户肉眼判断"到底拆没拆"的唯一依据。

        **过滤条件只露业务四项**（学科 / 文档类型 / 学段 / 年级）。安全三项
        （`tenant_id` / `acl_tags` / `kb_version`）由接入层决定、不参与用户可见的决策，
        多租户部署时把它们写进响应体，等于把别人的内部标识发给客户端。
        """
        return {
            "strategy": self.strategy.value,
            "decision": self.decision,
            "reason": self.reason,
            "clarification_reason": self.clarification_reason,
            "final_n": self.final_n,
            "rerank": self.rerank,
            "task_count": len(self.tasks),
            "tasks": [
                {
                    "task_id": t.task_id,
                    "query": t.query,
                    "query_role": t.query_role,
                    "top_k": t.top_k,
                    "weight": t.weight,
                    "note": t.note,
                    "filters": {
                        "subjects": list(t.filters.subjects),
                        "doc_types": list(t.filters.doc_types),
                        "stages": list(t.filters.stages),
                        "grades": list(t.filters.grades),
                    },
                }
                for t in self.tasks
            ],
        }


@dataclass(slots=True)
class Evidence:
    """一条可引用证据 = 一个父块 + 它在本次检索中的最好成绩。

    字段顺序是历史包袱（`tests/` 里按位置构造），**不要在中间插字段**；新增字段一律加在末尾。
    `doc_id / file_name / page_no / section / snippet` 是 §7.2.7-B 要求的对外名字，
    这里用属性从既有字段派生，避免同一个事实存两份、改一处忘另一处。
    """

    parent_id: str
    text: str
    score: float = 0.0
    rerank_score: float | None = None
    child_hits: list[str] = field(default_factory=list)
    title: str = ""
    source: str = ""          # 原始文件路径/URL
    subject: str = ""
    doc_type: str = ""
    heading: str = ""
    page: int = 0
    # 产生这条证据的检索任务（SUB_QUERY 时用来做"覆盖了哪几路"的判断）
    task_id: str = ""
    child_id: str = ""
    # 融合明细：dense / sparse 各自的名次与分数，供诊断接口与基线对比
    score_dense: float = 0.0
    score_sparse: float = 0.0
    # 学段 / 年级（新语料按 `阶段/学科/年级` 组织）：只在末尾追加，
    # 因为 `tests/` 里有按位置构造 Evidence 的地方（见类文档）。
    stage: str = ""
    grade: str = ""

    @property
    def file_name(self) -> str:
        from pathlib import PurePosixPath, PureWindowsPath

        if not self.source:
            return self.title or ""
        # 两种分隔符都试：数据里既有 Windows 路径也有 POSIX 路径
        name = PureWindowsPath(self.source).name
        return name or PurePosixPath(self.source).name

    @property
    def doc_id(self) -> str:
        """父块 id 形如 `{doc_id}:p{n}`；取前缀就是文档 id（证据门跨文档比较用同一口径）。"""
        return self.parent_id.split(":p", 1)[0]

    @property
    def page_no(self) -> int:
        return self.page

    @property
    def section(self) -> str:
        return self.heading

    @property
    def snippet(self) -> str:
        """命中片段：优先第一个子块命中，其次父块开头。引用卡片只需要一句。"""
        if self.child_hits:
            return self.child_hits[0]
        return self.text[:200]

    @property
    def locator_kind(self) -> str:
        """定位串的**来源**：page / section / title / file / none。

        单独暴露它是为了排障：前端看到的都是"能点开的一段字"，但
        「有页码」和「退化成章节」是两种完全不同的数据质量状态，
        混在一个字符串里就再也统计不出"语料到底有多少比例可精确到页"。
        """
        if self.page > 0:
            return "page"
        if self.heading:
            return "section"
        if self.title:
            return "title"
        if self.file_name:
            return "file"
        return "none"

    @property
    def locator(self) -> str:
        """引用卡片上的定位串：**有页码就用页码，没有就退化为「标题 · 章节」**。

        为什么必须退化：新语料的 `page` 字段恒为 0（HTML 教案/试卷本来就没有页码，
        教材走本地解析时页标记也没落到 chunk 上），继续按"第 N 页"渲染会得到
        "第 0 页"或干脆空白——那是**看起来有定位、实际点不开**，比明确写章节更糟。
        退化顺序写死在这里，是因为 `Evidence.citation()`（诊断接口）与
        `rag_system/core/citation.py`（对外引用）必须给出**同一句话**，
        否则同一个证据在两处显示成两个位置，用户与排障的人会互相怀疑。
        """
        kind = self.locator_kind
        if kind == "page":
            return f"第{self.page}页 · {self.heading}" if self.heading else f"第{self.page}页"
        if kind == "section":
            return f"{self.title} · {self.heading}" if self.title else self.heading
        if kind == "title":
            return self.title
        if kind == "file":
            return self.file_name
        return ""

    def citation(self) -> dict:
        return {
            "parent_id": self.parent_id,
            "title": self.title or self.file_name,
            "file": self.file_name,
            "locator": self.locator,
            "locator_kind": self.locator_kind,
            "subject": self.subject or None,
            "heading": self.heading or None,
            "page": self.page or None,
            "stage": self.stage or None,
            "grade": self.grade or None,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "page_no": self.page_no,
            "section": self.section,
            "parent_id": self.parent_id,
            "child_id": self.child_id or (self.child_hits[0][:32] if self.child_hits else ""),
            "snippet": self.snippet,
            "score": round(self.score, 4),
            "source": self.source,
            "task_id": self.task_id,
            "subject": self.subject or None,
            # 学段/年级：新语料按 `阶段/学科/年级` 组织，诊断接口要能看见它们
            "stage": self.stage or None,
            "grade": self.grade or None,
        }


@dataclass(slots=True)
class RetrievalResult:
    """检索收口结果。`stages` 是**实测**的分阶段耗时（§7.2.7-D 的键名），不许填估值。"""

    query: str
    evidences: list[Evidence] = field(default_factory=list)
    raw_hits: int = 0
    cost_ms: int = 0
    reranked: bool = False
    degraded: list[str] = field(default_factory=list)
    strategy: Strategy = Strategy.DIRECT
    tasks: tuple[str, ...] = ()
    # **召回期**就产出过命中的任务 id（融合与 `final_n` 裁剪之前）。`tasks - task_hits`
    # 才是"这一路真的什么都没找到"；不能用 `evidences` 反推——融合时同一个父块
    # 只保留一个 `task_id`，裁剪又会砍到 `final_n`，两路都命中的任务会因此被
    # 误判成"没产出"，进而触发一次多余的 `retry_direct`，把 SUB_QUERY 计划
    # 换成单路 DIRECT（策略名与任务数在响应体里随之丢失）
    task_hits: tuple[str, ...] = ()
    stages: dict[str, int] = field(default_factory=dict)

    @property
    def top_score(self) -> float:
        return self.evidences[0].score if self.evidences else 0.0

    def citations(self) -> list[dict]:
        return [e.citation() for e in self.evidences]


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    """证据门结论（§7.2.7-B 第 6 步）。

    `status` 四值 + `reasons` 多值：一次检索可能同时"分数不够"且"候选太少"，
    只留一个字符串会丢掉另一半病灶（这正是旧版 `reason: str` 的问题）。
    `retry_count` 记录**已经做过几次修复**，由 pipeline 写回，门自己不改它。
    """

    status: GateStatus = "insufficient"
    reasons: tuple[str, ...] = ()
    missing_tasks: tuple[str, ...] = ()
    retry_action: RetryAction = "fallback"
    retry_count: int = 0
    top_score: float = 0.0
    margin: float = 0.0
    n_docs: int = 0
    qcov: float = 0.0

    @property
    def passed(self) -> bool:
        """是否允许生成。派生字段：`status == "sufficient"` 是唯一口径。"""
        return self.status == "sufficient"

    @property
    def reason(self) -> str:
        """兼容旧调用方与日志：多原因取第一个，没有就是 ok / unknown。"""
        if self.status == "sufficient":
            return "ok"
        return self.reasons[0] if self.reasons else "unknown"

    def with_retry(self, action: RetryAction, *, count: int) -> "EvidenceDecision":
        """返回一份带修复记录的副本（frozen，不做原地修改）。"""
        from dataclasses import replace

        return replace(self, retry_action=action, retry_count=count)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "passed": self.passed,
            # 单个字符串是给"只想显示一句原因"的调用方（旧响应体也是这个形状）；
            # 判据细节看 `reasons`，别从 `reason` 反推是哪条判据挂了
            "reason": self.reason,
            "reasons": list(self.reasons),
            "missing_tasks": list(self.missing_tasks),
            "retry_action": self.retry_action,
            "retry_count": self.retry_count,
            "top_score": round(self.top_score, 4),
            "margin": round(self.margin, 4),
            "n_docs": self.n_docs,
            "qcov": round(self.qcov, 4),
        }


# `/v1/chat` 的**统一出口状态**（实施手册 §2.7 / §7.1）。所有分支必须落在这六档之一：
#
# | 值 | 含义 | 典型 `source` |
# |---|---|---|
# | `ok` | 正常作答（FAQ 逐字 / RAG 生成 + 引用） | faq / rag |
# | `clarify` | 证据不足或指代不明，改为追问 | clarify |
# | `refused` | 主动拒答：问题不在服务范围内（2026-10-30 新增） | refusal |
# | `no_evidence` | 检索成功但证据不够，明确说"知识库没有" | fallback |
# | `degraded` | 依赖降级但仍给出了可用答案（如换兜底模型、剔除无支撑句） | rag / faq |
# | `error` | 失败：检索不可用、生成不可用、引用不可用 | fallback |
#
# `refused` 必须独立于 `no_evidence`，这是 §7.8.5 "两套口径"的直接延伸：
# "明天成都天气"记成 `no_evidence` 会被推进**补语料**队列，教研据此去建语料；
# 而正确答案是"这类问题我们本来就不答"，该由**产品口径**决定边界。
# 它也不进 `error`：拒答是正常产品行为，挂到 5xx 口径里会把告警淹掉。
#
# 与 `error_code` 的分工：`status` 是**给前端分支用**的粗粒度开关，`error_code` 是
# 给运维与告警用的具体原因。二者正交——`degraded` 可以带 `error_code`（剔除半段答案
# 后仍作答），`no_evidence` 一定带 `no_evidence`。前端只读 `status` 就够，不必枚举几十种原因。
ChatStatus = Literal["ok", "clarify", "refused", "no_evidence", "degraded", "error"]
# 回答的来源通道。`refusal` 是固定文案（不调模型、不检索），与"检索没找到"的 `fallback` 分开，
# 否则拒答率会被算进知识库缺口指标里。
AnswerSource = Literal["faq", "rag", "clarify", "refusal", "fallback"]


@dataclass(slots=True)
class ChatResponse:
    """`/v1/chat` 的响应体（§7.8 协议 / 实施手册 §2.7）。

    **所有分支都返回这个结构**——FAQ 高置信、澄清、无证据兜底、检索异常、生成异常、
    引用不合格，一个都不许另拼形状。前端因此可以写一份渲染逻辑，而不是每加一条出口
    就在客户端补一个 `if`。

    `source` 四值 + `error_code` 是正交的：`source=fallback` 时 `error_code` 说明**为什么**兜底
    （`no_evidence` / `retrieval_unavailable` / `generation_unavailable` / `citation_invalid`）。
    把"检索系统异常"和"知识库没有证据"混成一句话，运营就永远分不清该修服务还是补语料。

    `degraded` 是机器码（grep / 告警 / 指标读它），`degraded_reasons` 是同一批码的中文说明
    （文案唯一出处 `base/log_messages.DEGRADED_LABELS`）：前者稳定可断言，后者给人看。
    """

    request_id: str
    # 身份与追踪（阶段 1 §3.3 / §3.5）：`role` 是**服务端解析出的**角色，不是请求体里那个；
    # 客户端据此渲染"学生 / 教师"视角。`trace_id` 让用户截图能一路对到 OTEL 与日志。
    trace_id: str = ""
    role: str = "ANONYMOUS"
    status: ChatStatus = "ok"
    source: AnswerSource = "fallback"
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    faq_id: int | None = None
    faq_suggestion: dict[str, Any] | None = None
    clarify: str | None = None
    error_code: str | None = None
    gate: dict[str, Any] | None = None
    # 实际执行过的检索计划（`RetrievalPlan.as_dict()`）。`strategy` 只是它的一个字段，
    # 单独露出策略名答不了"拆成了几路、每路查的是什么"——那正是复合问句要验收的东西。
    # 只有 FAQ 高置信 / 闲聊 / 澄清这些没走检索的出口才是 `None`（确实是没计划，不是漏填）
    plan: dict[str, Any] | None = None
    # 证据门不过时 pipeline 会至多修一次，修完换一份计划。`plan` 保留首轮那份
    # （"这句话被理解成几路"），这里给"最后实际怎么查的"；两者相同或没修过时为 `None`。
    # 只回传一份的话，复合问句的拆分会在修复后就地消失——`strategy` 还写着 `sub_query`，
    # 任务列表却只剩一条 `direct`，读到的人只能怀疑自己看错了。
    plan_after_repair: dict[str, Any] | None = None
    # 会话：`session_id` 回传便于下一轮带上；`rewritten` 只在"确实改写过"时给值，
    # 避免响应体里出现两份一模一样的问句
    session_id: str | None = None
    rewritten: str | None = None
    # 归一化/改写是怎么来的：`none`（原样）/ `normalize`（确定性归一化）/ `rule` / `llm`。
    # 与 `intent_method` 同组：不分开记，A/B 时"召回变差"分不清是归一化还是大模型改写造成的
    rewrite_method: str = "none"
    memory_tokens: int = 0
    # 理解层：标签只供日志与分意图指标，档位名才是下游读过的东西（§7.7 作用点 8）
    intent: str | None = None
    intent_method: str | None = None
    intent_confidence: float | None = None
    profile: str | None = None
    strategy: str | None = None
    degraded: list[str] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)
    stages: dict[str, int] = field(default_factory=dict)
    latency_ms: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "role": self.role,
            "status": self.status,
            "source": self.source,
            "answer": self.answer,
            "citations": self.citations,
            "faq_id": self.faq_id,
            "faq_suggestion": self.faq_suggestion,
            "clarify": self.clarify,
            "error_code": self.error_code,
            "gate": self.gate,
            "plan": self.plan,
            "plan_after_repair": self.plan_after_repair,
            "session_id": self.session_id,
            "rewritten": self.rewritten,
            "rewrite_method": self.rewrite_method,
            "memory_tokens": self.memory_tokens,
            "intent": self.intent,
            "intent_method": self.intent_method,
            "intent_confidence": self.intent_confidence,
            "profile": self.profile,
            "strategy": self.strategy,
            "degraded": self.degraded,
            "degraded_reasons": self.degraded_reasons,
            "stages": self.stages,
            "latency_ms": self.latency_ms,
            "usage": self.usage,
        }


@dataclass(frozen=True, slots=True)
class MemoryContext:
    """记忆模块注入 prompt 的背景信息，不得作为引用来源。"""
    profile: str = ""
    summary: str = ""
    atoms: tuple[str, ...] = ()
    tokens: int = 0
    degraded: bool = False
    extra: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        """拼成 prompt 的"记忆"段；空记忆返回空串，调用方据此跳过该段。

        标记用词与 CLAUDE.md §7.9 一致（`【会话摘要】` / `【历史记忆】`）：
        这三段是模型区分"背景"与"依据"的唯一线索，措辞不一致会让提示词审查
        与实际注入的内容对不上（§7.9 边界 1）。
        """
        parts: list[str] = []
        if self.profile:
            parts.append(f"【学员画像】{self.profile}")
        if self.summary:
            parts.append(f"【会话摘要】{self.summary}")
        if self.atoms:
            parts.append("【历史记忆】" + "；".join(self.atoms))
        return "\n".join(parts)


# ---- 理解层（CLAUDE.md §7.7）------------------------------------------------
# `IntentResult` 只存"分类结果"，**没有 subject 字段**：学科过滤是槽位与检索路由的职责
# （§7.1.8）。意图只允许收窄 doc_type，租户 / ACL / kb_version 由接入层决定。
IntentMethod = Literal["bert", "model", "rule", "fallback"]
# 四值的分工（§7.7 任务 2.2）：`bert` = 精调分类模型（初始分类器，快且确定）；
# `model` = 大模型（复杂语义与辅助判断）；`rule` = 关键词规则（最终兜底）；
# `fallback` = 三条路都没给出可靠结论，落 `unknown`。四者**不是同类可比的质量档位**，
# 只是"这次结论是谁给的"——排障时要能一眼分清"模型没加载"和"模型说不知道"。


@dataclass(frozen=True, slots=True)
class IntentResult:
    label: str                       # INTENT_LABELS 之一，或 "unknown"
    confidence: float
    method: IntentMethod
    profile: str                     # 档位名；下游只读这个

    @property
    def code(self) -> str:
        """对外英文代号（`EXAM_GENERATION` 之类），任务 2.2 的响应契约要用它。

        中文 `label` 是内部口径（`base/labels.py` 的唯一出处、Milvus 与日志里都用它），
        而对外接口与任务书说的是英文代号。映射只在这里做一次，**不在调用侧各拼一份**；
        `unknown` 没有代号，映射不到时统一给 `UNKNOWN`（那正是它的语义）。
        """
        from base.labels import INTENT_CODES

        return INTENT_CODES.get(self.label, "UNKNOWN")

    def as_dict(self) -> dict[str, Any]:
        """任务 2.2 要求的最小输出形状：`{"intent": code, "confidence": ..., "method": ...}`。

        额外带上中文 `label` 与 `profile`：前者是内部口径（日志与评测都按它统计），
        后者是下游唯一读的字段——**只给三个字段会让排障时看不到"为什么走了这个档位"**。
        """
        return {
            "intent": self.code,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "profile": self.profile,
        }


@dataclass(frozen=True, slots=True)
class IntentProfile:
    """档位 = 策略配置。下游节点读它，不读意图字符串，更不写 if/else。"""

    name: str
    retrieval: bool                  # 是否走检索
    doc_types: tuple[str, ...]       # 唯一可被意图影响的过滤条件；空元组 = 不限
    strategy: str                    # 策略先验 DIRECT / HyDE / SUB_QUERY / BACKTRACK
    require_citation: bool           # 策略：没有可验证证据时是否允许作答
    allow_general: bool              # 无证据时给通用说明还是转人工
    prompt_id: str                   # prompts/ 下的模板 id（带版本）
    cacheable: bool                  # 是否允许写共享缓存
    # 用哪个生成引擎。`default` = 主网关（云端）；`local` = 本地小模型（闲聊档）。
    # **写在这里而不是在编排层写 `if profile.name == "闲聊"`**：档位是数据表，
    # 换引擎是"这个档位的策略"，与"检索还是不检索"同一层级（§7.7 硬规则 1）。
    llm_target: Literal["default", "local"] = "default"


# ---- 业务权限（阶段 3）------------------------------------------------------
# 存的是**被拒绝**的意图集合，不是被允许的：这样 `allows()` 天然对
# "意图为 None / unknown / 低置信" 放行——权限判定**绝不在猜出来的意图上下刀**。
# 反过来存允许集，就得在每次判断里记得"unknown 要特事特办"，而那种"记得"
# 迟早在某个新调用点被忘掉，表现为"说不清的问题被拒掉了"。
PermissionSource = Literal["role", "disabled", "unknown_role"]


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    """角色 → 可用意图的判定结果（`orchestration/admission.py::permissions_for` 产出）。

    放在 `base/contracts.py` 而不是准入层，是为了让 `orchestration/state.py` 能引用它
    而不必 import `orchestration/admission.py`（那会把 `auth.tokens` 与 `qa_slots`
    一起拖进状态模块）。**判定逻辑不在这里**——本类只承载结论。
    """

    role: str
    denied: frozenset[str] = frozenset()
    source: PermissionSource = "role"

    def allows(self, intent: str | None) -> bool:
        """该意图是否放行。`intent` 为空（未分类/低置信）时恒放行。"""
        if not intent:
            return True
        return intent not in self.denied

    def as_dict(self) -> dict[str, Any]:
        return {"role": self.role, "denied": sorted(self.denied), "source": self.source}
