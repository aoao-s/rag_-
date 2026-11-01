"""意图 → 路由档位（CLAUDE.md §7.7）。**这是数据表，不是分支代码。**

加档位 = 加一行；加标签 = 改 `base/labels.py::INTENT_LABELS`。两者都不碰下游节点——
下游只读 `IntentProfile` 的字段，不读意图字符串，更不许写 `if label == ...`。

2026-10-30（阶段 2 任务 2.1）：标签由 K12 教学 6 类换成 7 类业务意图，
本表随之重排。旧对象 `KB_STRICT` / `GENERATE_ASSIST` 已删除——
它们在新口径下不再被任何标签引用，留着就是两段不会被执行的配置。
"""

from __future__ import annotations

from typing import Final

from base.contracts import IntentProfile
from base.labels import INTENT_FALLBACK, SUBJECTS

# ---- 7 个业务档位 ---------------------------------------------------------

# 教案制作：生成教案 / 课件 / PPT / 说课稿。
# 检索是为了**参考现成教案与教材**，但要产出一份新材料，所以不要求逐句引用，
# 且允许模型补充（`allow_general=True`）——纯靠知识库拼不出一份成型的教案。
MATERIAL_GENERATE = IntentProfile(
    name="material_generate",
    retrieval=True,
    doc_types=("lesson_plan", "textbook"),
    strategy="DIRECT",
    require_citation=False,
    allow_general=True,
    prompt_id="edurag-qa-v1",
    cacheable=True,
)

# 考试出题：出卷子 / 专项测试 / 题组。
# `doc_types` 故意**不含教材**：教材里没有成套题，搜进来只会稀释证据；
# 真题（exam）与题库（question_bank）才是出题的参照。
# `question_bank` 在本档位被显式点名，才不会被 `EXCLUDED_BY_DEFAULT` 排除。
EXAM_ASSEMBLE = IntentProfile(
    name="exam_assemble",
    retrieval=True,
    doc_types=("exam", "question_bank"),
    strategy="DIRECT",
    require_citation=False,
    allow_general=True,
    prompt_id="edurag-qa-v1",
    cacheable=True,
)

# 备课检索：教师找现成资源（备课资料、公开课、拓展素材）。
# 找材料 = 找现成的东西，必须有出处。
DOC_LOOKUP = IntentProfile(
    name="doc_lookup",
    retrieval=True,
    doc_types=("lesson_plan", "textbook"),
    strategy="DIRECT",
    require_citation=True,
    allow_general=False,
    prompt_id="edurag-qa-v1",
    cacheable=True,
)

# 学习讲解：讲知识点 + 辅导习题，两件事在旧口径里是两个标签，现合并为一。
# `doc_types` 取**并集**（题库 + 试卷 + 教材 + 教案）：
#   - `exam`（2026-10-29）：讲解类问题最直接的证据常是真题；
#   - `question_bank`（2026-10-29）：近 1.8 万块，是习题讲解最全的来源。
# 取并集是有代价的：题库占比 ≈47%，概念题也会搜到题库、可能挤掉教材证据。
# 这是"合并两个标签"的**已知代价**，要改得先用检索指标证明收窄不伤召回（见阶段 2 设计 §3.1）。
# `question_bank` 只出现在本档位——其余档位与默认检索一律排除它，否则会把教材证据挤掉。
KB_STEPWISE = IntentProfile(
    name="kb_stepwise",
    retrieval=True,
    doc_types=("question_bank", "exam", "textbook", "lesson_plan"),
    strategy="DIRECT",
    require_citation=True,
    allow_general=False,
    prompt_id="edurag-qa-v1",
    cacheable=True,
)

# 学习规划：个性化提分计划 / 复习安排。
# - `doc_types=()`：方法类证据散落在各类文档里，收窄只会把答案挡在外面。
# - `cacheable=False`（2026-10-30 改）：规划强依赖学情与记忆，同一句话对不同学员答案不同。
#   §7.7 作用点 7 要求"记忆参与过的答案一律不写共享缓存"，而本档位**必然**吃记忆。
GENERAL_ADVISORY = IntentProfile(
    name="general_advisory",
    retrieval=True,
    doc_types=(),
    strategy="DIRECT",
    require_citation=False,
    allow_general=True,
    prompt_id="edurag-qa-v1",
    cacheable=False,
)

# 闲聊：不检索，直接短答。不写共享缓存——同一句话在不同会话里意思可能不同。
# **走本地小模型**（`LLM_LOCAL_MODEL`，默认 qwen3.5:0.8b）：这是唯一一个既不需要知识库
# 证据、也不需要推理能力的档位，一句话短答用云端大模型是纯浪费（一次往返 + 一次计费）。
# 别的档位不跟着换：它们要引用、要忠实度，0.8B 扛不住，换过去只会把"答得贵"变成"答得错"。
NO_RETRIEVAL = IntentProfile(
    name="no_retrieval",
    retrieval=False,
    doc_types=(),
    strategy="",
    require_citation=False,
    allow_general=True,
    prompt_id="edurag-qa-v1",
    cacheable=False,
    llm_target="local",
)

# 超出范围：与教育完全无关的问题（天气、购物、出行、娱乐、生活琐事）。
# 这是**唯一一个 `allow_general=False` 的不检索档位**，编排层靠这个字段把拒答与闲聊分开：
#   - 不检索：查库对"明天天气"没有意义；
#   - 不生成：拒答文案是固定的，不该花一次大模型调用去"礼貌地拒绝"；
#   - 不缓存：拒绝话术与用户问法无关，缓存它只会污染命中率统计。
# `require_citation=False` 是如实描述（没有引用，也不需要引用），
# **不是**许可编造——它根本没走到生成。
OUT_OF_SCOPE = IntentProfile(
    name="out_of_scope",
    retrieval=False,
    doc_types=(),
    strategy="",
    require_citation=False,
    allow_general=False,
    prompt_id="edurag-refusal-v1",
    cacheable=False,
)

# unknown：低置信不猜。全库检索不收窄 + 通用模板 + 结尾追问（§7.7 硬规则 2、3）。
# `require_citation` 保持 True、`allow_general` 保持 False：不知道用户问的是什么，
# 恰恰是最该守住引用闸门的档位（与 §3.4「无证据 → 知识库暂无 + 转人工」一致）。
GENERIC_CLARIFY = IntentProfile(
    name="generic_clarify",
    retrieval=True,
    doc_types=(),
    strategy="DIRECT",
    require_citation=True,
    allow_general=False,
    prompt_id="edurag-qa-v1",
    cacheable=False,
)

PROFILES: dict[str, IntentProfile] = {
    "教案制作": MATERIAL_GENERATE,
    "考试出题": EXAM_ASSEMBLE,
    "备课检索": DOC_LOOKUP,
    "学习讲解": KB_STEPWISE,
    "学习规划": GENERAL_ADVISORY,
    "闲聊": NO_RETRIEVAL,
    "超出范围": OUT_OF_SCOPE,
    INTENT_FALLBACK: GENERIC_CLARIFY,
}

# 关掉意图（`FEATURE_INTENT=false`）时用的档位：不收窄 doc_type、必须检索、必须有引用。
# 它**就是** §7.7 A/B 里的"全量 DIRECT 基线"——灰度与对照实验都跑这一个对象，
# 免得"基线"在脚本里再实现一遍、与线上不一致。
BASELINE = IntentProfile(
    name="baseline",
    retrieval=True,
    doc_types=(),
    strategy="DIRECT",
    require_citation=True,
    allow_general=False,
    prompt_id="edurag-qa-v1",
    cacheable=True,
)


def profile_for(label: str) -> IntentProfile:
    """标签 → 档位。非法标签一律 `generic_clarify`，绝不硬路由到专业档位。"""
    return PROFILES.get(label, GENERIC_CLARIFY)


# ---- 两个方向相反的信号表 ------------------------------------------------
# 放在本模块而不是 `intent.py`：两者都是**路由安全**规则，不是分类规则；
# 分类器不该知道下游怎么用它。且 `intent.py` 已经 import 本模块，反向 import 会成环。

# 疑问形态：带疑问词的输入哪怕被分成"闲聊"，也不许因此跳过检索——宁可多查一次。
QUESTION_MARKERS: Final[tuple[str, ...]] = (
    "?", "？", "怎么", "如何", "什么", "为什么", "哪些", "哪", "是否",
    "能不能", "可以吗", "吗", "呢", "多少", "怎样", "介绍", "解释", "讲",
)

# 教育领域信号：准备拒答时，只要问句里有这些词之一，就**不许**拒答。
# 覆盖"教学场景"而不是只覆盖学科名——"明天成都天气怎么样"与"天气系统的基本原理"
# 差的不是学科词，是场景。
#
# 刻意不用单字"题"：`这个问题` / `没问题` 里都有它，会把几乎所有输入都判成教育问题，
# 守卫等于失效。用双字以上的题目类词。
_EDU_SCENE_WORDS: Final[tuple[str, ...]] = (
    # 教与学
    "教学", "教案", "课件", "说课", "备课", "讲义", "学案", "板书", "教材", "课本",
    "课程", "课标", "考纲", "考点", "知识点", "单元测试",
    # 题目与考试
    "题目", "习题", "试题", "考题", "真题", "试卷", "作业", "刷题", "做题",
    "复习", "预习", "中考", "高考", "月考", "期末考", "期中考",
    # 人物与场所
    "学生", "老师", "教师", "班级", "课堂", "上课", "讲课", "听课", "评课",
    # 学段与学期
    "小学", "初中", "高中", "年级", "学期", "开学", "期末", "期中",
    # 学习本身
    "学习", "上学", "读书", "成绩", "提分", "分数",
)

# 学科名直接引用 `base.labels.SUBJECTS`：学科词表只能有一处出处（§6.4），
# 在这里另抄一份，将来加学科就要改两处，漏一处就是"新学科的问题被拒答"。
EDU_SIGNALS: Final[tuple[str, ...]] = _EDU_SCENE_WORDS + SUBJECTS


def looks_like_question(query: str) -> bool:
    """带疑问形态吗。命中就不允许走"跳过检索且允许自由作答"的档位（当前只有 闲聊）。

    注意它**只**用于那个方向（§7.7 硬规则 3 的同一条道理：高风险路由要往"多查一轮"倒）。
    对"超出范围"用它是错的——那一类的输入天生就是疑问句，用它会把唯一的正确出口堵死。
    """
    return any(m in query for m in QUESTION_MARKERS)


def looks_educational(query: str) -> bool:
    """问句里有没有明确的教育信号。用于否决"拒答"这个高风险动作。"""
    return any(s in query for s in EDU_SIGNALS)


def resolve_profile(
    query: str, label: str, *, feature_intent: bool = True
) -> tuple[IntentProfile, tuple[str, ...]]:
    """`(标签, query)` → `(档位, 降级标记)`。**编排层与评估脚本共用这一份判定。**

    评估脚本如果自己再实现一遍"档位怎么选、安全阀怎么开"，
    量出来的就不是线上行为——这正是 A/B 最容易失真、也最没人发现的地方。

    两道守卫方向相反，因为这个方向上两种误判的代价不对称：
    - ① 放过（本该多查一轮却跳过了检索）→ 答非所问，可接受；
    - ② 拒答（本该检索却拒了）→ 用户拿不到本来就有的答案，不可接受。
    所以 ① 只作用于"跳过检索且允许自由作答"（闲聊），② 只作用于"准备拒答"（超出范围），
    两者都往"多查一轮"倒。
    """
    if not feature_intent:
        return BASELINE, ()
    profile = profile_for(label)

    # ① 疑问形态安全阀（既有行为）：闲聊档遇到疑问句 → 全库检索 + 追问。
    if not profile.retrieval and profile.allow_general and looks_like_question(query):
        return profile_for(INTENT_FALLBACK), ("intent_no_retrieval_guarded",)

    # ② 教育信号反向保护（新增）：准备拒答，但问句里有明确教育信号 → 不许拒答。
    if not profile.retrieval and not profile.allow_general and looks_educational(query):
        return profile_for(INTENT_FALLBACK), ("intent_out_of_scope_guarded",)

    return profile, ()
