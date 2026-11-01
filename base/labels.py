"""标签注册表：意图、学科。

为什么单独成文件（见 CLAUDE.md §3.2 / §7.7）：
意图标签集合可能随产品口径变化，全项目只允许在这里定义，
业务代码一律 import 常量，禁止在分支里写字符串字面量——否则换口径要改几十处。

当前口径（2026-10-30 定案，阶段 2 任务 2.1）：**7 类业务意图**，与
`datasets/06_intent`（训练 2 100 / 测试 500，7×300）逐字一致；原 K12 教学 6 类
（知识点查询 / 题目讲解 / 教案素材 / 学习方法 / 备课辅助 / 闲聊）已被取代，
只在 `INTENT_ALIASES` 里留作迁移表。改口径只改本文件与 `understanding/profiles.py` 的档位表。
"""

from __future__ import annotations

from typing import Final

# 角色：**只允许来自可信认证上下文**（JWT claim 或网关注入），
# 永远不从请求体读——请求体是用户写的，`{"role": "ADMIN"}` 谁都能发（阶段 1 §3.3）。
# `ANONYMOUS` 是第四个取值，**不会出现在任何 token 里**：它是"没有身份"这一事实的名字，
# 而不是一种可以被授予的角色。阶段 3 的口径是"ANONYMOUS 恒按最小权限处理"。
ROLES: Final[tuple[str, ...]] = ("STUDENT", "TEACHER", "ADMIN")
ROLE_ANONYMOUS: Final[str] = "ANONYMOUS"


def normalize_role(raw: str | None) -> str | None:
    """角色归一：去空白 + 转大写；空值返回 None。

    大小写不敏感是刻意的——`admin` / `Admin` / `ADMIN` 在 token 里都出现过，
    为这种差异拒掉一次合法请求，运维只会去翻日志而不会觉得"这是安全设计"。
    格式不合法的取值返回 None，由调用方决定是 401（token 里）还是匿名（请求头）。
    """
    text = (raw or "").strip().upper()
    return text or None


def is_valid_role(raw: str | None) -> bool:
    """可授予的角色（不含 ANONYMOUS）。"""
    return normalize_role(raw) in ROLES


# 意图：7 类业务意图（阶段 2 任务 2.1 的固定口径）
# 中文是规范名：数据集、request_log.intent、日志、指标、缓存键全部用它。
INTENT_LABELS: Final[tuple[str, ...]] = (
    "教案制作",
    "考试出题",
    "备课检索",
    "学习讲解",
    "学习规划",
    "闲聊",
    "超出范围",
)

# 中文规范名 → 对外英文代号（计划书给的名字，逐字保持）。
# 供对外 API 与将来的多语言前端使用；**不**用来替代中文规范名——
# 换名字要同时改数据集、提示词、日志、指标与全部断言，收益只是"看起来更规范"。
INTENT_CODES: Final[dict[str, str]] = {
    "教案制作": "TEACHING_MATERIAL_GENERATION",
    "考试出题": "EXAM_GENERATION",
    "备课检索": "TEACHER_PREPARATION",
    "学习讲解": "LEARNING_EXPLANATION",
    "学习规划": "STUDY_PLANNING",
    "闲聊": "CHITCHAT",
    "超出范围": "OUT_OF_DOMAIN",
}

# 标签 ↔ 模型输出下标的唯一映射：训练脚本、推理服务、评测都从这里取。
# 顺序由 INTENT_LABELS 决定，**不要**在别处再写一份 label2id。
# `rag_system/core/bert_{train,test}_model.py` 已按本表训练与推理（2026-10-30 对齐）；
# 它们原先各自写了 6 类 label2id，不会报错，只会把新口径的样本静默错分。
INTENT2ID: Final[dict[str, int]] = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2INTENT: Final[dict[int, str]] = {i: label for i, label in enumerate(INTENT_LABELS)}
# 对外英文代号 → 中文规范名（`INTENT_CODES` 的反查）。大模型偶尔照着任务书
# 直接吐英文代号，认它比让它落 unknown 划算；但**只认这 7 个**。
CODE2INTENT: Final[dict[str, str]] = {code: label for label, code in INTENT_CODES.items()}

# 兜底意图：分类模型不可用或置信度不足时使用，绝不硬路由到专业检索。
# 按 CLAUDE.md §7.7 硬规则 2 走 generic_clarify（全库检索 + 结尾追问）：
# 把"没听懂"记成"闲聊"会让指标失真，还会把错误答案写进缓存。
INTENT_FALLBACK: Final[str] = "unknown"

# 置信度闸门（§3.2 第 2 条）：低于它一律不猜，落 `INTENT_FALLBACK`。
# 三个调用方必须用同一个数，否则"模型 A 说 0.58 弃权、模型 B 说 0.58 定档"，
# 日志里两条记录看起来毫无差别、实际走了相反的路：
#   - `understanding/bert_intent.py`（精调分类器，任务 2.2）
#   - `understanding/llm_understanding.py`（大模型）
#   - `rag_system/core/bert_train_model.py`（训练/评测侧的弃权口径，本文件是它的上游出处；
#     该脚本另有同名常量 `MIN_CONFIDENCE`，值必须与这里一致，改一处就要改两处并重训）
INTENT_MIN_CONFIDENCE: Final[float] = 0.6

# **旧 6 类 → 新 7 类的迁移表**，不再是"数据集简称 → 规范名"的桥（数据集已经是新口径）。
# 只剩两个用途：① 滚动发布期间老进程/老缓存里的旧标签能被认出来；
# ② 度量口径切换时能回溯历史数据。**新代码不得依赖它做容错**——
# 大模型输出的标签必须逐字命中 INTENT_LABELS，否则按词表外处理（见 llm_understanding）。
INTENT_ALIASES: Final[dict[str, str]] = {
    "知识点查询": "学习讲解",   # 合并
    "题目讲解": "学习讲解",     # 合并
    "教案素材": "教案制作",     # 语义等价
    "学习方法": "学习规划",     # 语义等价
    "备课辅助": "备课检索",     # 语义收窄（原来的"出题/设计"部分归教案制作）
    "闲聊": "闲聊",
}

# 不需要检索的意图。
# 「超出范围」也在这里：查库对"明天天气"没有意义，但它**不是**闲聊——
# 闲聊允许自由作答（allow_general=True），超出范围必须拒答（allow_general=False）。
# 两者的分野在档位表，不在这个集合里。
INTENT_NO_RETRIEVAL: Final[frozenset[str]] = frozenset({"闲聊", "超出范围"})

# 学科：与 datasets/04_faq 的 subject_name 保持一致
# **权威配置**，不靠语料扫描自动扩展——否则一个错别字就成了合法学科（见 CLAUDE.md §7.1.8）。
SUBJECTS: Final[tuple[str, ...]] = (
    "语文",
    "数学",
    "英语",
    "物理",
    "化学",
    "生物",
    "道德与法治",  # 现行教材口径；语料 97 条即用此名
    "历史",
    "地理",
)

# 写法归一：教材改名/口语简称 → canonical 名。
# "政治" 是旧口径（标签表原先只有它），现行语料统一写"道德与法治"；两者是同一门课的
# 不同叫法，映射错会让那 97 条永远召不回来（见 CLAUDE.md §7.1.8）。
# 这是**配置**不是代码逻辑：口径若由项目方改回"政治"，只改这张表。
SUBJECT_ALIASES: Final[dict[str, str]] = {
    "政治": "道德与法治",
    "思想政治": "道德与法治",
    "思想品德": "道德与法治",
    "道法": "道德与法治",
    "品德": "道德与法治",
    # 教材目录里写的是"生物学"，词表里的规范名是"生物"；不改这一条，
    # 入库的 subject 就永远和检索过滤的取值对不上（§7.1.8 同一条教训）。
    "生物学": "生物",
}


# 学段与年级：新语料 `datasets/初高中教材教案试卷-按年级` 的目录就是
# `阶段/学科/年级/{教材,教案,试卷,题库}`，这两层必须作为字段落进 Milvus——
# 否则"高一物理的动量定理"和"高二物理的动量定理"在库里长得一模一样，引用卡片也分不清。
STAGES: Final[tuple[str, ...]] = ("初中", "高中")

GRADES: Final[tuple[str, ...]] = (
    "七年级",
    "八年级",
    "九年级",
    "高一",
    "高二",
    "高三",
)

# 年级 → 学段。年级本身就蕴含了学段，所以只认出年级时不必再猜阶段。
GRADE_STAGE: Final[dict[str, str]] = {
    "七年级": "初中",
    "八年级": "初中",
    "九年级": "初中",
    "高一": "高中",
    "高二": "高中",
    "高三": "高中",
}

# 目录/文件名里的其它写法 → 规范名（"初一"这类老口径教材里很常见）
GRADE_ALIASES: Final[dict[str, str]] = {
    "初一": "七年级",
    "初二": "八年级",
    "初三": "九年级",
    "初中一年级": "七年级",
    "初中二年级": "八年级",
    "初中三年级": "九年级",
    "高中一年级": "高一",
    "高中二年级": "高二",
    "高中三年级": "高三",
}

STAGE_ALIASES: Final[dict[str, str]] = {"初中阶段": "初中", "高中阶段": "高中"}


def normalize_stage(raw: str | None) -> str | None:
    """学段写法归一；未知返回 None（不猜）。"""
    if not raw:
        return None
    value = STAGE_ALIASES.get(raw.strip(), raw.strip())
    return value if value in STAGES else None


def normalize_grade(raw: str | None) -> str | None:
    """年级写法归一；未知返回 None（不猜）。"""
    if not raw:
        return None
    value = raw.strip()
    value = GRADE_ALIASES.get(value, value)
    return value if value in GRADES else None


def validate_intent(label: str) -> str:
    """校验意图标签，非法值返回兜底意图（不抛异常，避免污染请求路径）。"""
    return label if label in INTENT_LABELS else INTENT_FALLBACK


def normalize_intent_label(label: str) -> str:
    """意图标签归一：先认规范名，再认旧 6 类的迁移别名；都认不出返回兜底意图。

    先查 `INTENT_LABELS` 再看 `INTENT_ALIASES`，是因为别名表已经不是"全量映射"
    （它只是旧口径的迁移表），只查别名会让新口径的规范名自己也变成 unknown。
    """
    value = label.strip()
    if value in INTENT_LABELS:
        return value
    if value == INTENT_FALLBACK:
        return INTENT_FALLBACK
    return INTENT_ALIASES.get(value, INTENT_FALLBACK)


def normalize_subject(raw: str | None) -> str | None:
    """把学科写法归一（去空白/后缀"学科"/别名表），未知返回 None。"""
    if not raw:
        return None
    value = raw.strip().removesuffix("学科").strip()
    value = SUBJECT_ALIASES.get(value, value)
    return value if value in SUBJECTS else None


def is_valid_subject(raw: str | None) -> bool:
    """是否是已知学科写法（含别名）。公开 API 用它决定 400 还是放行。"""
    return normalize_subject(raw) is not None


# ---------------------------------------------------------------------------
# 槽位词表（阶段 2 任务 2.3）
#
# 这些词表与 `SUBJECTS` / `GRADES` 同一口径：**权威配置**，只在这里定义，
# 抽取端（`qa_system/res/qa_slots.py`）与校验端（`understanding/slots.py`）都从这里取。
#
# 为什么封闭词表而不是让模型自由发挥：槽位会被拼进检索过滤与提示词，
# 一个词表外的值不是"多一个写法"，而是"永远零命中"（§7.1.8 的 97 条 unlabeled 就是这么来的）。
# 因此每个 `normalize_*` 认不出**一律返回 None**，由调用方决定是"缺这个槽位"还是"丢弃模型给的这个词"。
#
# 难度按计划书的字面量用英文（`{"difficulty": "medium"}`），中文只作为输入侧别名：
# 输出侧若也允许中文，同一个字段就会有两种写法并存，下游比较时又得再归一一次。
# ---------------------------------------------------------------------------

TEXTBOOK_VERSIONS: Final[tuple[str, ...]] = (
    "人教版",
    "部编版",
    "北师大版",
    "苏教版",
    "外研版",
    "沪科版",
    "华东师大版",
    "湘教版",
    "冀教版",
    "鲁教版",
    "沪教版",
)

TEXTBOOK_VERSION_ALIASES: Final[dict[str, str]] = {
    "人教": "人教版",
    "人教社": "人教版",
    "人民教育出版社": "人教版",
    "北师大": "北师大版",
    "北京师范大学出版社": "北师大版",
    "苏教": "苏教版",
    "外研": "外研版",
    "外语教学与研究出版社": "外研版",
    "沪科": "沪科版",
    "华师大": "华东师大版",
    "华东师大": "华东师大版",
    "华东师范大学出版社": "华东师大版",
    "湘教": "湘教版",
    "冀教": "冀教版",
    "鲁教": "鲁教版",
    "沪教": "沪教版",
    "部编": "部编版",
    "统编": "部编版",
    "统编版": "部编版",
}

SEMESTERS: Final[tuple[str, ...]] = ("上册", "下册")

SEMESTER_ALIASES: Final[dict[str, str]] = {
    "上学期": "上册",
    "下学期": "下册",
    "第一学期": "上册",
    "第二学期": "下册",
    "上期": "上册",
    "下期": "下册",
}

DIFFICULTIES: Final[tuple[str, ...]] = ("easy", "medium", "hard")

# 只收多字词：单字"中/易/难"在教育问句里是高频普通词（"初中""中等职业学校"），
# 按词面抽取会把大量查询误标成某个难度。
DIFFICULTY_ALIASES: Final[dict[str, str]] = {
    "简单": "easy",
    "容易": "easy",
    "基础": "easy",
    "入门": "easy",
    "中等": "medium",
    "适中": "medium",
    "一般": "medium",
    "普通": "medium",
    "困难": "hard",
    "较难": "hard",
    "拔高": "hard",
    "压轴": "hard",
    "挑战": "hard",
}

EXAM_TYPES: Final[tuple[str, ...]] = (
    "期中考试",
    "期末考试",
    "月考",
    "单元测试",
    "中考",
    "高考",
    "模拟考试",
    "学业水平考试",
    "竞赛",
)

EXAM_TYPE_ALIASES: Final[dict[str, str]] = {
    "期中": "期中考试",
    "期末": "期末考试",
    "模拟考": "模拟考试",
    "模拟": "模拟考试",
    "会考": "学业水平考试",
    "学考": "学业水平考试",
}

QUESTION_TYPES: Final[tuple[str, ...]] = (
    "选择题",
    "填空题",
    "判断题",
    "计算题",
    "解答题",
    "证明题",
    "简答题",
    "实验题",
    "作图题",
)

LESSON_TYPES: Final[tuple[str, ...]] = ("新授课", "复习课", "习题课", "实验课", "讲评课")

RESOURCE_TYPES: Final[tuple[str, ...]] = (
    "教案",
    "课件",
    "学案",
    "习题",
    "试卷",
    "视频",
    "素材",
    "说课稿",
)

OUTPUT_FORMATS: Final[tuple[str, ...]] = ("markdown", "docx", "pptx", "pdf", "excel")

OUTPUT_FORMAT_ALIASES: Final[dict[str, str]] = {
    "word": "docx",
    "wps": "docx",
    "文档": "docx",
    "ppt": "pptx",
    "幻灯片": "pptx",
    "演示文稿": "pptx",
    "md": "markdown",
    "markdown格式": "markdown",
    "xlsx": "excel",
    "表格": "excel",
}


def _normalize_from(raw: str | None, canonical: tuple[str, ...], aliases: dict[str, str]) -> str | None:
    """共用归一：先原样命中规范名，再查别名表；都不中返回 None。

    五个 `normalize_*` 只差两张表，分头写五份等于把"先规范名后别名"的顺序
    抄五遍——其中一份写反了，就会出现"规范名被别名表吃掉"的诡异结果。
    """
    if not raw:
        return None
    value = raw.strip()
    if value in canonical:
        return value
    lowered = value.lower()
    if lowered in canonical:
        return lowered
    return aliases.get(value) or aliases.get(lowered)


def normalize_textbook_version(raw: str | None) -> str | None:
    """教材版本写法归一；未知返回 None（不猜）。"""
    return _normalize_from(raw, TEXTBOOK_VERSIONS, TEXTBOOK_VERSION_ALIASES)


def normalize_semester(raw: str | None) -> str | None:
    """上/下册写法归一；未知返回 None。"""
    return _normalize_from(raw, SEMESTERS, SEMESTER_ALIASES)


def normalize_difficulty(raw: str | None) -> str | None:
    """难度归一成 easy / medium / hard；未知返回 None。"""
    return _normalize_from(raw, DIFFICULTIES, DIFFICULTY_ALIASES)


def normalize_exam_type(raw: str | None) -> str | None:
    """考试类型归一；未知返回 None。"""
    return _normalize_from(raw, EXAM_TYPES, EXAM_TYPE_ALIASES)


def normalize_question_type(raw: str | None) -> str | None:
    """题型归一；未知返回 None。"""
    return _normalize_from(raw, QUESTION_TYPES, {})


def normalize_lesson_type(raw: str | None) -> str | None:
    """课型归一；未知返回 None。"""
    return _normalize_from(raw, LESSON_TYPES, {})


def normalize_resource_type(raw: str | None) -> str | None:
    """资源类型归一；未知返回 None。"""
    return _normalize_from(raw, RESOURCE_TYPES, {})


def normalize_output_format(raw: str | None) -> str | None:
    """输出格式归一；未知返回 None。"""
    return _normalize_from(raw, OUTPUT_FORMATS, OUTPUT_FORMAT_ALIASES)


def surfaces(canonical: tuple[str, ...], aliases: dict[str, str]) -> tuple[tuple[str, str], ...]:
    """展开成 `(词面, 规范名)` 对，按词面长度降序——抽取端据此做"最长优先"匹配。

    不做最长优先的后果是可复现的：`期中考试` 里包含 `中考`，
    两个词面都命中就会在"承认哪一个是考试类型"上取决于遍历顺序。
    """
    pairs = {**{name: name for name in canonical}, **aliases}
    return tuple(sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0])))
