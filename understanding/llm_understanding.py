"""理解层的一次大模型调用：查询改写 + 意图识别 + 检索策略 + 槽位补充（§3.2 / §7.7）。

**为什么合并成一次调用**：它们落在同一个流水线位置（FAQ 未高置信命中之后、检索之前），
各要一次 LLM 往返。拆成四次等于把同一份预算花四遍。合并**不牺牲独立性**——
改写 / 意图 / 策略 / 槽位各自校验、各自降级（见 `analyze()` 里四个 `if want_*` 分支）：
任意一项不合格都不影响其余项，全都不合格就等于"只做了归一化"。

**策略只是一条建议，不是执行指令**：这里只说"这句话适合怎么查"，
"能不能这么查"由 `rag_system.core.planner` 判——要拆成多路得有真拆出来的子问题，
要 HyDE 得模型可用。不可行的建议会被降级并留下原因码，不会假装跑过。

槽位这一项只负责**把模型给的原样交出去**，不在这里校验：过滤类槽位要被丢弃、
词表外的值要被归一，那套规则属于 `understanding/slots.py`（与规则的合并也在那里）。

边界（违反即打回）：

1. **不 import `rag_system`**（§7.0）：用一个本地窄协议 `_UnderstandLLM` 描述"能被调用的模型"，
   与 `understanding/rewrite.py::_TurnLike` 同一写法；
2. **不抛异常**：超时 / 未配置 / 上游失败一律转成 `degraded` 标记。改写与意图都是增强项，
   不该让检索挂掉；
3. **不改用户原话**：调用方拿到的只是"建议的检索问句"，`state.query` 永远保持原样
   （FAQ 的精确哈希依赖原话，§7.9 边界）；
4. **不写 if/else 路由**：这里只产出 `IntentResult`，档位由 `understanding/profiles.py` 决定。

一个刻意的取舍：`method` 用 `"model"`（`IntentResult.method` 的 Literal 取值），
不用 `"llm"`——`Literal["model","rule","fallback"]` 是契约，改它要动 §7.7 与全部消费方。
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from base.contracts import IntentResult
from base.labels import CODE2INTENT, INTENT_FALLBACK, INTENT_LABELS
from base.logger import get_logger
from understanding.profiles import profile_for

log = get_logger(__name__)

# 与 `understanding/rewrite.py::_MAX_REWRITTEN` 同一口径：超过它的"改写"已经是答题了
_MAX_REWRITTEN = 120
# §3.2 第 2 条：置信度 < 0.6 不猜。落到 unknown（走 generic_clarify），不硬塞进某一类
_MIN_INTENT_CONFIDENCE = 0.6
# 拒答式开头：模型偶尔会把"我不确定"写进 query 字段，那是答案不是问句
_REFUSAL_OPENERS = ("抱歉", "对不起", "我无法", "我不能", "作为一个", "作为AI", "作为 AI")
# `[1]` / `[12]` 是引用标记，出现在"检索问句"里说明模型把它当答案在写
_CITATION_MARK = re.compile(r"\[\s*\d+\s*\]")
# 改写至少要保留原句四成长度，否则大概率是把限定条件（学科/年级/公式）丢了
_MIN_KEEP_RATIO = 0.4

_SYSTEM = """你是 K12 教育问答系统的"问题理解"模块。给你一句用户提问，输出下面几件事。

【改写】把它改写成一句规范、完整、适合检索的现代汉语问句。
- 保留原问题的全部知识点与限定（学科、年级、章节、公式、题号），不增不减；
- 补全省略成分（"咋""啥"→"怎么""什么"），去掉语气词与客套；
- 已经足够规范就原样返回，不要为了改写而改写；
- 只输出问句本身：不回答问题、不解释、不加引号、不加编号，不超过 {max_chars} 字。

【意图】从下列 7 类中选恰好一个，原样输出中文名称。
- 教案制作：要产出**新的教学材料**——教案、课件、PPT、说课稿、导学案、板书设计、课堂活动、作业设计；
- 考试出题：要产出**题目或试卷**——出一套卷子、专项训练、变式题、模拟题、带答案解析；
- 备课检索：要找**已经存在的**教学资源——找课件、找公开课、找素材、有没有现成的教案可以参考；
- 学习讲解：讲**知识点或题目**——概念是什么、为什么、这道题怎么解、这个选项为什么错；
- 学习规划：给一条**学习路径**——还剩多久来得及吗、怎么复习、哪块薄弱怎么补、怎么提分；
- 闲聊：与教学内容无关的寒暄、情绪、日常对话（你好、谢谢、在吗、今天好累）；
- 超出范围：不涉及教育领域的实质问题（天气、购物、出行、医疗、金融、娱乐等）。

判断顺序（按此顺序逐条排除，不要跳步）：
1. 有没有教学内容？没有 → 是寒暄就"闲聊"，是实质问题就"超出范围"。
   注意：带任何知识点、学科、题目、课堂、学生的问句都属于"有教学内容"，一律不是这两类。
2. 要**产出**一份新东西吗？看产出物：材料类（教案/课件/讲义/PPT/说课稿）→ 教案制作；
   题目类（卷子/专项训练/练习题/带答案）→ 考试出题。
3. 要**找**已经存在的东西吗（"有没有""哪里能找到""帮我找"）→ 备课检索。
4. 要**把知识或题目讲清楚**吗 → 学习讲解。
5. 要**给一条学习或提分路径**吗（常带时间、分数、薄弱项、学情）→ 学习规划。

最容易混的两对，判据是**动词**不是名词：
- 教案制作 vs 备课检索：`帮我出一份《氧气》的讲义` 是教案制作（生成）；
  `帮我找一份《氧气》的讲义` 是备课检索（检索）。同一个名词，动词不同，类别不同。
- 考试出题 vs 学习讲解：`出一套关于二次函数的卷子` 是考试出题；
  `二次函数的顶点式怎么求` 是学习讲解。问"怎么做"不是"要题"。
"""
# 检索策略段落只在 `want_strategy=True` 时拼进提示词。**只列出会被真正执行的三种**：
# `backtrack` 是"含指代但解不开 → 追问"的标记、`faq` 是出口，两者都不是检索方式，
# 给模型看只会诱导它在这两条上乱选——而它们各自有结构性的前置条件（无历史 / 命中 FAQ）。
_STRATEGY_SECTION = """
【检索策略】从下列 3 种中选恰好一个，把英文名原样写进 "strategy" 字段。
- direct：整句是**一个**诉求，一次检索就能答。拿不准时一律选它。
- sub_query：一句话里有 2 件以上**互相独立**的事，各自需要单独查资料
  （例："比较一次函数和二次函数的区别，并分别给我出三道练习题"）。
  整句一起查会让两组证据互相稀释，谁都排不进 top-k。
- hyde：问句很短、很抽象，知识点名称本身不足以命中教材原文
  （例："什么是熵""为什么会有季节"），需要先写一段教材式说明再去查。
判据是**诉求的个数**，不是名词的个数：`二次函数和一次函数有什么区别` 是一个诉求（direct），
`二次函数怎么求顶点，一次函数怎么求斜率` 才是两个诉求（sub_query）。
"""
# 槽位段落只在 `want_slots=True` 时拼进提示词：不抽槽位时还让它输出，
# 既浪费输出 token，也会让"模型没给"与"我们没要"在日志里分不清。
_SLOTS_SECTION = """
【槽位】把用户**明确写出**的结构化信息放进 "slots" 对象。**没有的键直接省略**，
不要猜、不要为了填满而编——缺一个槽位不算错，编一个会让检索用错过滤条件。
- chapter：章节或课题名（"一次函数""第十七章勾股定理"），不超过 30 字；
- knowledge_points：知识点数组，最多 5 个，每个 2–20 字，必须是用户问句里出现的词；
- difficulty："easy" / "medium" / "hard" 之一，只在用户明确说了难度时给；
- exam_type：考试类型（期中考试 / 期末考试 / 月考 / 单元测试 / 中考 / 高考 / 模拟考试 / 学业水平考试 / 竞赛）；
- task：任务参数，只填与意图对应的那一项——
  教案制作 → {"lesson": {"duration_min": 40, "objectives": ["..."], "lesson_type": "新授课", "output_format": "docx"}}
  考试出题 → {"exam": {"question_count": 10, "question_types": ["选择题", "填空题"], "difficulty_mix": "3:5:2", "total_score": 100, "need_analysis": true}}
  备课检索 → {"prep": {"resource_types": ["教案", "课件"]}}
  学习规划 → {"plan": {"target_exam": "中考", "exam_date": "2027-06-20", "current_level": "中等", "weak_points": ["二次函数"], "weekly_hours": 6}}
- 数值字段给数字，不要写成"十道""半个小时"。

**不要把学科、年级、教材版本、学期写进 slots**：这四项由系统规则决定，你写了会被直接丢弃。
"""

_SHAPE_BASE = '{{"query": "...", "intent": "学习讲解", "confidence": 0.9}}'
_SHAPE_WITH_SLOTS = (
    '{{"query": "...", "intent": "考试出题", "confidence": 0.92,'
    ' "slots": {{"chapter": "一次函数", "knowledge_points": ["函数概念"],'
    ' "task": {{"exam": {{"question_count": 10}}}}}}}}'
)
# 策略字段插在**最外层对象**的收尾之前，所以取 `[:-2]` 而不是按最后一个 `}`
# 去插：示例里的花括号是成对写的（`{{` / `}}`），按单个 `}` 定位会插进收尾括号中间，
# 得到一个花的示例，而模型会照着花的示例写花的 JSON。
_STRATEGY_FIELD = ', "strategy": "direct"'

# 模型写策略的常见变体。只认这三种**会被真正执行**的取值，
# `backtrack` / `faq` 即使在别名表里出现也在这里被挡掉。
_STRATEGY_ALIASES = {
    "direct": "direct",
    "sub_query": "sub_query",
    "subquery": "sub_query",
    "hyde": "hyde",
}


# 段落头。单独提出来是因为下面要按它切段：把中文标题在切分处再抄一遍，
# 某天改了标题就会静默切不动（`partition` 找不到就返回整段，不报错）。
# 刻意不用"一、二、三"编号：按开关丢掉任一段之后，剩下的编号会出现空洞。
_H_REWRITE = "【改写】"
_H_INTENT = "【意图】"


def _shape(*, want_slots: bool, want_strategy: bool) -> str:
    """要哪些字段就给哪些字段的形状示例。不要的字段**不出现**：
    示例里留着它，模型就会照着填，而我们随后把它丢掉——白付输出 token，日志里还多一条"被拒"。"""
    sample = _SHAPE_WITH_SLOTS if want_slots else _SHAPE_BASE
    return sample[:-2] + _STRATEGY_FIELD + sample[-2:] if want_strategy else sample


def _system_prompt(
    *, want_slots: bool, want_rewrite: bool, want_intent: bool, want_strategy: bool
) -> str:
    """按本次真正要的项拼提示词。

    只用一个模板、靠 `str.replace` 填参数（不用 `str.format`）：JSON 示例里全是花括号，
    每加一个字段就要多一倍转义，写着写着就会出现"提示词里少了半个括号"这种只有线上才发现的错。

    **按段装配而不是逐条截断**：改写成两段（`<头>【改写】…【意图】…`）之后，
    "只要意图"就不能再写成 `partition(_H_INTENT)[0]`——那样会把**排在意图之后的**
    每一段（策略、槽位）一起切掉，且不报错。四项开关各拿各的段，顺序固定。
    """
    body = _SYSTEM.replace("{max_chars}", str(_MAX_REWRITTEN))
    head, _, tail = body.partition(_H_REWRITE)
    rewrite_seg, _, intent_seg = tail.partition(_H_INTENT)
    parts = [head]
    if want_rewrite:
        parts.append(_H_REWRITE + rewrite_seg)
    if want_intent:
        parts.append(_H_INTENT + intent_seg)
    if want_strategy:
        parts.append(_STRATEGY_SECTION)
    if want_slots:
        parts.append(_SLOTS_SECTION)
    parts.append("\n只输出 JSON，不要代码块，不要任何其他文字：\n")
    parts.append(_shape(want_slots=want_slots, want_strategy=want_strategy))
    return "".join(parts)




class _UnderstandLLM(Protocol):
    """本地窄协议：只声明本模块真正用到的一件事。

    不 import `rag_system.core.llm_gateway.LLMClient`——理解层不依赖检索包（§7.0）。
    `Any` 只出现在这个 SDK 边界上，拿到之后立刻收敛成 `str`（§5.2）。
    """

    @property
    def enabled(self) -> bool: ...

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class LLMUnderstanding:
    """一次理解调用的结论。

    字段为空 = 这一项没拿到：改写退归一化句、意图交规则分类器、槽位只剩规则抽到的那些。
    """

    rewritten: str | None = None
    intent: IntentResult | None = None
    # 模型给的槽位**原样**带出去，校验与合并全在 `understanding/slots.py`：
    # 在这里顺手校验会让"模型写错了"与"链路降级了"混成一个标记。
    slots: dict[str, Any] | None = None
    # 检索策略建议（`direct` / `sub_query` / `hyde`），空串 = 没给或不可用。
    # **只是一条建议**：能不能这么查由 `rag_system.core.planner` 判（要拆多路得有真拆出来的
    # 子问题，要 HyDE 得模型可用），这里不做可行性判断——那是两件事，混在一起就无法分辨
    # "模型选错了"与"选了但跑不了"。
    strategy: str = ""
    method: str = "none"            # none / llm
    degraded: tuple[str, ...] = ()


async def analyze(
    query: str,
    llm: _UnderstandLLM | None,
    *,
    budget_ms: int,
    model: str | None = None,
    want_rewrite: bool = True,
    want_intent: bool = True,
    want_slots: bool = False,
    want_strategy: bool = False,
) -> LLMUnderstanding:
    """一次调用拿回"改写后的检索问句"、"意图"与（可选）"模型补充的槽位 / 检索策略"。**永不抛异常。**

    `query` 应当是**已归一化**的问句（改写的输入），但原始问话由调用方保留——
    这里的返回值只是一条建议，改写是否采用由编排层决定。

    `want_slots` 默认 `False`：槽位的过滤类字段只认规则，模型只补章节名 / 知识点 /
    任务参数这几个规则覆盖不到的地方（任务 2.3）。不需要时还让它输出，既浪费 token，
    也会让"模型没给"与"我们没要"在日志里分不清。

    `want_strategy` 同样默认 `False`，但**调用方一律显式传**：`FEATURE_PLANNER_LLM` 打开时
    传 `True`，关掉时传 `False`。默认 `False` 是为了保住"什么都没要就一次调用都不发"这条
    既有契约（§6.1 的预算账按"零次调用"算过），不是为了让它悄悄关着。
    """
    text = (query or "").strip()
    if not text or not (want_rewrite or want_intent or want_slots or want_strategy):
        return LLMUnderstanding()
    if llm is None or not getattr(llm, "enabled", False):
        # 没配 key 是**配置状态**，不是本次请求的降级：不该出现在每一份 degraded 里。
        # 启动期的 `Settings.soft_warnings()` 已经提醒过一次了。
        return LLMUnderstanding()

    messages = [
        {
            "role": "system",
            "content": _system_prompt(
                want_slots=want_slots,
                want_rewrite=want_rewrite,
                want_intent=want_intent,
                want_strategy=want_strategy,
            ),
        },
        {"role": "user", "content": text},
    ]
    try:
        # 网关自己也有超时；这里再包一层是因为预算可能被上游调大，
        # 而理解层的 800 ms 是 §6.1 里写死的，不能靠网关的 60 s 兜底
        result = await asyncio.wait_for(
            llm.complete(messages, budget_ms=budget_ms, model=model),
            timeout=max(0.05, budget_ms / 1000.0),
        )
    except TimeoutError:
        return LLMUnderstanding(degraded=("understanding_timeout",))
    except Exception as exc:  # noqa: BLE001 —— 改写是增强项，任何异常都不该冒泡到检索
        # 只记一条 warn 再降级：`understanding_failed` 这一个码背后可能是 429 限流、
        # 连接被掐、网关 5xx，处置完全不同。评估脚本里 47% 的样本落到这个码而看不出原因，
        # 只能靠这行日志区分"模型不行"与"评估环境撞墙"。
        log.warning("understanding_failed", error=f"{type(exc).__name__}: {exc}"[:200])
        return LLMUnderstanding(degraded=("understanding_failed",))

    data = _extract_json(getattr(result, "text", "") or "")
    if data is None:
        return LLMUnderstanding(degraded=("understanding_bad_json",))

    degraded: list[str] = []
    rewritten: str | None = None
    if want_rewrite:
        rewritten, rejected = _valid_rewrite(_clean(data.get("query")), text)
        if rejected:
            degraded.append("rewrite_rejected")
    intent: IntentResult | None = None
    if want_intent:
        intent = _valid_intent(data.get("intent"), data.get("confidence"))
        if intent is None:
            degraded.append("intent_rejected")
    slots: dict[str, Any] | None = None
    if want_slots:
        raw_slots = data.get("slots")
        if isinstance(raw_slots, dict):
            slots = raw_slots
        elif raw_slots is not None:
            # 给了个数组或字符串：模型没按格式来。缺键是"没抽到"（常态），
            # 给了错类型才是"没照做"，两者在排障时完全不同
            degraded.append("slots_rejected")
    strategy = ""
    if want_strategy:
        # 认不出的取值一律退回规则规划，**不猜最近的那个**：
        # 把 `sub_query` 猜成 `direct` 会让该拆的句子一路查下去，结果是"看起来没拆"，
        # 而原因码只说"策略被拒"，排障时还得翻提示词。
        strategy = _valid_strategy(data.get("strategy"))
        if not strategy:
            degraded.append("strategy_rejected")
    return LLMUnderstanding(
        rewritten=rewritten,
        intent=intent,
        slots=slots,
        strategy=strategy,
        method="llm",
        degraded=tuple(degraded),
    )


# --------------------------------------------------------------------------- 校验
def _clean(raw: object) -> str:
    """去掉引号 / 代码块围栏 / 首尾空白。模型常见的两种多余包装。"""
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if text.startswith("```"):
        body = text.split("\n", 1)[1] if "\n" in text else ""
        text = body.rsplit("```", 1)[0] if "```" in body else body
    return text.strip().strip('"\'“”‘’').strip()


def _valid_rewrite(candidate: str, original: str) -> tuple[str | None, bool]:
    """`(改写结果, 是否被拒)`。相同 → `(None, False)`，那是"本来就规范"，不是失败。"""
    if not candidate:
        return None, True
    if candidate == original:
        return None, False
    if len(candidate) > _MAX_REWRITTEN:
        return None, True
    if "\n" in candidate or "\r" in candidate:
        return None, True
    if _CITATION_MARK.search(candidate):
        return None, True
    if candidate.startswith(_REFUSAL_OPENERS):
        return None, True
    # 砍掉限定词（"八年级数学的勾股定理" → "勾股定理"）会静默改变检索范围，
    # 这种"改写"比不改更危险：它看起来成功了，召回却丢了约束
    if len(original) >= 8 and len(candidate) < len(original) * _MIN_KEEP_RATIO:
        return None, True
    return candidate, False


def _valid_intent(label: object, confidence: object) -> IntentResult | None:
    """`IntentResult | None`。`None` = 模型没给出可用标签，调用方交回规则分类器。

    **不走 `normalize_intent_label()`**，那里会顺着 `INTENT_ALIASES` 把旧 6 类名
    （"知识点查询""教案素材"…）映射到新类上。那对滚动发布是对的，对**评估**是灾难：
    模型若照旧口径吐标签，我们会静默接受并在报告里记成"按新口径答对了"，
    于是任务 2.1 最想量的那件事（大模型到底认不认新口径）永远量不出来。
    这里只认逐字命中的 7 个规范名，外加任务书里的英文代号（`CODE2INTENT`）。
    """
    if not isinstance(label, str):
        return None
    raw = label.strip()
    resolved = raw if raw in INTENT_LABELS else CODE2INTENT.get(raw, INTENT_FALLBACK)
    if resolved == INTENT_FALLBACK and raw != INTENT_FALLBACK:
        # 词表外的标签（"数学题"/"提问"/旧 6 类名）：当它没给，别屈尊用它
        return None
    try:
        conf = float(confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        conf = 0.0
    conf = min(1.0, max(0.0, conf))
    if conf < _MIN_INTENT_CONFIDENCE:
        # §3.2 第 2 条：不猜。落 unknown 走 generic_clarify（全库检索 + 结尾追问），
        # 而不是硬路由到某一类——错误档位的答案会被写进缓存
        return IntentResult(
            label=INTENT_FALLBACK,
            confidence=conf,
            method="model",
            profile=profile_for(INTENT_FALLBACK).name,
        )
    return IntentResult(
        label=resolved, confidence=conf, method="model", profile=profile_for(resolved).name
    )


def _valid_strategy(raw: object) -> str:
    """`""` = 没给出可用的策略建议，调用方交回规则规划。

    只做**归一与认词**，不做可行性判断：`hyde` 在这儿是合法取值，能不能跑由 planner 判
    （模型不可用时它会被降级成 `direct` 并留下 `hyde_unavailable`）。两件事分开记，
    才分得清"模型选错了策略"与"选对了但这台机器跑不了"。
    """
    if not isinstance(raw, str):
        return ""
    key = raw.strip().lower().replace("-", "_").replace(" ", "_")
    return _STRATEGY_ALIASES.get(key, "")


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型输出里取出 JSON 对象。容忍代码块围栏与前后废话，不做"智能修复"。"""
    body = text.strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
