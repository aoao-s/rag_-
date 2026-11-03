"""复合问句拆分（任务 2.4）：**本地小模型优先，规则兜底**（2026-11-01 修订）。

拆的目的不是"把问题问得更全"，是让两件不相干的事**各自有自己的召回**：
「比较一次函数和二次函数的区别，并分别给我出三道练习题」如果整句进检索，
"区别"和"练习题"两组证据会互相稀释，谁都排不进 top-k。

**谁在拆**：`orchestration/chat.py::_split_compound()` 在**意图判定之后**调
`split_with_model()`——本地小模型（Ollama 的 OpenAI 兼容层，不计费、不出网）先拆，
规则 `split()` 兜底。三处位置/顺序都是刻意的：

- **排在意图之后**：拆句读的是"这句话里有几个诉求"，那是意图定性之后才有的判断；
  放在意图之前等于让一个还没定性的问题先决定要查几路。
- **用本地小模型而不是云端**：拆句的产出会直接变成检索任务（`planner` 下推成多路并行
  召回），一次请求多几十毫秒本地时延换一次云端往返与计费不划算，而且它**不该出网**。
- **规则不能删**：`FEATURE_COMPOUND_LLM=false` 时它就是唯一在跑的那条；模型超时 / 出错 /
  编词时它也是接住的那条。规则可复现、可单测、可逐条解释——出问题时能指着某条规则说
  "就是它"，而不是"模型今天心情不好"。

**来源可验（这条是硬约束，模型也绕不过去）**：所有子问题都由原问句的片段重排拼出，
出口处用**字符多重集**校验——子问题里出现的每一个字，都必须已经在原问句里出现过
（且次数不超过原问句）。所以"子问题必须有明确来源"不是一句承诺，是**结构上编不出来**：
没有任何路径（规则的或模型的）能引入原问句里没有的词。校验不过的候选一律丢弃并记进
`dropped`，不静默使用。

**模型只能多拆、不能把规则已经拆对的路拆没**：模型返回的可用子问题不足 2 条时退回规则，
而不是听它说"这不是复合句"。0.8b 的模型偶尔会把三诉求的句子当成一件事，让规则垫这个底
比让它否掉更安全。（这也意味着模型路径的收益是"拆得更准"或"拆出规则漏掉的"，
不会是"拆得更少"。）

⚠ **§5.4 的评估缺口（如实记录，别当成已完成）**：接通模型拆分属于检索行为改动，
按 §5.4 本该附黄金集对比报告（Recall/MRR/忠实度）。本轮**没有跑**——本机没有可用的
黄金集跑批环境，报告一个数字都不能编。现状是：规则路径有单测与固定的规格样本
（`tests/test_compound_query.py` 的三路黄金样本），模型路径只有正常 / 超时 / 坏 JSON /
编词四条结构性单测，**拆分质量的实测收益未量**。要做 A/B 时把 `FEATURE_COMPOUND_LLM`
当分组开关，规则组就是本文件里没被改动过的 `split()`。

与 `rag_system.core.planner` 的分工：本模块只负责"拆得对不对、来源可不可信"，
策略选择（BACKTRACK / SUB_QUERY / HYDE / DIRECT 的优先级）仍然只由 planner 决定。
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

# 与 `rag_system.core.planner.MAX_SUB_QUERIES` 同值。两处各留一份是刻意的：
# 它们守的是两件不同的事（这里守"别把一句话拆成五段"，planner 守"别发太多并行请求"），
# 合成一个常量就得让 understanding import rag_system——违反 §5.1 的分层方向。
MAX_SUB_QUERIES = 3

# 子问题的长度下限：太短的片段（"并"、"还有呢"）当检索问句只会拉低召回
_MIN_SUB_LEN = 6

_CLAUSE = re.compile(r"[，,。；;！!？?]|(?:并且|以及|还有|另外|同时|然后)")
# 「分别/各自」是**分发动作**的标记：它本身不是话题，说的是"上面提到的那几样，各来一份"
_DISTRIBUTE = re.compile(r"(分别|各自)")
_LEAD_CONNECTIVE = re.compile(r"^(并且|并|以及|还有|另外|同时|然后)\s*")
_LEAD_FRAME = re.compile(r"^(比较|对比|区分|分析|说说|讲讲|谈谈)\s*")
_TOPIC_SEP = re.compile(r"(?:和|与|跟|、|及)")
# 无标点并列诉求的连接词。「以及」不在此列（它由 `_CLAUSE` 整词消费，按"及"切会留下一个"以"）
_CONJUNCTION = re.compile(r"(?<!以)(?:和|与|跟|、)")
# 「A和B的区别/联系」——中文里最常见的对比句式，单独认，因为按分隔符硬切会切出
# 「二次函数的区别」这种带尾巴的伪话题（下面 `_looks_like_topic` 会挡掉它，但认得出更好）
_DIFF = re.compile(
    r"(?:比较|对比|区分|说说|讲讲)?\s*"
    r"(?P<a>[^，,。；;和与跟、的]{2,12})\s*(?:和|与|跟|、)\s*"
    r"(?P<b>[^，,。；;和与跟、的]{2,12})\s*的\s*"
    r"(?P<rel>区别|差异|不同|异同|联系|关系)"
)
# 裸名词短语的粗判据：不许带语气词与虚词，否则「二次函数的区别」也会被当成话题
_TOPIC_NOISE = re.compile(r"[的了吗呢吧啊着过是怎什哪多少]")
# 一段话像不像一个"独立诉求"（拆分的前提）。只有逗号不算数——
# 「老师，二次函数的顶点式怎么求」里"老师"不是第二件事
_CUE = re.compile(
    r"(什么|怎么|怎样|如何|为什么|多少|哪些|哪一|区别|差异|比较|对比|介绍|讲解|解释|"
    r"分析|总结|整理|生成|设计|出.{0,3}题|练习|证明|计算|举例|列举)"
)
# 数量的量词组：「给我出三道练习题」里的"三道"。话题要插在它后面
_NUM_CLASSIFIER = re.compile(r"[一二三四五六七八九十百千零两0-9]+\s*(?:道|个|条|篇|张|套|份)")
_TAIL_NOUNS = ("练习题", "题目", "试题", "练习", "教案", "素材", "课件", "方案")


@dataclass(frozen=True, slots=True)
class CompoundResult:
    """一次拆分的结论。`source` 说明**谁**拆的，排障时不至于只剩一个 bool。"""

    is_complex: bool = False
    sub_queries: tuple[str, ...] = ()
    # none / distribute / connective / conjunction / model
    source: str = "none"
    dropped: tuple[str, ...] = ()   # 被来源校验挡掉的候选（只用于诊断，不当结果用）
    # 这次拆分**退化了什么**：compound_timeout / compound_failed / compound_bad_json /
    # compound_rejected。非空 ⟺ 没拆成模型的样子（结果来自规则）。
    # **没有可用的本地模型不算降级**——那是部署形态（没配 Ollama 的机器照常跑规则），
    # 记成降级会让开发机上"这次请求没有任何降级"这条线恒假
    degraded: tuple[str, ...] = ()


def split(query: str) -> CompoundResult:
    """把一句可能含多个诉求的问题拆成 ≤3 个子问句。拆不动就原样返回 `is_complex=False`。"""
    text = (query or "").strip()
    if len(text) < _MIN_SUB_LEN:
        return CompoundResult()

    for source, builder in (
        ("distribute", _by_distribution),
        ("connective", _by_connective),
        ("conjunction", _by_conjunction),
    ):
        kept, dropped = _verified(text, builder(text))
        if len(kept) >= 2:
            return CompoundResult(
                is_complex=True,
                sub_queries=tuple(kept[:MAX_SUB_QUERIES]),
                source=source,
                dropped=tuple(dropped),
            )
    return CompoundResult()


# ---------------------------------------------------------------- 模型拆分
class _CompoundLLM(Protocol):
    """只声明本模块用到的那一点能力，**不 import 网关实现**（§5.1 的分层方向）。

    与 `understanding/llm_understanding.py::_UnderstandLLM` 同形：两边各自声明一份，
    是为了让"拆句只需要 complete"这件事不被网关后续新增的参数绑架。
    """

    @property
    def enabled(self) -> bool: ...

    async def complete(
        self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = None
    ) -> Any: ...


_SPLIT_SYSTEM = (
    "你是检索问句拆分器。用户问句里可能有好几件不同的事，请把它们拆成各自可独立检索的中文问句。\n"
    "硬性要求：\n"
    "1. 只能直接摘取原问句里已有的字词，**不许新增、替换、改写任何词**；\n"
    "2. 每条 4-30 字，必须是原问句的字面片段或片段拼接；\n"
    '3. 只输出 JSON：{"sub_queries": ["...", "..."]}，不要解释、不要代码块；\n'
    '4. 只有一件事就输出 {"sub_queries": []}。'
)


def looks_compound(text: str) -> bool:
    """结构预筛：判断是否值得调用本地模型。

    目标不是判断所有语义上的复合句，而是先挡住明显的单诉求句，
    避免每个带有“和、与、区别”的普通问题都调用一次 Ollama。

    允许进入模型的情况：
    1. 明确的分发诉求：分别、各自；
    2. 至少两个具有独立诉求特征的分句；
    3. 无标点并列结构，且规则能够识别出至少两个独立诉求。

    例如：
    - “比较一次函数和二次函数的区别” -> False
    - “求一次函数的值，还有二次函数的顶点式怎么求” -> True
    - “比较一次函数和二次函数的区别，并分别出三道题” -> True
    - “一次函数怎么求和二次函数有什么区别” -> True
    """
    text = (text or "").strip()

    if len(text) < _MIN_SUB_LEN:
        return False

    # ① 明确的分发标记：例如“分别出题”“各自介绍”
    if _DISTRIBUTE.search(text):
        return True

    # ② 标点或连接词分句后，至少有两个独立诉求
    clauses = [
        clause.strip()
        for clause in _CLAUSE.split(text)
        if len(clause.strip()) >= _MIN_SUB_LEN
    ]

    actionable_clauses = [
        clause for clause in clauses
        if _CUE.search(clause)
    ]

    if len(actionable_clauses) >= 2:
        return True

    # ③ 无标点并列诉求。
    # _by_conjunction 内部已经排除了：
    # “比较A和B的区别”这类单一对比诉求。
    if _by_conjunction(text):
        return True

    return False


async def split_with_model(
    query: str,
    llm: _CompoundLLM | None,
    *,
    budget_ms: int,
    model: str | None = None,
) -> CompoundResult:
    """本地模型优先拆分，任何异常都退回规则 split()。

    失败码：
    - compound_timeout：模型调用超时，或调用预算已经耗尽
    - compound_failed：模型网关调用异常
    - compound_bad_json：模型输出不是合法 JSON，或结构不符合约定
    - compound_rejected：模型提供了候选，但有效子问题不足两个，或候选未通过来源校验

    特殊情况：
    - 单诉求句：预筛直接返回，不调用模型
    - 没有配置本地模型：使用规则，不记 degraded
    - 模型明确返回空列表：使用规则，不记 degraded
    - 模型编造词语：拒绝模型结果，使用规则并记 compound_rejected
    """
    text = (query or "").strip()

    # ① 结构预筛：单诉求句不调用本地模型
    if not looks_compound(text):
        return CompoundResult()

    # ② 没有配置本地模型属于部署形态，不算运行故障
    if llm is None or not getattr(llm, "enabled", False):
        return split(text)

    # ③ 请求预算耗尽，不再发起无意义的模型请求
    if budget_ms <= 0:
        return _fallback(text, "compound_timeout")

    try:
        result = await asyncio.wait_for(
            llm.complete(
                _split_messages(text),
                budget_ms=budget_ms,
                model=model,
            ),
            timeout=budget_ms / 1000.0,
        )

    except asyncio.TimeoutError:
        return _fallback(text, "compound_timeout")

    except Exception:  # noqa: BLE001
        # 本地模型不可用不能影响主 RAG 链路
        return _fallback(text, "compound_failed")

    # ④ 提取模型文本
    payload = getattr(result, "text", None)

    if not isinstance(payload, str) or not payload.strip():
        return _fallback(text, "compound_bad_json")

    data = _json_object(payload)

    if data is None:
        return _fallback(text, "compound_bad_json")

    # ⑤ 校验 JSON 结构
    candidates = data.get("sub_queries")

    if not isinstance(candidates, list):
        return _fallback(text, "compound_bad_json")

    # 不允许静默丢弃非字符串数据。
    # 例如 ["一次函数怎么求", 123] 属于模型输出结构异常。
    if any(not isinstance(candidate, str) for candidate in candidates):
        return _fallback(text, "compound_bad_json")

    # ⑥ 模型明确表示没有复合诉求
    # 不能让模型的空列表覆盖规则已经拆对的结果。
    # 这是模型的正常判断，不记降级。
    if not candidates:
        return _fallback(text, "")

    # ⑦ 字符多重集来源校验
    kept, dropped = _verified(text, candidates)

    # ⑧ 模型给了候选，但最终不足两个有效子问题
    # 无论是全部编造、只有一个有效候选，还是重复候选，
    # 都不能把它当成一次成功的模型拆分。
    if len(kept) < 2:
        return _fallback(text, "compound_rejected")

    # ⑨ 模型拆分成功
    return CompoundResult(
        is_complex=True,
        sub_queries=tuple(kept[:MAX_SUB_QUERIES]),
        source="model",
        dropped=tuple(dropped),
    )

def _fallback(text: str, code: str) -> CompoundResult:
    """回落到规则结果，并把"模型那条路为什么没成"挂上去（`code` 为空表示不是失败）。"""
    base = split(text)
    if not code:
        return base
    return CompoundResult(
        is_complex=base.is_complex,
        sub_queries=base.sub_queries,
        source=base.source,
        dropped=base.dropped,
        degraded=(code,),
    )


def _split_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _SPLIT_SYSTEM},
        {"role": "user", "content": query},
    ]


def _json_object(text: str) -> dict[str, Any] | None:
    """从模型输出里取出 JSON 对象。容忍代码块围栏与前后废话，不做"智能修复"。

    不 import `llm_understanding._extract_json`：那会把意图 / 档位 / 模型句柄整条链
    连带拉进本模块，而拆分只需要读一个对象。同样的取舍在 `MAX_SUB_QUERIES` 那儿已经做过一次。
    """
    body = text.strip()
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------- 来源校验
def _verified(origin: str, candidates: list[str]) -> tuple[list[str], list[str]]:
    """字符多重集校验：子问题必须是原问句的**子多重集**。

    这是整个模块唯一一条不能让的规则。用多重集而不是集合，是因为「三道」的"道"和
    「一次函数」的"一"都可能只有一个——集合校验会放过"凭空多借一个字"的情形。
    """
    allowed = Counter(origin)
    kept: list[str] = []
    dropped: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if len(candidate) < 4:
            dropped.append(candidate)
            continue
        if Counter(candidate) - allowed:  # 多出来的字符 = 原问句里没有 → 可能有编造
            dropped.append(candidate)
            continue
        if candidate in kept or candidate == origin:
            continue
        kept.append(candidate)
    return kept, dropped


# ---------------------------------------------------------------- 规则一：分别/各自
def _by_distribution(text: str) -> list[str]:
    """「比较A和B…，并分别出三道练习题」→ 三个子问句（对比 + 每个话题各一条）。

    插不进去就**整条放弃**（`_insert` 返回空）——宁可退回单路检索，
    也不拼一条"看起来像子问题"的半成品，那等于用一个错问句占掉一路召回预算。
    """
    clauses = [c.strip() for c in _CLAUSE.split(text) if c.strip()]
    action_idx = next((i for i, c in enumerate(clauses) if _DISTRIBUTE.search(c)), None)
    if action_idx is None:
        return []

    others = [c for i, c in enumerate(clauses) if i != action_idx]
    topics = _topics_of(others)
    if len(topics) < 2:
        return []

    stem = _strip_frame(max(others, key=len))
    if len(stem) < _MIN_SUB_LEN:
        return []

    action = _LEAD_CONNECTIVE.sub("", _DISTRIBUTE.sub("", clauses[action_idx])).strip()
    if not action:
        return []

    built = [stem]
    for topic in topics[: MAX_SUB_QUERIES - 1]:
        placed = _insert(action, topic)
        if not placed:
            return []
        built.append(placed)
    return built


def _topics_of(clauses: list[str]) -> list[str]:
    """从一个子句里取出并列的话题。认不出就返回空——**不猜**。"""
    for clause in clauses:
        diff = _DIFF.search(clause)
        if diff is not None:
            pair = [diff.group("a").strip(), diff.group("b").strip()]
            if all(_looks_like_topic(p) for p in pair):
                return pair
        parts = [p.strip() for p in _TOPIC_SEP.split(clause) if p.strip()]
        parts = [_strip_frame(p) for p in parts]
        if len(parts) >= 2 and all(_looks_like_topic(p) for p in parts):
            return parts
    return []


def _looks_like_topic(part: str) -> bool:
    if not 2 <= len(part) <= 12:
        return False
    return _TOPIC_NOISE.search(part) is None


def _strip_frame(clause: str) -> str:
    """去掉句首的框架动词（比较/对比/说说…）：它们是**动作**，不是话题。"""
    return _LEAD_FRAME.sub("", clause).strip()


def _insert(action: str, topic: str) -> str:
    """把话题插进动作句：「给我出三道练习题」+「一次函数」→「给我出三道一次函数练习题」。"""
    match = _NUM_CLASSIFIER.search(action)
    if match is not None:
        at = match.end()
        return f"{action[:at]}{topic}{action[at:]}"
    for noun in _TAIL_NOUNS:
        pos = action.rfind(noun)
        if pos > 0:  # `> 0`：整句只有名词时插在前面等于没插，交给调用方放弃
            return f"{action[:pos]}{topic}{action[pos:]}"
    return ""


# ---------------------------------------------------------------- 规则二：并列诉求
def _by_connective(text: str) -> list[str]:
    """逗号/连接词切开后，凡是"像一件独立诉求"的分句都各自成路。

    判据是**分句自身**够不够独立（长度 + 诉求词），不是"有没有逗号"：
    「老师，二次函数的顶点式怎么求」也有逗号，但那是称呼，不该被拆成两路。
    """
    clauses = [c.strip() for c in _CLAUSE.split(text) if c.strip()]
    picked = [c for c in clauses if len(c) >= _MIN_SUB_LEN and _CUE.search(c)]
    if len(picked) < 2:
        return []
    return picked[:MAX_SUB_QUERIES]


# ---------------------------------------------------------------- 规则三：无标点的「A…和B…」
def _by_conjunction(text: str) -> list[str]:
    """「2次函数是怎么求解的和1次函数有什么区别」→ 两路，各自有自己的召回。

    为什么单开一条规则而不是把"和"加进 `_CLAUSE`：「和」既是**并列诉求**的连接词，
    也是**对比句式**的连接词（`_TOPIC_SEP` / `_DIFF` 正是靠它切话题的）。把它并进
    `_CLAUSE` 会让「比较一次函数和二次函数的区别」先被切成两个半截，`_topics_of()`
    再也拿不到那一对话题，规格里的三路黄金样本会退化成两路——用一条新规则、排在
    前两条**之后**，原有行为一个字节都不动。

    两条守卫说明"什么时候不切"（都不是判据本身，判据仍是"切完 ≥2 段站得住"）：

    ① 句中有 `_DIFF`（A和B的区别/关系…）或句首是框架动词（比较/对比/说说…）：
       整句是**一个**对比诉求，切开就变成两个半截问题，多付一路召回还答不好；
    ② 每一段都得**自己带诉求词**且够长：否则它只是话题碎片（「一次函数」），
       拿它当检索问句只会拉低召回。
    """
    if _DIFF.search(text) or _LEAD_FRAME.search(text):
        return []
    parts = [p.strip() for p in _CONJUNCTION.split(text) if p.strip()]
    picked = [p for p in parts if len(p) >= _MIN_SUB_LEN and _CUE.search(p)]
    if len(picked) < 2:
        return []
    return picked[:MAX_SUB_QUERIES]
