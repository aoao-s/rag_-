"""会话改写（CLAUDE.md §3.2 第 4 条 / §7.7）：把指代补全成可检索的问句。

**规则优先，不调模型**：触发条件很窄（含指代词 / 太短 / 无实体），命中后要做的动作也很窄
（把上一轮的用户问句接在前面）。这种"拼接"用小模型做既贵又不稳定——模型可能顺手
把问题改写成一个不同的意思，而检索只关心"关键词有没有被带进来"。

三条边界：

1. **只给检索用，不改用户问句**：FAQ 的命中建立在归一化串的精确匹配上（§7.1.2），
   把上下文拼进去会让原本能命中的标准问句变成一条谁都不认识的串。
   因此 `ChatPipeline` 里 FAQ 用 `state.query`，检索用 `state.rewritten`。
2. **不跨主题乱接**：上一轮是"谢谢"这类寒暄、或与当前问句没有任何共享词时，
   拼接会把噪音带进检索。用一条保守规则挡住：上一轮必须是**问句**且 ≥4 字。
3. **改写失败等于没改**：任何异常返回原句 + `method="none"`，绝不因为没有历史就报错。
"""

from __future__ import annotations

import re
from typing import Literal, Protocol

# 指代与上下文依赖标记：命中任一就认为这句话离开上文读不懂
REFERENTIAL = (
    "它", "他", "她", "这个", "那个", "这些", "那些", "此", "该",
    "上面", "刚才", "前面", "上面说", "你说的", "刚才说", "上一个", "前一个",
    "还有呢", "继续", "再详细", "再说说",
)

# 判定「指代有没有解开」用的是**更窄**的一组，与 `REFERENTIAL` 刻意分开：
# 后者是"要不要拼接上下文"的宽判据（宁多勿漏），上面那几个泛用字（此 / 该 / 继续）
# 在正常问句里出现率太高（「因此」「应该」「继续教育」），拿它们判"未解指代"
# 会把大量正常问句标成降级。`(?<!其)` 排掉「其他 / 其余」——那是泛指，不是指代。
_PRONOUN = re.compile(
    r"(?<!其)[它他她]|这个|那个|这些|那些|上述|上面|刚才|前面|你说的|刚才说|上一个|前一个"
)

# 上一轮要接进来的最短长度：太短的（"好的""嗯"）接进来只有噪音
_MIN_CONTEXT = 4
# 改写后的上限：检索 query 太长会让 BM25/向量都失去焦点
_MAX_REWRITTEN = 120

RewriteMethod = Literal["none", "rule", "llm"]


class _TurnLike(Protocol):
    role: str
    content: str


def has_reference(query: str) -> bool:
    """问句里有没有**真正的**代词 / 上文指称。

    用途只有一个：判断"这次改写到底有没有把指代解开"（任务 2.4 / 2.5 的
    `coreference_unresolved` 与 `is_follow_up`）。所以它必须比 `needs_rewrite()`
    严格——那个是**动作**判据（"要不要试着拼接"，误判的代价只是多接一句上下文），
    这个是**结论**判据（"没解开就记一笔降级"，误判会把正常问句标成故障）。
    """
    return bool(_PRONOUN.search(query or ""))


def needs_rewrite(query: str) -> bool:
    """§3.2：含指代词、长度 < 8 字、无实体时触发。这里实现前两条。"""
    text = (query or "").strip()
    if not text:
        return False
    if any(marker in text for marker in REFERENTIAL):
        return True
    return len(text) < 8


def rewrite(query: str, history: list[_TurnLike]) -> tuple[str, RewriteMethod]:
    """返回 `(用于检索的问句, 方法)`。无历史或不触发时原样返回。"""
    original = (query or "").strip()
    if not original or not needs_rewrite(original):
        return original, "none"

    previous = _previous_question(history)
    if previous is None:
        return original, "none"

    # 已经有完整问句形态，且不含指代 → 拼接只会稀释关键词
    if not any(marker in original for marker in REFERENTIAL) and len(original) >= 12:
        return original, "none"

    merged = f"{previous}，{original}"
    if len(merged) > _MAX_REWRITTEN:
        # 从上一轮尾部保留：越靠近当前的语境越相关
        keep = _MAX_REWRITTEN - len(original) - 1
        merged = f"{previous[-keep:]}，{original}"
    return merged, "rule"


def _previous_question(history: list[_TurnLike]) -> str | None:
    """最近一条**用户问句**的首句。跳过助手回答：助手答的是知识库内容，
    把它接进检索 query 等于让上一轮的回答污染这一轮的召回。"""
    for turn in reversed(history):
        if getattr(turn, "role", "") != "user":
            continue
        text = " ".join((getattr(turn, "content", "") or "").split())
        if len(text) < _MIN_CONTEXT:
            continue
        # 只取首句：上一轮常常是"什么是X？顺便问下Y"，整段接进来会跑题
        for sep in ("。", "？", "?", "！", "!", "\n"):
            idx = text.find(sep)
            if idx > 0:
                text = text[:idx]
                break
        return text[:60]
    return None
