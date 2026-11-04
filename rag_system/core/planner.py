"""策略规划：把"理解结果"翻译成"要跑哪几路检索"（CLAUDE.md §7.2.7-B 第 5 步）。

**规划层只做调度**：它不生成答案、不调 Milvus、不碰权限过滤（`tenant_id`/ACL/
`kb_version` 只从 `QueryContext` 来）。它回答两件事——**要不要检索**（`decision`）、
**检索该怎么做**（`strategy` + 任务列表）。

规则优先、档位先验兜底。**策略选择可以交给大模型**（`strategy_hint`，2026-11-01）：
理解层在那次合并调用里顺手判一句"这句话适合怎么查"，这里按**可行性**采纳或降级。
规划只有 300 ms 预算，所以模型的结果是**随理解层一起回来的**，不是在这里再发一次调用。

`decision`（阶段 4.1，`base.contracts.PlanDecision`）按下面的顺序判，**先到先得**：

| 顺序 | 条件 | decision | 说明 |
|---|---|---|---|
| 0 | `needs_retrieval=False`（闲聊 / 业务短路） | `no_retrieval` | 一路都不发，省整条检索链 |
| 1 | 含指代词且**无历史**，或调用方报 `unresolved_reference` | `clarify` | 检索拿原句必然召回不到，追问比乱检便宜 |
| 2 | 其余 | `retrieve` | 再看 `strategy` |

**这条分界是刻意的**：`decision` 编码的是**结构事实**——这个意图根本不检索、指代解不开。
它们不该由模型判：一次采样就能把"这句是闲聊，直接答"变成"再去检索一轮"，
而那是 §6.1 里算过的 350 ms。`strategy` 只编码**偏好**（一个诉求还是两个、要不要先写假答案），
这才是模型擅长的，也才是"选错了最多多查一轮"的那一类。

只有 `retrieve` 才有 `strategy`（`base.contracts.Strategy`）。判顺序：

| 顺序 | 条件 | 策略 | 任务数 |
|---|---|---|---|
| 2a | 模型给了可用建议（`strategy_hint` ∈ direct/sub_query/hyde） | 按建议，跑不了的降级 | 见下 |
| 2b | 含并列/对比连接词，或调用方给出 `sub_queries` | SUB_QUERY | ≤3 并行 |
| 2c | 问句短（<10 字）且抽象（"什么是X"）**且**有可用的生成模型 | HYDE | 1 |
| 2d | 其余（含档位先验 `sub_query`） | DIRECT | 1 |

2a 的可行性约束：`sub_query` 要有**真拆出来的**子问句（拆句器没拆出来就退成单路，
不临场编）；`hyde` 要有可用的生成模型（没有就落 `DIRECT` + `hyde_unavailable`）。
降级都留原因码，不假装跑过。`strategy_hint` 为空或取值不在上面三种时，整条 2a 不生效，
直接走 2b——这就是 `FEATURE_PLANNER_LLM=false` 的回滚路径，与加开关之前逐字节一致。

`BACKTRACK` 不在上表里：它**不是一个同级别的首轮策略**，而是首轮检索过不了证据门之后的
修复动作，决策点在 `RagPipeline`（阶段 4.4）。规划层只在"指代解不开"时用
`Strategy.BACKTRACK` 标记这一路（`decision` 仍是 `clarify`），让日志一眼看出是哪种追问。
同理，`backtrack` 与 `faq` 也不在给模型的候选里——它们各自有结构性的前置条件。

**HyDE 的落实差异**：HyDE 需要先让模型写一段"假答案"再拿它去检索。没有可用 LLM 时
（开发机常态）这里**不假装 HyDE**，直接落 `DIRECT` 并在 `reason` 里写 `hyde_unavailable`。
`hyde` 生成的文本只能用来检索，**永远不能作为证据**（§7.2.7-G）。
"""

from __future__ import annotations

import re

from understanding.rewrite import has_reference

from base.contracts import (
    IntentProfile,
    QueryContext,
    QueryRole,
    QueryTask,
    RetrievalPlan,
    Strategy,
)

# 指代词：出现且没有历史上下文时，检索拿原句必然召回不到东西，追问比乱检便宜
# 并列/对比：拆成多路并行召回，比把整句丢给 embedding 更准
_MULTI = re.compile(r"(以及|还有|和.+的|对比|比较|区别|分别|各自|差异)")
_ABSTRACT = re.compile(r"^(什么是|什么叫|为何|为什么说|顾名思义)")

MAX_SUB_QUERIES = 3
HYDE_MAX_CHARS = 10
# 模型可以选的检索方式。**不含 `backtrack` / `faq`**：前者是"指代解不开要追问"的标记与
# 修复动作，后者是出口，两者都不是首轮的检索方式。`_valid_strategy()` 那边也挡了同一批，
# 两处都挡是刻意的——规划层不假设上游一定校验过
_HINT_STRATEGIES = frozenset({"direct", "sub_query", "hyde"})


def looks_referential(query: str) -> bool:
    """兼容旧调用方；指代判断统一委托给 understanding.rewrite。"""
    return has_reference(query)


def looks_multi(query: str) -> bool:
    return bool(_MULTI.search(query or ""))


def build_plan(
    query: str,
    ctx: QueryContext,
    *,
    profile: IntentProfile | None = None,
    subjects: tuple[str, ...] = (),
    sub_queries: tuple[str, ...] = (),
    history_present: bool = False,
    hyde_available: bool = False,
    final_n: int | None = None,
    rerank: bool | None = None,
    query_role: QueryRole = "original",
    needs_retrieval: bool | None = None,
    unresolved_reference: bool = False,
    strategy_hint: str = "",
) -> RetrievalPlan:
    """`query_role` 描述**传进来的 `query` 是什么**（原话 / 改写 / 假设文），
    不是描述选了哪个策略——调用方拿改写句进来时必须传 `rewrite`，
    否则日志里所有任务都长得像"用原话检索"，诊断时无从判断改写到底帮了还是害了。

    `rerank` 默认 `None` = 不表态，由 `RAG_RERANK_ENABLED` 仲裁（`auto` 即 CUDA 探测）。
    默认写死 `True` 会让 `auto` 与 `.env` 全部失效：本机无 CUDA，重排 40 条候选实测
    227 s，一次检索拖到 232 s——正是 §5.6 那条"不要在 CPU 上跑 Reranker"。

    `needs_retrieval=None` 表示"调用方没表态"，此时**跟随档位**（`profile.retrieval`）：
    档位说不用检索就是不检索。给了布尔值就以调用方为准——编排层已经在更早的位置
    按档位短路过一次，这里的默认值只是让"没传"不会退化成"总是检索"。

    `is_complex` **不是**这个函数的入参，这是有意的：复杂 ≠ 要多路。要不要拆由
    `sub_queries` 决定（理解层拆出来了才拆，拆不出来就是一路）。拿 `is_complex=True`
    去开 SUB_QUERY，会让"这段问题很长"变成"必须拆成三路"，而三路里两路可能是同一个
    知识点——多付两次召回，证据门还得靠跨任务覆盖度把它们捡回来。

    `strategy_hint` 是模型给的**建议**，不是指令：这里只认 `direct` / `sub_query` / `hyde`
    三个取值（其余一律当没给，退回规则），且 `sub_query` 要有真拆出来的 `sub_queries`、
    `hyde` 要有可用的生成模型。跑不了的建议降级并留原因码，**不假装跑过**——
    传 `hyde` 就落 `Strategy.HYDE` 的话，HyDE 文本根本没人生成，
    这次检索会拿原句去查却标着自己跑过 HyDE，之后拿它做 A/B 就是拿日志骗自己。
    """
    final_n = final_n or ctx.final_n
    # 学科只从调用方给的 SlotResult / ctx.subject 来；意图**只允许**收窄 doc_type（§7.7 硬规则 4）
    filters = ctx.filters(
        subjects=tuple(subjects),
        doc_types=tuple(profile.doc_types) if profile else (),
    )
    prior = (profile.strategy if profile else "").lower()

    # 0) 只要调用方（或档位）说这次不需要检索，就一路都不发。
    #    这是**业务决策**，不属于任何一种检索策略，所以它出在 `decision` 而不是 `strategy`。
    if needs_retrieval is False or (needs_retrieval is None and profile and not profile.retrieval):
        return RetrievalPlan(
            strategy=Strategy.DIRECT,
            tasks=(),
            final_n=final_n,
            rerank=rerank,
            reason="no_retrieval",
            decision="no_retrieval",
        )

    # 1) 指代解不开：先追问，别硬检（有历史时理解层已经改写过，走到这里说明没改成）。
    #    两路来源合并成同一个 decision：规则自己看出来的（含指代 + 无历史），
    #    与理解层明确报上来的（`unresolved_reference`）——后者更可信，因为它知道
    #    "改写试过了还是指代不明"，而规则只知道"句子里有'它'"。
    if looks_referential(query) and not history_present:
        # `reason` 记的是"计划为什么长这样"（规则自己看出来的：有指代 + 没历史），
        # `clarification_reason` 记的是"该用哪句追问话术"的机器码。两者不是一回事：
        # 追问话术只有 `referential_without_context` / `unresolved_reference` 两种，
        # 而"没历史"只是其中一种成因。合成一个字段会让追问层分不清是谁报的。
        return _clarify_plan(
            query, filters, ctx, final_n, rerank, "referential_without_history",
            note="referential_without_history",
            clarification_reason="referential_without_context",
        )
    if unresolved_reference and looks_referential(query):
        # 上游的 unresolved 标记只有在当前问句仍含真正的指代词时才有阻断权。
        # 这样可以防止旧词表/陈旧标记把「那些物质有哪些」一类独立问句拦在检索前。
        # 真正的「它是什么」仍然追问，不能把证据门当成指代消解器。
        return _clarify_plan(
            query, filters, ctx, final_n, rerank, "unresolved_reference",
            note="unresolved_reference",
        )

    # 2) 多意图：调用方拆好的子问句优先；否则让调用方按连接词拆（这里只标策略）
    subs = tuple(dict.fromkeys(q.strip() for q in sub_queries if q.strip()))[:MAX_SUB_QUERIES]

    # 2a) 模型选的检索方式。**排在规则之前**：既然把这一层交给模型，就不能在它给了答案之后
    #     再让规则改主意——那样模型的输出只在"恰好与规则一致"时才起作用，等于没接。
    #     但**排在 decision 之后**：要不要检索不是它能决定的（见模块文档）。
    hint = (strategy_hint or "").strip().lower()
    if hint in _HINT_STRATEGIES:
        return _plan_from_hint(
            hint, query, subs, filters, ctx, final_n, rerank, query_role, hyde_available
        )

    # 2b) 规则兜底：没开开关、模型没给、或给的值认不出来，走的都是这里
    if len(subs) >= 2 or looks_multi(query):
        # 真的拆出了子问句才叫 `sub`；只有连接词、仍拿整句去检索，它还是原来那句话
        tasks = tuple(
            QueryTask(f"t{i}", q, Strategy.SUB_QUERY, filters=filters, top_k=ctx.top_k,
                      query_role="sub" if len(subs) >= 2 else query_role)
            for i, q in enumerate(subs if len(subs) >= 2 else (query,), start=1)
        )
        return _plan(Strategy.SUB_QUERY, tasks, final_n, rerank, "multi_intent")

    # 2c) 短且抽象 → HyDE（没有生成模型就诚实降级为 DIRECT）
    if len(query) < HYDE_MAX_CHARS and _ABSTRACT.search(query):
        if hyde_available:
            # 真跑 HyDE：这一路送进检索的是**假设性答案**，标 `hyde` 以便事后核验它没混进证据
            return _plan(
                Strategy.HYDE,
                [QueryTask("t1", query, Strategy.HYDE, filters=filters, top_k=ctx.top_k,
                           query_role="hyde")],
                final_n, rerank, "short_abstract",
            )
        return _plan(
            Strategy.DIRECT,
            [QueryTask("t1", query, Strategy.DIRECT, filters=filters, top_k=ctx.top_k,
                       note="hyde_unavailable", query_role=query_role)],
            final_n, rerank, "hyde_unavailable",
        )

    # 2d) 档位先验（只认 DIRECT / SUB_QUERY；HYDE/BACKTRACK 必须由上面的硬规则触发）
    if prior == Strategy.SUB_QUERY.value:
        return _plan(
            Strategy.SUB_QUERY,
            [QueryTask("t1", query, Strategy.SUB_QUERY, filters=filters, top_k=ctx.top_k,
                       query_role=query_role)],
            final_n, rerank, "profile_prior",
        )

    return _plan(
        Strategy.DIRECT,
        [QueryTask("t1", query, Strategy.DIRECT, filters=filters, top_k=ctx.top_k,
                   query_role=query_role)],
        final_n, rerank, "default",
    )


def _plan_from_hint(
    hint: str,
    query: str,
    subs: tuple[str, ...],
    filters,
    ctx: QueryContext,
    final_n: int,
    rerank: bool | None,
    query_role: QueryRole,
    hyde_available: bool,
) -> RetrievalPlan:
    """按模型给的 `strategy_hint` 出计划，**只做可行性收敛，不改它的选择**。

    三种建议各自的落点：

    | 建议 | 可行 | 不可行 |
    |---|---|---|
    | `direct` | 单路 DIRECT | — |
    | `sub_query` | 有 ≥2 条真拆出来的子问句 → 一任务一路 | 拆句器没拆出来 → 单路 SUB_QUERY，`note="sub_query_unsplit"` |
    | `hyde` | 有生成模型 → HYDE，任务标 `hyde` | 没有 → DIRECT，`note="hyde_unavailable"` |

    **不临场编子问句**：模型说"这句该拆成两路"但它自己没给出子问题，这里就只有一句原话可查。
    硬编一个"按逗号切"出来只会把同一句话拆成两半各查一遍，召回不增、预算翻倍；
    单路 SUB_QUERY 与规则在 `looks_multi()` 命中时是同一种形态，日志里也看得出"想拆没拆成"。
    """
    if hint == Strategy.HYDE.value:
        if hyde_available:
            return _plan(
                Strategy.HYDE,
                [QueryTask("t1", query, Strategy.HYDE, filters=filters, top_k=ctx.top_k,
                           query_role="hyde")],
                final_n, rerank, "llm_hyde",
            )
        # 想跑 HyDE 但这台机器跑不了：降级成 DIRECT 并留 `hyde_unavailable`。
        # 理由与规则那条同样的：落 `Strategy.HYDE` 而没有人生成假设文，等于日志声称跑过它
        return _plan(
            Strategy.DIRECT,
            [QueryTask("t1", query, Strategy.DIRECT, filters=filters, top_k=ctx.top_k,
                       note="hyde_unavailable", query_role=query_role)],
            final_n, rerank, "hyde_unavailable",
        )

    if hint == Strategy.SUB_QUERY.value:
        if len(subs) >= 2:
            tasks = tuple(
                QueryTask(f"t{i}", q, Strategy.SUB_QUERY, filters=filters, top_k=ctx.top_k,
                          query_role="sub")
                for i, q in enumerate(subs, start=1)
            )
            return _plan(Strategy.SUB_QUERY, list(tasks), final_n, rerank, "llm_sub_query")
        return _plan(
            Strategy.SUB_QUERY,
            [QueryTask("t1", query, Strategy.SUB_QUERY, filters=filters, top_k=ctx.top_k,
                       note="sub_query_unsplit", query_role=query_role)],
            final_n, rerank, "llm_sub_query",
        )

    return _plan(
        Strategy.DIRECT,
        [QueryTask("t1", query, Strategy.DIRECT, filters=filters, top_k=ctx.top_k,
                   query_role=query_role)],
        final_n, rerank, "llm_direct",
    )


def _clarify_plan(
    query: str,
    filters,
    ctx: QueryContext,
    final_n: int,
    rerank: bool | None,
    reason: str,
    *,
    note: str,
    clarification_reason: str | None = None,
) -> RetrievalPlan:
    """"先追问"的计划：`strategy` 仍标 `BACKTRACK`（这是历史行为，日志与旧单测都认它），
    但 `decision="clarify"` 才是**执行**依据——pipeline 见到它一路都不发。

    两处都写是有意的：`strategy` 说明"如果真要检，那属于回溯场景"，`decision` 说明
    "这次先别检"。只写前者会让"不检索"藏在策略名里，读代码的人得知道 BACKTRACK 的
    特殊语义才看得懂；只写后者会丢掉"为什么是回溯"这条线索。
    """
    return RetrievalPlan(
        strategy=Strategy.BACKTRACK,
        tasks=(
            QueryTask("t1", query, Strategy.BACKTRACK, filters=filters, top_k=ctx.top_k,
                      note=note),
        ),
        final_n=final_n,
        rerank=rerank,
        reason=reason,
        decision="clarify",
        clarification_reason=clarification_reason or reason,
    )


def _plan(
    strategy: Strategy,
    tasks: list[QueryTask],
    final_n: int,
    rerank: bool | None,
    reason: str,
) -> RetrievalPlan:
    return RetrievalPlan(
        strategy=strategy, tasks=tuple(tasks), final_n=final_n, rerank=rerank, reason=reason
    )
