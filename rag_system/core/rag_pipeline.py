"""检索流水线（CLAUDE.md §7.2.7-B 第 5–7 步）：计划 → 多路召回 → 融合 → 装配 → 证据门 → 至多一次修复。

**唯一持有"重试决策"的地方**。证据门只说"为什么不够"，pipeline 才决定"要不要再试一次、
怎么试"。三条硬约束：

1. **至多一次**：`max_repairs=1`。第二次失败就带着"已修复仍未过门"的结论往上走，
   由编排层决定追问 / 兜底 / 转人工。反复重试会把 8 s 总预算吃光，还会把
   "检索系统坏了"伪装成"知识库里没有"——这两种情形对运营的意义完全不同。
2. **修完必须重新过门**：不做"改了过滤就当过了"，否则等于绕过证据门。
3. **安全过滤不可放宽**：修复只会调 `filters.relaxed()`（只清 subjects / doc_types），
   `tenant_id / acl_tags / kb_version` 从头到尾不变。

错误语义：检索不可用抛 `RetrievalUnavailable`（对外 `retrieval_unavailable`），
与"检索成功但证据不足"（`no_evidence`）严格分开。

**`plan.decision` 先于 `plan.strategy` 判**（阶段 4.1）：`no_retrieval` / `clarify` 是
"根本别检索"，`strategy` 说的是"检索该怎么做"。两件事分两层，下游就不会出现
"该追问的请求跑了一遍全库检索"这种事——那正是把"不检索"塞进 `Strategy` 必然的错法。

**HyDE 只借道，不入库**（阶段 4.3）：假设性答案只用来**生成检索向量**，
`Evidence` 集合里只允许出现真实库块。`_execute` 里有一道显式过滤（见 `_drop_hyde_echo`），
不靠"我们没往里加"这种口头保证。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from typing import Awaitable, Callable

from base.contracts import (
    EvidenceDecision,
    IntentProfile,
    QueryContext,
    QueryRole,
    QueryTask,
    RetrievalFilters,
    RetrievalPlan,
    RetrievalResult,
    Strategy,
)
from base.errors import RetrievalUnavailable
from base.logger import get_logger
from rag_system.core import evidence_builder, planner as planner_mod
from rag_system.core.evidence_gate import evaluate as default_gate

log = get_logger(__name__)

# 修复动作 → 是否可修（不可修的直接把结论交给编排层）
_REPAIRABLE = {
    "retry_direct",
    "retry_original_query",
    "rewrite_query",
    "expand_neighborhood",
    "relax_business_filter",
}


@dataclass(slots=True)
class PipelineOutcome:
    """一次检索（含至多一次修复）的完整结论。"""

    result: RetrievalResult
    decision: EvidenceDecision
    plan: RetrievalPlan | None = None
    # 修复**之前**那份计划（首轮真正跑过的那份）。`plan` 会被修复覆盖成 `r1` 单路，
    # 于是"这句复合问句到底拆成了几路"在响应体里就看不见了——而那正是用户要核验的东西：
    # 他已经从 `strategy` 里读到过 `sub_query`，下一眼却只看到一条 `direct` 任务，
    # 两端对不上。`None` 只在没走到计划阶段时出现。
    initial_plan: RetrievalPlan | None = None
    stages: dict[str, int] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    attempts: int = 1

    @property
    def passed(self) -> bool:
        return self.decision.passed

    @property
    def plan_after_repair(self) -> RetrievalPlan | None:
        """修复确实换了计划时返回那份新计划；没换过（或没修）返回 `None`。

        只"存在一份不同的计划"才算修复过——`reason` 不同但任务完全相同的两份，
        并列展示会让报告里多出一条什么也没发生的"修复"。
        """
        if self.plan is None or self.plan == self.initial_plan:
            return None
        return self.plan


class RagPipeline:
    def __init__(
        self,
        settings,
        retriever,
        *,
        gate: Callable[..., EvidenceDecision] | None = None,
        max_repairs: int = 1,
        hyde: Callable[[str], Awaitable[str]] | None = None,
        hyde_budget_ms: int = 1500,
    ) -> None:
        """`hyde` 是"问句 → 假设性答案"的生成器（阶段 4.3）。

        注入而不是在这里 new 一个网关：pipeline 不 import 生成层，也不该知道模型是谁。
        **`None` 即没有生成能力**，此时规划层不会选 HYDE（`hyde_available=False`）；
        万一有人手工塞了一个 HYDE 计划进来，`_apply_hyde` 会降级成 DIRECT 并记
        `hyde_failed`，而不是让请求带着一条伪造的证据往下走。

        `hyde_budget_ms` 是**生成假设文**这一段的上限。HyDE 的价值是改善召回，
        它自己不该成为新的超时来源：超时就退回 DIRECT 用原句检索，那一问本来也答得出来。
        """
        self.settings = settings
        self.retriever = retriever
        self._gate = gate or default_gate
        self.max_repairs = max(0, max_repairs)
        self._hyde = hyde
        self.hyde_budget_ms = max(1, hyde_budget_ms)

    async def run(
        self,
        query: str,
        ctx: QueryContext,
        *,
        plan: RetrievalPlan | None = None,
        profile: IntentProfile | None = None,
        subjects: tuple[str, ...] = (),
        sub_queries: tuple[str, ...] = (),
        history_present: bool = False,
        hyde_available: bool = False,
        needs_retrieval: bool | None = None,
        unresolved_reference: bool = False,
        strategy_hint: str = "",
        budget_ms: int | None = None,
    ) -> PipelineOutcome:
        started = time.perf_counter()
        budget = budget_ms or ctx.budget_ms
        # 传进来的 query 是不是改写句：与 `ctx.original_query` 比即可，不需要调用方再多传一个参数
        # （多一个参数就多一处能忘传，而忘传的后果是日志里所有检索都长得像"用的原话"）
        query_role: QueryRole = (
            "rewrite" if ctx.original_query and query != ctx.original_query else "original"
        )
        stages: dict[str, int] = {}
        degraded: list[str] = []
        # 计划阶段也要计时（§11.4）：调用方传 plan 进来时这里接近 0 ms，
        # 那也是实话——本层确实没花时间，不是漏填
        t_plan = time.perf_counter()
        plan = plan or planner_mod.build_plan(
            query,
            ctx,
            profile=profile,
            subjects=subjects,
            sub_queries=sub_queries,
            history_present=history_present,
            hyde_available=hyde_available,
            final_n=ctx.final_n,
            query_role=query_role,
            # 阶段 4.1 的两个业务决定：要不要检索、要不要先追问。
            # 它们的**信息源在编排层**（档位说"这档不检索"，理解层说"指代没解开"），
            # 计划怎么落由 planner 定；本层只负责把信息原样递过去，不自己判。
            needs_retrieval=needs_retrieval,
            unresolved_reference=unresolved_reference,
            # 模型选的检索方式（2026-11-01）：**只是建议**，可行性由 planner 收敛。
            # 空串 = 没开那个开关、模型没给、或给的值认不出来，三条都退回规则规划
            strategy_hint=strategy_hint,
        )
        stages["plan"] = _ms(t_plan)

        # 0) 业务上就不需要检索（闲聊 / 超出教育范围 / 简单计算）：一路都不发，`attempts=0`。
        #    这里不发请求是**结构保证**——规划层给了 `tasks=()`，即便下游有人误调也不会真去查库。
        if plan.decision == "no_retrieval":
            decision = EvidenceDecision(
                status="insufficient",
                reasons=("retrieval_not_needed",),
                retry_action="none",
            )
            stages["total"] = _ms(started)
            log.info("rag_pipeline", strategy=plan.strategy.value, tasks=0,
                     status=decision.status, attempts=0, repairs=0, cost_ms=stages["total"])
            return PipelineOutcome(
                result=RetrievalResult(query=query, strategy=plan.strategy),
                decision=decision,
                plan=plan,
                initial_plan=plan,
                stages=stages,
                degraded=["retrieval_not_needed"],
                attempts=0,
            )

        # 1) 指代解不开：追问比乱检便宜，省下整条检索链（阶段 4.1）。
        #    两种来源共用一个 decision，但 reason 与 degraded 分开记——
        #    "句子里有'它'但没历史"和"改写试过了还是指代不明"是两条不同的排障线索。
        if plan.decision == "clarify" or plan.strategy == Strategy.BACKTRACK:
            reason = plan.clarification_reason or plan.reason or "referential_without_context"
            marker = (
                "backtrack_without_history"
                if reason == "referential_without_context"
                else "clarify_unresolved_reference"
            )
            decision = EvidenceDecision(
                status="ambiguous",
                reasons=(reason,),
                retry_action="ask_disambiguate",
            )
            stages["total"] = _ms(started)
            log.info("rag_pipeline", strategy=plan.strategy.value, tasks=0,
                     status=decision.status, attempts=0, repairs=0, cost_ms=stages["total"])
            return PipelineOutcome(
                result=RetrievalResult(query=query, strategy=plan.strategy),
                decision=decision,
                plan=plan,
                initial_plan=plan,
                stages=stages,
                degraded=[marker],
                attempts=0,
            )

        # 2) HyDE：先让模型写一段假设性答案，拿它去检索（阶段 4.3）。
        #    失败 / 没配生成器 → 退回 DIRECT 用原句，记 `hyde_failed`，不让它变成一个错误出口。
        hyde_text = ""
        if plan.strategy == Strategy.HYDE:
            plan, hyde_text = await self._apply_hyde(plan, stages, degraded)

        # 给的是**剩余预算**（不是总预算）：修复前后的两次召回共用同一个 8 s，
        # 各给一份总额会让"至多修复一次"变成"至多跑两倍预算"
        outcome = await self._execute(
            query, ctx, plan, stages, degraded,
            budget_ms=max(1, budget - _ms(started)), hyde_text=hyde_text,
        )
        outcome.plan = plan
        # 首轮计划在修复之前就留底：`outcome.plan` 下面会被 `plan2` 覆盖
        outcome.initial_plan = plan
        decision = self._evaluate_into(
            outcome.result, plan, stages, original_query=ctx.original_query
        )
        outcome.decision = decision

        repairs = 0
        while not decision.passed and repairs < self.max_repairs:
            repair = self._repair_plan(decision.retry_action, plan, query, ctx)
            if repair is None:
                break
            if _ms(started) >= budget:
                degraded.append("budget_exceeded")
                log.warning("rag_repair_skipped", reason="budget", action=decision.retry_action)
                break
            repairs += 1
            plan2, note = repair
            degraded.append(f"retry:{note}")
            log.info("rag_repair", action=decision.retry_action, note=note, attempt=repairs)
            retry_started = time.perf_counter()
            # 用**修复后计划自己的问句**重检：`retry_original_query` 换掉了问句，
            # 若这里仍传外层的 `query`，`RetrievalResult.query` 会与实际检索文本不一致，
            # 证据门的 qcov 也会拿错的问句去比
            retry_query = plan2.tasks[0].query if plan2.tasks else query
            try:
                outcome = await self._execute(
                    retry_query, ctx, plan2, stages, degraded,
                    budget_ms=max(1, budget - _ms(started)), hyde_text=hyde_text,
                )
            except RetrievalUnavailable as exc:
                # 修复那一次没跑完（多半是剩余预算不够）：保留**第一次**的结果与结论，
                # 不让"多试一次失败"把本来能答的请求降成 `retrieval_unavailable`
                log.warning(
                    "rag_repair_skipped",
                    reason="retrieval_unavailable",
                    action=decision.retry_action,
                    error=str(exc),
                )
                degraded.append("retry_failed")
                break
            outcome.plan = plan2
            # `_execute` 返回的是一份**新的** PipelineOutcome，`initial_plan` 在上面那次
            # 赋值（修复之前）留的那份随旧对象一起丢了。不补回来，暴露给响应体的
            # `plan` 就会退化成 `plan2`——首轮那份（比如拆成三路的 sub_query）看不见，
            # 而 `plan_after_repair` 又恰好等于它，等于同一份计划显示两次、
            # "修复前 / 修复后"这个对照彻底失效。
            outcome.initial_plan = plan
            fresh = self._evaluate_into(
                outcome.result, plan2, stages, original_query=ctx.original_query
            )
            # 成功就保留 `none`（说明修复起效了）；仍不过门才保留"试过什么"，供上层与报告读
            decision = fresh if fresh.passed else fresh.with_retry(
                decision.retry_action, count=repairs
            )
            outcome.decision = decision
            stages["retry"] = _ms(retry_started)

        if not decision.passed and repairs and repairs >= self.max_repairs:
            # 修过了还是不够：把动作收敛成 fallback，避免调用方以为"还能再试"。
            # 注意 `repairs and` 这个条件：一次都没试（max_repairs=0，或该动作本来就不可修）
            # 不能报成"重试耗尽"——那会让报告里的重试收益与真相不符
            decision = decision.with_retry("fallback", count=repairs)
            outcome.decision = decision
            degraded.append("retry_exhausted")

        outcome.stages = stages
        outcome.degraded = list(dict.fromkeys([*outcome.degraded, *degraded]))
        stages["total"] = _ms(started)
        log.info(
            "rag_pipeline",
            strategy=plan.strategy.value,
            tasks=len(plan.tasks),
            status=decision.status,
            attempts=outcome.attempts,
            repairs=repairs,
            cost_ms=stages["total"],
        )
        return outcome

    # ---- 执行 ----
    async def _apply_hyde(
        self, plan: RetrievalPlan, stages: dict[str, int], degraded: list[str]
    ) -> tuple[RetrievalPlan, str]:
        """生成假设性答案并把检索问句换掉；返回 `(新计划, 假设文)`，失败时 `(DIRECT 计划, "")`。

        换的是**任务的 query**，不是 `RetrievalResult.query`：后者必须一直是用户原话，
        因为证据门的覆盖率（`qcov`）拿它做分母，而 τ/qcov 是在原问句上标定的。
        拿假设文当分母会让"问题词覆盖率"变成一个自己跟自己比的值，门必然放行。

        假设文本身**只以 query 的形式**流出去：它不做任务、不进 `evidence_builder`、
        也不在任何地方被 append 到结果里。`_execute` 里另有一道精确匹配过滤
        （`_drop_hyde_echo`）兜住"知识库里恰好有一块和假设文一模一样"这种极端情况——
        那是唯一能让它变成证据的路径，而它一旦变成证据，引用卡片就会指向一段模型编的
        文字，且带着真实文件名与章节，看起来比编造本身更可信。
        """
        started = time.perf_counter()
        task = plan.tasks[0] if plan.tasks else None
        try:
            if self._hyde is None or task is None:
                raise RuntimeError("hyde generator not configured")
            text = await asyncio.wait_for(
                self._hyde(task.query), self.hyde_budget_ms / 1000
            )
            text = (text or "").strip()
            if not text:
                raise RuntimeError("empty hyde text")
        except Exception as exc:  # noqa: BLE001 —— 生成失败一律降级，不是错误出口
            degraded.append("hyde_failed")
            log.warning("rag_hyde_failed", error=type(exc).__name__)
            if task is None:
                return plan, ""
            # 退回 DIRECT 用**原句**：HyDE 的收益是召回，原句本来也是一条合法检索路径
            return (
                _single(plan, task.filters, task.query, task.top_k, Strategy.DIRECT,
                        reason="hyde_failed"),
                "",
            )
        finally:
            stages["hyde"] = _ms(started)

        log.info("rag_hyde", query_len=len(task.query), hyde_len=len(text))
        return (
            RetrievalPlan(
                strategy=Strategy.HYDE,
                tasks=(
                    QueryTask(
                        task.task_id, text, Strategy.HYDE,
                        weight=task.weight, top_k=task.top_k, filters=task.filters,
                        note="hyde", query_role="hyde",
                    ),
                ),
                final_n=plan.final_n,
                rerank=plan.rerank,
                reason=plan.reason,
            ),
            text,
        )

    def _spawn(
        self, plan: RetrievalPlan, use_rerank: bool | None
    ) -> asyncio.Future[list[RetrievalResult | BaseException]]:
        """按计划并行发出所有检索任务；`use_rerank` 原样下传（`None` = 跟随配置）。"""
        return asyncio.gather(
            *(
                self.retriever.search(
                    task.query,
                    top_k=task.top_k,
                    final_n=max(plan.final_n, 1) * (1 if len(plan.tasks) == 1 else 2),
                    subject=(task.filters.subjects[0] if len(task.filters.subjects) == 1 else None),
                    doc_types=list(task.filters.doc_types) or None,
                    rerank=use_rerank,
                    filters=task.filters,
                    task_id=task.task_id,
                    strategy=plan.strategy,
                )
                for task in plan.tasks
            ),
            return_exceptions=True,
        )

    async def _recall(
        self, plan: RetrievalPlan, budget_ms: int, degraded: list[str]
    ) -> list[RetrievalResult | BaseException]:
        """一次多路召回，**带硬超时**。

        §6.1 给检索 200 ms，但原实现里 `budget` 只在"要不要再修复一次"时被读一次，
        单次检索本身没有任何上限：本机无 CUDA 时重排 40 条候选要 227 s，一次
        `/v1/rag/search` 挂了 232 s 才返回，`budget_exceeded` 是**事后**才补上的标记。
        超时先砍最长的一环（重排）重检一次——`rerank=False` 时同一批候选只要
        几十毫秒，比"整条链超时 → 一条证据都没有"有用得多；再超时才判检索不可用。
        """
        timeout = max(0.05, budget_ms / 1000)
        try:
            return await asyncio.wait_for(self._spawn(plan, plan.rerank), timeout)
        except TimeoutError:
            if plan.rerank is False:
                raise RetrievalUnavailable(
                    f"检索超过预算 {budget_ms}ms（重排已关闭，无可再降）", node="rag_pipeline.recall"
                ) from None
            degraded.append("retrieval_timeout")
            log.warning(
                "rag_recall_timeout", budget_ms=budget_ms, rerank=plan.rerank, retry="rerank_off"
            )
            try:
                # 用同一份预算的一半重检：剩下一半留给"装配 + 过门 + 回答"
                return await asyncio.wait_for(self._spawn(plan, False), timeout / 2)
            except TimeoutError:
                raise RetrievalUnavailable(
                    f"检索超过预算 {budget_ms}ms（关闭重排后仍超时）", node="rag_pipeline.recall"
                ) from None

    async def _execute(
        self,
        query: str,
        ctx: QueryContext,
        plan: RetrievalPlan,
        stages: dict[str, int],
        degraded: list[str],
        *,
        budget_ms: int | None = None,
        hyde_text: str = "",
    ) -> "PipelineOutcome":
        stage = time.perf_counter()
        results = await self._recall(plan, budget_ms or ctx.budget_ms, degraded)

        ok: list[RetrievalResult] = []
        errors: list[BaseException] = []
        # 按 `plan.tasks` 的位置对齐取"这一路召回到没有"：`gather(return_exceptions=True)`
        # 保持位置不变，所以 `zip` 是可靠的对应关系。**必须在融合与 `final_n` 裁剪之前判**——
        # 融合时同一个父块只留一个 `task_id`，裁剪又会砍到 `final_n`，两路都命中的任务
        # 会因此看起来"一条证据都没有"，进而触发一次多余的 `retry_direct`，
        # 把 SUB_QUERY 计划换成单路 DIRECT（响应体里的 `strategy` 与任务数随之丢失）
        hits: list[str] = []
        for task, res in zip(plan.tasks, results):
            if isinstance(res, BaseException):
                errors.append(res)
                continue
            ok.append(res)
            if res.evidences:
                hits.append(task.task_id)
        if not ok and errors:
            # 全部失败才算"检索不可用"；部分失败按部分结果继续（降级而不是整体失败）
            raise RetrievalUnavailable(
                f"所有检索任务失败：{errors[0]}", node="rag_pipeline.recall"
            )
        if errors:
            degraded.append("partial_recall_failed")

        evidences = evidence_builder.build(ok, final_n=plan.final_n)
        if hyde_text:
            kept, dropped = _drop_hyde_echo(evidences, hyde_text)
            if dropped:
                degraded.append("hyde_text_dropped")
                log.warning("rag_hyde_evidence_dropped", dropped=dropped)
            evidences = kept
        merged = RetrievalResult(
            query=query,
            evidences=evidences,
            raw_hits=sum(r.raw_hits for r in ok),
            cost_ms=_ms(stage),
            reranked=any(r.reranked for r in ok),
            degraded=list(dict.fromkeys(d for r in ok for d in r.degraded)),
            strategy=plan.strategy,
            tasks=tuple(t.task_id for t in plan.tasks),
            task_hits=tuple(hits),
        )
        stages["recall"] = merged.cost_ms
        # 分阶段耗时从各路结果里聚合，**并行任务取最大值而不是相加**——
        # 相加会把并行跑的三路算成串行三倍，报告里的"检索耗时"立刻失真。
        # `fusion` 由 Milvus 内部完成（本层无独立阶段），retriever 会显式写 0，
        # 这里是如实透传，不是漏填。
        for res in ok:
            for key, value in res.stages.items():
                if key == "total":
                    continue
                stages[key] = max(stages.get(key, 0), value)
        return PipelineOutcome(result=merged, decision=EvidenceDecision(), attempts=1)

    def _evaluate(
        self, result: RetrievalResult, plan: RetrievalPlan, *, original_query: str = ""
    ) -> EvidenceDecision:
        s = self.settings
        filters: RetrievalFilters | None = plan.tasks[0].filters if plan.tasks else None
        conflicts = evidence_builder.detect_conflicts(
            result.evidences, threshold=s.rag_evidence_threshold
        )
        return self._gate(
            # 覆盖率的分母必须是**用户原话**：τ=0.74 / qcov=0.20 是在原问句上标定的。
            # 接上查询改写后 `result.query` 可能是改写句，分母变小 → 覆盖度虚高 → 门变松。
            # 同时也修掉一处既有不一致：`retry_original_query` 那一次本来就换回了原问句，
            # 于是同一请求两次过门用的是两套分母。
            _with_query(result, original_query),
            threshold=s.rag_evidence_threshold,
            margin=s.rag_evidence_margin,
            min_docs=s.rag_evidence_min_docs,
            min_qcov=s.rag_evidence_qcov,
            filters=filters,
            conflicts=conflicts,
        )

    def _evaluate_into(
        self,
        result: RetrievalResult,
        plan: RetrievalPlan,
        stages: dict[str, int],
        *,
        original_query: str = "",
    ) -> EvidenceDecision:
        """过门并累加 `evidence_gate` 阶段耗时。

        修复后还要再过一次门，两次都是**本次请求真的花了**的时间，所以取累加而不是覆盖；
        与并行召回取 max 是两回事（那边是同一件事被并行做了三遍，这边是先后做了两遍）。
        """
        started = time.perf_counter()
        decision = self._evaluate(result, plan, original_query=original_query)
        stages["evidence_gate"] = stages.get("evidence_gate", 0) + _ms(started)
        return decision

    # ---- 修复 ----
    def _repair_plan(
        self, action: str, plan: RetrievalPlan, original_query: str, ctx: QueryContext
    ) -> tuple[RetrievalPlan, str] | None:
        if action not in _REPAIRABLE or not plan.tasks:
            return None
        base = plan.tasks[0]
        if action == "retry_direct":
            return (
                _single(plan, base.filters, original_query, base.top_k, Strategy.DIRECT),
                "retry_direct",
            )
        if action in ("retry_original_query", "rewrite_query"):
            # `below_threshold` 的修复语义是"**对原 query 再召回**"（实施手册 §5.3）。
            # `run()` 收到的 query 可能已经是会话改写后的，直接拿它重检等于把同一句话
            # 又检一遍（分数不变、白花一次预算），还会让报告里"重试收益"永远是 0。
            original = ctx.original_query or original_query
            # 没有可用的改写模型时，"改写"退化为"用原问句再检一次"——如实标注，
            # 不假装做了一次语义改写（那会让报告里的"改写收益"变成假数）。
            note = "rewrite_to_original" if action == "rewrite_query" else "retry_original_query"
            return (_single(plan, base.filters, original, base.top_k, Strategy.DIRECT), note)
        if action == "expand_neighborhood":
            return (
                _single(plan, base.filters, base.query, max(base.top_k, 1) * 2, plan.strategy),
                "expand_neighborhood",
            )
        if action == "relax_business_filter":
            relaxed = base.filters.relaxed()
            return (
                _single(plan, relaxed, base.query, base.top_k, plan.strategy),
                "relax_business_filter",
            )
        return None


def _single(
    plan: RetrievalPlan,
    filters: RetrievalFilters,
    query: str,
    top_k: int,
    strategy: Strategy,
    *,
    query_role: QueryRole = "original",
    reason: str | None = None,
) -> RetrievalPlan:
    return RetrievalPlan(
        strategy=strategy,
        tasks=(
            QueryTask("r1", query, strategy, top_k=top_k, filters=filters, query_role=query_role),
        ),
        final_n=plan.final_n,
        rerank=plan.rerank,
        reason=reason or f"repair:{plan.reason}",
    )


def _drop_hyde_echo(evidences, hyde_text: str) -> tuple[list, int]:
    """剔除与假设文**完全相同**的证据块（阶段 4.3 的硬约束）。

    正常情况下这条路永远走不到：假设文是我们自己写的，不会出现在 Milvus 里。它存在的
    理由是"万一"——如果哪天有人在检索侧把假设文当成一条候选塞进去（比如为了对比实验），
    或者知识库里恰好有一份内容完全一致的文档，那一条就会带着**真实文件名的引用卡片**
    出现在答案后面，比凭空编造更难被读者识破。

    只做**整块完全相等**（空白归一后）这一种判定，不做相似度：
    相似度阈值会把"复习中抄了一遍定义"的真实教材块误杀，而那种块恰恰是我们想要的证据。
    宁可放过"微微改写过的假设文"，也不误删真实内容——前者是理论上不存在的路径，
    后者是每次请求都在发生的事。
    """
    needle = "".join(hyde_text.split())
    if not needle:
        return list(evidences), 0
    kept = [e for e in evidences if "".join((e.text or "").split()) != needle]
    return kept, len(evidences) - len(kept)


def _with_query(result: RetrievalResult, original_query: str) -> RetrievalResult:
    """只给证据门用的副本：把 `query` 换成原问句，**不改原对象**。

    结果对象仍要如实记录"实际检索用的是什么句"（`RetrievalResult.query`），
    否则日志与 `degraded` 会看不出改写发生过。改写句为空（未接改写）时原样返回。
    """
    if not original_query or original_query == result.query:
        return result
    return replace(result, query=original_query)


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


__all__ = ["RagPipeline", "PipelineOutcome"]
