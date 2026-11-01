"""编排层状态（CLAUDE.md §7.8）：节点之间只通过它通信。

它是**一次请求的临时记录**，不是业务实体：不做持久化、不跨请求复用。
`to_response()` 是 `/v1/chat` 响应体的唯一出口——想加字段就改这里，别在 api.py 里另拼一套；
`to_dict()` 只是它的字典投影，保留给"已经按 dict 写的调用方"（SSE 的 `done` 帧走自己的
`_done_frames`，不经过这里）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from base.contracts import (
    AnswerSource,
    ChatResponse,
    ChatStatus,
    EvidenceDecision,
    FAQHit,
    IntentProfile,
    IntentResult,
    MemoryContext,
    PermissionDecision,
    RetrievalPlan,
    RetrievalResult,
    SlotResult,
    UnderstandingResult,
)
from base.log_messages import degraded_reasons
from understanding.compound_query import CompoundResult

# 直接沿用 `base/contracts.AnswerSource`，不在本模块另立一份字面量：
# 两处枚举迟早只改一处，然后响应体的类型与这里的赋值对不上（`refusal` 就是这么漏的可能）。
Source = AnswerSource

# `source="fallback"` 时，`error_code` 落在这几个上就是**系统失败**（该告警、该重试），
# 其余（`no_evidence` / 未设置）是**知识库没有**（该补语料）。两者对用户都是兜底话术，
# 对运维是两条完全不同的处置线，所以 `status` 也必须分开（实施手册 §11.3）。
_HARD_ERRORS = frozenset(
    {"retrieval_unavailable", "generation_unavailable", "citation_invalid", "internal_error"}
)

# 哪些降级码意味着"**这条答案本身**比理想路径弱"。只有它们会把 `status` 推到 `degraded`。
# 刻意不含 `memory_degraded` / `session_unavailable` / `citation_out_of_range`：
# 前两个不影响答案正确性（答案照样来自知识库），后者已经被剔除、不影响发出去的正文。
# 判据是"用户读到的这句话是否因此变弱"，不是"这次请求有没有出过岔子"。
_DEGRADED_ANSWER = frozenset(
    {
        "llm_fallback_model",
        "llm_retry",
        "citation_repaired",
        "citation_unsupported",
        "rerank_failed",
        "partial_recall_failed",
        "doc_type_relaxed",
        "faq_unavailable",
        "intent_unavailable",
    }
)


@dataclass(slots=True)
class ChatState:
    request_id: str
    query: str
    slots: SlotResult
    # 大模型给的槽位**原样**放这里（任务 2.3）：校验与合并要等意图结论出来之后才做
    # （`TaskSlots.kind` 是意图的结论，模型不许写），所以先存、后处理，不在这里改 `slots`。
    model_slots: dict[str, Any] | None = None
    # 请求体里显式给的学科（`subject` 参数）。留在这里是因为它是**槽位重解析**的输入：
    # `_load_session()` 拿到会话历史后还要再解析一次槽位（任务 2.4 的
    # `history_hint`），而参数优先级高于问句与历史，重解析时不能把用户明确指定的学科弄丢
    subject_param: str | None = None
    # 会话与记忆（§7.9）。`query` 永远是用户原话，`rewritten` 才是检索用的问句——
    # FAQ 的精确命中建立在原话的归一化串上，把上下文拼进去会让它再也命不中（§7.1.2）
    user_id: str = ""
    session_id: str | None = None
    # 安全边界（§7.2.7 DoD 第 3 条）：由接入层写进来，**请求体里没有这两个字段**，
    # 调用方也改不了它们。落到这里而不是每次去读 `settings`，是因为同一进程可能
    # 同时服务多个租户；每处各读一次 settings，迟早有一处忘了改。
    tenant_id: str = ""
    acl_tags: tuple[str, ...] = ()
    # 身份与追踪（阶段 1）：由准入层写进来，**请求体里没有这些字段**。
    # `role` 不参与检索过滤，但**参与业务权限判定**——结论是下面那个 `permission`，
    # 下游只读 `permission`，不再自己拿 `role` 比表（同一张表比两遍，迟早比出两种结果）。
    # `trace_id` 让响应体、日志、OTEL 三者能用一个键串起来。
    role: str = "ANONYMOUS"
    trace_id: str = ""
    # 业务权限（阶段 3）：准入层算好的结论（角色 × 意图），随 `RequestContext` 一路带进来。
    # **None = 没走准入层的调用方**（脚本 / 单测 / 旧调用方），此时视为不设限，
    # 与加这个字段之前的行为一致——门禁只在"确实知道身份"的时候才收
    permission: PermissionDecision | None = None
    # 索引版本与预算：`kb_version` 决定在哪一版索引上检索（接入层定，不看 token）；
    # `deadline` 是准入层建的同一个时钟，节点按剩余预算自我裁量（§6.1）
    kb_version: str = ""
    deadline: Any = None
    rewritten: str = ""
    # 归一化后的问句（NFKC + 空白/重复标点折叠）。单独留一份是为了让
    # `UnderstandingResult.normalized` 与 `rewritten` 能区分开：前者是确定性变换的结果，
    # 后者可能还叠了一次大模型改写。两者都空时检索回落到 `query`
    normalized: str = ""
    # 改写/归一化是怎么来的：`none`（原样）/ `normalize`（确定性归一化）/ `rule`（多轮拼接）/ `llm`。
    # 与 `intent_method` 同组——不分开记，A/B 时"召回变差"分不清是归一化还是大模型改写造成的
    rewrite_method: str = "none"
    memory: MemoryContext | None = None
    history: list[tuple[str, str]] = field(default_factory=list)
    faq: FAQHit | None = None
    # 理解层（§7.7）：`intent` 只进日志/指标，`profile` 才是下游读取的档位。
    # 二者都由 `_classify()` 填。**在 FAQ 之前**就能定（阶段 3 定稿的时序是
    # 会话 → 理解 → 权限 → FAQ → 检索），所以 FAQ 高置信命中时它们同样有值——
    # 权限判定要读 `intent`，定了顺序就不可能再"FAQ 先返回、意图还没算"。
    # 为 None 只剩一种情况：准入层在更前面就拒了（空问句 / 超长 / 无权限），根本没进编排
    intent: IntentResult | None = None
    profile: IntentProfile | None = None
    # 统一理解结果（任务 2.5）。**只读**：`_prepare()` 在槽位收尾之后整车装一次，
    # 之后下游（检索、响应、日志）只读它，不再各自去猜"改写到底有没有生效"。
    # 与 `intent` 同理，它在 FAQ 之前就装配好了；为 None 表示没走到理解层（准入层已拒）
    understanding: UnderstandingResult | None = None
    # 复合问句拆分的结论（任务 2.4，2026-11-01 改由本地小模型拆）。
    # 单独存一份而不是就地塞进 `understanding`：拆分**排在意图之后**、装配理解结果之前，
    # 中间还隔着 `_note_coreference()`——把它变成 `_build_understanding()` 的入参，
    # 就能让"谁拆的 / 拆出几条 / 为什么退回规则"在日志与响应里各有一处出处。
    # `source` 说明谁拆的（`model` / `distribute` / `connective` / …），`degraded` 说明
    # 模型那条路为什么没成；`None` 表示没走到这一步（准入层已拒）
    compound: CompoundResult | None = None
    # 大模型给出的检索策略建议（`direct` / `sub_query` / `hyde`），空串 = 没给或没开。
    # **它只是建议**：planner 会按可行性（有没有真拆出子问题、HyDE 能不能跑）决定采纳还是降级，
    # 最终执行的是 `plan`。单独留一份是为了让"模型想这么查"与"最后这么查了"能对上——
    # 只留 `plan` 的话，一次降级看起来就像模型从来没提过这事。
    strategy_hint: str = ""
    retrieval: RetrievalResult | None = None
    plan: RetrievalPlan | None = None
    # 证据门不过时 pipeline 会**至多修一次**（重检 / 放宽过滤 / 扩邻域），修完换一份计划。
    # 两份都留着：`plan` 是"这句话被拆成了几路"的答案，这里才是"最后实际怎么查的"。
    # 只留一份的话，要么丢失拆分证据（旧行为），要么谎报最终执行的那一路。
    plan_after_repair: RetrievalPlan | None = None
    gate: EvidenceDecision | None = None
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    source: Source = "fallback"
    # 统一出口状态（实施手册 §2.7 / §7.1）。**一般不手写它**：`to_response()` 会按
    # `source` / `error_code` / `degraded` 归一化，漏设的路径不会因此谎报 `ok`。
    # 显式赋值的唯一场景是"外部没走编排"（如 api 层捕获到未预期异常），
    # 那种情况 `source` 与 `answer` 都是兜底拼出来的，推导不出来。
    status: ChatStatus = "ok"
    # 对外错误码（§7.2.7）：`retrieval_unavailable` 与 `no_evidence` 必须分得开——
    # "检索坏了"和"知识库里没有"对用户与运维是两件事
    error_code: str | None = None
    degraded: list[str] = field(default_factory=list)
    latency_ms: dict[str, int] = field(default_factory=dict)
    stages: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
    # 下面三个是响应体里的产品字段，不是内部状态：
    # mid 档的追问对象（只有 question，**不含 answer**，§7.1.9）
    faq_suggestion: dict[str, Any] | None = None
    # clarify 档的追问话术；fallback 档为空
    clarify: str | None = None
    faq_id: int | None = None

    def to_response(self) -> ChatResponse:
        """唯一出口：把内部状态整理成 `/v1/chat` 的响应体。

        `status` 的归一化规则（顺序即优先级，别调换）：

        1. `refusal` → `refused`。拒答是**确定的结论**，不随 `error_code` / `degraded` 变。
           排在 `clarify` 之前是因为拒答一定不是追问（两者互斥），而拒答最需要稳定的指标口径：
           它既不能进"知识库缺口"（`no_evidence`），也不能进"系统故障"（`error`）。
        2. `clarify` → `clarify`。追问就是追问，不管前面发生了什么。
        3. `fallback` → 看 `error_code`：系统类失败记 `error`，其余记 `no_evidence`。
           **兜底话术必须能被前端识别成"没答上"**，不能因为 HTTP 是 200 就当成成功。
        4. 其余（`faq` / `rag`）→ 有答案才算 `ok`；`degraded` 里出现"答案变弱"的码则记
           `degraded`。`answer` 为空却自称 `ok` 是自相矛盾，按 `error` 处理。

        规则写成推导而不是让每个分支自己赋值，是因为出口有十几条：靠人记着"这条别忘了设
        status"迟早会漏，而漏掉的那次恰好会是线上唯一一次兜底。
        """
        return ChatResponse(
            request_id=self.request_id,
            trace_id=self.trace_id,
            role=self.role,
            status=self.resolve_status(),
            source=self.source,
            answer=self.answer,
            citations=self.citations,
            faq_id=self.faq_id,
            faq_suggestion=self.faq_suggestion,
            clarify=self.clarify,
            error_code=self.error_code,
            gate=self.gate.as_dict() if self.gate else None,
            plan=self.plan.as_dict() if self.plan else None,
            plan_after_repair=(
                self.plan_after_repair.as_dict() if self.plan_after_repair else None
            ),
            session_id=self.session_id,
            # 只有"确实改写过"才回传：等于原句时给 None，免得响应体里出现两份一样的问句
            rewritten=(
                self.rewritten if self.rewritten and self.rewritten != self.query else None
            ),
            rewrite_method=self.rewrite_method,
            memory_tokens=self.memory.tokens if self.memory else 0,
            intent=self.intent.label if self.intent else None,
            intent_method=self.intent.method if self.intent else None,
            intent_confidence=self.intent.confidence if self.intent else None,
            profile=self.profile.name if self.profile else None,
            strategy=self.plan.strategy.value if self.plan else None,
            degraded=list(self.degraded),
            degraded_reasons=degraded_reasons(self.degraded),
            stages=self.stages,
            latency_ms=self.latency_ms,
            usage=self.usage,
        )

    def to_dict(self) -> dict[str, Any]:
        """`to_response()` 的字典投影。

        保留它是因为已经有一批调用方（脚本、单测、评估）按 dict 取值；
        新代码请直接用 `to_response()`，`as_dict()` 的字段名与顺序以那里为准。
        """
        return self.to_response().as_dict()

    def resolve_status(self) -> ChatStatus:
        if self.source == "refusal":
            return "refused"
        if self.source == "clarify":
            return "clarify"
        if self.source == "fallback":
            return "error" if self.error_code in _HARD_ERRORS else "no_evidence"
        if self.status != "ok":
            # 显式设过的（如 api 层兜底）尊重它，但仍不许把兜底说成成功
            return self.status
        if not self.answer:
            return "error"
        if any(code in _DEGRADED_ANSWER for code in self.degraded):
            return "degraded"
        return "ok"
