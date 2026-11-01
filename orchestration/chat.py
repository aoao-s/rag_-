"""编排内核（CLAUDE.md §7.8）：一次提问从 FAQ 到生成答案的完整编排，同步与流式共用。

**为什么需要这一层**：`qa_system` 与 `rag_system` 按宪法互不 import（§5.1）。
"先问 FAQ，没中再查文档，最后决定生成还是兜底"这件事不属于任何一方，
所以放在 `orchestration/`——它是唯一同时依赖两个业务包的包。

三档读法（§7.1.9）：
- `high`：逐字返回 FAQ 答案，**跳过检索**（省约 350 ms），不调 LLM；
- `mid`：跑 RAG。证据门不过 → 追问 top1 候选（**不返回 FAQ 原文**）；过了 → RAG 答案 + FAQ 卡片。
  生成时 top3 候选的**问答对**可以进 prompt 当辅助上下文（`FEATURE_FAQ_CANDIDATES`），
  但它不是证据：不编号、不进【资料】、不进 citations，`FAQHit.answer` 仍是 None；
- `low`：跑 RAG。证据门不过 → 兜底 + 转人工；候选**不进** prompt（§7.1.9 修订后仍如此）。

顺序是硬约束：**身份与权限先于一切检索**（阶段 3）——会话 → 理解 → 权限 → FAQ → 检索，
`FAQ` 也是一条检索路径，所以它排在权限之后，"高置信命中直接返回"不再能绕过权限判定。
`_prepare()` 里的注释逐条写明为什么是这样排的。

**两条通道一份内核**：`ask()`（`POST /v1/chat`）与 `ask_stream()`（`GET /v1/chat/stream`）
共用 `_prepare()`（理解 → 权限 → FAQ → 检索 → 证据门，决定出口）与同样的三档动作；差别只在"生成"这一步
是拿整段还是逐句。要改分支策略，只改 `_prepare()`——两处各写一遍必然对不上。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from base.config import Settings
from base.contracts import (
    FAQCandidate,
    FAQHit,
    IntentProfile,
    IntentResult,
    MemoryContext,
    PermissionDecision,
    QueryContext,
    RetrievalResult,
    SlotResult,
    UnderstandingResult,
)
from base.errors import BudgetExceeded, UpstreamUnavailable
from base.labels import GRADE_STAGE, ROLE_ANONYMOUS
from base.logger import get_logger, new_request_id
from memory.service import MemoryService
from orchestration.admission import permissions_for
from orchestration.state import ChatState
from qa_system.res import qa_slots
from qa_system.res.qa_search import FAQService
from rag_system.core import query_processor
from rag_system.core.citation import align, repair
from rag_system.core.llm_gateway import LLMClient
from rag_system.core.prompts import PROMPT_VERSION, build_messages
from understanding import compound_query
from understanding import llm_understanding
from understanding import slots as slot_fusion
from understanding.bert_intent import BertIntentClassifier
from understanding.intent import IntentClassifier
from understanding.profiles import BASELINE, resolve_profile
from understanding.rewrite import has_reference, rewrite as rewrite_query

log = get_logger(__name__)

# 理解层自己产生的降级码（任务 2.5）。`UnderstandingResult.degraded` 只装这些：
# 它是"理解这一环退化了什么"，不是整条链路的降级清单——后者在 `state.degraded` 里。
_UNDERSTANDING_CODES = frozenset(
    {
        "understanding_timeout",
        "understanding_failed",
        "understanding_bad_json",
        "rewrite_rejected",
        "intent_rejected",
        "slots_rejected",
        "strategy_rejected",
        "slot_model_invalid",
        "slot_model_filter_dropped",
        "slot_model_value_rejected",
        "coreference_unresolved",
        "session_unavailable",
        # 复合问句拆分（任务 2.4）走本地小模型的那四个失败码。**都不是硬失败**：
        # 它们只说明"这次是规则拆的"，结果照样能用，所以不进 `_UNDERSTANDING_HARD_FAILURES`
        "compound_timeout",
        "compound_failed",
        "compound_bad_json",
        "compound_rejected",
    }
)

# 只有这三个意味着"理解**整段**没跑成"，其余都是"某一项退回了更保守的做法"。
# 分开是因为它们对 `UnderstandingResult.success` 的意义不同：前者该告警，
# 后者是设计内的降级路径（每次都会发生几十次，不该拉响警报）。
_UNDERSTANDING_HARD_FAILURES = frozenset(
    {"understanding_timeout", "understanding_failed", "understanding_bad_json"}
)

_NO_EVIDENCE = "抱歉，知识库里暂时没有找到能支撑这个问题的资料。你可以换个问法，或联系{support}由人工帮你确认。"
_SYSTEM_ERROR = "抱歉，答案生成服务暂时不可用，请稍后重试，或联系{support}由人工处理。"
_CONFLICT = "知识库里对这个问题的说法不一致（{reasons}），我不替你挑一个。请参考下方来源，或联系{support}人工确认。"
_CLARIFY = "你是不是想问：{question}？如果是，请直接确认；如果不是，可以再补充一点背景，我帮你重新查。"
# 证据门 `ambiguous` 的追问：没有一条"值得推荐的 top1 候选"可问（FAQ 那边才有），
# 只能请用户补充区分条件。**不返回检索片段当答案**——那等于让用户自己挑一条，而不是我们答。
_GATE_CLARIFY = (
    "这个问题在知识点上分得不够开：我找到了几处相近的资料，但没法确定你说的是哪一处。"
    "能再具体一点吗（比如哪个年级、哪一章，或者想问哪个方面）？"
    "如果补充后还是查不准，可以联系{support}由人工帮你确认。"
)
# 超出教育领域的固定拒答话术（§7.7 `OUT_OF_SCOPE`）。
# **刻意不带客服电话与"转人工"**：转人工是"知识库没这条"的出口（客服确实能帮上），
# 而越界问题人工也答不了、也不该答——写上电话等于承诺了一件我们不做的事，
# 还会让用户以为换个渠道就能绕过这条边界。这里只做一件事：把服务范围说清楚。
_OUT_OF_SCOPE = (
    "抱歉，这个问题超出了我的服务范围。我主要处理中小学教学相关的问题："
    "查找备课与教学资源、制作教案与课件、出题组卷、知识点讲解与习题辅导、制定学习规划。"
    "如果有这类问题，欢迎继续提问。"
)

# 流式按句切分：模型吐到句末标点就发一句，用户不用等整段。
_SENTENCE_ENDS = "。！？；!?;\n"

# 权限不足的固定话术（阶段 3：角色 × 意图）。
# 三条刻意的写法：
# - **不说"超出服务范围"**：那是越界拒答（`_OUT_OF_SCOPE`）的语义，而这里的问题本身在范围内的，
#   只是这个账号没有这项功能；混用会让用户以为"这类问题不能问"，其实是"你暂时不能用"。
# - **不暗示"换个账号就行了"**：那等于教人绕过权限。只说清是账号权限问题、给出正路（联系老师/顾问）。
# - **不列被拒的意图名**：那等于向外披露权限矩阵，也把内部枚举名（"备课检索"）暴露给终端用户。
_PERMISSION_DENIED = (
    "抱歉，当前账号没有使用这项功能的权限。如需开通，请联系老师或课程顾问。"
)


class ChatPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        faq_service: FAQService,
        rag: dict[str, Any] | None,
        llm: LLMClient,
        llm_local: LLMClient | None = None,
        intent: IntentClassifier | None = None,
        bert: BertIntentClassifier | None = None,
        memory: MemoryService | None = None,
    ) -> None:
        self.settings = settings
        self.faq_service = faq_service
        self.rag = rag or {}
        self.llm = llm
        # 本地小模型（闲聊档）。**默认 None**：不注入就与加它之前逐字节一致——
        # 所有档位都走 `self.llm`，闲聊也不例外（回滚口径）。
        # 它的可用性**与 `self.llm` 无关**：云端 key 缺失时闲聊仍能由本地作答。
        self.llm_local = llm_local
        self.intent = intent or IntentClassifier()
        # BERT 初始分类器（任务 2.2）。**默认 None 是刻意的**：它是一次可选的前置判断，
        # 不注入就整条链路退回"大模型 → 规则"，与加它之前逐字节一致（回滚口径）。
        # 加载在启动期完成（`main.py::Container.build_intent`），请求路径上不碰磁盘。
        self.bert = bert
        self.memory = memory
        # 回写任务要留强引用，否则任务可能在完成前被 GC 掉（asyncio 只持弱引用）
        self._bg: set[asyncio.Task[Any]] = set()
        # 会话归属的**进程内**快表（阶段 1 §3.4）。它存在的理由是两种 `memory` 挡不住的情况：
        # ① `memory=None`（单测、无库模式）；② FAQ 高置信短路——那条路径在 `_load_session()`
        # 之前就返回了，会话归属根本轮不到持久层。快表先查，查不到再问持久层
        self._session_owners: dict[str, str] = {}

    # ---- 会话归属（阶段 1 §3.4）----
    def claim_session(self, session_id: str | None, user_id: str) -> None:
        """把会话绑定到首个使用它的用户。**先到者胜**，不覆盖。

        用 `setdefault` 而不是赋值：后续请求沿用首轮归属。覆盖等于让后到的 B
        用一次就把 A 的会话抢走——而这正是这条规则要挡住的事。
        """
        if not session_id or not user_id:
            return
        self._session_owners.setdefault(session_id, user_id)

    async def session_owner(self, session_id: str | None) -> str | None:
        """返回会话归属人；`None` 表示**还没有人认领**（首轮，允许绑定）。

        异常**不在这里吞**：查不到归属与查不了归属是两件事，后者由准入层记
        `session_owner_unchecked` 后 fail-open——吞在这一层就再也没人知道发生过
        """
        if not session_id:
            return None
        owner = self._session_owners.get(session_id)
        if owner is not None:
            return owner
        if self.memory is None:
            return None
        return await self.memory.session_owner(session_id)

    async def _claim_session(self, state: ChatState) -> None:
        """首次使用即绑定（规则 6 的另一半：判归属的人也得有人去写归属）。

        两处都写：进程内快表兜住"持久层写进之前的下一次请求"（同一 worker 内立即可判），
        持久层让重启与多 worker 之后仍判得出归属。持久层失败只降级——
        "写不进归属"不该让用户问不了问题
        """
        self.claim_session(state.session_id, state.user_id)
        if self.memory is None or not state.session_id or not state.user_id:
            return
        try:
            await self.memory.claim_session(state.session_id, state.user_id)
        except Exception as exc:
            log.warning("session_claim_failed", error=str(exc))

    # ---- 同步通道 ----
    async def ask(
        self,
        query: str,
        *,
        subject: str | None = None,
        selected_faq_id: int | None = None,
        request_id: str | None = None,
        require_citation: bool | None = None,
        user_id: str = "",
        session_id: str | None = None,
        tenant_id: str = "",
        acl_tags: tuple[str, ...] = (),
        role: str | None = None,
        trace_id: str = "",
        kb_version: str = "",
        deadline: Any = None,
        permission: PermissionDecision | None = None,
    ) -> ChatState:
        started = time.perf_counter()
        state = _new_state(
            query,
            subject=subject,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            acl_tags=acl_tags,
            role=role,
            trace_id=trace_id,
            kb_version=kb_version,
            deadline=deadline,
            permission=permission if permission is not None else self._permission_for(role),
        )
        await self._claim_session(state)

        early = await self._prepare(state, started, selected_faq_id=selected_faq_id)
        if early is not None:
            return early
        return await self._generate(
            state, started, require_citation=self._require_citation(state, require_citation)
        )

    # ---- 流式通道（SSE，§7.8 第 5 项）----
    async def ask_stream(
        self,
        query: str,
        *,
        subject: str | None = None,
        selected_faq_id: int | None = None,
        request_id: str | None = None,
        require_citation: bool | None = None,
        user_id: str = "",
        session_id: str | None = None,
        tenant_id: str = "",
        acl_tags: tuple[str, ...] = (),
        role: str | None = None,
        trace_id: str = "",
        kb_version: str = "",
        deadline: Any = None,
        permission: PermissionDecision | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """产出四类帧：`meta` / `delta` / `citations` / `done`。

        契约：**`done` 是权威结果**。`done.answer` 是最终该展示的全文、`done.source` 是最终出口。
        流式下"引用不过关就整段拒答"没法把已发出的字节收回来，所以约定：客户端收到
        `done.source != "rag"` 时丢弃此前 delta，改显示 `done.answer`。这样既保留了逐字体验，
        又不会让一条没通过的答案留在屏幕上。
        """
        started = time.perf_counter()
        state = _new_state(
            query,
            subject=subject,
            request_id=request_id,
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            acl_tags=acl_tags,
            role=role,
            trace_id=trace_id,
            kb_version=kb_version,
            deadline=deadline,
            permission=permission if permission is not None else self._permission_for(role),
        )
        await self._claim_session(state)

        try:
            early = await self._prepare(state, started, selected_faq_id=selected_faq_id)
        except asyncio.CancelledError:
            log.info("chat_stream_cancelled", request_id=state.request_id, stage="prepare")
            raise

        if early is not None:
            for frame in _frames_of(early):
                yield frame
            return

        yield {
            "type": "meta",
            "request_id": state.request_id,
            "source": "rag",
            "tier": state.faq.tier if state.faq else "-",
            "cached": False,
            "session_id": state.session_id,
        }

        async for frame in self._stream_generate(
            state, started, require_citation=self._require_citation(state, require_citation)
        ):
            yield frame

    # ---- 准备阶段：理解 → 权限 → FAQ → 检索 → 证据门 ----
    async def _prepare(
        self,
        state: ChatState,
        started: float,
        *,
        selected_faq_id: int | None = None,
    ) -> ChatState | None:
        """返回 `None` 表示"证据已就绪、可以生成"；其余返回值是提前敲定的出口。

        执行顺序（阶段 3 定稿，CLAUDE.md §7.4）：会话 → 理解 → **权限** → FAQ → 检索 → 证据门。
        三处顺序都不是随手排的：

        - **权限在 FAQ 之前**：FAQ 是一条检索路径，"先查库再判权限"等于把知识库答案先交给
          一个没权限的人（`high` 档还会逐字返回）。判定只需要 `state.intent`，所以紧跟
          `_classify()` 就能定论，不必等到检索那一刻。
        - **会话与理解在 FAQ 之前**：代价是 FAQ 命中也要付一次理解调用（受
          `UNDERSTANDING_TIMEOUT_MS` 约束，BERT 本地判得出意图时大模型只做改写）。换来的
          是"FAQ 与检索看到同一套意图与槽位"——此前 FAQ 用的是 `_new_state()` 里那份只有
          规则、没有 `history_hint` 的槽位，同一次请求里两条路可能按不同学科过滤。
        - **追问排在权限之后**：没权限的人不该被问"你想做哪一份教案"，那等于先确认他要
          用一个没开通的功能。
        """
        # ---- 1. 会话窗口 → 改写（§3.2 第 4 条）----
        # 排在所有检索之前：指代消解（"它的公式"）要用上一轮，而改写结果决定后面每条路
        # 拿哪句话去查。FAQ 的精确命中仍建立在**原话**的归一化串上（§7.1.2），不受影响。
        await self._load_session(state)

        # ---- 2. 归一化 + 一次大模型理解（改写 + 意图，§3.2 / §7.7）----
        # 位置是硬约束：必须在 `_load_session` **之后**（它会先把 rewritten 重置成原话），
        # 且在 `_classify` **之前**（后者只认 `state.intent`，不再自己调模型）。
        await self._understand(state, started)

        # ---- 3. 意图 → 档位 ----
        # 大模型结论上面已就绪就用它；否则退回规则分类（纯 CPU 微秒级）。
        profile = self._classify(state)

        # ---- 3.5 槽位收尾（任务 2.3）----
        # 排在 `_classify` 之后是硬约束：`TaskSlots.kind` 是**意图的结论**（模型不许写），
        # 没拿到意图就 finalize，等于让下游按"没有任务"去执行。
        question = self._apply_slots(state)

        # ---- 3.55 复合问句拆分（任务 2.4，本地小模型）----
        # 排在**意图之后**是这次改动的核心要求：拆不拆、往哪拆都取决于已经判出来的意图。
        # 也排在槽位收尾之后：`_apply_slots()` 给出追问话术时这次请求会在下面早退，
        # 那时候拆出来也没人用，先拆只是白付一次本地调用。
        if question is None:
            await self._split_compound(state, started)

        # ---- 3.6 指代 + 统一理解结果（任务 2.4 / 2.5）----
        # 放在槽位收尾**之后**：`UnderstandingResult.slots` 要的是合并过的最终槽位，
        # 而不是 `_new_state()` 里那份"只有规则"的中间态。
        # 放在追问早退**之前**：`ambiguity` 就是那个追问话术，早退了就永远是 None。
        self._note_coreference(state)
        state.understanding = self._build_understanding(state, question)
        log.info(
            "understanding_ready",
            request_id=state.request_id,
            success=state.understanding.success,
            intent=state.intent.label if state.intent is not None else "unknown",
            intent_method=state.understanding.method,
            confidence=f"{state.understanding.confidence:.2f}",
            profile=profile.name,
            is_complex=state.understanding.is_complex,
            sub_queries=len(state.understanding.sub_queries),
            is_follow_up=state.understanding.is_follow_up,
        )

        # ---- 3.7 业务权限（阶段 3）：角色 × 意图 ----
        # 「先身份与权限，再 FAQ」的落点。日志先记 `understanding_ready` 是有意的：
        # 被拒的请求也要留下"当时判成了什么意图"，否则线上看到一批 permission_denied
        # 却不知道是哪条规则挡的，只能靠复现去猜。
        if not self._allowed(state):
            return self._permission_denied(state, started)

        if question is not None:
            return self._slot_clarify(state, question, started)

        # ---- 4. 不查库的档位（闲聊 / 超出范围）----
        # **排在 FAQ 之前**：这两档本来就不该读知识库——超范围要固定拒答，
        # 闲聊要短答；让 FAQ 先插一句"知识库答案"会把这两条出口的语义搅乱。
        if not profile.retrieval:
            # 两个档位共用"不查库"（省掉检索+重排 ≈350 ms），出口却完全不同。
            # `allow_general` 是它们之间**唯一**的区别字段，所以分叉只认它，不认标签字符串
            # （认标签就等于把意图写成了 if/else，§7.7 硬规则 1）。
            if not profile.allow_general:
                # 超出范围：固定文案拒答。
                # - 不调模型：拒答话术是产品文案，"礼貌地拒绝"不值得一次大模型往返，
                #   而且模型一旦自由组织话术，偶尔会把问题顺手答了——边界就漏了；
                # - 不召回记忆：记忆只会让模型更想"聊两句"，与拒答互相拆台；
                # - 不检索：库里没有也不该有"明天成都天气"这类内容。
                state.degraded.append("intent_out_of_scope")
                return self._refusal(state, started)
            # 闲聊档：不查库，直接用通用模板短答
            state.retrieval = RetrievalResult(query=state.rewritten)
            state.degraded.append("intent_no_retrieval")
            # 不检索也要召回记忆：闲聊最需要"这个学员是谁"，否则画像等于没建
            state.memory = await self._recall(state)
            # 这里判的是**本档位要用的那台引擎**，不是主网关：闲聊走本地小模型，
            # 云端 key 缺失不该让"你好"变成兜底话术
            if not getattr(self._llm_for(state), "enabled", False):
                state.degraded.append("llm_unavailable")
                state.error_code = "generation_unavailable"
                return self._fallback(state, self._system_error(), started)
            return None

        # ---- 5. FAQ（权限之后、检索之前）----
        t0 = time.perf_counter()
        try:
            faq = await self.faq_service.search(
                state.query, slots=state.slots, selected_faq_id=selected_faq_id
            )
        except Exception as exc:  # noqa: BLE001
            # FAQ 是"省钱的那条捷径"，不是必需品：它挂了就退回全量 RAG（§7.1.4），
            # 用户的提问照常能答，只是慢一点、贵一点。
            # 但必须留痕——不记 degraded 的话，"今天 FAQ 命中率为什么是 0"要查半天。
            log.warning("faq_search_failed", error=str(exc), request_id=state.request_id)
            state.degraded.append("faq_unavailable")
            faq = FAQHit(reason="faq_unavailable")
        state.faq = faq
        state.latency_ms["faq"] = _ms(t0)
        state.degraded.extend(faq.degraded)

        if faq.tier == "high" and faq.answer:
            state.answer = faq.answer
            state.source = "faq"
            state.faq_id = faq.faq_id
            return self._finish(state, started)

        suggestion = _suggestion(faq)

        # ---- 6. 检索（与记忆召回并行，§7.4 第 4 步）----
        if not self.rag.get("ready"):
            # RAG 挂了不等于 FAQ 就变可信了：mid 只能追问，不许升级为 high 直接作答（§7.1.9）
            state.degraded.append("rag_unavailable")
            # 这是**系统问题**，不是"知识库没有"：`error_code` 要让运维看见该去修服务
            state.error_code = "retrieval_unavailable"
            if faq.tier == "mid" and suggestion:
                return self._clarify(state, suggestion, started)
            return self._fallback(state, self._no_evidence(), started)

        t0 = time.perf_counter()
        # 记忆召回先起成任务再等检索：两者互不依赖，串行做等于白送一次超时预算（§7.4）
        memory_task = asyncio.create_task(self._recall(state))
        ctx = self._context(state, profile)
        try:
            # 一条管线做完：计划 → 多路召回 → 融合 → 装配 → 证据门 → 至多一次修复。
            # doc_type 收窄把候选清空时由 pipeline 自己放宽一次（§7.7 硬规则 6），
            # 编排层不再手写"再检一次"的兜底分支——分支写两份必然对不上。
            outcome = await self.rag["pipeline"].run(
                state.rewritten,
                ctx,
                profile=profile,
                # 多学科要整组下推：`subjects` 非空时 subject 只是"第一个"，
                # 只传 subject 会让"物理和数学的区别"只查物理，另一门学科的证据永远进不了候选。
                # `_build_expr` 对 >1 已生成 `subject in [...]`，这里补齐即可。
                subjects=state.slots.subjects
                or ((state.slots.subject,) if state.slots.subject else ()),
                # 复合问句的子问题（任务 2.4）：每一条**各自一路召回**，
                # 合成一句话的检索会让"区别"与"练习题"两组证据互相稀释。
                # 这里只负责把已拆好的递下去——"要不要拆、拆出来几路合不合法"全在
                # `compound_query.split()` 里定稿，planner 只按 `len(subs) >= 2` 决定走 SUB_QUERY。
                # 传空元组表示"没拆"，不表示"不让拆"：planner 仍会看连接词自己标策略
                sub_queries=state.understanding.sub_queries if state.understanding else (),
                history_present=bool(state.history),
                # 要不要检索 / 要不要先追问（阶段 4.1）：这两个决定在 planner 里，
                # 但它们的信息源在编排层——档位说"这档不检索"，理解层说"指代没解开"。
                # `needs_retrieval` 显式传而不是让 planner 读档位：`FEATURE_INTENT=off`
                # 时基线档位恒 `retrieval=True`，靠默认值会悄悄绕开基线。
                needs_retrieval=bool(profile.retrieval),
                unresolved_reference="coreference_unresolved" in state.degraded,
                # HyDE（阶段 4.3）：**默认关闭**（`FEATURE_HYDE=false`），且没有可用模型时
                # 一律不选——`hyde_available=True` 却没人能生成假设文，会让 telemetry 与
                # `plan.strategy` 声称跑了一个没跑的策略。关着时 planner 如实降级成
                # `DIRECT + hyde_unavailable`；开着时假设文只做检索查询，
                # 绝不进证据（`RagPipeline._drop_hyde_echo` 兜底，§7.2.7-G）。
                hyde_available=self.settings.feature_hyde
                and bool(getattr(self.llm, "enabled", False)),
                # 模型选的检索方式（2026-11-01）：**只递建议，不在这里下结论**。
                # 跑不跑得了由 planner 判（要拆多路得有真拆出来的子问句、要 HyDE 得有模型），
                # 最终执行的是 `outcome.plan`；两者对不上时 `strategy_hint` 与
                # `plan.reason`（`llm_*` / `hyde_unavailable`）并排看就知道是"选错了"还是"跑不了"
                strategy_hint=state.strategy_hint,
                budget_ms=self.settings.total_budget_ms - _ms(started),
            )
        except Exception as exc:  # noqa: BLE001 - 检索失败必须落到兜底，不能把 5xx 抛给用户
            memory_task.cancel()
            log.warning(
                "retrieve_failed", error=str(exc), request_id=state.request_id,
                code=getattr(exc, "code", "-"),
            )
            state.degraded.append("retrieve_failed")
            # "检索系统异常"与"知识库没有证据"必须分开（§7.2.7）：前者该告警修服务，
            # 后者该补语料，混成一个错误码会让两条完全不同的处置线合流
            state.error_code = getattr(exc, "code", None) or "retrieval_unavailable"
            if faq.tier == "mid" and suggestion:
                return self._clarify(state, suggestion, started)
            return self._fallback(state, self._system_error(), started)

        state.memory = await memory_task
        state.retrieval = outcome.result
        # 对外显示**首轮**那份计划：它是"这句话被理解成了几路、每路查什么"的答案，
        # 也是复合问句拆分有没有生效的唯一可核验面。修复换掉的那份单列进
        # `plan_after_repair`，两份都露出来才对得上账——否则用户先读到
        # `strategy=sub_query`、再看到一条 `direct` 任务，会以为拆分没发生。
        state.plan = outcome.initial_plan or outcome.plan
        state.plan_after_repair = outcome.plan_after_repair
        state.gate = outcome.decision
        state.stages = dict(outcome.stages)
        state.latency_ms["retrieve"] = _ms(t0)
        state.degraded.extend(outcome.degraded)
        state.degraded.extend(outcome.result.degraded)

        decision = outcome.decision
        if not decision.passed:
            state.degraded.append(f"gate_{decision.reason}")
            if faq.tier == "mid" and suggestion:
                return self._clarify(state, suggestion, started)
            if decision.status == "conflicting":
                # 两篇文档对同一个量给了不同的数：并列摆出来比按分数挑一条更诚实（§7.2.7-C）
                state.citations = outcome.result.citations()[:3]
                return self._fallback(state, self._conflict(decision), started, keep_citations=True)
            if decision.status == "ambiguous":
                # 两篇文档咬得太紧：按分数挑一条就是抽签（§7.2.7-B）。追问一轮比给一个
                # "可能是这个"的答案代价小——用户补一句背景就能收敛，而错答会被当成事实。
                state.citations = outcome.result.citations()[:3]
                return self._gate_clarify(state, started, keep_citations=True)
            # 到这里才是"知识库里确实没有能支撑的证据"
            state.error_code = "no_evidence"
            return self._fallback(state, self._no_evidence(), started)

        # ---- 3. 生成前置检查（两通道共用，避免流式忘了这一条）----
        # 与闲聊分支同一个口径：判本档位要用的引擎（此处档位都是 `default`，即主网关，
        # 但走 `_llm_for` 是为了将来有别的档位换引擎时不必再改一遍）
        if not getattr(self._llm_for(state), "enabled", False):
            state.degraded.append("llm_unavailable")
            state.error_code = "generation_unavailable"
            return self._fallback(state, self._system_error(), started)
        return None

    # ---- 生成阶段：同步 ----
    async def _generate(
        self, state: ChatState, started: float, *, require_citation: bool
    ) -> ChatState:
        result = state.retrieval
        assert result is not None  # _prepare 返回 None 时必然已填好

        budget_ms = self.settings.total_budget_ms - _ms(started)
        messages = build_messages(
            state.query,
            result.evidences,
            memory=_memory_text(state),
            history=state.history,
            faq_candidates=_faq_prompt_candidates(state, self.settings),
        )
        t0 = time.perf_counter()
        gateway = self._llm_for(state)
        try:
            completion = await gateway.complete(messages, budget_ms=budget_ms)
        except BudgetExceeded:
            state.degraded.append("budget_exceeded")
            state.error_code = "generation_unavailable"
            return self._fallback(state, self._system_error(), started)
        except UpstreamUnavailable as exc:
            log.warning("generate_failed", error=str(exc), prompt=PROMPT_VERSION)
            state.degraded.append("llm_unavailable")
            # 生成不可用是**系统失败**，必须与"知识库没有证据"分开上报（§11.3）：
            # 混成 `no_evidence` 会让"模型网关挂了"变成"该补语料"，处置方向完全反了
            state.error_code = "generation_unavailable"
            return self._fallback(state, self._system_error(), started)
        state.latency_ms["generate"] = _ms(t0)
        state.usage = {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
        }
        state.degraded.extend(completion.degraded)

        # ---- 4. 引用对齐（最多一次受控修复，§6.3）----
        t_cite = time.perf_counter()
        alignment = align(
            completion.text,
            result.evidences,
            min_support=self.settings.citation_min_support,
        )
        if alignment.out_of_range:
            state.degraded.append("citation_out_of_range")
            log.warning("citation_out_of_range", refs=alignment.out_of_range, request_id=state.request_id)

        if alignment.unsupported:
            # 第一遍有点对不上依据的句子 → **一次**受控修复：整句剔掉再验一遍。
            # 修复的边界写在 `citation.repair()` 里：不调模型改写、不改来源元数据、
            # 不把编号改挂到别的证据上——只剩"删"这一种改动，所以修完的正文仍逐句可验。
            alignment, dropped = repair(
                completion.text,
                result.evidences,
                min_support=self.settings.citation_min_support,
            )
            state.degraded.append("citation_unsupported")
            if dropped and alignment.citations and not alignment.unsupported:
                # 修完还有引用可用：这是"可用但削弱了"的答案，不是失败，单独记 `citation_repaired`
                state.degraded.append("citation_repaired")

        if require_citation and not alignment.citations:
            # 修完一句话都没对上，或压根没引用：整段答案都不能用（§7.8 引用契约）。
            # 不"挑掉几句再拼回去"——那正是受控修复已经试过并失败的那一步。
            state.degraded.append("citation_missing" if not alignment.unsupported else "citation_unsupported")
            state.error_code = "citation_invalid"
            state.latency_ms["citation"] = _ms(t_cite)
            return self._fallback(state, self._no_evidence(), started)
        state.latency_ms["citation"] = _ms(t_cite)

        state.answer = alignment.text
        state.citations = alignment.citations
        state.source = "rag"
        # mid 档即便 RAG 过了门，也要把 FAQ 卡片带上（不含 answer）：用户可以顺带确认那条 FAQ
        if state.faq and state.faq.tier == "mid":
            state.faq_suggestion = _suggestion(state.faq)
        return self._finish(state, started)

    # ---- 生成阶段：流式 ----
    async def _stream_generate(
        self, state: ChatState, started: float, *, require_citation: bool
    ) -> AsyncIterator[dict[str, Any]]:
        """逐句发送：**只有通过引用校验的句子才会发给用户**。

        与同步通道的唯一差异写在 `ask_stream()` 的 docstring 里：同步是"整段不合格就整段拒答"，
        流式做不到回收已发出的字节，于是改成"逐句剔除 + `done` 兜底"。判定用的是同一个
        `align()`，所以"哪句算有依据"的口径两通道一致。
        """
        result = state.retrieval
        assert result is not None
        evidences = result.evidences

        budget_ms = self.settings.total_budget_ms - _ms(started)
        messages = build_messages(
            state.query,
            evidences,
            memory=_memory_text(state),
            history=state.history,
            faq_candidates=_faq_prompt_candidates(state, self.settings),
        )

        buffer = ""
        emitted: list[str] = []
        citations: dict[int, dict] = {}
        out_of_range: list[int] = []
        unsupported = 0
        t0 = time.perf_counter()

        try:
            async for chunk in self._llm_for(state).stream(messages, budget_ms=budget_ms):
                buffer += chunk
                while True:
                    sentence, buffer = _take_sentence(buffer)
                    if sentence is None:
                        break
                    frames, dropped = _align_and_emit(
                        sentence, evidences, self.settings.citation_min_support,
                        emitted, citations, out_of_range,
                    )
                    unsupported += dropped
                    for frame in frames:
                        yield frame
            # 模型最后一句经常没有句号：缓冲区剩余部分按最后一句处理
            if buffer.strip():
                frames, dropped = _align_and_emit(
                    buffer, evidences, self.settings.citation_min_support,
                    emitted, citations, out_of_range,
                )
                unsupported += dropped
                for frame in frames:
                    yield frame
        except asyncio.CancelledError:
            # 客户端断开 → 停止拉取上游，等于停止计费（§3.1 第 3 条）
            log.info(
                "chat_stream_cancelled",
                request_id=state.request_id,
                stage="generate",
                chars=len("".join(emitted)),
            )
            raise
        except BudgetExceeded:
            state.degraded.append("budget_exceeded")
            state.source = "fallback"
            state.error_code = "generation_unavailable"
            state.answer = self._system_error()
            for frame in self._stream_finish(state, started):
                yield frame
            return
        except UpstreamUnavailable as exc:
            log.warning("generate_failed", error=str(exc), prompt=PROMPT_VERSION)
            state.degraded.append("llm_unavailable")
            state.source = "fallback"
            state.error_code = "generation_unavailable"
            state.answer = self._system_error()
            for frame in self._stream_finish(state, started):
                yield frame
            return

        state.latency_ms["generate"] = _ms(t0)
        if out_of_range:
            state.degraded.append("citation_out_of_range")
            log.warning("citation_out_of_range", refs=out_of_range, request_id=state.request_id)
        if unsupported:
            state.degraded.append("citation_unsupported")

        if require_citation and not citations:
            # 一句话都没对齐上：把出口改成兜底，由 `done.answer` 覆盖掉已发出的内容
            state.degraded.append("citation_missing" if not unsupported else "citation_unsupported")
            state.source = "fallback"
            state.error_code = "citation_invalid"
            state.answer = self._no_evidence()
            state.citations = []
        else:
            state.source = "rag"
            state.answer = "".join(emitted)
            state.citations = [citations[n] for n in sorted(citations)]
            if state.faq and state.faq.tier == "mid":
                state.faq_suggestion = _suggestion(state.faq)

        for frame in self._stream_finish(state, started):
            yield frame

    # ---- 出口 ----
    def _classify(self, state: ChatState) -> IntentProfile:
        """意图 → 档位。关掉 `FEATURE_INTENT` 就用 `BASELINE`（全量 DIRECT），
        这样 A/B 的对照组与回滚路径都是同一个对象，不会各写一份"基线"。

        "标签怎么变档位、安全阀怎么开"全部在 `understanding.profiles.resolve_profile`——
        评估脚本调的是同一个函数，否则量出来的就不是线上行为。
        """
        if not self.settings.feature_intent:
            state.profile = BASELINE
            return BASELINE
        # 大模型已经在 `_understand()` 里给过结论就直接用：再调一次规则分类器不但浪费，
        # 还会让日志里出现两个不同的标签（一个进了 `request_log`，一个进了 prompt 选择）
        if state.intent is not None:
            profile, degraded = resolve_profile(state.query, state.intent.label)
            state.degraded.extend(degraded)
            state.profile = profile
            return profile
        try:
            intent = self.intent.classify(state.query)
        except Exception as exc:  # noqa: BLE001
            # 意图只是"挑一条先验"，挂了不影响能不能答（§7.7 降级表）。
            # 退回 `BASELINE`（不收窄、必检索、必引用）而不是随手挑一个档位：
            # 猜错档位会把正确答案挡在 `doc_type` 过滤外面，而且答案还会被写进缓存。
            log.warning("intent_failed", error=str(exc), request_id=state.request_id)
            state.degraded.append("intent_unavailable")
            state.profile = BASELINE
            return BASELINE
        state.intent = intent
        profile, degraded = resolve_profile(state.query, intent.label)
        state.degraded.extend(degraded)
        state.profile = profile
        return profile

    def _context(self, state: ChatState, profile: IntentProfile | None) -> QueryContext:
        """把一次请求的"安全边界 + 预算"打包给检索层（§7.2.7-A）。

        `tenant_id / acl_tags / kb_version` 从这里进 Milvus 的标量过滤，且**只进不出**：
        修复阶段只允许放宽 `subjects` / `doc_types` / `stages` / `grades`（`RetrievalFilters.relaxed()`），
        这三个永远不动——放宽租户或 ACL 等于跨用户泄露。
        """
        # 年级 → 学段（`labels.GRADE_STAGE` 是唯一出处）。只从槽位来，意图不得覆写（§7.7 硬规则 4）。
        # `FEATURE_GRADE_FILTER=false` 是误抽年级时的回滚开关：过滤会把正确答案挡在索引外面，
        # 而"有结果但都是错的"不会触发 `relax_business_filter`（那只在空结果时生效）
        grade = state.slots.grade if self.settings.feature_grade_filter else None
        grades = (grade,) if grade else ()
        stages = (GRADE_STAGE[grade],) if grade in GRADE_STAGE else ()
        return QueryContext(
            request_id=state.request_id,
            # 身份优先：接入层给了就用接入层的，没给才回落到部署配置（单租户部署）
            tenant_id=state.tenant_id or self.settings.tenant_id,
            user_id=state.user_id,
            session_id=state.session_id,
            acl_tags=tuple(state.acl_tags),
            kb_version=self.settings.kb_version,
            subject=state.slots.subject,
            stages=stages,
            grades=grades,
            top_k=self.settings.rag_top_k,
            final_n=self.settings.rag_final_n,
            require_citation=(profile.require_citation if profile else True),
            budget_ms=self.settings.total_budget_ms,
            # 修复动作 `retry_original_query` 要的是**用户原话**（实施手册 §5.3），
            # `query` 可能已经被会话改写过，不能拿它当"原问句"重检
            original_query=state.query,
        )

    # ---- 槽位收尾（任务 2.3）----
    def _apply_slots(self, state: ChatState) -> str | None:
        """校验模型槽位 → 合并 → 按意图定任务种类；返回澄清问句（`None` = 不用追问）。

        规则那一半在 `_new_state()` 里就抽好了（`qa_slots.resolve`），这里只做三件事：
        ① 把模型给的槽位过一遍 Schema 校验（词表外的值、过滤类字段一律丢弃并计数）；
        ② 规则优先、模型补空；③ 拿到意图结论后补 `TaskSlots.kind` 与 `PrepSlots` 投影。

        **为什么模型槽位不打一开始就合并**：`TaskSlots.kind` 是意图的结论，而意图要等
        `_classify()`。先合并再 `finalize()` 会让任务种类恒为 `None`——下游就按"这次没有任务"
        去执行，而且不会有任何报错。所以 `ChatState` 里存的是**原样**的 `model_slots`。

        过滤类槽位（学科/年级/教材版本/学期）模型给的一律丢弃（§7.1.8 的安全不对称）：
        规则抽不到只是"全库多查一轮"，模型抽错是"正确答案被挡在索引外面"。
        """
        model, codes = slot_fusion.parse_model(state.model_slots)
        # 降级码进 `state.degraded`（响应体的 `degraded_reasons` 就吃它）。
        # 不塞回 `SlotResult.degraded` 之外另立一套：规则侧的降级同样并进来，
        # 否则"抽槽位时降级了"与"模型那一半被丢掉"会分成两个没人看的角落。
        state.degraded.extend(codes)
        state.degraded.extend(state.slots.degraded)
        state.slots = slot_fusion.finalize(
            slot_fusion.merge(state.slots, model),
            state.intent.label if state.intent is not None else None,
        )
        log.info(
            "slots_ready",
            request_id=state.request_id,
            subject=state.slots.subject or "-",
            subject_source=state.slots.subject_source,
            grade=state.slots.grade or "-",
            chapter=state.slots.chapter or "-",
            knowledge_points=state.slots.knowledge_points or "-",
            task_kind=state.slots.task.kind or "-",
            slot_source="规则+模型" if not model.empty else "规则",
            degraded=list(dict.fromkeys((*codes, *state.slots.degraded))) or "-",
        )

        # 追问默认关闭（`FEATURE_SLOT_CLARIFY=false`）：先按"不拦"发布。
        # 缺槽位时大模型多半也能给个可用的通用答案，而追问一次就是一次交互成本——
        # 该不该为此拦住请求，要拿真实流量判断，不能在这里替产品拍板。
        if not self.settings.feature_slot_clarify:
            return None
        question = slot_fusion.clarification(state.slots)
        if question is None:
            return None
        state.degraded.append("slot_clarify")
        return question

    # ---- 指代与统一理解结果（任务 2.4 / 2.5）----
    # ---- 复合问句拆分（任务 2.4）----
    async def _split_compound(self, state: ChatState, started: float) -> None:
        """把一句里的多个诉求拆成子问句。**本地小模型优先，规则兜底。永不抛异常。**

        为什么是本地小模型而不是云端：拆句是**每次请求都可能发生**的一步，它只需要
        "按标点切开、去掉连接词"这种低阶能力，付一次云端往返与计费不划算；而它又排在
        意图之后、检索之前，正好卡在首 token 预算里（§6.1），本地几十毫秒还能接受。
        格式错、超时、模型抽风都退回规则——规则拆得出来的句子，本来就不需要模型。

        为什么排在意图之后：拆出来的子问题要**一路一路去检索**，而检索档位（收窄哪些
        `doc_type`、要不要跑）已经由意图定死了。先拆再判意图，等于让拆分结果去猜检索范围。

        `state.compound` 一定被填（哪怕是"没拆"），`_build_understanding()` 只读它，
        不再自己调一次 `split()`——两处各拆一遍，日志里就会出现两种"拆出了几条"。
        """
        t0 = time.perf_counter()
        if not self.settings.feature_compound_llm:
            # 关掉开关 = 纯规则，与接模型之前**逐字节一致**。它是灰度回滚与 A/B 的分组开关
            state.compound = compound_query.split(state.query)
            state.latency_ms["compound"] = _ms(t0)
            return
        # 预算与理解层同一套算法：取小值，前面步骤慢了这一步就该少花
        budget_ms = min(
            self.settings.compound_timeout_ms,
            max(0, self.settings.total_budget_ms - _ms(started)),
        )
        result = await compound_query.split_with_model(
            state.query,
            self.llm_local,
            budget_ms=budget_ms,
            model=self.settings.llm_local_model or None,
        )
        state.latency_ms["compound"] = _ms(t0)
        state.compound = result
        state.degraded.extend(result.degraded)
        if result.is_complex:
            # 只记"真的拆开了"：绝大多数请求不是复合句，每次都记一行等于把日志淹掉
            log.info(
                "compound_split",
                request_id=state.request_id,
                source=result.source,
                sub_queries=len(result.sub_queries),
                dropped=len(result.dropped),
                latency_ms=state.latency_ms["compound"],
            )

    def _note_coreference(self, state: ChatState) -> None:
        """指代没解开就**记一笔**，绝不猜主语（任务 2.5 第 4 条）。

        判据是"问句里有真代词，且改写没落地"：`rule`（会话拼接）与 `llm`（模型改写）
        都算解开；`none` 与 `normalize` 都没有引入上文，代词原样留在检索问句里。

        记完**照样检索**——降级不等于放弃，只是让日志与指标能看见"这句话是裸着出去的"。
        编一个主语补上才是真故障：那会用一个没出现过的限定词去收窄检索范围，而且事后
        从响应体里完全看不出来（`rewritten` 看起来像一次正常改写）。
        """
        if not has_reference(state.query):
            return
        if state.rewrite_method in ("rule", "llm"):
            return
        state.degraded.append("coreference_unresolved")
        log.info(
            "coreference_unresolved",
            request_id=state.request_id,
            history_turns=len(state.history),
            rewrite_method=state.rewrite_method,
        )

    def _build_understanding(self, state: ChatState, clarify: str | None) -> UnderstandingResult:
        """把"这次请求理解出了什么"装成一个只读对象（任务 2.5）。

        为什么必须集中在一处装：`query / normalized / rewritten / slots / intent / profile`
        分别由六处写入，谁都不知道别人最后写成了什么。下游（响应体、日志、评估脚本）
        各自去拼，就会拼出三种"这次到底用哪句话检索"的说法。这里一次定稿，
        之后想知道就只读 `state.understanding`。

        三条口径写在这里，别改：

        - `rewritten` 与原话相同就**留空**：响应体里摆两份一样的问句，
          "改写到底生效没有"就再也看不出来了。
        - `success` 只看**整体性失败**（超时 / 异常 / JSON 坏）。没配 key 是配置状态，
          某个字段退回是部分成功——把它们也算失败，等于把"降级可用"说成"理解不可用"。
        - `degraded` 只留理解层的码：它是"理解这一环退化了什么"，
          不是整条链路的降级清单（那份在 `state.degraded` 里，别处已经暴露）。
        """
        # 拆分结论由 `_split_compound()` 在**意图之后、装配之前**产好（任务 2.4）。
        # 这里只读，不再自己拆一次：两处各拆一遍，日志里就会出现两种"拆出了几条"。
        # 兜底那半句是给"没走 `_prepare` 的调用方"（单测、将来的别的入口）留的，
        # 它们没机会跑模型，至少要拿到规则的结果
        split = state.compound or compound_query.split(state.query)
        return UnderstandingResult(
            query=state.query,
            normalized=state.normalized,
            rewritten=state.rewritten if state.rewritten != state.query else "",
            slots=state.slots,
            intent=state.intent,
            profile=state.profile,
            method=state.intent.method if state.intent is not None else "rule",
            degraded=tuple(
                d for d in dict.fromkeys(state.degraded) if d in _UNDERSTANDING_CODES
            ),
            success=not (set(state.degraded) & _UNDERSTANDING_HARD_FAILURES),
            confidence=state.intent.confidence if state.intent is not None else 0.0,
            is_complex=split.is_complex,
            # 拆的是**用户原话**而不是 `retrieval_query`：子问题的来源必须是用户自己写下的字。
            # 拿改写句去拆，来源校验就变成"相对于模型改写结果"，那等于让上游模型替下游作保
            sub_queries=split.sub_queries,
            # 有历史 + 有真代词才算追问。「它的图像有什么特点」在第一轮是断章取义，
            # 但那时 `history` 是空的，"是不是追问"本身就无从谈起
            is_follow_up=bool(state.history) and has_reference(state.query),
            ambiguity=clarify,
        )

    # ---- 会话与记忆（§7.9）----
    async def _load_session(self, state: ChatState) -> None:
        """读窗口 → 改写检索问句。失败一律退回原句，多轮只是增强项。"""
        state.rewritten = state.query
        if self.memory is None or not state.session_id:
            return
        t0 = time.perf_counter()
        try:
            turns = await self.memory.session.window(state.session_id)
            state.history = _recent_history(turns, self.settings)
            state.rewritten, method = rewrite_query(state.query, turns)
            if method != "none":
                state.rewrite_method = method
                log.info(
                    "query_rewritten",
                    request_id=state.request_id,
                    method=method,
                    q_len=len(state.query),
                    rewritten_len=len(state.rewritten),
                )
            # ---- 槽位重解析（任务 2.4）：历史里已经确认过的学科 ----
            # 只在问句**含指代词**时重解析：「那它的图像有什么特点」的"它"要靠上文；
            # 而「二次函数怎么求」这种自成一句的问句，拿上文的学科去收窄检索范围
            # 等于用一句不相干的话决定这一轮查哪一门——这是最典型的错误过滤。
            if has_reference(state.query):
                hint = _history_subject_hint(turns)
                if hint:
                    # 覆盖 `_new_state()` 里那份"只有参数与问句"的结论：`resolve()` 内部
                    # 仍然是参数 > 问句 > 历史的优先级，所以问句里点名的学科不会被历史盖掉，
                    # 只有"问句没说、参数没给"的那一档才会落到 `context`
                    state.slots = qa_slots.resolve(
                        state.query, param=state.subject_param, history_hint=hint
                    )
        except Exception as exc:  # 会话读失败不许影响本轮回答
            log.warning("session_load_failed", error=str(exc))
            state.degraded.append("session_unavailable")
        state.latency_ms["session"] = _ms(t0)

    async def _understand(self, state: ChatState, started: float) -> None:
        """归一化 + 一次大模型理解（查询改写 + 意图识别）。**永不抛异常。**

        两步的分工：

        1. **归一化**（确定性、微秒级）：NFKC + 空白折叠 + 重复标点折叠。它不进 `state.query`
           ——FAQ 的精确哈希与日志里的原话都依赖用户打的那串字，污染了就再也对不上。
        2. **大模型理解**（可变、百毫秒级）：改写检索问句 + 给意图。改写只是**建议**，
           出口校验不过就退回归一化结果；意图不合法就 `state.intent = None`，让 `_classify()`
           退回规则分类器。两项各自降级，互不影响。

        中间还夹着一步 **BERT 初始分类**（任务 2.2，`_bert_intent()`）：本地毫秒级，
        置信就给意图并让大模型少问一个字段，弃权就当没跑。它**不依赖大模型是否可用**。

        预算取 `UNDERSTANDING_TIMEOUT_MS` 与剩余总预算的**小值**：改写抢的是同一个
        `TOTAL_BUDGET_MS`，前面步骤慢了就该少花，而不是让生成没时间。
        """
        if not state.rewritten:
            state.rewritten = state.query

        normalized = query_processor.normalize(state.rewritten)
        # 留一份归一化结果给 `UnderstandingResult.normalized`：`rewritten` 后面可能还会被
        # 大模型覆盖，而"确定性归一化得到的那个串"是它被覆盖之前的唯一证据。
        # 两者都空时 `retrieval_query` 回落到 `query`——这就是"理解模型超时回退到原问题"
        # 的实现方式：**字段为空**，不是一条单独的分支
        state.normalized = normalized or state.rewritten
        if normalized and normalized != state.rewritten:
            state.rewritten = normalized
            # 只用归一化改过才算 `normalize`：多轮拼接（`rule`）已经把问句改写过一次，
            # 归一化只是在它上面收个尾，记成 `normalize` 会让 A/B 里"规则改写"的收益凭空消失
            if state.rewrite_method == "none":
                state.rewrite_method = "normalize"

        # ---- BERT 初始分类（任务 2.2）----
        # 位置：归一化之后、大模型之前。它在**大模型不可用**时同样要跑——没有 key
        # 的部署恰恰最需要这条本地分类器，所以不能塞进下面那个 `self.llm is None` 的早退之后。
        bert = await self._bert_intent(state)
        if bert is not None:
            state.intent = bert

        want_intent = self.settings.feature_intent and self.settings.feature_intent_llm
        want_rewrite = self.settings.feature_query_rewrite
        # 槽位的"模型补空"（任务 2.3）：章节名 / 知识点 / 任务参数。**与改写、意图共用一次调用**
        # ——三者落在流水线的同一个位置（FAQ 未命中之后、检索之前），拆成三次等于同一份
        # 上下文让模型读三遍。这四类里只有自由文本类走模型，过滤类永远走规则。
        want_slots = self.settings.feature_slot_llm
        # 检索策略（`direct` / `sub_query` / `hyde`）交给大模型选（2026-11-01）。
        # **只有"选哪种检索方式"这一件**：`no_retrieval` 与 `clarify` 两条决策仍由 planner
        # 按规则判——它们编码的是结构事实（这个意图不检索 / 指代解不开），不是偏好。
        want_strategy = self.settings.feature_planner_llm
        if state.intent is not None:
            # BERT 已经给了可靠结论：这次调用里**少问一个字段**（任务 2.2 的级联）。
            # 省的不只是 token——同一件事问两遍，两个模型给出不一致的答案时
            # 还得再定一条"听谁的"规则，而那条规则没有数据支撑。
            want_intent = False
        if not (want_intent or want_rewrite or want_slots or want_strategy) or self.llm is None:
            return

        t0 = time.perf_counter()
        budget_ms = min(
            self.settings.understanding_timeout_ms,
            max(0, self.settings.total_budget_ms - _ms(started)),
        )
        result = await llm_understanding.analyze(
            state.rewritten,
            self.llm,
            budget_ms=budget_ms,
            model=self.settings.llm_router_model or None,
            want_rewrite=want_rewrite,
            want_intent=want_intent,
            want_slots=want_slots,
            want_strategy=want_strategy,
        )
        state.latency_ms["understand"] = _ms(t0)
        state.degraded.extend(result.degraded)
        if result.rewritten:
            # 模型给的是自由文本，和归一化那一半**不是同一套写法**（它常把 `?` 写成 `？`）。
            # 这里再折一次，下游只认一种拼法：两条路径各自产出不同形态的"同一句话"，
            # 就是 §7.9.1 坐标收口要消灭的那类静默不一致。
            state.rewritten = query_processor.normalize(result.rewritten) or result.rewritten
            state.rewrite_method = "llm"
            log.info(
                "query_rewritten",
                request_id=state.request_id,
                method="llm",
                q_len=len(state.query),
                rewritten_len=len(state.rewritten),
            )
        if result.intent is not None and state.intent is None:
            # **先到者的结论优先**：BERT 在上面已经定过档时，大模型的意图不覆盖它。
            # 这次本来也不会问（上面把 `want_intent` 关了），这一行是把那条口径写死——
            # 谁将来把 `want_intent` 改回 True，也不会在这里静默变成"后者覆盖前者"。
            state.intent = result.intent
        # 槽位**原样存**，校验与合并交给 `_apply_slots()`（在 `_classify()` 之后）：
        # `TaskSlots.kind` 是意图的结论，那时候才拿得到。这里只做搬运，不动 `state.slots`。
        state.model_slots = result.slots
        # 检索策略建议同样只搬运、不下结论：可行性（有没有真拆出子问句、HyDE 能不能跑）
        # 由 `planner.build_plan()` 判，最终执行的是 `outcome.plan`。这里不校验，
        # 因为"认词"已经在 `analyze()` 里做过一次了，两处各认一遍迟早认成两种结果
        state.strategy_hint = result.strategy

    async def _bert_intent(self, state: ChatState) -> IntentResult | None:
        """BERT 初始分类（任务 2.2）。`None` = 没有可靠意见，交给大模型/规则。

        `FEATURE_INTENT=false` 时整段跳过：那个开关的语义是"不做意图路由"，
        不是"换个分类器"——关掉它就该连 BERT 也不跑，否则档位照样被收窄。

        弃权（`None`）**不记 `degraded`**：它是级联里的正常一步，下一段本来就要接手。
        把它记成降级，等于让一个每秒钟发生几十次的正常分支变成告警噪音。
        """
        if self.bert is None or not self.settings.feature_intent:
            return None
        t0 = time.perf_counter()
        try:
            result = await self.bert.classify(
                state.rewritten, timeout_ms=self.settings.intent_bert_timeout_ms
            )
        except Exception as exc:  # noqa: BLE001
            # 分类器自己的契约是"永不抛"，但它可以被替换（评测桩、将来的别的模型）。
            # 意图只是"挑一条先验"，任何异常都不许连坐整条请求（§7.7 降级表）。
            log.warning("bert_intent_failed", error=str(exc), request_id=state.request_id)
            result = None
        state.latency_ms["intent"] = _ms(t0)
        if result is None:
            return None
        log.info(
            "intent_bert_hit",
            request_id=state.request_id,
            label=result.label,
            confidence=result.confidence,
            latency_ms=state.latency_ms["intent"],
        )
        return result

    async def _recall(self, state: ChatState) -> MemoryContext | None:
        """记忆召回。`MemoryService` 自己带超时与 fail-open，这里只做接线。"""
        if self.memory is None or not state.user_id:
            return None
        t0 = time.perf_counter()
        context = await self.memory.recall(state.user_id, state.query, sid=state.session_id)
        state.latency_ms["memory"] = _ms(t0)
        if context.degraded:
            state.degraded.append("memory_degraded")
        return context

    def _persist_later(self, state: ChatState) -> None:
        """把"回写历史 + 抽记忆"挂到后台。

        **不能在响应路径上 await**：写库与抽取对用户没有任何可见价值，
        让它们占一次 MySQL 往返只会把 p95 拉长；而失败也不该影响已经产出的答案
        （§7.4 第 10 步"异步，不阻塞"）。
        """
        if self.memory is None or not state.session_id:
            return
        task = asyncio.create_task(self._persist(state))
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)

    async def _persist(self, state: ChatState) -> None:
        try:
            sid = state.session_id
            assert sid is not None
            await self.memory.session.append(sid, "user", state.query, user_id=state.user_id)
            if state.answer:
                await self.memory.session.append(
                    sid, "assistant", state.answer, user_id=state.user_id
                )
            # 只从**用户原话**抽事实：答案来自知识库，不是用户属性（memory/extractor.py）
            await self.memory.remember(
                state.user_id or "", sid, state.query, turn_index=len(state.history)
            )
        except Exception as exc:  # 后台任务抛异常会静默丢日志，必须自己接住
            log.warning("memory_persist_failed", error=str(exc), request_id=state.request_id)

    def _require_citation(self, state: ChatState, param: bool | None) -> bool:
        """参数为空时由档位决定（§7.7 作用点 6：`require_citation` 是策略，不是格式）。
        显式传参只用于评估与调试，不给前端开关——前端不该能绕过引用闸门。"""
        if param is not None:
            return param
        return state.profile.require_citation if state.profile else True

    def _llm_for(self, state: ChatState, *, record: bool = True) -> LLMClient:
        """按档位选生成引擎：**只认 `profile.llm_target` 字段**，不认标签字符串。

        闲聊档（`llm_target="local"`）交给本地小模型。选引擎是档位策略的一部分，
        与"检不检索"同一层级——所以它写在 `IntentProfile` 的数据表里（§7.7 硬规则 1），
        下游只读字段，不写 `if label == "闲聊"`。

        本地不可用时**回落主网关**，不把闲聊降成拒答：用户说的是"你好"，
        这台机器没装 Ollama 不该让他收到"服务暂时不可用"。回落事实记
        `llm_local_unavailable`（进日志与 `degraded`），不静默。

        `record=False` 只用于事后看"这一轮会由谁生成"（日志），不改变请求状态——
        否则一个连生成都没发生的请求（FAQ 高置信直返）也会被标成"本地降级"。
        """
        profile = state.profile
        if profile is None or profile.llm_target != "local":
            return self.llm
        local = self.llm_local
        if local is not None and getattr(local, "enabled", False):
            return local
        # 去重：`_prepare` 与 `_generate` 各调一次，不去重会在 `degraded` 里出现两遍
        if record and "llm_local_unavailable" not in state.degraded:
            state.degraded.append("llm_local_unavailable")
        return self.llm


    def _clarify(self, state: ChatState, suggestion: dict[str, Any], started: float) -> ChatState:
        state.source = "clarify"
        state.faq_suggestion = suggestion
        state.clarify = _CLARIFY.format(question=suggestion["question"])
        state.answer = state.clarify
        return self._finish(state, started)

    def _gate_clarify(
        self, state: ChatState, started: float, *, keep_citations: bool = False
    ) -> ChatState:
        """证据门 `ambiguous` 的追问：没有 top1 候选可问，只能请用户补充区分条件。

        与 `_clarify()` 分开是因为两者**问的东西不一样**：一个问"你是不是想问这条 FAQ"
        （有具体候选，用户点一下就能确认），一个问"能不能说细一点"（没有候选，
        只能靠用户补信息）。合成一个函数会让前端没法区分"要不要渲染候选卡片"。
        """
        state.source = "clarify"
        state.clarify = _GATE_CLARIFY.format(
            support=self.settings.chat_support_contact or "课程顾问"
        )
        state.answer = state.clarify
        if not keep_citations:
            state.citations = []
        return self._finish(state, started)

    def _slot_clarify(self, state: ChatState, question: str, started: float) -> ChatState:
        """槽位缺失的追问（任务 2.3）：缺了就没法执行，先问一句再干活。

        三个追问出口**问的是三件不同的事**，所以各占一个方法而不是共用一个：
        `_clarify()` 问"你是不是想问这条 FAQ"（有候选可点，要渲染卡片）、
        `_gate_clarify()` 问"能不能说细一点"（证据门 ambiguous，没有候选）、
        这里问"还差一个必要参数"。合成一个函数，前端就没法区分要不要渲染候选卡片。

        话术原样使用 `understanding/slots.py::clarification()` 给出的那一句——
        问什么由槽位缺什么决定，不在这一层再拼一遍文案。
        """
        state.source = "clarify"
        state.clarify = question
        state.answer = question
        return self._finish(state, started)

    def _fallback(
        self, state: ChatState, text: str, started: float, *, keep_citations: bool = False
    ) -> ChatState:
        state.source = "fallback"
        state.answer = text
        if not keep_citations:
            state.citations = []
        return self._finish(state, started)

    def _refusal(self, state: ChatState, started: float) -> ChatState:
        """超出教育领域：固定文案拒答（§7.7 `OUT_OF_SCOPE`）。

        **为什么不复用 `_fallback()`**：兜底与拒答对下游是三种不同的处置线（§7.8.5），
        混成一条会让指标说谎——拒答进了 `no_evidence`，"明天成都天气"就会被计进
        "知识库缺口"，教研照着这个数去为天气建语料，而那本来就不是我们要答的题。
        所以这里单独设 `source="refusal"`：`resolve_status()` 据此给出 `refused`，
        既不是 `no_evidence`（补语料），也不是 `error`（修服务）。
        """
        state.source = "refusal"
        state.error_code = "out_of_scope"
        state.answer = _OUT_OF_SCOPE
        # 拒答不带任何来源：给了引用就等于暗示"这些资料和你的问题有关"
        state.citations = []
        return self._finish(state, started)

    # ---- 业务权限（阶段 3）----
    def _permission_for(self, role: str | None) -> PermissionDecision | None:
        """Service 层的**再次**校验：调用方没传准入层结论时，这里按角色重算一遍。

        为什么要有这一层（用户要求"在实际业务 API、Service 和数据访问层再次校验"）：
        准入层是**唯一的正常入口**，但它是可以被绕过的——脚本、单测、将来某个新的接口
        都可能直接 `pipeline.ask(query)`。只要它给了角色，这里就按同一张表重新算一次，
        而不是信任"上游一定算过了"。`permissions_for()` 是纯函数，重算的代价是一次集合运算。

        `role is None` = **没有呈现任何身份**（进程内直调、旧调用方），此时返回 `None`，
        `_allowed()` 因此放行——与加这个门禁之前的行为一致。这与"匿名用户"不是一回事：
        真实的匿名终端用户经准入层进来，`role` 是 `ANONYMOUS`，照样受限。
        """
        if role is None:
            return None
        return permissions_for(role, enabled=self.settings.feature_permission)

    def _allowed(self, state: ChatState) -> bool:
        """这次请求的意图是否在当前角色的权限内。`True` 才允许继续往下走。

        放在 `_prepare()` 里**紧跟理解之后、FAQ 之前**：FAQ 也是一条检索路径，
        所以它必须排在权限之后（用户要求「先完成权限和身份校验，再进行 FAQ」）。

        三处刻意的宽松：
        - `permission is None`（没走准入层、也没给角色）→ 放行，见 `_permission_for()`；
        - 意图为空 / `unknown` / 低置信 → 放行：权限**绝不在猜出来的意图上下刀**，
          错分类把"能问的问题"拒掉，比错放行一次正常回答糟糕得多；
        - `FEATURE_PERMISSION=false` → 拒绝集为空 → 全部放行（紧急回滚开关）。
        """
        if state.permission is None:
            return True
        return state.permission.allows(state.intent.label if state.intent is not None else None)

    def _permission_denied(self, state: ChatState, started: float) -> ChatState:
        """权限不足：固定文案拒答，且**不进入任何检索**（FAQ、RAG 都不查）。

        与 `_refusal()` 共用 `source="refusal"`（都是确定的结论，不是故障、也不是知识库缺口），
        但 `error_code` 必须区分开：`out_of_scope` 是"这类问题不该答"，
        `permission_denied` 是"这个问题可以答，但这个账号没有这项功能"——
        两者的处置完全不同（前者补语料无用，后者是开通权限），混在一个错误码里就再也分不开。

        顺带说明为什么不进 `degraded` 之外的任何状态：被拒的请求不产生 `faq` / `retrieval`，
        所以 `latency_ms` 里只有 `total`，这在报表上正好能被识别成"没查库的请求"。
        """
        state.source = "refusal"
        state.error_code = "permission_denied"
        state.answer = _PERMISSION_DENIED
        state.citations = []
        state.degraded.append("permission_denied")
        log.info(
            "permission_denied",
            request_id=state.request_id,
            role=state.role,
            intent=state.intent.label if state.intent is not None else "-",
            intent_method=state.intent.method if state.intent is not None else "-",
            profile=state.profile.name if state.profile else "-",
        )
        return self._finish(state, started)

    def _finish(self, state: ChatState, started: float, *, stream: bool = False) -> ChatState:
        state.latency_ms["total"] = _ms(started)
        # 这一轮到底是谁生成的：`cloud:deepseek-chat` / `local:qwen3.5:0.8b`。
        # 不看这一行，"闲聊省了一次云端调用"只能靠推断，日志里看不出实际用的是哪台。
        engine = self._llm_for(state, record=False)
        log.info(
            "chat_done",
            request_id=state.request_id,
            source=state.source,
            status=state.resolve_status(),
            llm=f"{getattr(engine, 'target_name', '-')}:{getattr(engine, 'model', '-')}",
            tier=state.faq.tier if state.faq else "-",
            citations=len(state.citations),
            degraded=state.degraded or "-",
            latency_ms=state.latency_ms["total"],
            prompt=PROMPT_VERSION,
            stream=stream,
            # §7.7 作用点 8：意图进日志与指标，但不参与任何分支——下游只读档位
            intent=state.intent.label if state.intent else "-",
            profile=state.profile.name if state.profile else "-",
            memory_tokens=state.memory.tokens if state.memory else 0,
            # §7.2.7：证据门的四态与重试动作进日志，"为什么没答"才查得清
            gate=state.gate.status if state.gate else "-",
            retry=state.gate.retry_action if state.gate else "-",
            error_code=state.error_code or "-",
            strategy=state.plan.strategy.value if state.plan else "-",
        )
        self._persist_later(state)
        return state

    def _stream_finish(self, state: ChatState, started: float) -> list[dict[str, Any]]:
        """流式收尾：日志与回写挂载点与同步通道共用 `_finish`，只是帧不同。"""
        self._finish(state, started, stream=True)
        return _done_frames(state)

    def _no_evidence(self) -> str:
        return _NO_EVIDENCE.format(support=self.settings.chat_support_contact or "课程顾问")

    def _system_error(self) -> str:
        return _SYSTEM_ERROR.format(support=self.settings.chat_support_contact or "课程顾问")

    def system_error(self) -> str:
        """系统异常兜底话术的**唯一出处**：编排内的生成失败与接口层的未捕获异常共用。

        接口层本来可以自己拼一句，但那样同一个故障会在两处写出两种说法，
        用户看到的取决于异常恰好在哪一层被抓住——这不该影响措辞。
        """
        return self._system_error()

    def _conflict(self, decision: Any) -> str:
        """冲突话术：把"哪一根量对不上"直接说给用户，而不是含糊的"没找到"。"""
        units = [r.split(":", 1)[1] for r in (getattr(decision, "reasons", ()) or ()) if ":" in r]
        reasons = "、".join(dict.fromkeys(units)) or "数值"
        return _CONFLICT.format(
            reasons=reasons, support=self.settings.chat_support_contact or "课程顾问"
        )


def _new_state(
    query: str,
    *,
    subject: str | None,
    request_id: str | None,
    user_id: str = "",
    session_id: str | None = None,
    tenant_id: str = "",
    acl_tags: tuple[str, ...] = (),
    role: str | None = None,
    trace_id: str = "",
    kb_version: str = "",
    deadline: Any = None,
    permission: PermissionDecision | None = None,
) -> ChatState:
    return ChatState(
        request_id=request_id or new_request_id(),
        query=query,
        # 学科：参数优先，其次问句（`inferred/context/multi` 四条路由因此才在 /v1/chat 上可达）。
        # 这里**还不能**传 `history_hint`——历史在 `_load_session()` 里才读得到。所以此处
        # 只解析参数与问句，读完会话再由 `_load_session()` 带着 `history_hint` 重解析一次
        # （任务 2.4）。`FAQ` 也排在那之后，因此 FAQ 与检索看到的是同一份最终槽位
        slots=qa_slots.resolve(query, param=subject),
        # 重解析时要用它：参数优先级高于问句与历史，丢了它用户明确指定的学科就会被历史覆盖
        subject_param=subject,
        user_id=user_id,
        session_id=session_id,
        # 空值表示"接入层没给"，由 `_context()` 回落到 settings.tenant_id；
        # 这里不直接读 settings，是为了让 `ChatState` 保持纯数据（可序列化、可断言）
        tenant_id=tenant_id,
        acl_tags=tuple(acl_tags),
        # 身份/追踪/预算（阶段 1）：一律由准入层给，**这里不读请求体、也不读 settings**。
        # `role=None` 表示"没有呈现任何身份"（脚本、单测、旧调用方），落成 ANONYMOUS
        # 只是为了响应体里有个可读的取值；**它不等于"匿名用户"**，权限判定看的是下面的
        # `permission`（`None` = 不设限），而不是这个显示用的角色名
        role=role or ROLE_ANONYMOUS,
        trace_id=trace_id,
        kb_version=kb_version,
        deadline=deadline,
        # 业务权限（阶段 3）：准入层算好的结论，或者 `_permission_for()` 按角色重算的那份。
        # **只在这里落一次**，下游（FAQ / 检索 / 生成）都读它，不再自己拿 role 比表
        permission=permission,
        rewritten=query,
    )


def _recent_history(turns: list[Any], settings: Settings) -> list[tuple[str, str]]:
    """取最近的逐字轮次做 prompt 历史，按**剩余 token 预算**从旧到新砍。

    预算口径（§3.7 第 3 条）：系统 600 + 画像 300 + 摘要 400 + 原子 300 + 知识 6 000，
    逐字历史吃剩下的。这里只做一个保守上限，真正的裁剪点在 `build_messages` 的证据截断；
    两边都按上限算会让"历史 + 证据"反复互相挤，反而装不满。
    """
    budget = max(settings.mem_window_tokens - 8_000, 500)
    picked: list[tuple[str, str]] = []
    used = 0
    for turn in reversed(turns):
        cost = len(turn.content)
        if used + cost > budget:
            break
        picked.append((turn.role, turn.content))
        used += cost
    picked.reverse()
    return picked


def _history_subject_hint(turns: list[Any], *, limit: int = 6) -> str | None:
    """从最近几轮**用户**话里找已经确认过的学科，喂给 `qa_slots.resolve(history_hint=)`。

    这是在补一条**已经存在但线上不可达**的通路（任务 2.4）：`qa_slots.resolve` 早就支持
    `history_hint` 并会据此给出 `subject_source="context"`，只是编排层从来没传过。
    所以这里只做"把历史里现成的学科递过去"，不新造一套指代系统。

    三条口径：

    - **只看用户话**。助手答的是知识库内容，把它的学科当成用户的学科，
      等于让上一轮检索结果替用户决定这一轮查哪一门。
    - **倒序、先命中者胜**。越靠近当前的话题越可能是"它"指的东西。
    - **抽不到就返回 `None`**，不猜、不取默认值：`resolve()` 拿到 `None` 的行为就是
      "只看问句"，与没有历史完全一致。
    """
    seen = 0
    for turn in reversed(turns):
        if getattr(turn, "role", "") != "user":
            continue
        seen += 1
        if seen > limit:
            break
        subject = qa_slots.extract(getattr(turn, "content", "") or "").subject
        if subject:
            return subject
    return None


def _memory_text(state: ChatState) -> str:
    """`MemoryContext.render()` 的空安全包装：没有记忆时返回空串，模板据此跳过该段。"""
    return state.memory.render() if state.memory else ""


def _take_sentence(buffer: str) -> tuple[str | None, str]:
    """按句末标点切一句；还没到句末就返回 None。

    只在句末切，不在标点中间切——`[1]` 这种引用标记被拆到两个 chunk 时，
    切在半截会让 `align()` 把 `[1` 当成正文，引用就漏了。
    """
    for i, ch in enumerate(buffer):
        if ch in _SENTENCE_ENDS:
            return buffer[: i + 1], buffer[i + 1 :]
    return None, buffer


def _align_and_emit(
    sentence: str,
    evidences: list[Any],
    min_support: float,
    emitted: list[str],
    citations: dict[int, dict],
    out_of_range: list[int],
) -> tuple[list[dict[str, Any]], int]:
    """校验一句：通过就产出 delta 帧，不通过就不发。

    返回 `(帧列表, 被剔除的句数)`——第二项只用于计数，不影响控制流。
    """
    alignment = align(sentence, evidences, min_support=min_support)
    for n in alignment.out_of_range:
        if n not in out_of_range:
            out_of_range.append(n)
    if alignment.unsupported:
        return [], 1  # 无依据的结论直接不发给用户（§7.8 边界 3）
    text = alignment.text
    if not text.strip():
        return [], 0
    emitted.append(text)
    for citation in alignment.citations:
        citations.setdefault(citation["n"], citation)
    return [{"type": "delta", "text": text}], 0


def _frames_of(state: ChatState) -> list[dict[str, Any]]:
    """提前出口（faq / clarify / fallback）翻译成帧：meta → delta → citations → done。"""
    frames: list[dict[str, Any]] = [
        {
            "type": "meta",
            "request_id": state.request_id,
            "source": state.source,
            "tier": state.faq.tier if state.faq else "-",
            "cached": False,
            "session_id": state.session_id,
        }
    ]
    if state.answer:
        frames.append({"type": "delta", "text": state.answer})
    return frames + _done_frames(state)


def _done_frames(state: ChatState) -> list[dict[str, Any]]:
    """`citations` 与 `done` 两帧。

    `done` 里再带一份 `answer` 与 `citations`：它才是权威结果，客户端只读这一帧也能完整渲染，
    不必依赖前面收到的 delta（流式下"引用不过关"只能靠它覆盖）。
    """
    return [
        {"type": "citations", "citations": state.citations},
        {
            "type": "done",
            "request_id": state.request_id,
            # `session_id` 也必须出现在这里：`meta` 与 `done` 是两次独立的读点，
            # 只读 `done` 的客户端（重试、断线重连后重放）否则拿不到会话 id 继续多轮
            "session_id": state.session_id,
            # `role` 与 `trace_id` 同 `meta` 一样带在 `done` 上：`done` 是权威结果，
            # 只读这一帧的渲染器（以及排障时的用户截图）也要拿得到这两个键
            "role": state.role,
            "trace_id": state.trace_id,
            "source": state.source,
            "answer": state.answer,
            "citations": state.citations,
            "faq_id": state.faq_id,
            "faq_suggestion": state.faq_suggestion,
            "error_code": state.error_code,
            "gate": state.gate.as_dict() if state.gate else None,
            "plan": state.plan.as_dict() if state.plan else None,
            "plan_after_repair": (
                state.plan_after_repair.as_dict() if state.plan_after_repair else None
            ),
            "strategy": state.plan.strategy.value if state.plan else None,
            "degraded": state.degraded,
            "latency_ms": state.latency_ms,
            "usage": state.usage,
        },
    ]


def _suggestion(faq: FAQHit | None) -> dict[str, Any] | None:
    """mid 档的追问对象：只给候选**问题**，绝不给答案（§7.1.9 边界 2）。

    与外发响应里的 `faq_suggestion` 是同一件事：它进的是**给用户看的卡片**，
    卡片点开走 `GET /v1/faq/{id}`，所以这里一个字都不能带 `answer`——
    中置信的答案可以被模型"参考"，但绝不能被接口直接当成结论交付。
    """
    if faq is None or faq.tier != "mid":
        return None
    question = faq.matched_question
    if not question and faq.candidates:
        question = faq.candidates[0].question
    if not question:
        return None
    return {
        "faq_id": faq.faq_id or (faq.candidates[0].faq_id if faq.candidates else None),
        "question": question,
        "score": round(faq.score, 4),
        "candidates": [
            {"faq_id": c.faq_id, "question": c.question, "score": round(c.score, 4)}
            for c in faq.candidates[:3]
        ],
    }


def _faq_prompt_candidates(
    state: ChatState, settings: Settings
) -> tuple[FAQCandidate, ...]:
    """中置信档交给生成器的 FAQ 候选（问答对）；其余情况一律返回空元组。

    三条口径（§7.1.9 修订）：

    - **只有 mid 档**。high 已经逐字作答、压根不生成；low 连阈值都没过，
      把它塞进 prompt 等于用"没把握的匹配"污染一次本就证据不足的回答。
    - **不是证据**。候选只进 `render_faq_candidates()` 渲染的那条独立 system 消息，
      用的是「候选N：」而不是 `[n]`，所以永远进不了 `citation.align()` 的对齐表，
      也不会出现在响应的 `citations` 里。
    - **开关关掉就等于回到旧行为**（`FEATURE_FAQ_CANDIDATES=false`），
      这是一次干净的 A/B 分组，也是唯一的回滚手段（§5.3）。

    注入条数进日志：模型答差了要能分清"没给候选"和"给了候选但它带偏了"。
    """
    faq = state.faq
    if faq is None or faq.tier != "mid" or not faq.candidates:
        return ()
    if not settings.feature_faq_candidates:
        return ()
    with_answer = tuple(c for c in faq.candidates[:3] if (c.answer or "").strip())
    if not with_answer:
        return ()
    log.info(
        "faq_candidates_injected",
        request_id=state.request_id,
        count=len(with_answer),
        faq_ids=[c.faq_id for c in with_answer],
    )
    return with_answer


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
