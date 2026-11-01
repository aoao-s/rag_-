"""rag_system 的 HTTP 接口（检索侧，不含生成）。

只暴露"检索 + 证据门"两个动作给编排层与调试台：
- `GET /v1/rag/search` 一次混合检索 → 父块证据 + 引用卡片 + 证据门判定；
- `GET /v1/rag/stats`  集合规模与索引状态，用于验证入库是否真的写进去了。

为什么不在这里生成答案：生成是编排层的事（选模型、控预算、拼记忆、流式回吐），
放在检索模块里会让 rag_system 依赖 LLM 网关，破坏"三模块平级"的边界。
"""

from __future__ import annotations

from dataclasses import replace

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from base.contracts import QueryContext, RetrievalPlan
from base.logger import get_logger
from rag_system.core import query_processor
from rag_system.core.planner import build_plan

log = get_logger(__name__)

router = APIRouter(prefix="/v1/rag", tags=["rag"])


def _rag(request: Request):
    return getattr(request.app.state, "rag", None)


def _plan_body(plan: RetrievalPlan) -> dict:
    """把计划原样摊开给调试台（阶段 4.1「结构化 RetrievalPlan」的可观测面）。

    这里刻意**不**做任何二次解释：不把 `decision` 翻译成中文、不给缺省的
    `reason` 补一句话。调试接口的价值就在于看到的与 `build_plan()` 返回的是同一份，
    一旦在输出层再加工，排查时就得同时怀疑两处逻辑。

    任务里带 `filters` 全部字段（含安全三项）：验收要求「所有策略都保留安全过滤条件」，
    只有把它们摆出来，才能一眼看出多路任务有没有哪一路漏了 `tenant_id` / ACL。
    """
    return {
        "decision": plan.decision,
        "strategy": plan.strategy.value,
        "reason": plan.reason,
        "clarification_reason": plan.clarification_reason,
        "final_n": plan.final_n,
        # `None` = 不表态（跟随 `RAG_RERANK_ENABLED`），与 `false` 不是一回事
        "rerank": plan.rerank,
        "tasks": [
            {
                "task_id": t.task_id,
                "query": t.query,
                "query_role": t.query_role,
                "strategy": t.strategy.value,
                "top_k": t.top_k,
                "note": t.note,
                "filters": {
                    "subjects": list(t.filters.subjects),
                    "doc_types": list(t.filters.doc_types),
                    "stages": list(t.filters.stages),
                    "grades": list(t.filters.grades),
                    "tenant_id": t.filters.tenant_id,
                    "acl_tags": list(t.filters.acl_tags),
                    "kb_version": t.filters.kb_version,
                },
            }
            for t in plan.tasks
        ],
    }


@router.get("/search")
async def search(
    request: Request,
    q: str = Query(..., min_length=1, description="用户问题原文"),
    top_k: int | None = Query(None, ge=1, le=100, description="混合召回条数"),
    final_n: int | None = Query(None, ge=1, le=50, description="最终保留的父块数"),
    subject: str | None = Query(None, description="按学科过滤，如 数学"),
    doc_type: str | None = Query(None, description="按资料类型过滤，逗号分隔，如 textbook,faq"),
    rerank: bool | None = Query(None, description="是否重排；默认跟随 RAG_RERANK_ENABLED"),
) -> JSONResponse:
    rag = _rag(request)
    if rag is None or not rag.get("ready"):
        return JSONResponse(status_code=503, content={"detail": "rag 未就绪", "reason": rag.get("reason") if rag else "startup"})

    doc_types = [t.strip() for t in doc_type.split(",") if t.strip()] if doc_type else None
    container = getattr(request.app.state, "container", None)
    settings = container.settings if container is not None else None
    ctx = QueryContext(
        request_id=getattr(request.state, "request_id", "") or "",
        top_k=top_k or (settings.rag_top_k if settings else 40),
        final_n=final_n or (settings.rag_final_n if settings else 6),
        subject=subject,
    )
    # 与 `/v1/chat` 同一口径的归一化（NFKC / 空白 / 重复标点）。不调 LLM：
    # 调试接口要能复现线上**检索**那一半的行为，两个入口各走一套归一化时，
    # 同一句问话在两个地方量出来的分数不一样（§7.2.7-G 的教训）
    normalized = query_processor.normalize(q)
    # 复合问句必须在这里先拆：`/v1/chat` 的子问题由理解层（`compound_query.split()`）
    # 产出后传给 planner，调试接口不拆的话，同一个问句在两个入口会得到两种计划
    # （这里只有"整句一路"，线上是三路），调试台展示的就不再是线上行为。
    # 拆分函数由组合根注入（`main.py` 把 `understanding.compound_query.split` 放进 container）：
    # 检索侧不 import 理解层，否则两者就成了互相依赖的平级模块（§5.1）。
    splitter = getattr(container, "compound_split", None)
    split = splitter(normalized) if splitter is not None else None
    plan = build_plan(
        normalized,
        ctx,
        subjects=(subject,) if subject else (),
        # 拆不出来就是空元组 = "没拆"，不是"不许拆"：planner 仍会看连接词自己标策略
        sub_queries=split.sub_queries if split is not None else (),
        # 调试接口没有会话，按"无历史"处理：于是带指代词的问句会如实落 `decision="clarify"`，
        # 而不是假装能回溯到上下文
        history_present=False,
        final_n=ctx.final_n,
        # 不传就是"跟随配置"：这里补一个 `else True` 会让 `rerank=false` 的配置与
        # `RAG_RERANK_ENABLED=auto`（无 CUDA → 不重排）同时失效，调试接口于是
        # 永远跑 CPU 重排（实测 227 s/次），和线上行为正好相反
        rerank=rerank,
    )
    if doc_types:
        # 调试台的 doc_type 直接下推为业务过滤（生产路径由意图档位给）
        plan = replace(
            plan,
            tasks=tuple(
                replace(t, filters=replace(t.filters, doc_types=tuple(doc_types)))
                for t in plan.tasks
            ),
        )
    # 与 `/v1/chat` 同一条管线：调试接口如果绕开 pipeline 自己调 retriever + gate，
    # 那它展示的就不再是线上行为（重试、过滤放宽都不会发生）
    outcome = await rag["pipeline"].run(normalized, ctx, plan=plan)
    result, decision = outcome.result, outcome.decision
    body = {
        "query": normalized,
        "original_query": q,
        "top_score": round(result.top_score, 4),
        "cost_ms": result.cost_ms,
        "raw_hits": result.raw_hits,
        "reranked": result.reranked,
        "degraded": list(dict.fromkeys([*result.degraded, *outcome.degraded])),
        "strategy": plan.strategy.value,
        # 结构化检索计划（阶段 4.1）。顶层 `strategy` 保留不动：它早就在这里，
        # 换掉会打断已有的调试习惯；`plan` 只是把同一个决定的来龙去脉摊开——
        # 为什么检/不检（decision）、为什么选这个策略（reason）、追问该用哪句话术
        # （clarification_reason）、每一路任务各自带着哪些过滤条件。
        "plan": _plan_body(plan),
        # 修复过才有这一项：`RagPipeline` 至多修一次（放宽过滤 / 换回原句重检），
        # 修完的过滤条件可能已经和 planner 当初给的不一样。不摆出来，
        # 就会把"放宽之后才检到的证据"当成"按原过滤检到的"，误判检索质量
        "plan_after_repair": (
            _plan_body(outcome.plan)
            if outcome.plan is not None and outcome.plan != plan
            else None
        ),
        "stages": outcome.stages,
        "attempts": outcome.attempts,
        "evidence": decision.as_dict(),
        "citations": result.citations(),
        "evidences": [
            {
                "parent_id": e.parent_id,
                "score": round(e.score, 4),
                "rerank_score": round(e.rerank_score, 4) if e.rerank_score is not None else None,
                "title": e.title,
                "subject": e.subject,
                "doc_type": e.doc_type,
                "heading": e.heading,
                "page": e.page,
                "child_hits": len(e.child_hits),
                "text": e.text,
            }
            for e in result.evidences
        ],
    }
    log.info(
        "rag_search_http",
        q_len=len(q),
        kept=len(result.evidences),
        top_score=round(result.top_score, 4),
        passed=decision.passed,
        gate_reason=decision.reason,
        gate_status=decision.status,
        retry_action=decision.retry_action,
        cost_ms=result.cost_ms,
    )
    return JSONResponse(content=body)


@router.get("/stats")
async def stats(request: Request) -> JSONResponse:
    rag = _rag(request)
    if rag is None or not rag.get("ready"):
        return JSONResponse(status_code=503, content={"detail": "rag 未就绪", "reason": rag.get("reason") if rag else "startup"})
    store = rag["retriever"].store
    body = store.stats()
    body["embedder"] = rag["embedder_name"]
    body["rerank_enabled"] = rag["rerank_enabled"]
    return JSONResponse(content=body)
