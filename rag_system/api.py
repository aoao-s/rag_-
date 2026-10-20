"""rag_system 的 HTTP 接口（检索侧，不含生成）。

只暴露"检索 + 证据门"两个动作给编排层与调试台：
- `GET /v1/rag/search` 一次混合检索 → 父块证据 + 引用卡片 + 证据门判定；
- `GET /v1/rag/stats`  集合规模与索引状态，用于验证入库是否真的写进去了。

为什么不在这里生成答案：生成是编排层的事（选模型、控预算、拼记忆、流式回吐），
放在检索模块里会让 rag_system 依赖 LLM 网关，破坏"三模块平级"的边界。
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from base.logger import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/v1/rag", tags=["rag"])


def _rag(request: Request):
    return getattr(request.app.state, "rag", None)


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
    result = await rag["retriever"].search(
        q, top_k=top_k, final_n=final_n, subject=subject, doc_types=doc_types, rerank=rerank
    )
    decision = rag["gate"](result)
    body = {
        "query": q,
        "top_score": round(result.top_score, 4),
        "cost_ms": result.cost_ms,
        "raw_hits": result.raw_hits,
        "reranked": result.reranked,
        "degraded": result.degraded,
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
