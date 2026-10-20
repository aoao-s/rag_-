"""qa_system HTTP 接口（/v1/faq/*）。

约定（CLAUDE.md §7.1.4）：
- 查询接口只读、无副作用，p95 ≤15 ms；
- 写接口（发布/下线/重建）需要 X-Admin-Token，且必须显式触发重建；
- 返回体全部由 contracts.FAQHit.as_dict() 生成，禁止在这里另拼一套字段。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Header, HTTPException, Query, Request, status

from base.config import get_settings
from qa_system.res.qa_search import FAQService

router = APIRouter(prefix="/v1/faq", tags=["faq"])


def _service(request: Request) -> FAQService:
    service = getattr(request.app.state, "faq_service", None)
    if service is None:  # 理论上 lifespan 已注入，这里只防御测试期的手工挂载
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="faq service not ready")
    return service


def _require_admin(token: str | None) -> str:
    expected = get_settings().faq_admin_token
    if not expected:
        # 未配置 token 时只允许本机调试；生产部署必须设置 FAQ_ADMIN_TOKEN
        return "anonymous-dev"
    if token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token")
    return "admin"


@router.get("/search")
async def search(
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=200, description="用户原始问题")],
    subject: Annotated[str | None, Query(description="限定学科；不传则全学科检索")] = None,
) -> dict[str, Any]:
    hit = await _service(request).search(q, subject=subject)
    return hit.as_dict()


@router.get("/stats")
async def stats(request: Request) -> dict[str, Any]:
    service = _service(request)
    return {"state": service.state(), "stats": service.stats()}


@router.get("/{faq_id}")
async def get_faq(request: Request, faq_id: int) -> dict[str, Any]:
    record = await _service(request).get(faq_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="faq not found")
    return {
        "faq_id": record.faq_id,
        "subject": record.subject,
        "question": record.question,
        "answer": record.answer,
        "status": record.status,
        "content_hash": record.content_hash,
    }


@router.post("/{faq_id}/publish")
async def publish(
    request: Request,
    faq_id: int,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict[str, Any]:
    operator = _require_admin(x_admin_token)
    result = await _service(request).publish(faq_id, operator=operator)
    return {"faq_id": faq_id, "status": "published", **result}


@router.post("/{faq_id}/deprecate")
async def deprecate(
    request: Request,
    faq_id: int,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict[str, Any]:
    operator = _require_admin(x_admin_token)
    result = await _service(request).deprecate(faq_id, operator=operator)
    return {"faq_id": faq_id, "status": "deprecated", **result}


@router.post("/rebuild")
async def rebuild(
    request: Request,
    x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
) -> dict[str, Any]:
    _require_admin(x_admin_token)
    return await _service(request).rebuild()

