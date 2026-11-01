"""记忆页接口（CLAUDE.md §7.3.4 / §3.7 第 7 条）。

动作：查看 / 筛选 / 编辑 / 删除 / 链视图 / 清空 / 导出，加上顾问只读授权（M4）。

**身份与目标要分开看，这是本文件最容易写错的地方**：

| 请求头 | 是什么 | 谁能决定 | 带不带权限 |
|---|---|---|---|
| `X-User-Id` | **调用方身份** | 白名单外的环境只认网关注入的 `request.state.user_id` | 带 |
| `X-Memory-Owner-Id` | **目标资源 id** | 调用方自报 | **不带**（能不能读由 `memory_grant` 决定） |

两者混成一个是本模块最危险的错法：把 `X-Memory-Owner-Id` 当身份，等于让任何人自报
"我是张三"就读张三的记忆。所以这里必须分成 `_user()`（身份）与 `_owner()`（目标）两件事，
而不是一个头兼两用。

`X-User-Id` 是**开发期身份**——接入层做鉴权后由网关注入、**不从外部透传**。
`base/config.py::is_trusted_env` 的白名单之外，客户端自报的这个头一律被忽略（不是"报错"，是"当没传"）：
与其指望上线前有人记得删掉那段透传代码，不如让它在生产配置下**默认不生效**。

删除的 `mode` 是必填而不是给默认值（§3.7 第 5 条）：`revive` 与 `purge_chain`
对用户是两个完全不同的意思（"记错了，恢复上一条" vs "这件事别再提了"），
默认任何一边都会让另一边变成事故。
"""

from __future__ import annotations

import time
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from base.config import is_trusted_env
from base.labels import GRADES, SUBJECTS, normalize_grade, normalize_subject
from base.logger import get_logger
from memory.service import MemoryAccessDenied, MemoryReadOnly, MemoryService

log = get_logger(__name__)

router = APIRouter(prefix="/v1/memory", tags=["memory"])


def _service(request: Request) -> MemoryService:
    service = getattr(request.app.state, "memory", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="memory not ready"
        )
    return service


def _trust_headers(request: Request) -> bool:
    """本环境是否信任客户端自报的身份头。规则本体在 `base/config.py::is_trusted_env`。

    这里**不复制白名单**：`memory/` 不许 import `orchestration/`（§7.0 硬边界 1），
    所以规则只能从 `base/` 拿；两份白名单迟早只改一处，然后两个接口在某个环境上分叉。
    """
    container = getattr(request.app.state, "container", None)
    return is_trusted_env(getattr(container, "settings", None))


def _user(request: Request, header: str | None) -> str:
    """调用方身份。空值一律 401，**不允许**退化成共享的 "anonymous"——
    那会让所有未登录用户共用一份长期记忆，是比"没有记忆"严重得多的隐私事故。"""
    state_user = (getattr(request.state, "user_id", "") or "").strip()
    header_user = (header or "").strip() if _trust_headers(request) else ""
    user = state_user or header_user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少可用的调用方身份（生产环境由网关注入 request.state.user_id）",
        )
    return user


def _owner(caller: str, header: str | None) -> str:
    """要操作的是**谁的**记忆，默认就是调用方自己。

    `X-Memory-Owner-Id` 只是资源 id，带不来任何权限——别人的记忆能不能读，
    由 `MemoryService.authorize_read` 查 `memory_grant` 决定，查不到就 404。
    """
    return (header or "").strip() or caller


def _self_only(caller: str, header: str | None) -> str:
    """授权关系的管理入口只许本人操作。

    顾问不能给自己发授权（那等于自己开门），管理员也不能——"管理员默认不看原文"
    这条约束在本版本靠"表里只有 owner→grantee 一个方向"保证（§3.7 第 7 条 / M4 不做项）。
    """
    owner = (header or "").strip()
    if owner and owner != caller:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="授权关系只能由本人查看与修改",
        )
    return caller


async def _call(coro: Any) -> Any:
    """把服务的越权异常翻成 HTTP：**无 grant → 404，有 grant 但写入 → 403**。

    404 不是"藏起来"这么简单——403 会告诉对方"这条记忆存在，只是你没权看"，
    而"某个学员有这条记忆"本身就是隐私。只有当对方已经拿到读权限（403 那条路）
    时，承认资源存在才不额外泄漏任何东西。
    """
    try:
        return await coro
    except MemoryAccessDenied as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="not found") from exc
    except MemoryReadOnly as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc


def _coordinate(kind: str, value: str | None, canonical: tuple[str, ...], normalize) -> str | None:
    """把学科/年级筛选值归一成**库里的写法**，认不出就 400。

    为什么不在服务层"先归一不上就原样查"：库里的值全是规范名（`九年级` / `道德与法治`），
    传 `初三` 原样查询会**命中 0 条且不报错**——用户会以为"我没有这条记忆"，
    而那正是这次 M0 要消灭的那类静默不一致。空串是合法查询（筛未归类的）。
    """
    if value is None:
        return None
    text = value.strip()
    if not text:
        return ""
    resolved = normalize(text)
    if resolved is None:
        raise HTTPException(
            status_code=400,
            detail=f"{kind} 不在词表内：{text}；可选：{' / '.join(canonical)}",
        )
    return resolved


@router.get("/atoms")
async def list_atoms(
    request: Request,
    fact_type: str | None = Query(None, description="preference/identity/constraint/decision/progress"),
    subject: str | None = Query(None, description="学科（接受别名，如 道法/政治）；空串=未归类"),
    grade: str | None = Query(None, description="年级（接受别名，如 初三 → 九年级）；空串=未归类"),
    q: str | None = Query(None, description="内容包含"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    include_dead: bool = Query(False, description="包含已被取代/删除的条目（下钻历史用）"),
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _user(request, x_user_id)
    body = await _call(
        _service(request).list_atoms(
            _owner(caller, x_memory_owner_id),
            fact_type=fact_type,
            subject=_coordinate("subject", subject, SUBJECTS, normalize_subject),
            grade=_coordinate("grade", grade, GRADES, normalize_grade),
            q=q,
            page=page,
            page_size=page_size,
            include_dead=include_dead,
            viewer=caller,
        )
    )
    return JSONResponse(content=body)


@router.patch("/atoms/{atom_id}")
async def patch_atom(
    atom_id: str,
    request: Request,
    payload: dict[str, Any],
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    content = payload.get("content")
    confidence = payload.get("confidence")
    if content is None and confidence is None:
        raise HTTPException(status_code=400, detail="至少要改 content 或 confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise HTTPException(status_code=400, detail="confidence 必须是 0~1 的数字")
    caller = _user(request, x_user_id)
    updated = await _call(
        _service(request).patch(
            _owner(caller, x_memory_owner_id),
            atom_id,
            content=str(content) if content is not None else None,
            confidence=float(confidence) if confidence is not None else None,
            actor=caller,
        )
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse(content=updated)


@router.delete("/atoms/{atom_id}")
async def delete_atom(
    atom_id: str,
    request: Request,
    mode: Literal["revive", "purge_chain"] = Query(..., description="revive=恢复前驱；purge_chain=整链失效"),
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _user(request, x_user_id)
    body = await _call(
        _service(request).forget(
            _owner(caller, x_memory_owner_id), atom_id, mode, actor=caller
        )
    )
    if not body.get("deleted"):
        raise HTTPException(status_code=404, detail=body.get("reason", "not found"))
    return JSONResponse(content=body)


@router.get("/chains/{atom_id}")
async def get_chain(
    atom_id: str,
    request: Request,
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _user(request, x_user_id)
    body = await _call(
        _service(request).chain(
            _owner(caller, x_memory_owner_id), atom_id, viewer=caller
        )
    )
    if not body.get("items"):
        raise HTTPException(status_code=404, detail="not found")
    return JSONResponse(content=body)


@router.post("/export")
async def export(
    request: Request,
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _user(request, x_user_id)
    body = await _call(
        _service(request).export(_owner(caller, x_memory_owner_id), viewer=caller)
    )
    log.info("memory_export", user_id=body["user_id"], viewer=caller, n=len(body["atoms"]))
    return JSONResponse(content=body)


@router.delete("/all")
async def clear_all(
    request: Request,
    confirm: bool = Query(False, description="二次确认，必须显式传 true"),
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    if not confirm:
        raise HTTPException(status_code=400, detail="清空全部记忆需要 confirm=true")
    caller = _user(request, x_user_id)
    deleted = await _call(
        _service(request).clear_all(_owner(caller, x_memory_owner_id), actor=caller)
    )
    return JSONResponse(content={"deleted": deleted})


# ---- L3 用户画像（§3.7 第 1 条）--------------------------------------------
class ProfileRequest(BaseModel):
    content: str = Field(
        default="", max_length=4000, description="画像正文；空串等于清除（取消人工画像）"
    )


@router.get("/profile")
async def get_profile(
    request: Request,
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    """`source` 告诉前端这一份是哪来的：`manual`（用户/顾问写的，落库）或
    `derived`（从身份类原子记忆推的，不落库）。两者合并显示，但只有前者能 PATCH/DELETE。"""
    caller = _user(request, x_user_id)
    body = await _call(
        _service(request).get_profile(_owner(caller, x_memory_owner_id), viewer=caller)
    )
    return JSONResponse(content=body)


@router.patch("/profile")
async def set_profile(
    payload: ProfileRequest,
    request: Request,
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _user(request, x_user_id)
    body = await _call(
        _service(request).set_profile(
            _owner(caller, x_memory_owner_id), payload.content, actor=caller
        )
    )
    return JSONResponse(content=body)


@router.delete("/profile")
async def clear_profile(
    request: Request,
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _user(request, x_user_id)
    body = await _call(
        _service(request).clear_profile(_owner(caller, x_memory_owner_id), actor=caller)
    )
    return JSONResponse(content=body)


# ---- 顾问只读授权（M4）-----------------------------------------------------
class GrantRequest(BaseModel):
    grantee_id: str = Field(min_length=1, max_length=32, description="被授权的顾问 id")
    expires_at: float | None = Field(
        default=None, description="到期时间（epoch 秒）；不传表示长期有效"
    )


@router.post("/grants")
async def create_grant(
    payload: GrantRequest,
    request: Request,
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _self_only(_user(request, x_user_id), x_memory_owner_id)
    if payload.expires_at is not None and payload.expires_at <= time.time():
        # 立刻过期的授权等于没授权，但它会出现在列表里让用户以为已经开好了
        raise HTTPException(status_code=400, detail="expires_at 必须是未来的时间")
    try:
        body = await _service(request).grant_read(
            caller, payload.grantee_id, expires_at=payload.expires_at, granted_by=caller
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return JSONResponse(content=body)


@router.delete("/grants/{grantee_id}")
async def delete_grant(
    grantee_id: str,
    request: Request,
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _self_only(_user(request, x_user_id), x_memory_owner_id)
    body = await _service(request).revoke_read(caller, grantee_id)
    if not body.get("revoked"):
        raise HTTPException(status_code=404, detail=body.get("reason", "not found"))
    return JSONResponse(content=body)


@router.get("/grants")
async def list_grants(
    request: Request,
    direction: Literal["owner", "grantee"] = Query(
        "owner", description="owner=我授权给了谁；grantee=我能看谁的记忆"
    ),
    x_user_id: str | None = Header(default=None),
    x_memory_owner_id: str | None = Header(default=None),
) -> JSONResponse:
    caller = _self_only(_user(request, x_user_id), x_memory_owner_id)
    body = await _service(request).list_grants(caller, direction=direction)
    return JSONResponse(content=body)
