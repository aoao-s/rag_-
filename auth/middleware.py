from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from base.config import (
    AUTH_REQUIRED,
    AUTH_PUBLIC_PATHS,
    JWT_SECRET,
)

from auth.tokens import decode_access_token


class AuthenticationMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):

        path = request.url.path

        # 1. 匿名白名单
        if path in AUTH_PUBLIC_PATHS:
            return await call_next(request)

        # 2. 不允许认证关闭时意外开放生产环境
        if not AUTH_REQUIRED:
            return await call_next(request)

        # 3. JWT 密钥缺失，拒绝业务请求
        if not JWT_SECRET:
            return JSONResponse(
                status_code=503,
                content={
                    "code": "AUTH_NOT_CONFIGURED",
                    "message": "认证服务未正确配置"
                }
            )

        # 4. 获取 Authorization
        authorization = request.headers.get("Authorization", "")

        if not authorization.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content={
                    "code": "TOKEN_MISSING",
                    "message": "请先登录"
                }
            )

        token = authorization[7:].strip()

        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "code": "TOKEN_INVALID",
                    "message": "Token 不能为空"
                }
            )

        # 5. 验证 JWT
        try:
            payload = decode_access_token(token)

            user_id = payload.get("sub")

            if not user_id:
                raise ValueError("Token 缺少用户标识")

        except Exception:
            return JSONResponse(
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
                content={
                    "code": "TOKEN_INVALID",
                    "message": "登录已失效，请重新登录"
                }
            )

        # 6. 注入认证身份
        request.state.user_id = str(user_id)

        # 不接受客户端传入的 X-User-Id / X-User-Role
        # 角色、租户、ACL 后续必须从数据库读取

        return await call_next(request)