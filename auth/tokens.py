"""JWT（HS256）**唯一**的签发与验签实现。

为什么用标准库自己算而不用 `pyjwt`：本机 `ragpro` 环境没有装它（阶段 1 施工时实测），
而 HS256 的验签就是"HMAC-SHA256 比一次"。代价是**必须只用 HS256**：
`alg: none` 与 RS256 都在这里被显式拒绝（只认白名单里的一个值，不信 token 自己声明的算法族）。

两条与调用方约定死的口径：

1. **签发与验签同一个模块**。`orchestration/admission.py` 从这里 import `decode_token`，
   自己不再留一份。两份实现迟早会在一处修了、另一处忘改，而"准入层验得过、认证层
   验不过"这种差异，只会在有人构造特制 token 时才暴露。
2. **`kb_version` / `tenant_id` 之类决定"看哪份数据"的 claim 由服务端写**，
   `decode_token` 只管签名与时间，不管语义——它没有业务知识，就不该猜业务。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Final

# 只允许这一种算法。白名单而非黑名单：将来 JWT 生态再出现新的 `alg` 取值，
# 默认行为都是"不认识 → 拒绝"。
ALG: Final[str] = "HS256"


class TokenError(Exception):
    """令牌不可用（结构 / 签名 / 时间任一不过）。

    `code` 是给上层翻译用的稳定取值，**一律对应 HTTP 401**。它刻意不是
    `EduRAGError` 的子类：本模块不认识"业务错误码"这套词汇，把翻译留给调用方
    （`admission.py` 转 `AdmissionError`，`auth/service.py` 转 `AuthError`）。
    """

    def __init__(self, message: str, *, code: str = "invalid_token") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


# ---------------------------------------------------------------------------
# 签发
# ---------------------------------------------------------------------------


def encode_token(claims: dict[str, Any], *, secret: str, expires_in_s: int) -> str:
    """签一个 HS256 JWT。`iat` / `exp` 由本函数写，调用方不能自带——否则
    一条 claim 就能让令牌活到明年。其余 claim 原样使用。
    """
    if not secret:
        # 空密钥签名等于不签名（任何人都能伪造），所以宁可抛
        raise TokenError("服务端未配置签发密钥", code="auth_not_configured")
    ttl = max(1, int(expires_in_s))
    now = int(time.time())
    payload = {k: v for k, v in claims.items() if k not in ("iat", "exp")}
    payload["iat"] = now
    payload["exp"] = now + ttl

    header_seg = _b64encode(
        json.dumps({"alg": ALG, "typ": "JWT"}, separators=(",", ":"), ensure_ascii=False).encode()
    )
    payload_seg = _b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    )
    signing_input = f"{header_seg}.{payload_seg}".encode("ascii")
    signature = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_seg}.{payload_seg}.{_b64encode(signature)}"


def new_jti() -> str:
    """令牌唯一标识（32 位十六进制）。refresh 的吊销靠它——JWT 本身无法撤销，
    所以"签发过哪些、哪些已作废"必须落在库里，这个 id 就是那把钥匙。"""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# 验签
# ---------------------------------------------------------------------------


def decode_token(token: str, *, secret: str, leeway_s: int = 60) -> dict[str, Any]:
    """校验 HS256 JWT 并返回 payload。任何一步不过都抛 `TokenError`，**不返回部分结果**。

    `leeway_s` 给时钟漂移留余量；`exp` 必需，`nbf` / `iat` 可选但**出现就必须合法**。
    数值判断一律排除 `bool`：Python 里 `isinstance(True, int)` 为真，
    `{"exp": true}` 会被算成"1970 年过期"或"永不过期"取决于写法，两种都很糟。
    """
    if not secret:
        # 没配密钥却收到 token：这是配置问题，但对外只能是 401——
        # 降级为匿名等于"带凭证反而降权"，客户端会以为 token 生效了。
        raise TokenError("服务端未启用令牌校验", code="invalid_token")

    parts = token.split(".")
    if len(parts) != 3:
        raise TokenError("令牌不是三段式", code="invalid_token")
    header_seg, payload_seg, sig_seg = parts

    header = _json_segment(header_seg)
    payload = _json_segment(payload_seg)
    signature = _b64decode(sig_seg)
    if not isinstance(header, dict) or not isinstance(payload, dict):
        raise TokenError("令牌结构不正确", code="invalid_token")
    if header.get("alg") != ALG:
        raise TokenError("令牌算法不在允许范围内", code="invalid_token")

    expected = hmac.new(
        secret.encode("utf-8"),
        f"{header_seg}.{payload_seg}".encode("ascii", "ignore"),
        hashlib.sha256,
    ).digest()
    # `compare_digest` 而不是 `==`：后者在首个不同字节处返回，逐字节的耗时差
    # 足以让攻击者离线还原签名。
    if not hmac.compare_digest(expected, signature):
        raise TokenError("令牌签名不匹配", code="invalid_token")

    now = time.time()
    exp = payload.get("exp")
    if not _is_number(exp):
        raise TokenError("令牌缺少 exp", code="invalid_token")
    if now > float(exp) + leeway_s:
        raise TokenError("令牌已过期", code="invalid_token")

    nbf = payload.get("nbf")
    if nbf is not None:
        if not _is_number(nbf):
            raise TokenError("令牌的 nbf 不是数字", code="invalid_token")
        if now + leeway_s < float(nbf):
            raise TokenError("令牌尚未生效", code="invalid_token")

    iat = payload.get("iat")
    if iat is not None:
        if not _is_number(iat):
            raise TokenError("令牌的 iat 不是数字", code="invalid_token")
        if float(iat) > now + leeway_s:
            raise TokenError("令牌签发时间在未来", code="invalid_token")

    return payload


def peek_claims(token: str) -> dict[str, Any]:
    """**只解 payload、不验签**。仅用于日志与排障（"这个 token 声称是谁"），
    永远不能拿它做身份判定——签名没验过，内容就是用户自己写的。
    """
    parts = token.split(".") if token else []
    if len(parts) != 3:
        return {}
    payload = _json_segment(parts[1])
    return payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# base64url
# ---------------------------------------------------------------------------


def _b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64decode(seg: str) -> bytes:
    try:
        # urlsafe_b64decode 要求长度是 4 的倍数，JWT 规范允许去掉尾部 `=`。
        return base64.urlsafe_b64decode(seg + "=" * (-len(seg) % 4))
    except (binascii.Error, ValueError):
        raise TokenError("令牌不是合法 base64url", code="invalid_token") from None


def _json_segment(seg: str) -> Any:
    try:
        return json.loads(_b64decode(seg))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        raise TokenError("令牌段落不是合法 JSON", code="invalid_token") from None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
