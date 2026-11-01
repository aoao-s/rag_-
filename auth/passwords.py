"""口令哈希（Argon2id）。

**为什么是 argon2-cffi 而不是 stdlib**（§5.5 要求给理由）：`hashlib.scrypt` 是通用 KDF，
它没有 Argon2id 那套"参数随硬件升级而更新"的迁移机制，也没有 `check_needs_rehash`
这样的在线换参入口；交付方明确要求 "Argon2id 或 bcrypt"，而 bcrypt 的 72 字节截断
与 72 位口令的硬上限是中文本地口令场景里最常见的坑。argon2-cffi 是这条要求下
最小、最标准的选择（纯 C 后端，无额外系统依赖）。

**本模块的三个函数都是阻塞的**（Argon2 是刻意设计成算得慢的）。按 §5.2 的要求，
调用方必须放进 `asyncio.to_thread`——直接 `await` 会在一次登录里卡住整个事件循环，
表现是"一个人登录，所有人卡住 100 ms"。`auth/service.py` 统一走 `to_thread`。
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import Argon2Error

# 默认参数即 Argon2id（argon2-cffi 的 `PasswordHasher` 默认 type=Type.ID）。
# 不在这里写死 time_cost/memory_cost：让库跟着它的版本给默认值，
# 升级后靠 `needs_rehash()` 在用户下次登录时静默换参——参数一旦硬编码在代码里，
# 换它就变成一次全量重置口令的大工程。
_hasher = PasswordHasher()

# 口令长度口径。下限 8 是产品要求的最小声称；上限 128 挡的不是"更长的口令不安全"，
# 而是"一次登录塞 1 MB 进来，让 Argon2 在服务端算到天荒地老"——那是免费的 CPU DoS。
PASSWORD_MIN_CHARS = 8
PASSWORD_MAX_CHARS = 128

# 用户不存在时也要跑一次 Argon2，把"查无此人"与"口令错误"的耗时拉平。
# 不做这件事，登录接口就是一个用户名字典：响应快 80 ms 的那个用户名不存在。
_DUMMY_HASH = _hasher.hash("edurag-dummy-password-not-a-real-account")


def hash_password(raw: str) -> str:
    """算哈希。返回的字符串自带算法与参数（`$argon2id$v=19$m=...`），
    所以将来换参数不需要额外的字段来记"这条是用什么参数算的"。"""
    return _hasher.hash(raw)


def verify_password(stored_hash: str, raw: str) -> bool:
    """明文与库里那条哈希是否匹配。**任何异常都算不匹配**——
    哈希格式坏了（`InvalidHashError`）、参数不合法（`VerificationError`）、
    明文是 None（`TypeError`）都不该让 500 冒到接口上：那是配置问题，
    对外一律"用户名或密码错误"，把线索留给日志。"""
    try:
        return _hasher.verify(stored_hash, raw)
    except (Argon2Error, TypeError, ValueError):
        return False


def needs_rehash(stored_hash: str) -> bool:
    """库里这条哈希是不是用过期参数算的（库升级后自动跟进）。"""
    try:
        return _hasher.check_needs_rehash(stored_hash)
    except (Argon2Error, TypeError, ValueError):
        return True


def burn_equivalent_time(raw: str) -> None:
    """用户不存在时调用，制造与真实校验相当的耗时。

    只为一个目的：**不通过响应时间泄露"这个名字在不在库里"**。
    查无此人就立刻返回 401，是最常见的账号枚举漏洞——攻击者不需要看响应体，
    拿秒表就够了。
    """
    verify_password(_DUMMY_HASH, raw)


def validate_password(raw: str) -> str | None:
    """返回不合法原因；合法返回 `None`。**不校验复杂度**（大小写/数字/符号）：

    复杂度规则把用户推向 `Abc123!@#` 这类可预测的口令，收益远小于成本。
    真正的防线是"足够长 + 不被复用"，前者由这里管，后者由 Argon2 的慢来兜。
    """
    if len(raw) < PASSWORD_MIN_CHARS:
        return f"密码至少 {PASSWORD_MIN_CHARS} 位"
    if len(raw) > PASSWORD_MAX_CHARS:
        return f"密码不能超过 {PASSWORD_MAX_CHARS} 位"
    if not raw.strip():
        return "密码不能全为空白"
    return None
