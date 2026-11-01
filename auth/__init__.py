"""认证模块（登录 / 登出 / 令牌）。

分层：`auth/` 是**平级业务包**，与 `qa_system` / `rag_system` / `memory` 同级，
只依赖 `base/`。它**不 import `orchestration/`**——编排层依赖它、不是反过来
（CLAUDE.md §5.1 的依赖方向）。

唯一的上行依赖是 `orchestration/admission.py` 从这里取 `decode_token`：
HS256 的验签实现只能有一份，否则"准入层验得过、认证层验不过"这类分叉
会变成一个只有攻击者发现的差异。
"""
