# EduRAG-Pro 阶段 1 设计：请求上下文、鉴权与 FAQ 入口

> 对应《EduRag-Pro项目详细执行计划.md》**阶段 1**（批次 **B2**）。
> 本文件是施工图：把该阶段 6 条任务落到具体文件、契约、判定顺序与测试。
> 改架构决策时先改本文件，再改代码（CLAUDE.md 开篇约定）。

---

## 1. 目标与验收

**计划原文目标**：统一身份和安全上下文，为权限、缓存和检索提供可靠基础。

**本阶段回答三个问题**：

1. 这个请求**合不合法**（能不能进来）；
2. 这个请求**是谁发起的**（身份、角色、租户、ACL、知识库版本可信且可追踪）；
3. 这次请求的**时间预算**从哪来（一个 `Deadline` 对象，而不是四处各自算）。

**验收（对齐计划，逐条可测）**：

| # | 验收项 | 判据 |
|---|---|---|
| 1 | 身份不能由客户端随意伪造 | 非白名单环境忽略 `X-User-Id`；JWT 验签失败 401，不回落匿名 |
| 2 | 租户、ACL、知识库版本可追踪 | `ChatState` / `QueryContext` / 日志 / 响应体四面同值，单测逐字节断言 |
| 3 | FAQ 高置信度和降级路径都有测试 | `tests/test_faq_entry.py` 覆盖 high 短路、mid 追问、low 交棒、FAQ 抛异常降级 |
| 4 | FAQ 异常不会拖垮主流程 | FAQ 抛异常 → `degraded=["faq_unavailable"]` 且继续走 RAG |
| 5 | 每个请求有 `request_id` 与 `trace_id` | 响应体、日志、`EduRAGError` 三处都能取到 |
| 6 | 总超时预算约束所有子阶段 | 全部子阶段从同一个 `Deadline` 取剩余预算 |
| 7 | 角色不能由客户端伪造 | `role` 只从 JWT / 网关读；请求体带 `role` → 400；非法值 → 401 |
| 8 | 不同权限的用户不共享不该看到的缓存 | FAQ 缓存键含 `tenant`；两个租户问同一句话命中**两条**缓存；C2 / C3 未实现时文档明写，不谎称"已有缓存体系" |

---

## 2. 现状盘点（真实代码，2026-10-30）

> 原则：代码里存在某个字段/分支，不等于它在真实请求链路上生效。下表每一条都读代码核过。

| 项 | 现状 | 位置 |
|---|---|---|
| 入口长度闸门 | pydantic `Field(min_length=1, max_length=200)` → **422** | `orchestration/api.py::ChatRequest` |
| 内部长度策略 | `MAX_QUERY_CHARS = 400` + **截断** | `rag_system/core/query_processor.py` |
| 截断是否生效 | **否**。`process()` 生产路径**零调用**，只有 `normalize()` 被用 | `chat.py:612,644`、`rag_system/api.py:58` |
| 空 query | POST → 422（pydantic）；GET 流式 → 手工 `HTTPException(422)` | `orchestration/api.py` |
| 错误体形状 | **三套**：pydantic 422 体 / `subject_invalid` 自定义 400 体 / `_internal_error` 的 23 字段 500 体 | `orchestration/api.py` |
| `request_id` | 有。`X-Request-Id` 或 `uuid4().hex`，中间件绑定并回写响应头 | `main.py::trace_middleware` |
| `trace_id` | **完全不存在**。`QueryContext.trace_id` 字段在，但从没人填；`EduRAGError.__init__` 收 `trace_id`，零调用方传 | 全仓 |
| `role` | **不存在**。`ChatState` / `QueryContext` / `ChatResponse` 都没有 | 全仓 |
| `kb_version` | `QueryContext` 有（默认 `v1`）；`ChatState` / `ChatResponse` **没有** | `base/contracts.py:270` |
| `tenant_id` / `acl_tags` | `ChatState` 有；来自 `X-Tenant-Id` / `X-ACL-Tags`，**白名单环境才信任** | `chat.py::_identity`、`base/config.py::is_trusted_env` |
| `session_id` | **无格式校验、无归属校验**。传别人的 sid 就能读到别人的历史 | `orchestration/api.py::_identity` |
| 有 sid 但无身份 | **静默丢弃** sid，不报错 | 同上 |
| 预算 | 无 `Deadline` 对象；`total_budget_ms - _ms(started)` 在 **4 处**各自计算 | `chat.py:273,330,414,626` |
| JWT | **无配置、无依赖**。`JWT_SECRET` 只在 CLAUDE.md §2.5 出现过 | `base/config.py`（grep `jwt` 无命中） |
| 限流 | **完全没有** | — |
| `SessionState` | 只有 `turns` / `summary` / `compressed`，**无 `user_id`** | `memory/session.py:46` |
| MySQL `request_log` | **表不存在**（只有 CLAUDE.md §6.4 的规划） | `scripts/init_db.py` |
| `chat_message` | 只有 `session_id`，无 `request_id` / `trace_id` 列 | `scripts/init_db.py` |
| 缓存层 | **只有 C1（FAQ 结论）**；C2 完整答案与 C3 检索结果**均不存在**（详见 §3.9.1） | `qa_system/res/qa_cache.py` |
| FAQ 缓存键 | `edurag:{env}:faq:v{N}:{scope}:{sha1}`——**无 `tenant`**；今天不泄漏仅因 `faq` 表本身无租户列 | `qa_system/res/qa_search.py:748` |

### 2.1 三个真实缺陷（不是"看起来不优雅"）

1. **`session_id` 可越权**：`_identity()` 直接 `return user_id, session_id or uuid4().hex`，既不校验格式也不校验归属。A 用户拿到 B 的 sid 就能读到 B 的对话历史，且 `chat_message` 会以 A 的身份写进 B 的会话。
2. **两套长度口径**：入口按 200 字拒绝（422），内部按 400 字截断。当前截断是死代码，一旦有人把 `process()` 接回主链路，就会出现"日志里是 400 字、FAQ 哈希是 400 字、用户以为问的是 600 字"的分裂。
3. **身份校验的点位分散**：pydantic 管长度、`_check_subject` 管学科、`_identity` 管身份、中间件管 `request_id`——四个地方各管一段，新增一条规则时不知道往哪加。

---

## 3. 设计

### 3.1 层的归属

新增 **`orchestration/admission.py`**，它是**唯一**的准入判定点：

```
HTTP / SSE  ──►  orchestration/api.py  ──►  admit()  ──►  RequestContext  ──►  ChatPipeline.ask()
                （只做 HTTP 转接）       （唯一校验/身份/预算）
```

- **为什么不在 `orchestration/api.py` 里堆校验**：该文件已 245 行且混着 SSE 帧逻辑，再塞 150 行校验会变成第二个大文件；`/v1/rag/search`、`/v1/memory/*` 将来复用时只能复制粘贴。
- **为什么不走 FastAPI `Depends` + 全局异常处理器**：异常处理器里还要再写一遍 23 字段响应体组装，比直接调 `admit()` 多一层而没有收益。
- **依赖方向不变**：`admission.py` 只 import `base/`（config / errors / contracts）与 `qa_system/res/qa_slots`（学科合法性）。**不 import `memory/`**——会话归属通过注入的 `owner_of` 回调查询（§7.0 硬边界 1）。

### 3.2 数据契约

```python
# orchestration/admission.py

@dataclass(frozen=True, slots=True)
class Deadline:
    """单次请求的时间预算。全链路只有一个，子阶段从它取剩余。"""
    started: float          # time.perf_counter()
    budget_ms: int

    @classmethod
    def start(cls, budget_ms: int) -> "Deadline": ...

    def elapsed_ms(self) -> int: ...
    def remaining_ms(self) -> int:            # max(0, budget - elapsed)
    def child(self, cap_ms: int) -> "Deadline":  # budget 取 min(budget, cap)，started 不变
    def expired(self) -> bool: ...

@dataclass(frozen=True, slots=True)
class Identity:
    user_id: str            # 匿名时为 ""
    role: str               # STUDENT | TEACHER | ADMIN | ANONYMOUS（见 §3.3.1）
    tenant_id: str
    acl_tags: tuple[str, ...]
    source: str             # jwt | header | anonymous

@dataclass(frozen=True, slots=True)
class RequestContext:       # 字段与顺序对齐计划 §5「任务 1.2」的示例
    request_id: str
    trace_id: str
    user_id: str
    tenant_id: str
    role: str               # STUDENT | TEACHER | ADMIN | ANONYMOUS
    session_id: str | None
    acl_tags: tuple[str, ...]
    kb_version: str
    # 以下是执行必需、计划示例里没有的四项
    deadline: Deadline
    query: str              # **strip 之后的用户原话**，未归一化
    subject: str | None
    selected_faq_id: int | None
    require_citation: bool | None

async def admit(
    payload: ChatRequest,
    request: Request,
    *,
    settings: Settings,
    owner_of: Callable[[str], Awaitable[str | None]] | None = None,
) -> RequestContext: ...
```

`owner_of` 是注入的会话归属查询（生产实现读 `SessionStore`），返回 `None` 表示"该会话尚未绑定"。

**`RequestContext` 与 `QueryContext` 的分工**：`RequestContext` 是**入口**的产物（含请求体字段与 deadline），`QueryContext` 是**检索**的产物（含 `top_k` / `final_n` / `stages` 等检索参数）。前者不进 `rag_system`，后者不进 `api.py`。不合并：合并会让检索层拿到 `subject`/`selected_faq_id` 这类与它无关的字段。

### 3.3 身份解析（三级，先严后宽）

```python
def resolve_identity(request, settings) -> Identity
```

| 优先级 | 条件 | 结果 | 失败行为 |
|---|---|---|---|
| 1 | `Authorization: Bearer <jwt>` | 验签 → `sub` / `role` / `tenant` / `acl` | **401 `invalid_token`**，不回落匿名 |
| 2 | `settings.jwt_secret` 为空 且 `is_trusted_env(settings)` | `X-User-Id` / `X-User-Role` / `X-Tenant-Id` / `X-ACL-Tags` | 头缺失 → 回落 3；`role` 非法 → 401 `invalid_role` |
| 3 | 兜底 | `user_id=""`、`role=ROLE_ANONYMOUS`、`tenant=settings.tenant_id`、`acl=()` | — |

四条硬规则：

- **坏 token 不许当匿名**。验签失败 / `exp` 过期 / `alg` 不符 → 401。回落匿名等于把"身份不可信"变成"身份是空的"，攻击者可以用坏 token 反复探测而不被审计。
- **非白名单环境只认 JWT**。`X-User-Id` 一类的头仅在 `dev/development/local/test/testing` 生效（`base/config.py::is_trusted_env` 已有白名单，直接复用，不复制）。
- **`role` 只从 token / 网关注入读**，请求体里**没有** `role` 字段（`ChatRequest` 加 `extra="forbid"` 后传了直接 400）。计划「角色必须由服务端认证体系提供，不能信任用户请求体中自行填写的 role」由此在入口处就成立，不依赖下游记得忽略。
- **非法 role 一律 401，不静默降级**。`SUPERADMIN` 这类不在枚举内的值意味着签发方与本服务版本不一致，宁可让调用方立刻报错，也不要把它映射成 `STUDENT` 悄悄放过去（那正是"降级成匿名"的另一种写法）。

#### 3.3.1 `role` 取值与 JWT claims 契约

```python
# base/labels.py —— 与 SUBJECTS / is_valid_subject 同一处放"唯一出处"
ROLES = ("STUDENT", "TEACHER", "ADMIN")   # 计划给出的三种业务角色
ROLE_ANONYMOUS = "ANONYMOUS"              # 未认证：**不是**业务角色，见下
def is_valid_role(value: str) -> bool: ...
```

- **大写、原样比较**。来源里写 `teacher` 会在入口 `.strip().upper()` 一次再比对——大小写不是权限语义，但存储与日志一律大写，避免 `student` / `Student` / `STUDENT` 三种写法在下游分支里各认一种（M0 年级三套写法踩过同款坑，§7.9.1）。
- **`ANONYMOUS` 是本设计新增的第四个值**（计划只列了三个），需要你确认。理由：现在 `/v1/chat` 允许无身份调用（FAQ / RAG 都不依赖身份），如果不给一个显式值，办法只剩"用空串当角色"——那是把所有 `role ==` 的分支都变成隐式的空值判断，阶段 3 的 `PermissionService` 会写成 `if not role: ...`，而漏写这行就是越权。**`ANONYMOUS` 恒定按最小权限处理，且它永远不会出现在 token 里**（出现即 401）。

| claim | 必填 | 映射到 | 缺失 / 非法 |
|---|---|---|---|
| `sub` | 是 | `user_id` | 401 `invalid_token` |
| `role` | 是 | `role`（`.strip().upper()`） | 不在 `ROLES` → 401 `invalid_role` |
| `tenant` | 否 | `tenant_id`，缺失回落 `settings.tenant_id` | 空串按缺失处理 |
| `acl` | 否 | `acl_tags`（`list[str]`，去空去重，≤20 项） | 非 `list` → 401 `invalid_token` |
| `exp` | 是 | — | 过期 → 401 `invalid_token` |
| `iat` / `nbf` | 否 | — | `leeway=60` 容忍时钟偏差 |

**`kb_version` 不从 token 读**，只取 `settings.kb_version`（部署配置）。索引版本是部署事实，不是身份属性；放进 token 会让同一个进程同时服务两个版本，缓存键与 alias 切换的语义随之失效。将来真要做按租户灰度索引，那是"租户 → 版本"的服务端映射，仍然不是客户端能自报的东西。

JWT 实现：新增依赖 **`pyjwt`**。理由（CLAUDE.md §5.5 要求说明）：手写 HS256 验签要自己处理 `alg: none` 混淆、`exp`/`nbf` 时钟偏差、base64url 补齐三个坑，每一个都是已知的鉴权漏洞来源；PyJWT 是标准库无替代方案的最小选择。只允许 `HS256`（`algorithms=["HS256"]` 显式传入，杜绝 `none` 与 RS/HS 混淆）。

### 3.4 校验规则（判定顺序固定，先便宜后昂贵）

| # | 规则 | 判定 | `error_code` | HTTP |
|---|---|---|---|---|
| 1 | 空 / 纯空白 | `not query.strip()` | `empty_query` | 400 |
| 2 | 超长 | `len(query) > settings.chat_max_query_chars` | `query_too_long` | 400 |
| 3 | 无可见字符（只有标点/控制符） | `not any(c.isalnum() for c in query)` | `empty_query` | 400 |
| 4 | 学科非法 | `qa_slots.from_param(s).subject_source == "invalid"` | `subject_invalid` | 400 |
| 5 | session 格式 | 不匹配 `^[0-9a-f]{32}$` | `invalid_session` | 400 |
| 6 | session 归属 | `owner_of(sid)` 非空且 ≠ `user_id` | `invalid_session` | 400 |
| 7 | 有 sid 无身份 | `session_id and not user_id` | `invalid_session` | 400 |

**顺序：先鉴权（§3.3，401），再下表（400）**。身份不成立时不继续做请求体语义校验——否则攻击者能用"这个 sid 格式对不对"这类差异化响应探测。`/v1/chat` 与 `/v1/chat/stream` 走同一个 `admit()`，两处不再各判一遍。

**框架层的 422 也必须变成 400**：`ChatRequest` 缺 `query`、多出未知字段、类型不对时，FastAPI 在进 `admit()` 之前就返回了 pydantic 的 422 体——这与"全部 400 + 统一 `ChatResponse`"冲突，前端要写两套解析，而 422 这条路径恰恰最难被测到。做法是在 `main.py` 注册 `RequestValidationError` 处理器，映射成 400 + `error_code="invalid_request"`，体里带 `request_id`（中间件已生成）。

**为什么规则 6 用 400 而不是 403**：403 等于向调用方确认"这个会话存在，只是不是你的"。与 §7.9.3 记忆模块用 404 而不 403 是同一条理由——"某个 sid 存在"本身就是信息。

**规则 7 是行为变更**：现状是静默丢弃 sid。静默降级会掩盖客户端 bug，且让"无主会话不落库"（§7.9 边界 3）变成一条谁也不知道自己被拦了的规则。

**判定顺序不能调换**：先判空再判长（空串长度必然合规）、先判格式再判归属（归属查询要打 Redis，别在格式都不对时去打）。

### 3.5 长度契约：三个概念，各一个数

| 概念 | 值 | 负责方 | 失败表现 |
|---|---|---|---|
| 传输层防炸 | body ≤ 64 KB | 中间件（新增） | 413 |
| 业务硬限 | `CHAT_MAX_QUERY_CHARS`，默认 **400** | `admit()` | 400 `query_too_long` |
| 模型上下文预算 | token 预算 | Context Builder | 裁剪，不报错 |

动作：

- `base/config.py` 增 `chat_max_query_chars: int = Field(default=400, alias="CHAT_MAX_QUERY_CHARS")`，`self_check()` 校验 `100 ≤ x ≤ 4000`。
- **删除** `rag_system/core/query_processor.MAX_QUERY_CHARS` 与 `process()` 的截断分支；`process()` 改为超限抛 `InvalidRequest(code="query_too_long")`。入口与内部从此读同一个配置值，不再各有一个常量。
- **`state.query` 全链路只用 strip 后的原话一串文本**：FAQ 精确哈希、日志、`qcov` 分母、citation snippet 的比对基准全部指向它。这是本条设计真正要消灭的东西——同一次请求里出现三种长度的问句。

### 3.6 两个 ID 的产生与贯穿

| id | 来源优先级 | 校验 |
|---|---|---|
| `request_id` | `X-Request-Id` → `uuid4().hex` | ≤64 且只含 `[A-Za-z0-9_.:-]`；不合法**重新生成**（不报错，但要记 `request_id_rejected`） |
| `trace_id` | `X-Trace-Id` → W3C `traceparent` 第 2 段（32 位 hex）→ **缺省 `= request_id`** | 不符合则同样回落到 `= request_id` |

7 个贯穿点与落点：

| # | 位置 | 现状 | 本阶段动作 | 阶段 |
|---|---|---|---|---|
| 1 | `ChatState` | 只有 `request_id` | 增 `trace_id: str = ""` | 1 |
| 2 | `QueryContext` | 有 `trace_id` 字段，**从没被填** | 由 `RequestContext` 填入 | 1 |
| 3 | 结构化日志 | `bind(request_id=, path=)` | 增 `bind(trace_id=)`；中间件从请求头取，`admit` 修正后的值再 bind 一次 | 1 |
| 4 | `EduRAGError` | 收 `trace_id` 但零调用方 | `admit()` 与 `ChatPipeline` 边界统一注入 | 1 |
| 5 | 下游请求头 | `llm_gateway` 的 httpx 不带 | 增 `X-Request-Id` / `X-Trace-Id` | 1 |
| 6 | MySQL `request_log` | **表不存在** | 建表 + 写入 | **1b** |
| 7 | 历史 `chat_message` | 无两列 | 加列 + 写入 | **1b** |

`ChatResponse` 增 `trace_id: str = ""`（23 → 24 字段）。理由：这是错误定位的唯一入口——用户报错截图里只有 `request_id` 时，跨服务链路就断了。

### 3.7 deadline 接线

| 接线点 | 现状 | 改为 |
|---|---|---|
| `chat.py:273` / `330` / `414` | `self.settings.total_budget_ms - _ms(started)` | `state.deadline.remaining_ms()` |
| `chat.py:626` | `min(timeout, remaining)` | `min(timeout, deadline.remaining_ms())` |
| 理解层（改写 + 意图） | 直接传 `understanding_timeout_ms` | `deadline.child(understanding_timeout_ms)` |

`Deadline` 随 `ChatState` 传递（`ChatState` 增 `deadline: Deadline | None = None`，`None` 时按 `total_budget_ms` 现建，兼容既有单测直接构造 `ChatState` 的写法）。

### 3.8 FAQ 入口（计划任务 5、6）

**置信分层现状即契约**（§7.1.9，已实现，本阶段**不改判定逻辑**）：

| FAQ 结果 | 处理 | 现状 |
|---|---|---|
| high（`exact`/`alias_exact`，或过①–③且 `margin ≥ FAQ_HIGH_MARGIN`） | 短路返回，**零次大模型调用** | ✅ `chat.py::_prepare` |
| mid | 跑 RAG；不过门 → 追问 top1 候选（`answer=None`） | ✅ |
| low | 交棒 RAG | ✅ |
| FAQ 抛异常 | `degraded=["faq_unavailable"]`，继续 | ✅（`except Exception` 收口，本阶段补**测试**） |

计划任务 6 的验收"FAQ 异常不会拖垮主流程"当前**没有测试**守（只有代码里的 `except`）。本阶段补 `tests/test_faq_entry.py`：让 `FAQService.search` 抛 `RuntimeError`，断言请求仍返回 `source="rag"` 且 `degraded` 含 `faq_unavailable`。

#### 3.8.1 三档动作（措辞对齐计划任务 1.3 的 FAQ 部分）

| 档 | 动作 | 返回给调用方的东西 |
|---|---|---|
| **高置信** | 直接返回 FAQ 答案（逐字，`source="faq"`），**跳过检索与生成** | `answer` + `faq_id` |
| **中置信** | 跑 RAG；候选**问题**进 `faq_suggestion`（query 扩展**未实现**，见下） | RAG 答案 + 引用 + `faq_suggestion{faq_id, question, candidates[≤3]}`；不过门则只给 `clarify` + 候选 |
| **低置信** | 继续正常理解 + RAG | RAG 答案 + 引用；不过门 → 兜底 + 转人工 |

**与计划的差距（需如实记录）**：计划说中置信"返回候选问题，作为后续**理解或生成**的参考"。实际代码里候选**只有一个去处**——`chat.py::_suggestion()` 把它塞进 `faq_suggestion` 回给前端（调用点三处：`209` clarify 出口、`397` / `498` 两条兜底出口）。`candidates` 在这三个位置之外**没有任何读者**：既不进检索的 query 扩展（CLAUDE.md §7.2.1 检索链第②步写着「FAQ top3 候选做 query 扩展」，**代码里没有这一步**），也不进 prompt。

所以计划这一句目前是**半落地**：候选能当"给用户的参考"，不能当"给系统的参考"。本阶段**不接**这条线（接上会改检索输入，属阶段 5；且没有 A/B 数据证明扩展有用），但要在文档与 CLAUDE.md §7.2.1 里把它标成"未实现"，而不是留着让人以为检索已经在用候选。

#### 3.8.2 三种"参考物"的边界（计划明文要求，当前只在代码注释里）

计划原文：**「中置信 FAQ 只能作为参考，不能直接作为可信证据，也不能把候选答案当成知识库原文。」**
落地成三条硬规则。**现状是有代码、没测试**：第一条由类型保证（`answer=None`），第二、三条只有 `_suggestion()` 的注释在说，没有任何断言守着——`tests/` 里搜不到一次针对"FAQ 答案进 prompt"或"FAQ 进 citations"的检查。本阶段补上（§6）。

| 物 | 能否进 prompt 的**知识**段 | 能否进 `citations` | 允许的用法 |
|---|---|---|---|
| 高置信 FAQ **答案** | **不进**（高置信根本不生成，直接逐字返回） | 不作为 citation chunk，以 `source="faq"` + `faq_id` 溯源 | 逐字返回 |
| 中/低置信 FAQ 候选**问题** | 不进（它是问句，不是知识） | **不能** | 只做 `faq_suggestion`（query 扩展**未实现**，见 §3.8.1） |
| 中置信候选**答案** | **严禁**（CLAUDE.md §5.6 第 9 条：模型会整段照抄，等于用引用的形式输出未经确认的答案） | **不能** | 无 |

- 中置信的 `FAQHit.answer` **恒为 `None`**（§7.1.9 边界 2 已实现），所以"候选答案进 prompt"在数据上就取不到——这是**靠类型不给**而不是靠调用方记得不传。
- 新增测试：构造一次 mid 命中，断言最终 prompt 文本里**不出现**该 FAQ 的 `answer` 原文；同时断言 `citations` 里没有任何一项的 `doc_id` 等于 `faq:{id}`。第二条防的是"把 FAQ 包成证据塞进引用卡片"这种绕过方式。

### 3.9 缓存分层（计划任务 1.3）

计划原文：**「缓存需要区分：FAQ 自身的缓存。完整问答结果缓存。RAG 检索结果缓存。不要把三者都叫 Redis 缓存，然后认为已经实现了完整缓存体系。」**

这句话在当前仓库是**准确的技术批评**：三层里只有一层存在，而它恰好被最常引用。

#### 3.9.1 现状盘点

| 层 | 缓存什么 | 现状 | 证据 |
|---|---|---|---|
| **C1 FAQ 结论缓存** | FAQ 命中**结论**（`faq_id` / `tier` / `score` / 候选），不含答案正文 | ✅ **已实现** | `qa_system/res/qa_cache.py`（`RedisFAQCache` / `InMemoryFAQCache` / `SingleFlight`）+ `qa_search.py::_cache_key` |
| **C2 完整问答结果缓存** | 一次 `/v1/chat` 的 `ChatResponse`（答案 + 引用 + 档位） | ❌ **不存在**。grep `:exact:` / `:sem:` 零命中 | `orchestration/` 与 `rag_system/` 无任何缓存层 |
| **C3 RAG 检索结果缓存** | 一次检索的 `RetrievalResult`（证据块 + 分数 + `stages`） | ❌ **不存在** | 同上；每请求都真跑 embedding + Milvus |

`base/config.py` 里只有 `faq_cache_*` 一组配置——**配置面也证实只有一层**。CLAUDE.md §6.3 写着"语义缓存阈值 0.95 / 精确 24 h / 语义 24 h"，那是**设计值**，代码里没有对应实现。

#### 3.9.2 三层的键、失效与归属

| | C1 FAQ 结论 | C2 完整答案 | C3 检索结果 |
|---|---|---|---|
| 写入时机 | FAQ 判定完成 | 一次回答**成功**且 `cacheable` 为真 | 一次检索过门（或虽未过门但结果可用） |
| 必含键项 | `env` / **`tenant`** / FAQ 语料版本 / 检索范围 / `sha1(normalized_query)`（无 `acl_scope`，见 §3.9.3） | `env` / `tenant` / `acl_scope` / `kb_version` / `model_ver` / `prompt_ver` / `sha1(normalized_query)` | `env` / `tenant` / `acl_scope` / `kb_version` / `sha1(retrieval_query)` + **`filters_sig`** |
| TTL | 正 1 h / 绝对 6 h；负 30 s（现有） | 24 h（§6.3 设计值） | **待定**，阶段 10 标定（不在这里拍数字） |
| 失效手段 | 语料版本 +1 即路由到新键；旧键靠 TTL 淘汰 | `kb_version` 变即整体失效；`prompt_ver` 变亦同 | `kb_version` 变即失效 |
| 实现阶段 | 已实现 | 计划阶段 10 | 计划阶段 10 |

**C3 为什么必须带 `filters_sig`**：同一句话在不同过滤下是两次不同的检索——"勾股定理"在 `subjects=("数学",)` 与全库下召回的证据集合不同。`filters_sig = sha1(tenant|acl|kb_version|sorted(subjects)|sorted(doc_types)|sorted(stages)|sorted(grades))`。少这一项，C3 会把"收窄到数学"的结果喂给"全库检索"的请求，而这批证据已经过了证据门——**它不会被下游任何一道闸门再挡一次**。这正是 §7.2.7-G 同一个教训的另一面。

**C1 不缓存答案正文**是既有设计里最值得保留的一条（`CacheEntry` 的 docstring 原文：「只存结论与溯源信息，答案正文由快照按 `faq_id` 取，避免正文被缓存后改不动」）。C2 如果要缓存正文，就必须自带 `model_ver` / `prompt_ver`——否则改一次提示词，所有用户继续拿到旧话术，而且**没有任何机制会让他们拿到新的**。

#### 3.9.3 共享缓存的键契约（先行成文，实现时照抄）

```
edurag:{env}:{layer}:{tenant}:{acl_scope}:{version}:{...层私有项...}:{sha1(query)}
          C1|C2|C3                      语料版本(C1) | kb_version(C2/C3)
```

- `layer` ∈ `faq` / `answer` / `retrieve`，**三层的键空间不重叠**，任何一层删库（如 `DEL edurag:*:answer:*`）不影响另两层。
- `acl_scope = sha1(",".join(sorted(acl_tags)))[:8]`；**ACL 为空时写 `da39a3ee`（`sha1("")` 前 8 位），不省略这一段**。省略会让字段数随请求变化，早晚有人用 `split(":")` 按位置取值，然后在空 ACL 的请求上取错字段。
- **C1 是这条形状的例外**：只带 `tenant`，**不带 `acl_scope`**（`faq` 表没有 `acl_tags` 列，加一个恒等字段只会制造"以为做了 ACL 隔离"的错觉）。C2 / C3 必须按上面的完整形状。
- `tenant` **不可省**（见 §3.9.4）。版本段对 C1 是 **FAQ 语料版本**（`snapshot.version`，自增 int），对 C2 / C3 是 `KB_VERSION`——两个版本号管两套语料，混用会让"RAG 索引没变、FAQ 改了"的场景下缓存不失效。
- `model_ver` / `prompt_ver` 只进 C2：C1 是判定结论（BM25 分数，与生成模型无关），C3 是检索结果（embedding 模型变则 `kb_version` 必变，因为向量要重算）。
- 统一走 `qa_cache.py::cache_key()`——**这个 helper 目前零调用方**，C1 是自己手拼的字符串（`qa_search.py:748`）。阶段 1 把 C1 改成调用它，从此"键长什么样"只有一处定义。C2 / C3 落地时同样调它，不另写一套。

#### 3.9.4 C1 的真实缺陷：键里没有租户

现状（`qa_system/res/qa_search.py:740-748`）：

```python
digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:20]
return f"{self._cache_prefix}:faq:v{version}:{scope}:{digest}"   # = edurag:{env}:faq:v{N}:{scope}:{sha1}
```

**没有 `tenant_id`，也没有 `acl_scope`** —— 两个租户问同一句话会命中同一条缓存。

**必须诚实标注它的严重程度**：今天这**不构成正在发生的泄漏**，因为 `faq` 表没有 `tenant_id` 列（`scripts/init_db.py:29-43`），`faq_alias` 的 `UNIQUE KEY uk_alias (alias)` 也是全局的——语料本身就是全租户共用的一份。所以现在这个键"碰巧正确"。

真正的风险在语料分租户的那一刻：计划的知识管理后台（阶段 2）一旦给 `faq` 加 `tenant_id`，这条键立刻变成**跨租户答案泄漏**，而且不会有任何报错——表现是"另一个学校的问题被答出来了"。这类问题的排障成本极高（要同时想到缓存与语料两层），所以**趁语料还是单租户时把键改对**，是成本最低的时机。

改动：`_cache_key(...)` 增 `tenant` 参数，`FAQService.search()` 增 `tenant_id: str = ""` 关键字参（空 → `settings.tenant_id`），键变为

```
edurag:{env}:faq:{tenant}:v{语料版本}:{检索范围}:{sha1(normalized_query)[:20]}
```

**ACL 不加入 C1**：`faq` 表没有 `acl_tags` 列，加一个恒为空的字段只会制造"以为过滤了"的错觉；等 FAQ 真的带 ACL 再按 §3.9.3 补齐，并同时处理 `uk_alias` 的全局唯一键（那才是分租户语料的下一个坑）。

#### 3.9.5 本阶段的边界

| 做 | 不做 |
|---|---|
| C1 键加 `tenant`；改走 `cache_key()` 统一构造 | C2 / C3 的任何实现（属计划阶段 10） |
| §3.9.3 的键契约写进文档，阶段 10 照抄 | 语义缓存（0.95 阈值）——没有 C2 就无从谈起 |
| C2 的 `cacheable` 闸门规则先写清（见下），避免阶段 10 又漏 | 给 `faq` 表加 `tenant_id` / `acl_tags`（知识管理，阶段 2） |

**`cacheable` 闸门**（C2 落地时唯一可写共享缓存的条件，现在先定死，参照 §7.7 作用点 7）：

1. `IntentProfile.cacheable` 为假 → 不写（`unknown` / `generic_clarify` 档：分类器自己说没听懂，不能把它的答案发给下一个人）；
2. 记忆参与过（`MemoryContext` 非空）→ 不写（C2 的键里没有 `user_id`，会把 A 的画像答案发给 B）；
3. 会话是多轮（有 `session_id` 且窗口非空）→ **不写**——这一条是我的判断，需要你确认：多轮答案的语义依赖前面几轮，而 C2 的键只含当前 `normalized_query`，两轮里同一句话的正确答案不同。若你也认为多轮答案不该进共享缓存，阶段 10 照此执行。

### 3.10 一处与计划的偏离（需要你确认或否决）

计划 §5 阶段 1 任务 5 写：**「FAQ 与 Query Understanding 可并行执行」**。

**本设计保持串行（FAQ 先跑、未命中才开始理解）**，理由是实测数据而不是偏好：

- 台账 763 题里 **404 题（53%）由 FAQ 高置信直接答掉**（CLAUDE.md §7.8.2）。并行意味着这一半请求白付一次大模型调用（`UNDERSTANDING_TIMEOUT_MS` 最多 800 ms 的往返），并且违背 CLAUDE.md §7.7 硬规则 5「FAQ 不等意图」。
- 串行在**命中**时比并行更快（省掉调用本身），在**未命中**时只多出不存在的启动开销——理解本来就要在检索之前完成。
- 计划原文是「**可**并行」而非「必须并行」，串行满足它要求的能力（两者都执行、都可降级），只是不做无用并行。

如果产品上确实要求并行（例如要求"未命中时首 token 更早"），改动点只有一个：`chat.py::_prepare` 里把 `_understand()` 提成 `asyncio.create_task`，FAQ 命中时 `task.cancel()`。请在评审时明确，我不会默认加。

---

## 4. 不改什么（本阶段明确不碰）

| 不碰 | 归属 |
|---|---|
| 限流（429） | 另开阶段（需 Redis 键族 + 话术 + fail-open） |
| 敏感词 / 越界拦截 | §7.7 前置节点，另开阶段 |
| `PermissionService.check()` 与角色权限矩阵 | 计划阶段 3（本阶段只把 `role` 带进来） |
| FAQ 三档阈值（`0.25 / 0.10 / 0.22`） | 已标定，不动 |
| 检索过滤的四项业务放宽逻辑 | §7.2.7，不动 |
| `/v1/rag/search`、`/v1/memory/*` 走准入层 | 复用留到后续；本阶段只覆盖 `/v1/chat` 与 `/v1/chat/stream` |
| `request_log` 建表、`chat_message` 加列 | **阶段 1b** |
| C2 完整答案缓存、C3 检索结果缓存、语义缓存 | 计划阶段 10（本阶段只把键契约与 `cacheable` 闸门写死） |
| 给 `faq` / `faq_alias` 加租户列 | 计划阶段 2（知识管理） |

---

## 5. 改动清单

| 文件 | 改动 |
|---|---|
| `orchestration/admission.py` | **新建**：`Deadline` / `Identity` / `RequestContext` / `resolve_identity()` / `admit()` |
| `orchestration/api.py` | `ChatRequest` 去掉 `max_length=200`、加 `model_config = ConfigDict(extra="forbid")`；两个端点改走 `admit()`；删 `_check_subject`（并入 `admit`）；`_identity` 删除（并入 `resolve_identity`）；新增 400 统一出口 `_reject(exc, ctx)` |
| `base/labels.py` | 新增 `ROLES` / `ROLE_ANONYMOUS` / `is_valid_role()` —— 与 `SUBJECTS` / `is_valid_subject` 同一处放"唯一出处" |
| `base/errors.py` | `InvalidRequest` 支持实例级 `code` 覆盖；新增 `Unauthorized`（`http_status=401`）承载 `invalid_token` / `invalid_role` |
| `base/contracts.py` | `ChatResponse` 增 `trace_id`；`QueryContext` 无字段变化（`trace_id` 已存在，只是开始被填） |
| `orchestration/state.py` | `ChatState` 增 `trace_id` / `role` / `kb_version` / `deadline`；`to_response()` 带出 `trace_id` |
| `orchestration/chat.py` | `_new_context()` 从 `RequestContext` 构造 `QueryContext`；4 处预算改读 `deadline`；理解层用 `deadline.child()` |
| `base/config.py` | 增 `chat_max_query_chars`、`jwt_secret`、`jwt_algorithms`(固定 HS256)、`max_request_bytes`、`jwt_leeway_s`(60)；`self_check()` 校验长度区间，并在 `jwt_secret` 为空且 `APP_ENV` 非白名单时**告警**（生产没配密钥等于所有 token 都 401） |
| `rag_system/core/query_processor.py` | 删 `MAX_QUERY_CHARS` 常量与截断分支，`process()` 超限抛错 |
| `qa_system/res/qa_search.py` | `_cache_key()` 增 `tenant` 参数并改为调用 `qa_cache.cache_key()`；`FAQService.search()` 增 `tenant_id: str = ""` 关键字参（空则取 `settings.tenant_id`），由编排层从 `RequestContext` 传入 |
| `qa_system/res/qa_cache.py` | `cache_key()` 从"零调用方"变为 C1 的唯一构造出口 |
| `rag_system/core/llm_gateway.py` | httpx 调用增 `X-Request-Id` / `X-Trace-Id` 头 |
| `main.py` | `trace_middleware` 增 `trace_id` 解析与 bind；新增 body 大小守卫（64 KB → 413）；注册 `RequestValidationError` 处理器（422 → 400 统一体，`error_code="invalid_request"`） |
| `memory/session.py` | `SessionState` 增 `owner_id: str = ""`；`bind_owner(sid, user_id) -> bool`（幂等：未绑定则写入，已绑定则比对） |
| `memory/store.py` | `session_summary` 读写带 `owner_id`（同 RAG M0 教训：`_COLS` / `insert` / `_update` / `_row_to_*` 四处同时带） |
| `requirements.txt` | 增 `pyjwt` |
| `.env.example` | 增 `CHAT_MAX_QUERY_CHARS` / `JWT_SECRET`（空值）/ `MAX_REQUEST_BYTES` |
| `CLAUDE.md` | §2.5 环境变量表、§3.1 接口约定、§6.1 预算表、§6.4 缓存键（三层键契约 + C1 加 `tenant`）、§7.2.1（候选 query 扩展标注"未实现"）、§7.4 时序、§7.8 响应体字段数、§8 变更记录 |

---

## 6. 测试清单

**新增 `tests/test_admission.py`（26 条，逐条对应上表）**

| 断言 | 为什么必须有 |
|---|---|
| 空 / 纯空白 / 只有标点 → 400 `empty_query` | 三条不同的入口现在各自 422/200 不一 |
| 401 字 → 400 `query_too_long`，400 字 → 通过 | 边界（正好等于上限必须通过） |
| `subject="物理课"` → 400 `subject_invalid` 且带合法学科列表 | 现有行为不变 |
| sid 非 32 位 hex → 400 `invalid_session` | 新增 |
| sid 格式正确但 `owner_of` 返回他人 → 400 `invalid_session` | **越权防线** |
| sid 格式正确、`owner_of` 返回 `None` → 通过（首用绑定） | 正常路径 |
| 有 sid 无 `user_id` → 400 `invalid_session` | 行为变更 |
| `owner_of` 抛异常 → **放行** + `degraded=["session_owner_unchecked"]` | fail-open，Redis 抖动不能让人问不了问题 |
| 请求体含未知字段（`"qurey"`）→ 400 | `extra="forbid"` |
| `X-Request-Id: ../../etc` → 重新生成，不落进日志 | 日志注入 |
| 无 `X-Trace-Id` / 无 `traceparent` → `trace_id == request_id` | 缺省规则 |
| `traceparent: 00-<32hex>-...` → 取第 2 段 | W3C 兼容 |
| 非白名单环境忽略 `X-User-Id` → `user_id=""` | 生产不可伪造 |
| 白名单环境读 `X-User-Id` / `X-User-Role` | 开发期可用 |
| `jwt_secret` 配置 + 有效 token → `source="jwt"` 且 `role` 来自 token | 主路径 |
| `role="teacher"`（小写）→ 归一为 `TEACHER` | 大小写不是权限语义，但落库/日志只允许一种写法 |
| `role="SUPERADMIN"` → 401 `invalid_role`，**不**降级成 `STUDENT` | 静默降级就是越权的另一种写法 |
| 无任何身份 → `role == "ANONYMOUS"` 且 `source == "anonymous"` | 匿名必须是显式值，不能是空串 |
| 请求体带 `"role":"ADMIN"` → 400（`extra="forbid"`） | 「不信任请求体中的 role」的落地断言 |
| `jwt_secret` 未配置 + 带 `Authorization` 头 → 忽略该头，走白名单/匿名 | 开发机不该因没配密钥就打不开服务 |
| `jwt_secret` 配置 + **过期** token → 401 `invalid_token` | 不回匿名 |
| `jwt_secret` 配置 + `alg: none` token → 401 | 算法混淆 |
| `Deadline.remaining_ms()` 单调递减且不为负 | 预算不能变成负数 |
| `Deadline.child(800)` 取 min 而非覆盖 | 子预算不能超过剩余 |
| 400 响应体是完整 `ChatResponse` 形状（24 字段） | 前端只写一份解析 |
| 缺 `query` 字段 → **400** `invalid_request`（不是 pydantic 的 422） | 框架层的拒绝也必须走同一形状 |

**新增 `tests/test_faq_entry.py`（约 7 条）**：high 短路（`llm.complete` 调用 0 次）/ mid 追问且 `answer is None` / low 走 RAG / `FAQService.search` 抛异常 → `source="rag"` + `degraded=["faq_unavailable"]` / `selected_faq_id` 无效 → 忽略参数并记 `faq_selected_invalid` / **mid 命中时最终 prompt 文本里不出现该 FAQ 的 `answer` 原文** / **`citations` 中不存在指向 FAQ 的条目**（后两条守 §3.8.2 的禁令，当前零覆盖）。

**新增 `tests/test_faq_cache.py`（约 5 条）**

| 断言 | 为什么必须有 |
|---|---|
| 两个 `tenant_id` 问同一句话 → 两次写入**不同 key** | §3.9.4 的那条键，跨租户必须不共用 |
| 同租户、同问法、同范围 → 两次调用 key 相同 | 改动不能把缓存打成恒未命中（性能回归） |
| 检索范围不同（`数学` vs 全库）→ key 不同 | 既有正确性，防改键时丢掉 |
| key 里**不含原始问法**（只有 20 位 sha1） | 学生问话不进 Redis 键空间 |
| `tenant_id=""` → 回落 `settings.tenant_id`，**不出现空段**（不存在 `::faq::v1:` 这种） | 空租户等于把"没传"变成"全租户共用一份" |

**新增 `tests/test_ids.py`（约 5 条）**：`request_id` 贯穿 `ChatState` → `QueryContext` → 日志 → 响应体；`trace_id` 同上；`EduRAGError.trace_id` 非空；LLM 网关收到的头含两个 id；`ChatResponse.trace_id` 与 `X-Trace-Id` 一致。

**必须修改的既有测试（3 条）**

| 文件 | 现状 | 改为 |
|---|---|---|
| `tests/test_chat_stream.py::test_http_stream_blank_query_is_422` | 断言 422 | 断言 400 + 统一体 |
| `tests/test_query_processor.py`（"超 400 字截断而非报错"） | 断言截断 | 断言抛 `InvalidRequest` |
| `tests/test_permissions_contract.py`（无 user_id + sid） | 断言 sid 被丢弃 | 断言 400 |

基线：**530 passed** → 预期 **≈573 passed**（`test_admission` +26 / `test_faq_entry` +7 / `test_faq_cache` +5 / `test_ids` +5）。

---

## 7. 阶段 1b 的范围（本阶段不做，但已定）

| 项 | 内容 |
|---|---|
| `request_log` 建表 | `request_id` PK、`trace_id`、`session_id`、`user_id`、`role`、`tenant_id`、`kb_version`、`query_hash`、`source`、`status`、`error_code`、`latency_json`、`created_at`；写入 fail-open（写失败只记日志，不能把请求打挂） |
| `chat_message` 加列 | `request_id` / `trace_id` + 索引；`scripts/init_db.py` 走幂等迁移（照 `migrate_memory()` 的写法） |
| 拒绝原因入表 | 400 也要落一行（`status="error"`），否则"用户说问不了"时查不到任何记录 |

---

## 8. 遗留与诚实标注

1. **`role`（含 `ANONYMOUS`）在本阶段只被携带，不被使用**。权限矩阵是计划阶段 3；现在把它塞进 `ChatState` 是为了让阶段 3 不必再改一次入口与本阶段的测试。**阶段 3 的铁律先写在这里**：`ANONYMOUS` 恒按最小权限处理，任何 `if role == "TEACHER"` 之类的新增分支都必须能回答"`ANONYMOUS` 走到这里会怎样"。
2. **`request_id` / `trace_id` 在 MySQL 两处落点缺席（阶段 1b）**，所以在 1b 完成前，"按 id 反查历史请求"只能靠日志。
3. **JWT 只是"能验"，不是"已接"**。生产网关是否签发 token、`JWT_SECRET` 如何轮换，不在本阶段范围；`jwt_secret` 为空时 JWT 分支直接跳过。
4. **限流仍然没有**。计划 §7.4 第 1 步写的"鉴权 + 限流 20 ms"目前只有鉴权一半。
5. **长度上限 400 是 FAQ 语料时代的基线**，对"生成教案"这类长指令可能偏紧；作为配置项放开，等阶段 2/3 有真实样本再定。
6. **本文档不构成检索质量结论**。本阶段不碰检索与证据门，τ / qcov / Recall 均沿用既有基线，也未复跑。
7. **三层缓存只做了一层**。C1（FAQ 结论）存在，C2（完整答案）与 C3（检索结果）**完全不存在**，属计划阶段 10。§3.9.2 里 C3 的 `filters_sig` 与 §3.9.3 的键形状是**契约不是现状**——阶段 10 落地时照抄，届时不要有人以为"我们有 Redis 缓存了"就等于三层都有。
8. **C1 加 `tenant` 是防御性改动，不是修复正在发生的泄漏**。今天 `faq` / `faq_alias` 两张表都没有租户列，语料全租户共用一份，所以旧键"碰巧正确"。真正的风险在阶段 2 给 `faq` 加 `tenant_id` 的那一刻——那时 `uk_alias` 的全局唯一键是下一个坑。这一段写在这里，免得将来把"当年加了 tenant 就安全了"当成结论。
9. **`cacheable` 三条件里的第三条（多轮答案不进共享缓存）请你在评审时确认**。前两条有 CLAUDE.md 依据（§7.7 作用点 7），第三条是我的判断：C2 的键只含 `normalized_query`，不含会话上下文，两轮里同一句话的正确答案可能不同。若你不同意，阶段 10 改这一行即可。
10. **FAQ 候选问题没接进检索（计划任务 1.3 只落地了一半）**。计划要求中置信候选成为"后续理解或生成的参考"，代码里它只回给前端（`faq_suggestion`），**不进 query 扩展、不进 prompt**——而 CLAUDE.md §7.2.1 的检索链写着「FAQ top3 候选做 query 扩展」。本阶段不接（属阶段 5），但 CLAUDE.md 那一步要改成"未实现"，否则读文档的人会以为检索已经在用候选。
