# EduRAG-Pro 智能教育知识库系统

> 项目代号：EduRAG-Pro 1.41　|　形态：**交付项目**（非演示 Demo）　|　硬件：**有 GPU**　|　状态：架构设计完成，待实现
> 本文件是本仓库的"项目宪法"：给人和给 AI 编码助手共用。改动架构决策时，**先改本文件，再改代码**。

---

## 1. 项目概述

### 1.1 一句话定位

EduRAG-Pro 是面向职业教育机构的**智能教育知识库问答系统**：学员/顾问用自然语言提问，系统在"人工维护的 FAQ 库 + 机构自有教学文档"上做检索增强生成，给出**可溯源到文件与页码**的答案，答不了就明确说不知道并转人工。

### 1.2 与旧项目的关系

| | 旧 `F:\heima\RAG\Ragpro` | 本仓库 `F:\heima\RAG\RAGpro_智能教育系统` |
|---|---|---|
| 定位 | 学习基线（CLI 单进程，能跑通 demo） | **交付仓库**，所有新代码/文档只写在这里 |
| 编排 | `main.py` 里 if/else 串行调用 | LangGraph 状态机，节点级超时/重试/降级 |
| 模型 | 请求时懒加载，CPU 推理 | **启动时加载到 GPU 并预热**，请求路径零加载 |
| 接口 | `input()` 阻塞式 CLI | FastAPI + WebSocket/SSE 流式 |
| 质量 | 人工肉眼看 | 黄金集回归 + 线上反馈闭环 |

旧仓库可继续作为参考实现（`vector_store.py` 的父子块回填、`rag_pipeline.py` 的四种策略都值得复用），但**新文件一律不再写入旧目录**。

### 1.3 核心链路（在线）

```
用户提问
  → 接入层：鉴权 / 限流 / request_id / trace_id
  → 归一化 → 会话改写（指代消解）
  → L1 精确缓存 ──命中──> 直接返回
  → 并行：FAQ BM25 检索 ‖ 意图分类(5类) ‖ 槽位抽取
       └─ FAQ 高置信 ──> 返回 FAQ 答案（带 faq_id 来源）
  → L2 语义缓存 ──命中──> 直接返回
  → 策略规划（规则优先，LLM 兜底）：DIRECT / HyDE / SUB_QUERY / BACKTRACK
  → 并行多路混合检索（Milvus dense+sparse，k=40，标量过滤）
  → 父块去重 → 高亮命中片段 → cross-encoder 重排 top8 → token 预算裁剪到 4–6 父块
  → 证据门（top1 分数 < τ 或覆盖度不足 → 兜底 / 追问 / 转人工）
  → LLM 流式生成 → 引用对齐 → 忠实度抽查 → 写缓存 / 写历史 / 上报指标
```

### 1.4 交付级 SLO（写进验收标准，不是口号）

| 指标 | 目标 | 告警线 |
|---|---|---|
| 首 token p95 | ≤ 1.2 s | > 2 s 持续 5 min |
| 完整答案 p95 | ≤ 4 s | > 6 s 持续 5 min |
| 错误率（5xx + 超时兜底） | ≤ 0.5% | > 2% |
| 检索 Recall@10（黄金集） | ≥ 0.90 | < 0.85 |
| 答案忠实度（人工抽检） | ≥ 0.95 | < 0.90 |
| 缓存命中率 | ≥ 25% | < 10% |
| 转人工率 | ≤ 8% | > 15% |

### 1.5 用户与场景

| 角色 | 场景 | 系统要求 |
|---|---|---|
| 学员 | "AI 大模型课程学多久""学费多少""零基础能学吗" | 快（缓存+FAQ）、准（引用大纲/价目表） |
| 咨询顾问 | 多轮追问、对比课程、问就业 | 多轮改写、多文档融合、可转人工 |
| 教研/运营 | 维护 FAQ、上传新课件、看答不好的问题 | 知识管理后台、反馈闭环 |
| 运维 | 扩容、回滚、排障 | 探针、指标、trace、索引版本切换 |

---

## 2. 技术栈与环境

### 2.1 后端

| 组件 | 选型 | 说明 |
|---|---|---|
| 语言 | **Python 3.12+** | 旧代码用了 3.12 的 f-string 嵌套引号语法，禁止降级到 3.11 |
| Web | FastAPI + Uvicorn（`uvicorn[standard]`） | REST + WebSocket；生产用 gunicorn/uvicorn worker 多进程 |
| 编排 | LangGraph + langchain-core | 状态机 + 检查点；不用 LCEL 拼长链（不可观测） |
| 向量库 | pymilvus 2.4.x（Milvus 2.4.0 standalone） | dense HNSW + sparse inverted，父子块 schema |
| 缓存/会话 | redis-py（Redis 7+，compose 用 8.10） | 精确/语义缓存、会话、限流、singleflight 锁 |
| 关系库 | pymysql + dbutils 连接池（MySQL 8.0） | FAQ、会话落库、请求日志、反馈、审计 |
| 对象存储 | MinIO（S3 兼容） | 原始文档、解析中间件、导出报告 |
| 配置 | pydantic-settings + `.env` | 所有可调参数集中，禁止散落硬编码 |
| 可观测 | prometheus-client + OpenTelemetry + structlog | 指标 / trace / JSON 日志 |
| 重试熔断 | tenacity + 自研 circuit breaker | 仅幂等节点重试 |
| 测试 | pytest + pytest-asyncio + httpx | 单测 + 黄金集回归 |

### 2.2 模型（全部在启动时加载到 GPU）

| 用途 | 模型 | 形态 | 显存估算 |
|---|---|---|---|
| 稠密+稀疏向量 | BAAI/bge-m3（1024 维） | FP16，常驻 | ~2.5 GB |
| 精排 | BAAI/bge-reranker-v2-m3（568 M） | FP16，常驻，批量 | ~1.5 GB |
| 意图分类 | BERT-small 蒸馏（5 类 + other） | ONNX Runtime GPU/CPU 二选一 | <0.3 GB |
| 生成 | DeepSeek API（`deepseek-flash` 主 / `deepseek-v4-pro` 兜底） | 经 LLM 网关 | 0（外部） |

加载顺序（`bootstrap/startup.py`）：`embedding → reranker → intent → milvus.load_collection → FAQ 语料 → warmup 三条链路`。
**任一关键模型加载失败 → 进程 fail-fast 退出**（不要带着半残状态接流量）；非关键模型（intent 可用规则替代）失败则降级并告警。

### 2.3 前端

Vue 3 + Vite + TypeScript + Tailwind CSS。核心三件事：会话流式渲染（打字机）、引用卡片（点开定位到文件页码）、反馈按钮（点赞/点踩/纠错）。

### 2.4 目录结构（沿用旧项目 Ragpro 的分层惯例：根 main.py + base/ + 各业务包）

```
RAGpro_智能教育系统/
├─ CLAUDE.md                          # 本文件
├─ main.py                            # FastAPI 入口（python main.py 或 uvicorn main:app）
├─ requirements.txt                   # 依赖（与旧项目一致，用 requirements 不用 pyproject）
├─ pytest.ini                         # pythonpath=. / testpaths=tests
├─ .env.example                       # 环境变量模板（真实 .env 不入库）
├─ base/                              # 公共底座，谁都能用，不含业务语义
│  ├─ config.py                       # pydantic-settings，全部阈值/TTL 的唯一出处
│  ├─ logger.py                       # 结构化日志 + trace/request_id 上下文
│  ├─ labels.py                       # 意图标签 / 学科词表的唯一出处
│  ├─ contracts.py                    # FAQHit / Chunk / Evidence / MemoryContext
│  └─ db.py                           # MySQL 连接池（PooledDB）
├─ qa_system/                         # FAQ 精准问答（已实现）
│  ├─ api.py                          # /v1/faq/* 路由
│  └─ res/                            # 对标旧项目 qa_system/res
│     ├─ qa_normalize.py              # 归一化（全半角/标点/语气词/数字/口语/别名）
│     ├─ qa_corpus.py                 # 分学科 BM25 索引
│     ├─ qa_matcher.py                # 命中三条件（阈值/间隔/学科）
│     ├─ qa_db.py                     # 仓库（MySQL + 内存两实现）
│     ├─ qa_cache.py                  # Redis 连接与锁
│     └─ qa_search.py                 # FAQService 门面（装载/检索/发布/下线/重建/预热）
├─ rag_system/                        # 知识库检索与生成（PR3 待实现）
│  └─ core/                           # rag_pipeline / retriever / reranker / evidence_gate / citation
├─ docker/
│  ├─ docker-compose.yml              # mysql(3307) + redis(6380) + etcd/minio/milvus/attu；全部 pull_policy: never
│  ├─ offline/                        # 离线镜像：save.ps1 导出 / load.ps1 导入 / images/*.tar（不入库）
│  ├─ docker-compose.gpu.yml          # 待实现：应用 + 推理容器（CUDA 12.1）
│  ├─ Dockerfile                      # 待实现：多阶段构建，非 root 运行
│  └─ volumes/                        # 容器数据（不入库；down -v 或 -Purge 才清）
├─ scripts/                           # docker_up/down.ps1、建库建表、导入 FAQ、阈值标定、压测
├─ tests/                             # 单测、契约测试、降级演练
├─ datasets/                          # 项目方提供的语料（只读，不改不删）
├─ logs/                              # 运行日志（不入库）
└─ docs/
   ├─ EduRAG-Pro-1.41-企业级架构设计.md
   ├─ EduRAG-Pro-记忆系统设计.md          # 定稿 v1.0
   ├─ EduRAG-Pro-QAF与标签体系设计.md
   ├─ EduRAG-Pro-意图数据体检与训练方案.md
   ├─ qa_system-运行手册.md
   ├─ analysis/                        # 数据体检产出
   └─ diagrams/                        # 06 目标架构 / 07 记忆模块 + 绘制脚本
```

> 与旧项目的对应关系：`base/config.py`↔旧 `base/config.py`、`base/logger.py`↔旧 `base/logger.py`、
> `qa_system/res/qa_search.py`↔旧 `qa_search.py`、`qa_system/res/qa_db.py`↔旧 `qa_db.py`、
> `qa_system/res/qa_cache.py`↔旧 `qa_cache.py`、`rag_system/core/`↔旧 `rag_system/core/`。
> 差异只有一处：旧项目是 CLI（`python main.py` 进命令行问答），本项目 `main.py` 是可运行的 FastAPI 服务。

### 2.5 环境变量（`.env`，密钥只从环境注入）

| 变量 | 说明 |
|---|---|
| `APP_ENV` / `APP_VERSION` | `dev` / `staging` / `prod`；版本号用于日志与灰度 |
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE` | MySQL 连接；**同时也是 compose 的启动参数**，默认 3307/rag_sbc |
| `REDIS_HOST/PORT/PASSWORD/DB` | Redis 连接（compose 的 `requirepass` 取同一个 `REDIS_PASSWORD`），默认 6380 |
| `FAQ_THRESHOLD` / `FAQ_MARGIN` | FAQ 命中绝对分与领先幅度；已标定为 0.25 / 0.10，改 `FAQ_NORM_K` 必须重标 |
| `FAQ_CORPUS_CSV` | 无库模式：从该 CSV 装载 FAQ 语料；生产留空（MySQL 是真相源） |
| `MILVUS_HOST/PORT/DATABASE/COLLECTION` | Milvus 连接与集合名 |
| `MINIO_ENDPOINT/ACCESS_KEY/SECRET_KEY/BUCKET` | 原始文档存储 |
| `LLM_API_KEY` / `LLM_BASE_URL` | 生成模型网关（不写死、不进镜像） |
| `LLM_QA_MODEL_NAME` / `LLM_ROUTER_MODEL_NAME` | 生成 / 路由决策模型名 |
| `EMBEDDING_DEVICE` / `RERANK_DEVICE` | `cuda:0` / `cpu` |
| `EMBEDDING_BATCH_SIZE` / `RERANK_BATCH_SIZE` | GPU 批大小（OOM 时下调） |
| `KB_VERSION` | 当前生效索引版本，进缓存 key，切索引必改 |
| `RETRIEVAL_K` / `RERANK_TOP_N` / `FINAL_PARENT_N` / `RERANK_MAX_TOKENS` | 检索与重排基线参数（见 §6.3） |
| `EVIDENCE_THRESHOLD` | 证据门阈值 τ |
| `TOTAL_BUDGET_MS` | 单请求总预算（默认 8000） |
| `JWT_SECRET` / `JWT_EXPIRE_MIN` | 接入层鉴权 |
| `LOG_LEVEL` / `LOG_FILE_LEVEL` / `LOG_DIR` / `LOG_MAX_BYTES` / `LOG_BACKUP_COUNT` | 控制台级别 `INFO`、文件级别 `DEBUG`、`logs`、50 MB、10 份 |
| `LOG_QUERY_TEXT` | 是否把原始 query 写进日志；dev `true` 便于排查，**生产必须 `false`**（避免学生隐私落盘） |
| `FAQ_CACHE_ENABLED` / `FAQ_CACHE_TTL` | FAQ 二级缓存开关与正缓存 TTL（默认 `true` / 3600 s） |
| `FAQ_CACHE_MAX_TTL` / `FAQ_CACHE_SLIDE_RATIO` | 绝对寿命 21600 s；剩余寿命 ≤ TTL×0.5 才续命 |
| `FAQ_CACHE_NEGATIVE_TTL` | 未命中的负缓存 TTL（默认 30 s，带 top3 候选） |
| `MEM_WINDOW_TURNS` / `MEM_WINDOW_TOKENS` | 单窗口 40 轮 / 28 000 token，先到先触发压缩 |
| `MEM_COMPRESS_BATCH_TURNS` / `MEM_ROLLUP_TOKENS` | 每次压缩 10 轮；摘要滚动次级阈值 3 000 token |
| `MEM_SUMMARY_MAX_TOKENS` / `MEM_RECALL_TOP_K` / `MEM_MIN_CONFIDENCE` | 会话摘要 400 token；原子记忆召回 5 条；置信度门槛 0.6 |
| `MEM_RECALL_BUDGET_MS` / `MEM_PROFILE_MAX_TOKENS` / `MEM_ATOM_MAX_TOKENS` | 召回超时 120 ms；画像 300 / 原子记忆 300 token |
| `MEM_DEDUPE_SIM` / `MEM_ATOM_LIMIT_PER_USER` | 合并阈值 0.92；活跃记忆上限 500 条 |
| `MEM_EXTRACT_MODEL` | 抽取与摘要模型，默认 `deepseek-flash` |

> 启动自检：`LLM_API_KEY` 为空、`REDIS_PASSWORD` 与容器不一致、Milvus 集合不存在，三者任一命中直接打印可执行的修复提示并退出——不要留到第一个用户请求才 401。

### 2.6 硬件与运行环境

- GPU：≥ 8 GB 显存（BGE-M3 + Reranker FP16 常驻约 4–5 GB，留 2 GB 给 batch 峰值）；
- CPU：≥ 4 核（FAQ BM25 + 编排）；内存 ≥ 16 GB；
- 磁盘：模型 ~5 GB + 文档与索引；
- CUDA 12.1 / 驱动 535+ / torch 2.3+；Windows 开发、Linux 交付（路径不要出现中文，交付机若必须中文路径需在 Dockerfile 里显式 UTF-8 locale）。

---

## 3. 核心功能需求

标注：`P0` 必须交付，`P1` 交付前尽量完成，`P2` 下一版本。

### 3.1 接入与流式问答（P0）

1. `POST /v1/chat`（非流式）与 `GET /v1/ws/chat`（流式）双通道，同一编排内核。
2. 流式帧协议固定四类：`meta`（request_id/source/strategy/cached）、`delta`（增量文本）、`citations`（引用数组，可多次）、`done`（latency/usage/done 原因）。
3. 客户端断开 → 立即取消生成（停止计费），已生成部分落 `request_log`。
4. 同一 `request_id` 重发幂等，不重复计费、不重复写历史。
5. **验收**：WS 首帧 `meta` 在 100 ms 内到达；断开后 1 s 内 GPU 占用回落。

### 3.2 理解层（P0 意图/改写，P1 槽位）

1. 意图 5 类 + `other`：课程咨询 / 考试辅导 / 学习方法 / 升学规划 / 技能培训。
2. 置信度 < 0.6 → 不猜，走通用回答 + 追问澄清（**禁止**低置信硬路由到专业咨询，会把错误答案写进缓存）。
3. 会话改写：含指代词、长度 < 8 字、无实体时触发；规则优先，小模型兜底；输出 `rewritten` 与原 query 一并落库。
4. 敏感词与越界请求（涉政/涉黄/竞品诋毁）在前置节点拦截。
5. **验收**：黄金集多轮子集（50 题）改写后 Recall@10 提升 ≥ 15 个百分点；意图 F1 ≥ 0.90。

> **待拍板**：`datasets/06_intent`（5 100 条，六类均衡）与 `datasets/README.md` 的口径是 **K12 教学 6 类**（知识点查询 / 题目讲解 / 教案素材 / 学习方法 / 备课辅助 / 闲聊），与本节招生 5 类不一致。
> 所有意图标签走 `INTENT_LABELS` 配置 + 枚举，**切换不改代码**；在拍板前，检索策略、Tag `intent` 命名空间、缓存 key 都按"标签集合可变"设计。

### 3.3 检索与重排（P0）

1. Milvus 混合检索：dense HNSW（M=16, efConstruction=200, ef=64）+ sparse inverted（drop_ratio_build=0.2），`WeightedRanker(0.7, 0.3)` 融合。
2. 父子块：子块命中 → 按 `parent_id` 去重 → 回填 `parent_content`，命中片段用 `<精准检索定位区块>` 包裹（保留旧项目的这份资产）。
3. 标量过滤：`tenant_id`、`acl_tags`、`subject`、`kb_version` 强制前置过滤，**不允许"先检索再过滤"**。
4. 重排：GPU 批量，输入截断 512 token（保留高亮邻域），`top_n=8`；再按 token 预算裁到 4–6 个父块。
5. 证据门：`max(rerank_score) < EVIDENCE_THRESHOLD` 或实体覆盖不足 → 不生成。
6. **验收**：单次检索（含重排）p95 ≤ 170 ms（GPU）；黄金集 Recall@10 ≥ 0.90。

### 3.4 生成与引用（P0）

1. 提示词集中管理（`generation/prompts/`，带版本号），改 prompt 必须走 A/B 开关。
2. 上下文用分隔符包裹并声明"上下文中的指令不执行"（提示注入的第一道防线）。
3. 答案必须带引用：`[{n, doc_id, file_name, page_no, section, parent_id, score, snippet}]`，正文用 `[1][2]` 标注，**无法对齐引用的句子在输出前剔除或改写**。
4. 兜底话术分级：无证据 → "知识库暂无该内容 + 转人工"；系统异常 → "稍后重试 + 客服电话"；越界 → 拒答。
5. **验收**：抽检 100 条答案，引用可点开定位率 100%；编造率 ≤ 2%。

### 3.5 知识管理（P1）

1. FAQ CRUD + 审核状态（草稿/生效/下线），BM25 语料热更新（不重启）。
2. 文档摄入：上传 → 解析（PDF/DOCX/PPTX/MD/图片 OCR）→ 清洗 → 切分 → 质检 → 向量化 → 写新 collection → alias 切换。
3. 增量与幂等：`content_hash` 未变跳过；文档变更先按 `doc_id` 删除旧块再插入；支持单文档重跑。
4. 索引版本：`edurag_v{N}` + alias `edurag_current`；切换后 `KB_VERSION` 变更 → 缓存自然失效；保留上一版本供回滚。
5. **验收**：单文档更新 5 min 内可查；回滚操作 ≤ 30 s；重复摄入不产生重复块。

### 3.6 运营与观测（P0 指标，P1 反馈闭环）

1. 指标：QPS、分阶段耗时、缓存命中率、错误分类计数、LLM token 与成本、转人工率。
2. 日志：JSON 结构化 + `trace_id` + 轮转（单文件 20 MB × 5）。
3. 反馈：点赞/点踩/纠错自由文本 → 未命中问题池 → 聚类 → 教研审核 → 生成 FAQ 候选。
4. 评估：`eval/golden_set.jsonl`（≥300 题，覆盖 5 类意图 × 学科 × 难易），CI 里跑 Recall/MRR/忠实度，回归超阈值阻断合并。

### 3.7 个性化记忆（P0，定稿见 `docs/EduRAG-Pro-记忆系统设计.md`）

每个用户一个私有助手：`tenant_id + user_id` 唯一标识，记忆默认私有、共享需授权。

1. **四层记忆**：L0 工作记忆（当前 session 逐字，Redis 热 + MySQL 真相源）、L1 会话摘要（每 session 一份，≤400 token）、L2 原子记忆（跨会话事实/偏好，Milvus 向量召回）、L3 用户画像（稳定属性，常驻 300 token）。
2. **窗口双上限**：单会话 40 轮 **或** 28 000 token，先到先压；用户级不存对话窗口，禁止两级窗口并存。
3. **token 是硬预算**：单请求输入 ≤28 K（系统 600 + 画像 300 + 摘要 400 + 原子记忆 300 + 知识 6 000 + 逐字历史吃剩余），超出先压缩再装配。
4. **写入门控 + 异步**：仅含事实/偏好/决定/承诺/未完成任务的轮次才抽取；抽取（`deepseek-flash`，JSON Schema 强约束）、去重、冲突消解、摘要滚动全部走 Redis Stream 异步，p95 ≤1.5 s，不阻塞回答。
5. **冲突与遗忘**：新事实 supersede 旧事实（`valid_to` + `superseded_by` + `chain_root_id`）；删除链上最新一条必须二选一：`revive`（恢复上一条生效）或 `purge_chain`（整链失效），软删 30 天后物理清理。
6. **记忆页（P0）**：查看/筛选/编辑/删除/链视图/清空/导出，每条可下钻到来源轮次；删除 2 s 内对召回生效。
7. **权限**：学员本人读写删；顾问经 `memory_grant` 授权后只读（写审计）；管理员只读聚合、默认不看原文；不做团队记忆（v2）。
8. **召回与主检索并行**，p95 ≤120 ms，超时降级为"画像 + 逐字历史"；原子记忆只用 `valid_to is null 且 confidence ≥0.6`。
9. **验收**：MemRecall@5 ≥0.85（100 题记忆回归集）、事实准确率 ≥0.95、过期污染率 ≤2%、记忆 token 占比 ≤10%。

### 3.8 QAF 与标签体系（P1，见 `docs/EduRAG-Pro-QAF与标签体系设计.md`）

QAF = Question–Answer–Feedback 闭环数据集；Tag 是贯穿路由、检索、评测、权限、记忆的横向元数据。

1. **QA 生命周期**：`draft → pending → published → deprecated`，发布即建索引（BM25 内存语料 + Milvus `qa` collection），5 分钟内生效。
2. **命中判定**三条件：归一化分数 ≥0.25、top1−top2 ≥0.10、学科一致；未命中转 RAG 并携带 top3 候选（不要硬阈值 8.0 那种写法；0.25/0.10 由 `scripts/eval_faq.py calibrate` 在本语料 480 条同义问法上标定，precision 0.95）。
3. **反馈必须带 `request_id`**，标签固定枚举，日聚合后触发"QA 修订 / 未命中池 / 文档缺口"三类任务。
4. **Tag 九个命名空间**：intent / subject / course / level / scenario / audience / doc_type / risk / acl；打标 = AC 自动机 + 正则 + `deepseek-flash` 零样本（置信度 <0.6 转人工）。
5. **发布硬门槛**：`published` 的 QA 必须至少有一个 `subject` 标签，否则不发布（检索过滤要用）。

---

## 4. 开发与运行指南

> 以下命令在 v1 代码落地后生效；当前仓库只有设计文档与本文件。

### 4.1 首次准备

```bash
# 1) Python 3.12 虚拟环境
py -3.12 -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 2) 环境变量（端口/账号只写这一份，compose 与 python 共用）
copy .env.example .env            # 填入真实 key，禁止提交

# 3) 数据面：MySQL + Redis（离线可用，镜像由 docker/offline/save.ps1 导出）
powershell -File scripts/docker_up.ps1          # 起 mysql+redis 并等健康检查
powershell -File scripts/docker_up.ps1 -Full    # 含 etcd/minio/milvus/attu（PR3）
# 离线机器：先 docker/offline/load.ps1 导入镜像，再跑上面这条

# 4) 建库建表 + 导入 FAQ + 建 collection
python scripts/init_db.py
python scripts/init_milvus.py --collection edurag_v1
python scripts/import_faq.py --csv datasets/04_faq/edu_faq.csv --publish
python scripts/ingest.py --src data/docs --kb-version v1

# 5) 启动（启动期加载模型，见 4.2）
python main.py                     # 或 uvicorn main:app --host 0.0.0.0 --port 8000
```

端口基线：MySQL **3307**、Redis **6380**（刻意避开 3306/6379，本机旧项目容器占着），
全部由 `.env` 的 `MYSQL_PORT` / `REDIS_PORT` 决定，详见 `docs/docker-离线部署.md`。

### 4.2 启动期的三件事（本项目的硬性要求）

1. **加载**：BGE-M3 / Reranker / 意图模型 → 显存（FP16），打印每个模型的加载耗时与显存占用。
2. **预热**：跑三条真实链路（FAQ 命中 / 通用知识 / 专业咨询），把各自耗时写日志；**不预热就等于没加载**。
3. **就绪门控**：`/healthz` 进程存活即 200；`/readyz` 仅在"模型加载 + 预热 + Milvus load + Redis ping"全部通过后 200，否则 503。负载均衡只把流量给 ready 实例。

```bash
curl -s localhost:8000/healthz   # {"status":"ok","uptime_s":42}
curl -s localhost:8000/readyz    # {"status":"ready","models":["bge-m3","reranker","intent"],"kb_version":"v1"}
```

失败策略：关键模型加载失败 → 打印修复指引并 `exit(1)`；intent 模型失败 → 降级为规则分类并告警；Milvus 未就绪 → 最多重试 60 s，仍失败则退出。

### 4.3 前端

```bash
cd web
npm install
npm run dev        # http://127.0.0.1:5173，代理到 :8000
npm run build
```

### 4.4 验证与压测

```bash
# 端到端自测（含降级演练：杀掉 LLM key、断开 Redis）
pytest -q
python scripts/smoke.py --ws        # 打印 meta/delta/citations/done 四类帧样例
python eval/run_eval.py --set eval/golden_set.jsonl --report out/report.md
python scripts/loadtest.py --qps 20 --duration 120s
```

### 4.5 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| 启动卡在"加载模型" | 模型路径错 / 下载中 / 显存不足 | 看启动日志的显存行；下调 `EMBEDDING_BATCH_SIZE` |
| `/readyz` 一直 503 | 预热链路失败（Milvus 未 load / Redis 密码错） | 查 `readyz` 返回体里的 `failed_stage` |
| `1045 Access denied` | `.env` 密码与容器不一致 | 对齐 `MYSQL_PASSWORD` 后重启应用 |
| `NOAUTH` / Redis 连接抖动 | compose `requirepass` 与 `.env` 不一致 | 统一密码，检查连接池上限 |
| Milvus 连不上 | etcd/minio 未起或端口占用 | `docker compose ps` + `logs milvus-standalone` |
| 首次请求偶发 1–3 s 抖动 | 预热不充分（只加载未前向） | 预热必须跑完整链路；检查 CUDA context 初始化日志 |
| CUDA OOM | 重排批过大 / 并发高 | 降 batch、降 `RETRIEVAL_K`、开 `torch.cuda.set_per_process_memory_fraction` |
| 401 Authentication Fails | IDE/系统环境变量里的旧 key 覆盖 `.env` | 代码统一读 `LLM_API_KEY`，启动自检会打印告警 |

---

## 5. 给 Claude 的开发指令与编码规范

### 5.1 分层与依赖方向（强约束）

```
main.py(api) → 编排层 → {理解, 检索, 生成, 知识} → base/ → 外部服务
```

- **禁止反向依赖**：`base/` 不许 import 业务包；`qa_system` 与 `rag_system` 互不 import；节点之间只通过 `ChatState` 通信。
- **禁止跨层访问数据库**：只有 `base/db.py`、`qa_system/res/qa_cache.py` 能建连接，业务层拿 repository。
- 新增一个能力时，先问：它属于理解、检索、生成、知识还是基础设施？放错层的代码一律打回。

### 5.2 Python 编码规范

- 全量类型注解，`mypy` 基本严格；禁止 `Any` 泛滥（外部 SDK 边界可用一次并就地收敛）。
- 一律 `async def`；**禁止在协程里做阻塞调用**（`requests`、`time.sleep`、同步 DB 驱动、`torch` 前向都要走 `run_in_executor` 或专用 GPU 线程）。
- 异常：定义业务异常体系（`BudgetExceeded / UpstreamUnavailable / EmptyRetrieval / PolicyViolation / InternalError`），**禁止裸 `except Exception: pass`**；异常必须带 `trace_id` 与节点名。
- 日志格式固定为**人可读单行**（与旧项目 Ragpro 对齐，便于沿用既有 grep 习惯）：

  ```
  2026-10-20 19:21:30.840 [INFO] [15624:MainThread] EduRAG.qa_system.res.qa_search (qa_search.py:537): 检索命中 [来源: Redis 缓存] 问题: [勾股定理怎么证明] | faq_id: 180 | 学科: 数学 | 得分: 1.0 | 耗时: 6.19ms
  ```

  - 字段顺序：`时间.毫秒 [级别] [pid:线程] logger名 (文件:行号): 文案`；文案由 `base/log_messages.py` 的模板表渲染（事件名 → 中文），调用侧只传结构化字段，**禁止在业务代码里手写格式化字符串**。
  - 控制台 `LOG_LEVEL`（默认 `INFO`，看结论）、文件 `LOG_FILE_LEVEL`（默认 `DEBUG`，留现场：缓存 key、SQL、续命）。
  - 检索日志必须能一眼看出**命中来源**：`[来源: Redis 缓存]` / `[来源: MySQL 索引]` / `[来源: RAG 检索]`；未命中再补一行「FAQ 未命中，转交 RAG 检索」便于按行统计交接率。
- 日志：`structlog` + 标准库 logging 单出口，字段固定 `trace_id/request_id/node/latency_ms/extra`；**禁止 `print`**。
  - 必须**双出口**：stdout（容器采集）+ `logs/app.log`（`RotatingFileHandler`，50 MB × 10 份，UTF-8）；只写 stdout 等于线上没有留档（m01712 踩过：配了 logger 但 `logs/` 是空的）。
  - 业务日志与 uvicorn/第三方库日志走**同一套 handler**：structlog 经 `stdlib.LoggerFactory` 落到标准库 logging，由 handler 上的 `ProcessorFormatter` 统一渲染。否则会出现"我们的日志有、访问日志没有"的半截现场。
  - `configure()` 必须可重复调用且**允许后补文件路径**（模块导入期先初始化 console，lifespan 拿到配置后补文件 handler）。
  - 请求内用 `bind(request_id=..., path=...)` 绑定上下文，响应用 `X-Request-Id` 回传；**默认**只记 `q_hash`，需要排查时用 `LOG_QUERY_TEXT=true` 打开原始问法（dev 默认已开）。**生产必须关闭**：学生会话里常带姓名/手机号/学校。
  - `ProcessorFormatter` 必须显式传 `fmt`（它的默认值是 `"%(message)s"`，不传就只剩裸文案，没有时间/级别/行号）；调用点的 `filename/lineno` 由 `CallsiteParameterAdder` 注入，**不能**在字段过滤时把它们一起丢掉。
  - 控制台流要设 `errors="backslashreplace"`：GBK 控制台遇到 `m/s²` 会抛 `UnicodeEncodeError`，logging 吞掉异常后**整行丢失**（实测踩过）。
- 缓存：Redis 只放"可重算"的东西（FAQ 结论、会话热点、限流计数），MySQL 永远是真相源；**缓存读写必须 fail-open**（异常只记 `cache_errors` 指标并继续走索引），不允许因为 Redis 抖动把查询打挂。
- 配置：只从 `base/config.py`（pydantic-settings）取；禁止在业务代码里写死 host/阈值/TTL。
- 格式化与静态检查：`ruff format` + `ruff check --fix` + `mypy app`，提交前必须全绿。

### 5.3 编排节点的写法（统一模板）

每个节点必须显式声明四件事：**超时、重试、fallback、指标**。

```python
@node(name="retrieve", timeout_ms=800, retries=1, fallback="retrieve_degraded")
async def retrieve_node(state: ChatState, ctx: Ctx) -> ChatState:
    ctx.check_budget(min_left_ms=200)      # 预算不足直接抛 BudgetExceeded
    ...
```

- 重试只允许出现在**幂等节点**（检索、重排、FAQ、缓存读）；**生成节点绝不重试**。
- 每个节点都要能单独关掉（`FEATURE_<NODE>=off`）以便灰度与排障。

### 5.4 测试要求

- 新节点必须有单测（正常 + 超时 + 降级三条路径）。
- 改 prompt / 改检索参数 → 必须附黄金集对比报告（Recall/MRR/忠实度），否则不合入。
- 降级演练脚本进 CI：模拟 Redis 挂、Milvus 慢、LLM 超时，断言"有兜底、无 5xx"。

### 5.5 提交与文档

- 提交信息：`feat|fix|perf|refactor|docs|test(scope): 描述`；一个提交只做一件事。
- 改架构决策 → 同步改 `CLAUDE.md` 与 `docs/EduRAG-Pro-1.41-企业级架构设计.md`，**文档滞后视为缺陷**。
- 新增依赖 → 先在 PR 描述里说明"为什么不能 stdlib/现有依赖解决"。

### 5.6 明确禁止（踩过坑）

1. 不要在请求路径上加载模型或 `from_pretrained`（启动期完成）。
2. 不要在 CPU 上跑 Reranker（有 GPU 却用 CPU 是本项目最大的性能事故来源）。
3. 不要用 `md5(原始query)` 当缓存 key（缺 `kb_version` / `tenant` / `model`）。
4. 不要把全文父块一股脑塞进 prompt（按 token 预算裁剪）。
5. 不要在检索后用 Python 过滤租户/权限（必须走 Milvus 标量过滤）。
6. 不要把"信息不足请转人工"只写在 prompt 里（模型不遵守，必须代码硬门禁）。
7. 不要为了"看起来智能"引入多智能体 / GraphRAG（当前瓶颈是延迟与工程）。
8. 不要提交 `.env`、模型权重、日志、`volumes/`。

---

## 6. 架构约束与交付标准

### 6.1 单请求预算（逐节点扣减，超时即降级）

| 阶段 | 预算 | 降级动作 |
|---|---|---|
| 鉴权 + 限流 | 20 ms | 429 / 401 |
| 归一化 + 会话改写 | 80 ms | 跳过改写，用原 query |
| L1 精确缓存 | 10 ms | 视为未命中 |
| FAQ ∥ 意图 | 60 ms | 只走 RAG |
| L2 语义缓存 | 60 ms | 视为未命中 |
| 策略规划 | 300 ms | 直接 DIRECT |
| 检索（含多路并行） | 200 ms | 减少路数 |
| 重排 | 150 ms | 用融合分排序 |
| 首 token | 600 ms | 切备用模型 |
| 生成 | 剩余预算 | 截断并提示"回答不完整" |
| **总计** | **8000 ms** | 兜底话术 + 转人工 |

### 6.2 稳定性四件套（缺一不可）

1. **熔断**：LLM 网关按 provider 统计失败率，>50% 持续 30 s 打开，半开探测恢复；Redis/Milvus 同理。
2. **单飞**：同一 `md5(rewritten+kb_version+tenant)` 并发只跑一次（Redis 锁 + 结果等待），防缓存击穿。
3. **幂等**：`request_id` 贯穿缓存、历史、工单、计费。
4. **优雅停机**：收到 SIGTERM 先 `/readyz` 置 503，等在途请求 ≤ 10 s，再释放 GPU。

### 6.3 参数基线（改之前先跑评估）

| 参数 | 基线值 | 说明 |
|---|---|---|
| `RETRIEVAL_K` | 40 | 每路召回数（dense / sparse 各 40） |
| `RERANK_TOP_N` | 8 | 重排保留数 |
| `FINAL_PARENT_N` | 6 | 进 prompt 的父块数上限 |
| `RERANK_MAX_TOKENS` | 512 | 重排输入截断 |
| `PARENT_CHUNK_SIZE / OVERLAP` | 1200 / 100 | 父块 |
| `CHILD_CHUNK_SIZE / OVERLAP` | 300 / 20 | 子块 |
| 融合权重 | dense 0.7 / sparse 0.3 | 或改 RRF |
| `EVIDENCE_THRESHOLD` | 0.35（待标定） | 证据门 |
| 语义缓存阈值 | 0.95 | 低于此不开 |
| 缓存 TTL | 精确 24 h / 语义 24 h | 随 `kb_version` 失效 |
| 会话窗口 | 40 轮 / 28 000 token（先到先压） | Redis LIST + 压缩任务 |
| 记忆注入 | 画像 300 / 摘要 400 / 原子 300 | 合计 ≤1 000 token |
| 并发重排 batch | 8（GPU） | OOM 时下调 |

### 6.4 数据契约

- **Milvus 字段**：`id(child_id), text, dense_vector, sparse_vector, parent_id, parent_content, timestamp, doc_id, file_name, page_no, section, subject, tenant_id, acl_tags, kb_version, content_hash`。
- **MySQL 表**：`faq`、`chat_message`、`request_log`、`feedback`、`doc_meta`、`audit_log`（字段见架构设计书 §8.2）；记忆模块 `user_profile`、`session_summary`、`memory_atom`（含 `chain_root_id` / `valid_to` / `superseded_by` / `deleted_at`）、`memory_grant`、`memory_audit`；QAF 模块 `qa_pair`、`qa_version`、`miss_pool`、`tag`、`tag_synonym`、`qa_tag`、`doc_tag`。
- **Redis key**：`edurag:{env}:exact:{kb_version}:{tenant}:{md5}`、`:sem:`、`:session:{sid}`、`:faq:corpus:{subject}`、`:rl:`、`:lock:`、`:mem:hot:{user_id}`、`:mem:sum:{sid}`、`:tag:tree:{ns}`。

### 6.5 交付物清单

1. 源码 + `pyproject.toml`（可 `pip install -e .`）；
2. `docker/` 全栈 compose（含 GPU 应用镜像）与一键启动脚本；
3. 数据库初始化脚本 + FAQ/文档导入脚本；
4. 黄金集 + 评估报告（Recall/MRR/忠实度分切片）；
5. 压测报告（QPS / P95 / 显存 / 成本每千次）；
6. 运维手册（扩容、回滚、索引切换、故障处置）；
7. 架构设计书与部署图（`docs/`）；
8. 验收记录：SLO 表逐项打勾。

---

## 7. 三大模块实施方案（qa_system / rag_system / 记忆模块）

> 本章是**动手实现的施工图**：每个模块的职责边界、目录、类与函数契约、数据结构、关键算法、参数、降级、验收、落地顺序。
> 原则：三模块**平级、互不 import**，只通过 `base`（DB/缓存/模型句柄）与编排层的 `ChatState` 协作。

### 7.0 总览与边界

| 模块 | 代码落位 | 回答什么问题 | 延迟预算 | 失败时 |
|---|---|---|---|---|
| **qa_system** | `qa_system/`（`res/` + `api.py`） | 高频标准问题（学费、学制、开班、报名） | p95 ≤ 15 ms | 未命中即交棒 RAG，不报错 |
| **rag_system** | `rag_system/core/` | 文档型问题（课程大纲、政策、教材细节） | 检索 p95 ≤170 ms，首 token ≤1.2 s | 证据不足→兜底/转人工 |
| **记忆模块** | `memory/`（待建） | 这个用户是谁、聊过什么、偏好什么 | 召回 p95 ≤120 ms | 降级为画像+逐字历史 |

三条硬边界（违反即打回）：

1. qa_system **不调用 LLM**（除发布期的同义句生成，离线）；rag_system **不读 FAQ 表**；记忆模块 **不参与事实性回答**（只提供背景，不作为引用来源）。
2. 三者都不得直接 `MilvusClient(...)` / `pymysql.connect(...)`，只拿 `base/` 注入的 repository 或 client。
3. 编排层只依赖三者的**接口（Protocol / 数据类）**，不依赖其内部实现，便于单独替换与压测。

```
RAGpro_智能教育系统/
├─ base/                     # 公共底座（config / logger / labels / contracts / db）
├─ qa_system/                # 已实现
│  ├─ api.py                 # /v1/faq/* 路由
│  └─ res/
│     ├─ qa_db.py            # MySQL faq/faq_alias 读写 + 内存实现
│     ├─ qa_corpus.py        # 分学科 BM25 索引 + 双缓冲热更新
│     ├─ qa_normalize.py     # 归一化、别名、口语同义
│     ├─ qa_matcher.py       # 三条件命中判定
│     ├─ qa_cache.py         # Redis 连接与锁
│     └─ qa_search.py        # FAQService（对外唯一入口）
├─ rag_system/               # PR3
│  └─ core/
│     ├─ rag_pipeline.py / retriever.py / reranker.py
│     ├─ evidence_gate.py / citation.py / llm_gateway.py
│     └─ ingest/（loader/parser/cleaner/splitter/quality/indexer）
└─ memory/                   # PR4（待建）
   ├─ service.py             # MemoryService（读）
   ├─ writer.py              # 异步写（Redis Stream 消费）
   ├─ extractor.py / dedupe.py / conflict.py / summarizer.py / policy.py
   └─ store/{mysql_repo,redis_repo,vector_repo}.py
```


---

### 7.1 qa_system（FAQ 精准问答）

**目标**：把 40%+ 的重复问题在 15 ms 内答掉，且答案**逐字一致**（FAQ 是人工审过的，不允许模型改写）。

#### 7.1.1 数据结构

```sql
-- 主表沿用旧项目字段，避免数据迁移成本
faq(id BIGINT PK AI, subject_name VARCHAR(20), question VARCHAR(1000),
    answer VARCHAR(10000), status ENUM('draft','pending','published','deprecated') DEFAULT 'draft',
    qa_pair_id BIGINT NULL, content_hash CHAR(32), updated_at DATETIME, updated_by VARCHAR(64),
    KEY idx_status_subject(status, subject_name));

faq_alias(id BIGINT PK AI, faq_id BIGINT, alias VARCHAR(1000),
          source ENUM('manual','log','llm'), created_at DATETIME, KEY idx_faq(faq_id));
```

- `faq_alias` 是**同义问法表**（"学费多少/多少钱/价格"）：BM25 命中 alias 时，答案回落到所属 `faq_id`。这是旧项目缺失、但最能提升命中率的一环。
- `content_hash = md5(normalized_question)`，发布时计算，用于幂等与去重。

#### 7.1.2 索引与匹配

```
归一化 → ① Redis 二级缓存（key 含语料版本）→ ② alias 精确命中（O(1) 哈希）
      → ③ 分学科 BM25 top20 → 三条件判定 → 命中返回 / 未命中交棒（带 top3 候选）
```

**缓存层（2026-10-20 补，见 §7.1.7）**：
- key = `edurag:{env}:faq:v{语料版本}:{学科|*}:{归一化问法 sha1 前 20}`；带版本号是刻意的——发布/下线后新版本自然路由到新键，旧键靠 TTL 淘汰，不需要遍历删除。
- 正缓存 TTL 1 h（`FAQ_CACHE_TTL`），**绝对寿命** 6 h（`FAQ_CACHE_MAX_TTL`，`born` 字段判定），避免"一直命中一直续命"导致下线的答案永远不消失。
- **续命**：命中时若剩余寿命 ≤ `TTL × 0.5`（`FAQ_CACHE_SLIDE_RATIO`）才写一次 `EXPIRE`，省掉每次命中都写 Redis。实测 100 s → 3600 s。
- **负缓存** TTL 30 s（`FAQ_CACHE_NEGATIVE_TTL`）：未命中也要缓存（挡住重复脏查询反复打分），但**必须一并缓存 top3 候选**——候选是交棒 RAG 做 query 扩展的输入，丢了等于把下游线索省掉。
- 未命中时先看缓存指向的 `faq_id` 是否还在快照里；不在（被下线）则当未命中重算，记 `faq_cache_stale`。

- **分学科索引**：每个 `subject_name` 一个 `BM25Okapi`，避免全库 IDF 漂移（旧项目单索引，长尾学科分数被稀释）。
- 归一化：全半角、大小写、去标点与语气词（"请问/一下/呢"）、数字统一（"三个月"↔"3个月"）、口语同义（咋→怎么、啥→什么）、别名整串映射。
- 分词：jieba + 领域词典（课程名、证书名、学科术语），加载时一次性建好，请求期不加载。
- **分数归一化**（旧项目直接用原始 BM25 分数对比 8.0，是最大缺陷）：`score_norm = bm25 / (bm25 + 12.0)`，落在 (0,1)。
- 命中三条件（全部满足才算命中）：`score_norm ≥ 0.25`、`top1 − top2 ≥ 0.10`、`subject` 一致或无冲突。

```python
@dataclass(frozen=True)
class FAQHit:
    hit: bool
    faq_id: int | None
    answer: str | None
    score: float
    subject: str | None
    candidates: list[tuple[int, float]]   # 未命中时给 RAG 的 query 扩展候选

class FAQService:                                   # qa_system/res/qa_search.py
    async def search(self, query: str, subject: str | None = None) -> FAQHit: ...
    async def publish(self, faq_id: int, operator: str) -> None: ...   # 建索引 + 版本号 +1
    async def deprecate(self, faq_id: int, operator: str) -> None: ...
    async def rebuild(self, subject: str | None = None) -> dict[str, int]: ...
    async def warmup(self, samples: list[str] | None = None) -> None: ...  # 启动预热
```

#### 7.1.3 热更新（不重启、不加锁阻塞请求）

1. 发布事件写 Redis Stream `edurag:faq:events`；
2. 消费端重建**该学科**索引（增量：追加后重算 IDF）；
3. 新索引对象整体替换旧引用（双缓冲，Python 引用赋值原子）；
4. `edurag:faq:corpus:{subject}` 版本号 +1，写 `audit_log`。

重建 1 000 条 p99 < 200 ms；重建期间读请求走旧索引，**不允许持有全局锁**。

#### 7.1.4 降级

| 故障 | 表现 | 降级 |
|---|---|---|
| Redis 不可用 | 拿不到事件流 | 发布后 30 s 定时全量校验；命中仍走内存索引 |
| 索引未就绪 | 启动中 | `search` 直接返回未命中，交棒 RAG（不阻塞） |
| MySQL 慢 | 发布失败 | 读路径不受影响（索引在内存），发布重试 3 次入死信 |
| 别名表缺失 | 只精确匹配 | 仅走 BM25，记 `faq_degraded_total` 指标 |
| Redis 缓存不可用 | 拿不到热数据 | fail-open：读写异常只记 `FAQ_CACHE_ERRORS`/`edurag_faq_cache_errors_total`，直接走内存索引，回答不受影响 |
| 缓存值损坏 | `from_json` 失败 | 当未命中处理并覆盖写（记 `faq_cache_bad_value`），不抛错 |

#### 7.1.5 验收

单次 `search` p95 ≤15 ms（内存索引，无 IO）；黄金 FAQ 集 200 题 top1 准确率 ≥0.95；误命中（答非所问）≤2%；发布到可查 ≤5 min（含索引重建）。

#### 7.1.6 实现现状（PR2 v1，已跑通）

| 项 | 计划（§7.1.1–7.1.5） | 已实现 | 说明 |
|---|---|---|---|
| 索引 | 每学科一个 BM25Okapi | 自研 `BM25Index`（IDF = log(1+…)，保证非负） | 小语料下 rank_bm25 会出负分，且要控制空语料行为 |
| 别名 | BM25 命中 alias 回落 faq_id | `faq_alias` → 归一化后的**整串等价**映射 | 子串替换会产出"学费学费多少"，明确不做；近义走 BM25 |
| 热更新 | Redis Stream + 按学科重建 | 发布/下线后**全量重建 + 原子换快照**（160 条 ≈50 ms） | 语料 >5 万条再上事件流与增量 |
| 归一化 | 语气词/数字/别名 | 另加口语表（咋→怎么、啥→什么、甭→不用） | "勾股定理咋证明"因此能精确命中 |
| 表结构 | faq / faq_alias / faq_feedback | 同左，`faq` 自动补列（旧库可平滑迁移） | `--prune-legacy-subjects` 一次性下线非 K12 历史行 |
| 接口 | search/publish/deprecate/rebuild + healthz/readyz/metrics | 全部实现 | `X-Admin-Token` 保护写接口，未配置时仅本机调试 |
| 数据面 | compose 起 mysql+redis | 独立端口 3307/6380，`pull_policy: never`，offline save/load 脚本 | 见 `docs/docker-离线部署.md` |
| **Redis 二级缓存** | 可选缓存层（§5.2 缓存规范） | `RedisFAQCache` + `InMemoryFAQCache`（测试/无 Redis 时）+ `SingleFlight` | 正缓存 1 h/绝对 6 h、负缓存 30 s、命中续命（剩余 ≤50% 才写 EXPIRE）、负缓存带 top3 候选 |
| **可读日志** | structlog + 标准库 logging 单出口 | `base/logger.py`（行格式 + 轮转 + 落盘）、`base/log_messages.py`（事件 → 中文文案） | 控制台 `INFO` / 文件 `DEBUG`；检索行必须带 `[来源: Redis 缓存 / MySQL 索引 / RAG 检索]`；`faq_cache_write/faq_cache_slide/faq_db_query/faq_search_hit/faq_search_miss/faq_handoff_rag` |

代码位置：`base/{config,labels,contracts,logger,db}.py`（地基）、`qa_system/res/{qa_normalize,qa_corpus,qa_matcher,qa_db,qa_cache,qa_search,qa_csv}.py`（qa_system）、`qa_system/api.py` + `main.py`（HTTP）、`scripts/{init_db,import_faq,eval_faq,docker_up,docker_down}.py|ps1`、`tests/`（46 条用例）。

实测：**2000 条** K12 FAQ（`datasets/04_faq/edu_faq.csv` 已扩充，其中 97 条无学科，`state()` 单列 `unlabeled` 计数，不混进 `subjects`）；服务端 `search` 平均 0.3 ms；端到端（keep-alive）p50 5.7 ms / p95 9.1 ms（160 条语料时测得）。
`FAQ_THRESHOLD` 已标定：`scripts/eval_faq.py` 用 deepseek-flash 为 160 条 FAQ 生成 480 条同义问法（+80 条题库负样本），二维扫描后取 **threshold=0.25 + margin=0.10**（召回 0.637 / 精度 0.950）。分数尺度依赖 `FAQ_NORM_K=12`，改 K 必须重跑标定；线上真实流量积累 2 周后要用真实点击重标。

---

### 7.2 rag_system（知识检索与生成）

**目标**：文档型问题给出**带引用、可定位、不编造**的答案；证据不足时明确说不知道。

#### 7.2.1 检索链路（在线）

```
① 归一化 query（复用 qa 的 normalize）
② 并行：dense ANN(k=40) ‖ sparse ANN(k=40) ‖ FAQ top3 候选做 query 扩展
③ 融合：WeightedRanker(0.7, 0.3) → 候选 ≤80
④ 按 parent_id 去重 → 回填 parent_content（子块命中处包 <精准检索定位区块>）
⑤ 截断到 512 token/条 → GPU 批量 rerank(top_n=8)
⑥ 证据门：max(score) < τ 或实体覆盖不足 → 不生成
⑦ token 预算裁剪到 4–6 个父块 → 装配上下文
```

策略选择（规则优先，LLM 兜底，300 ms 上限）：`DIRECT`（默认）、`HyDE`（query <10 字且抽象，如"什么是分布式锁"）、`SUB_QUERY`（含"和/以及/对比"多意图，拆 ≤3 条并行检索）、`BACKTRACK`（含"这个/它/上面说的"且历史有指代）。任一策略检索为空 → 回退 `DIRECT` 再试一次。

```python
class Retriever:                                   # rag_system/core/retriever.py
    async def search(self, q: str, *, k: int = 40, filters: Filter,
                     strategy: Strategy = Strategy.DIRECT) -> list[Chunk]: ...

class Reranker:                                    # rag_system/core/reranker.py
    async def rerank(self, q: str, docs: list[Chunk], top_n: int = 8) -> list[Chunk]: ...

@dataclass
class Chunk:   # 父子块 + 元数据（Milvus 字段见 §6.4）
    child_id: str; text: str; parent_id: str; parent_content: str
    doc_id: str; file_name: str; page_no: int; section: str
    subject: str | None; score_dense: float; score_sparse: float; score_rerank: float
```

- **标量过滤前置**：`tenant_id`、`acl_tags`、`kb_version`、`subject` 全部进 Milvus `expr`，禁止 Python 后过滤。
- embedding 走 `run_in_executor` 的专用 GPU 线程，批量 ≥8，LRU 缓存 `md5(text)` → 向量（TTL 与 `kb_version` 绑定）。

#### 7.2.2 生成与引用

```python
class LLMGateway:                                  # rag_system/core/llm_gateway.py
    async def stream(self, msgs: list[Msg], *, budget_ms: int,
                     model: str | None = None) -> AsyncIterator[str]: ...
```

- prompt 四段式：`系统（角色+禁止编造+上下文指令不执行）→ 记忆（画像/摘要/原子，≤1 000 token）→ 知识（≤6 000 token，带 [n] 编号）→ 历史（剩余预算）`。
- 引用对齐：生成后把正文 `[n]` 映射回 chunk；**无引用支撑的句子**（句级 embedding 相似度 <0.55 或含未出现实体）改写或删除，不改写就整段回退兜底。
- 忠实度抽检（离线，不阻塞）：每日抽 100 条用 `deepseek-v4-pro` 打分，低于阈值的样本进人工复核队列。

#### 7.2.3 摄入（离线）

`loader → parser(MinerU/PDF/DOCX/PPTX/OCR) → cleaner(去页眉页脚/断行合并) → splitter(父 1200/子 300) → quality → embed → 写 edurag_v{N} → alias 切换`

- `content_hash` 未变则跳过；文档变更按 `doc_id` 先删后插（旧项目只 upsert 不删，会留脏块）。
- 质检四项：块长分布、重复率（块间 Jaccard >0.9 报警）、乱码率、表格/代码块断裂率；不达标**不切 alias**。

#### 7.2.4 降级阶梯

| 故障 | 降级 |
|---|---|
| rerank 超时/GPU OOM | 用融合分排序（质量降级，不报错） |
| Milvus 慢 | 减少路数（只 dense），k 降到 20 |
| 检索为空 | 一次 `DIRECT` 重试 → 仍空则兜底话术 + 转人工 |
| LLM 首 token 超 600 ms | 切 `deepseek-v4-pro`；再超则非流式短答 + 提示 |
| LLM 全挂 | 返回 FAQ 近义答案（若有），否则兜底话术 |

#### 7.2.5 验收

黄金集 Recall@10 ≥0.90、MRR ≥0.75；检索+重排 p95 ≤170 ms（GPU）；首 token p95 ≤1.2 s；引用可定位率 100%；编造率抽检 ≤2%。

#### 7.2.6 实现现状（PR3 第一批，切分/入库/检索）

| 能力 | 代码位置 | 说明 |
|---|---|---|
| 加载分发 | `rag_system/document_loaders/thy_factory.py` | 按后缀选 loader；`.md/.txt` 多编码兜底（utf-8-sig/gbk/gb18030，乱码率 >0 直接报错而不是悄悄进库）；PDF 双解析器 + `_MinerUWithLocalFallback` |
| 文本/字幕 | `loader_text.py` | 字幕去时间轴与序号；front-matter 剥离；空文件报错 |
| Office | `loader_office.py` | `.docx/.pptx` 按文档流顺序取块（段落与表格交错、形状按 top/left）；标题样式转 Markdown 标题；表格转 Markdown 表格；内嵌图 OCR 由 `RAG_OCR_IMAGES` 控制（默认关） |
| 图片 | `loader_image.py` | `.jpg/.png/.bmp/.webp` 走 OCR；OCR 不可用或识别为空**抛错**而不是返回空串 |
| OCR 引擎 | `document_loaders/ocr.py` | 进程内单例，优先 `rapidocr_paddle`（GPU）、退 `rapidocr_onnxruntime`（CPU）；超大图先缩到 2000px |
| PDF（云） | `loader_pdf_mineru.py` | MinerU 云 API，结果按内容 sha256 缓存到 `data/mineru/`；Markdown 里的图片引用入库前清掉（避免污染向量） |
| PDF（本地） | `loader_pdf_local.py` | PyMuPDF 逐页抽取 + 大图/扫描页 OCR；每页打 `<!-- page: n -->` 供引用对齐；`split_pages()` 取页号 |
| 学科识别 | `document_loaders/base.py::guess_subject` | 先按 `-` 分词命中，再整串子串兜底；认不出返回 None（不猜） |
| 父/子块切分 | `rag_system/text_splitters/parent_child.py` | 父块按标题树聚合、按标点装箱；子块继承父块标题；每子块携带 `parent_text` |
| 向量化 | `rag_system/embedding/bge_m3.py` | `BGEM3FlagModel` 一次出 dense+sparse；无 torch/无权重时退 `HashEmbedder` 并打 warning |
| 入库 | `rag_system/ingest.py` | CLI + `IngestPipeline.run()`；先按 `source` 删后写；单文件失败隔离；`--dry-run` 不连库；落 `data/ingest_manifest.json` |
| 向量库 | `rag_system/store/milvus_store.py` | `ensure_collection`（HNSW COSINE + SPARSE_INVERTED_INDEX）、`upsert` 分批 64、`delete_by_source`、`hybrid_search`（WeightedRanker 0.7/0.3）、标量过滤 `subject/doc_type` |
| 在线检索 | `rag_system/core/retriever.py` | 混合召回 → 父块合并去重 → 可选重排 → `final_n`；重排失败降级为融合分并记 `degraded` |
| 重排 | `rag_system/core/reranker.py` | `FlagReranker` 懒加载 + 单例，`normalize=True` |
| 证据门 | `rag_system/core/evidence_gate.py` | 三条件：分数、top1-top2 间隔、证据数；孤证额外 +0.10 门槛 |
| 检索接口 | `rag_system/api.py` | `GET /v1/rag/search`（证据 + 引用卡片 + 证据门结论）、`GET /v1/rag/stats`；未就绪 503 带 reason，不拖垮 FAQ |
| 启动装载 | `main.py::Container.load_rag` | 启动期建 Milvus 句柄 + 装载向量化模型（GPU 15–40 s）；失败只置 `rag=false`，`/readyz` 仍 200 并暴露 `rag_reason` |
| 冒烟脚本 | `scripts/smoke_rag.py` | `--backend hash` 验证管线；默认验证真实质量；拒答按设计行为打印 |

**已验证（hash）**：hash 后端跑通「6 份教材 → 17 子块 → 写入 Milvus → 混合检索 → 证据门」，全链路 4.5 s（含建库建集合）；`/v1/rag/search` 返回 evidence+citations（`{"passed":false,"reason":"below_threshold"}` 为设计行为）；单测 84 条全绿（切分 7 + 入库 6 + 检索/证据门/接口 14 + qa 57）。

**已验证（真 BGE-M3，2026-10-20，CPU）**：

- 入库：14 份文档（教材 6 + 教案 6 + 字幕 2）→ **110 个子块**，耗时 82 s（≈0.75 s/块，含 4 类解析；CPU 单进程）
- 编码：模型装载 7.3–7.9 s；首条 4.05 s（含 warmup）；批量 8 条 0.64 s（≈80 ms/条）
- 检索（`/v1/rag/search?q=顶点式怎么写`）：top1 **0.9794**（命中"通过配方，一般式可化为顶点式"原文），evidence `passed=true`，其余 4 条按分数递降且与主题相关；`cost_ms=11529` 为**首次请求含重排冷启动**
- 重排开销（CPU）：40 候选 × bge-reranker-v2-m3 约 **4.5 s**；`rerank=false` 时同一查询 **599 ms**
- 结论：**CPU 只能用于功能验证，线上必须 GPU**（手册 §1 已写清换 cu121 wheel 的方式）

**未做**：黄金集 Recall/MRR 基线；索引版本化与 alias 切换；GPU 下的重排延迟基线。

**加载器矩阵（第二批，2026-10-20）**：`.md/.txt/.markdown/.srt/.docx/.pptx/.pdf/.jpg/.jpeg/.png/.bmp/.webp` 全覆盖；PDF 解析器由 `RAG_PDF_PARSER=auto|mineru|local` 决定，`auto` 下云解析任何异常都退本地并打 `pdf_parser_fallback`（同时在 `meta` 里留 `parser_fallback`/`mineru_error`，不静默）；两条都失败时把两个原因一起抛。加载器单测 16 条（含 docx 标题+表格、pptx 形状顺序、本地 PDF 页标记与空 PDF、工厂分派与双失败信息）；全量 **100 passed**。

---

### 7.3 记忆模块

**目标**：让系统记住"这个学员是谁、报了什么、纠结什么"，但**绝不进入事实性回答**。定稿细节见 `docs/EduRAG-Pro-记忆系统设计.md`，此处只写施工要点。

#### 7.3.1 读写两链路

```
读（请求路径，并行，≤120 ms）
  L3 画像（Redis → MySQL） ‖ L1 摘要（Redis → MySQL） ‖ L2 原子记忆（Milvus top5）
  → 预算装配（300 + 400 + 300 token）→ 注入 prompt 的"记忆"段
写（异步，Redis Stream 消费，p95 ≤1.5 s）
  门控 → 抽取(deepseek-flash, JSON Schema) → 去重(0.92) → 冲突消解 → 摘要滚动 → 落 Milvus/MySQL
```

```python
@dataclass(frozen=True)
class MemoryContext:
    profile: str          # ≤300 token
    summary: str          # ≤400 token
    atoms: list[Atom]     # ≤5 条，≤300 token
    tokens: int

class MemoryService:                               # memory/service.py
    async def recall(self, user_id: str, sid: str, q: str, *, budget_ms: int = 120) -> MemoryContext: ...
    async def forget(self, user_id: str, atom_id: str, mode: Literal['revive','purge_chain']) -> None: ...

class MemoryWriter:                                # memory/writer.py
    async def on_turn(self, user_id: str, sid: str, turn: Turn) -> None: ...   # 入 Stream，立即返回
```

#### 7.3.2 写入六步的关键规则

| 步 | 规则 | 失败处理 |
|---|---|---|
| 门控 | 命中事实/偏好/决定/承诺信号才抽；寒暄与重复问丢弃（约 40%） | 保守丢弃 |
| 抽取 | 五类 fact_type：偏好/身份/约束/决定/进度；**必须带 `source_turn` 与 `confidence`** | JSON 解析失败重试 1 次，再失败入死信 |
| 去重 | 同 `fact_key` 且向量相似度 >0.92 → 合并（保留信息量大的表述，`mention_count+1`） | 失败则新增 |
| 冲突 | 同 `fact_key` 不同值：数值/时间类规则判定，其余 `deepseek-flash` 判定；新者胜 → 旧者 `valid_to=now, superseded_by=新id` | 判不出则都保留，标 `conflict=true` 待人工 |
| 摘要 | 每 10 轮或 3 000 token 滚动一次，保留 topics/open_items/decisions | 失败保留原文，下轮重试 |
| 落库 | MySQL 事务写真相 → Milvus 异步写向量（可补） | 向量失败不影响 MySQL，由补偿任务重试 |

#### 7.3.3 数据结构（与 §6.4 一致，此处给字段语义）

```sql
memory_atom(id VARCHAR(40) PK, tenant_id VARCHAR(32), user_id VARCHAR(32),
  fact_type ENUM('preference','identity','constraint','decision','progress'),
  fact_key VARCHAR(64),            -- 归一化槽位，如 goal_exam / budget / subject_weak
  content VARCHAR(500),            -- 一句事实
  confidence DECIMAL(3,2), mention_count INT DEFAULT 1,
  source_sid VARCHAR(40), source_turn INT, source_span VARCHAR(200),
  valid_from DATETIME, valid_to DATETIME NULL,
  superseded_by VARCHAR(40) NULL, chain_root_id VARCHAR(40) NULL,
  deleted_at DATETIME NULL, created_at DATETIME,
  KEY idx_user_alive(user_id, valid_to, deleted_at), KEY idx_chain(chain_root_id));
```

`valid_to is null and deleted_at is null and confidence ≥ 0.6` 是召回的唯一可见条件——**三个条件少一个就会出现"已删除记忆还在回答"**。

#### 7.3.4 记忆页接口（P0）

`GET /v1/memory/atoms?type=&q=&page=`｜`PATCH /v1/memory/atoms/{id}`｜`DELETE /v1/memory/atoms/{id}?mode=revive|purge_chain`｜`GET /v1/memory/chains/{id}`｜`POST /v1/memory/export`｜`DELETE /v1/memory/all`（二次确认 + 审计）。

删除写 `memory_audit`，2 s 内对召回生效（Redis `:mem:hot:{user_id}` 同步失效 + Milvus 侧 `valid_to` 过滤，**不依赖向量删除的时效**）。

#### 7.3.5 降级与验收

Milvus 慢/挂 → 只用画像+摘要（记忆 token 降到 700）；Redis 挂 → 直读 MySQL（p95 允许到 300 ms）；抽取模型挂 → 积压 Stream，不丢数据。

验收：MemRecall@5 ≥0.85、事实准确率 ≥0.95、过期污染率 ≤2%、记忆 token 占比 ≤10%、召回 p95 ≤120 ms。

---

### 7.4 三模块协同：一次请求的时序

| # | 阶段 | 模块 | 输出 | 预算 | 失败降级 |
|---|---|---|---|---|---|
| 1 | 鉴权/限流 | 接入层 | user_id/tenant_id | 20 ms | 401/429 |
| 2 | 归一化+改写 | understanding | rewritten | 80 ms | 用原 query |
| 3 | L1 缓存 | infra | answer | 10 ms | 未命中 |
| 4 | **FAQ ∥ 意图 ∥ 记忆召回** | qa + understanding + memory | FAQHit / intent / MemoryContext | 120 ms（并行） | 任一失败则该路为空 |
| 5 | FAQ 命中？ | qa | 直接返回 | — | 落空继续 |
| 6 | L2 语义缓存 | infra | answer | 60 ms | 未命中 |
| 7 | 检索+重排 | rag | chunks | 350 ms | 降级排序 |
| 8 | 证据门 | rag | pass/fail | 5 ms | 兜底转人工 |
| 9 | 生成+引用 | rag + memory(上下文) | 流式文本 | 剩余预算 | 切模型/兜底 |
| 10 | 回写 | memory + infra | 历史/缓存/指标 | 异步 | 不阻塞 |

**关键约束**：第 4 步三条并行分支必须**各自独立超时**（`asyncio.wait` + `timeout`），不能因为记忆召回慢而拖住 FAQ 命中——FAQ 本该 15 ms 返回。

### 7.5 落地顺序（按 PR 切，每个 PR 可独立验收）

| PR | 内容 | 完成定义（DoD） |
|---|---|---|
| **PR1 地基** | `infra/`（config/db/cache/obs/inference）、DDL、契约数据类、`/healthz` `/readyz`、启动预加载+预热 | 空壳服务能起、readyz 三模型就绪、DDL 可重复执行 |
| **PR2 qa_system** | repository/corpus/matcher/service + 热更新 + import_faq | 200 题黄金 FAQ top1 ≥0.95、p95 ≤15 ms |
| **PR3 rag_system** | 摄入流水线 + 检索/重排/证据门 + 生成/引用 + 流式 | 黄金集 Recall@10 ≥0.90、引用可定位 100% |
| **PR4 记忆** | 读写链路 + 记忆页 API + 链删除 + 召回降级 | MemRecall@5 ≥0.85、删除 2 s 生效 |
| **PR5 编排+前端** | LangGraph 12 节点、WS 四类帧、前端三件套、反馈闭环 | SLO 表逐项达标、降级演练全绿 |

顺序不可乱：qa_system 先做，因为它**不依赖 GPU 与 LLM**，可以最快验证"链路 + 观测 + 降级"这套骨架是否成立。

### 7.6 契约速查（实现时以此为准）

```python
FAQService.search(q, subject) -> FAQHit
Retriever.search(q, k, filters, strategy) -> list[Chunk]
Reranker.rerank(q, docs, top_n) -> list[Chunk]
EvidenceGate.check(chunks, intent) -> Evidence(ok: bool, reason: str, top_score: float)
LLMGateway.stream(msgs, budget_ms, model) -> AsyncIterator[str]
MemoryService.recall(user_id, sid, q, budget_ms) -> MemoryContext
MemoryWriter.on_turn(user_id, sid, turn) -> None
```

四个数据类 `FAQHit / Chunk / Evidence / MemoryContext` 定义在 `base/contracts.py`，**三模块只依赖它，不互相依赖**。

---

## 8. 变更记录

| 日期 | 变更 | 决策人 |
|---|---|---|
| 2026-10-19 | 建立交付仓库；确定"模型启动预加载 + GPU 常驻"、"五类意图"、"引用可溯源"三条基线 | 项目组 |
| 2026-10-19 | 记忆系统进入设计：四层记忆（L0/L1/L2/L3）、session 级窗口、token 硬预算、异步写入与 supersede 遗忘；讨论稿见 `docs/EduRAG-Pro-记忆系统设计.md` | 项目组 |
| 2026-10-19 | **记忆系统定稿 v1.0**：单窗口 40 轮 / 28 K token 双上限、记忆页（含链式删除 `revive` / `purge_chain`）、顾问授权只读、不做团队记忆、抽取用 `deepseek-flash`；新增 §3.8 QAF 与 Tag 体系（P1） | 项目组 |
| 2026-10-19 | **新增 §7 三大模块实施方案**：qa_system（分学科 BM25 + 别名表 + 分数归一化 + 双缓冲热更新）、rag_system（检索/重排/证据门/引用对齐/摄入质检）、记忆模块（读写两链路 + 六步写入 + 链删除）的目录、契约、数据结构、降级与验收；明确"三模块平级不互相 import，只共享 `infra/contracts.py`"；落地拆 PR1–PR5 | 项目组 |
| 2026-10-19 | 意图标签改为 `INTENT_LABELS` 配置驱动：数据集口径（K12 教学 6 类）与文档口径（招生 5 类）并存，拍板前不得把标签写死进代码或索引 | 项目组 |
| 2026-10-20 | **PR1 地基 + PR2 qa_system 落地**：`base/` 五个基础模块（config/labels/contracts/logger/db）、`qa_system/res/` 六个模块、`/v1/faq/*` 六个接口、`/healthz` `/readyz` `/metrics`；`scripts/init_db.py`（含旧库补列与非 K12 历史行下线）、`scripts/import_faq.py`；46 条单测全绿；实测 160 条 K12 语料、search 端到端 p95 9.1 ms | 项目组 |
| 2026-10-20 | **目录结构对齐旧项目 Ragpro**：根 `main.py` + `base/`（config/logger/labels/contracts/db）+ `qa_system/{api.py,res/qa_*.py}` + `rag_system/core/` + `docker/` + `logs/`；去掉 `app/`、`pyproject.toml`、`services/`、`eval/`、`web/`，改用 `requirements.txt` + `pytest.ini`；`python main.py` 可直接起服务 | 项目组 |
| 2026-10-20 | 实现层三处收敛并写回 §7.1.6：别名只做整串等价、热更新先全量重建（>5 万条再上事件流）、归一化加口语同义词表 | 项目组 |
| 2026-10-20 | **FAQ 阈值标定完成**：`scripts/eval_faq.py`（gen 480 条同义问法 + calibrate 二维扫描），报告 `docs/analysis/faq_threshold_report.md`；定 `FAQ_THRESHOLD=0.25`、`FAQ_MARGIN=0.10`（precision 0.950 / recall 0.637），`MatchPolicy` 与 `.env` 同步 | 项目组 |
| 2026-10-20 | 新增无库模式：`FAQ_CORPUS_CSV` 从 CSV 装载语料（`qa_system/res/qa_csv.py` 与 `scripts/eval_faq.py` 共用），MySQL 仍为生产真相源 | 项目组 |
| 2026-10-20 | **离线数据面落地**：`docker/docker-compose.yml` 端口/账号全部取自 `.env`（MySQL 3307 / Redis 6380，避开旧项目占用的 3306/6379），每个服务 `pull_policy: never`；`docker/offline/{save,load}.ps1` 导出/导入镜像（mysql 237 MB + redis 55 MB）；`scripts/{docker_up,docker_down}.ps1` 封装 `--env-file` 与健康等待；文档 `docs/docker-离线部署.md`。实测 MySQL 8.0.46 + Redis 健康，2000 条 FAQ 入库 | 项目组 |
| 2026-10-20 | **全局日志 + FAQ 二级缓存**：`base/logger.py` 改为 stdout + `logs/app.log`（50 MB×10 轮转），structlog 经 `stdlib.LoggerFactory` 与 uvicorn 日志共用 handler（修掉"配了 logger 但文件为空"）；`qa_system/res/qa_cache.py` 新增 Redis 正/负缓存、绝对寿命、命中续命、`SingleFlight`；缓存 key 加 `edurag:{env}` 前缀；补 `LOG_*` / `FAQ_CACHE_*` 环境变量；57 条单测全绿 | 项目组 |
| 2026-10-20 | **日志改为人读行格式**：`base/log_messages.py` 事件→中文文案表；行格式 `时间.毫秒 [级别] [pid:线程] logger (文件:行): 文案`（与旧项目 Ragpro 对齐）；检索结果单行带 `[来源: Redis 缓存 / MySQL 索引]`，未命中补一行「转交 RAG 检索」；分 `LOG_LEVEL`（控制台）/`LOG_FILE_LEVEL`（文件）；`LOG_QUERY_TEXT` dev 置 `true` | 项目组 |
| 2026-10-20 | **PR3 第一批（切分/入库/检索）**：`rag_system/` 新增 document_loaders（md/txt/docx/pptx/srt/pdf-MinerU）、text_splitters（父 1200/子 300 + 标题继承）、embedding（BGE-M3 稠密+稀疏，无 torch 时退 hash）、store（Milvus HNSW+稀疏倒排、hybrid_search、先删后写）、ingest（CLI + 流水线 + manifest）、core（retriever 父块合并去重/可选重排/证据门）；`scripts/smoke_rag.py` 冒烟；79 条单测全绿；hash 后端跑通 6 份教材 → 17 子块 → 检索全链路 | 项目组 |
| 2026-10-20 | **PR3 第二批（检索接入服务）**：新增 `rag_system/api.py` 的 `/v1/rag/search`、`/v1/rag/stats`；`main.py` 启动期 `Container.load_rag()` 预加载 Milvus 句柄与向量化模型，失败只置 `rag=false`、`/readyz` 暴露 `rag/rag_docs/rag_reason`、`/metrics` 增 `edurag_rag_ready`/`edurag_rag_docs`；新增 `tests/test_rag_api.py` 5 条；单测 84 条全绿 | 项目组 |
| 2026-10-20 | **PR3 真 BGE-M3 端到端跑通（CPU）**：离线分段拼装 torch 2.5.1+cpu；锁定 transformers 4.44.2 / sentencepiece **0.2.0**（0.2.2 在 Windows cp312 上 import 即崩）/ datasets 2.21.0 / huggingface-hub 0.36.2 / peft 0.12.0 / pymilvus 2.4.9 + `setuptools<81` + `marshmallow 3.22`；14 份文档 → 110 子块入库 82 s；`/v1/rag/search` top1 0.9794 命中原文、evidence 通过；实测重排 40 候选 CPU 约 4.5 s（`rerank=false` 599 ms）；`requirements.txt` 补 RAG 段、`.gitignore` 忽略 `rag_system/models/` | 项目组 |
| 2026-10-20 | **PR3 第三种：加载器矩阵补齐**（参考旧项目 `Ragpro/rag_system/document_loaders`）：新增 `ocr.py`（rapidocr 进程内单例，paddle 优先、onnxruntime 兜底）、`loader_office.py`（docx/pptx 按文档流顺序、标题样式转 Markdown、表格转 Markdown 表格、内嵌图 OCR 可选）、`loader_image.py`（图片 OCR，失败抛错不返回空串）、`loader_pdf_local.py`（PyMuPDF 逐页 + 大图/扫描页 OCR + `<!-- page: n -->` 页标记）；工厂支持 12 种后缀，PDF 增 `RAG_PDF_PARSER=auto|mineru|local` 与云→本地自动降级（打 `pdf_parser_fallback`、meta 留痕）；新增 16 条加载器单测，全量 100 passed | 项目组 |

