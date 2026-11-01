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
  → 并行：FAQ BM25 检索 ‖ 意图分类(6类) ‖ 槽位抽取
       ├─ FAQ 高置信 ──> 返回 FAQ 答案（带 faq_id 来源；不等意图/记忆）
       └─ FAQ 中置信 ──> **不返回答案**，带 top3 候选继续往下走（§7.1.9）
  → L2 语义缓存 ──命中──> 直接返回
  → 策略规划（规则优先，LLM 兜底）：DIRECT / HyDE / SUB_QUERY / BACKTRACK
  → 并行多路混合检索（Milvus dense+sparse，k=40，标量过滤）
  → 父块去重 → 高亮命中片段 → cross-encoder 重排 top8 → token 预算裁剪到 4–6 父块
  → 证据门（top1 分数 < τ 或覆盖度不足 → 兜底 / 追问 / 转人工；由 FAQ 中置信进来的走「你是不是想问：<候选问句>？」追问，不答 FAQ 原文）
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
| 向量库 | pymilvus（Milvus 2.4.0 standalone）；`requirements.txt` 钉 2.4.9、**本机实装 3.0.2**，见 §7.2.7-G | dense HNSW + sparse inverted，父子块 schema。**`MilvusClient.hybrid_search` 没有 `filter` 参数**（3.0.2 实测），标量过滤只能挂在每路 `AnnSearchRequest(expr=...)` 上，写错不报错、静默全库检索 |
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
| 意图分类 | BERT-small 蒸馏（K12 教学 6 类） | ONNX Runtime GPU/CPU 二选一 | <0.3 GB |
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
│  ├─ errors.py                       # 业务异常体系（BudgetExceeded / UpstreamUnavailable / …）
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
├─ rag_system/                        # 知识库检索与生成（PR3）
│  └─ core/                           # retriever / reranker / evidence_gate / llm_gateway / prompts / citation
├─ orchestration/                     # 编排层（§7.8）：唯一同时 import qa_system 与 rag_system 的包
│  ├─ state.py                        # ChatState
│  ├─ chat.py                         # ChatPipeline（归一化 → FAQ → RAG → 证据门 → 生成/兜底）
│  └─ api.py                          # POST /v1/chat；GET /v1/chat/stream（SSE）
├─ understanding/                     # 理解层（PR5，2026-10-30 已接线）：llm_understanding / intent / rewrite / profiles / slots
├─ docker/
│  ├─ docker-compose.yml              # mysql(3307) + redis(6380) + etcd/minio/milvus/attu；全部 pull_policy: never
│  ├─ offline/                        # 离线镜像：save.ps1 导出 / load.ps1 导入 / images/*.tar（不入库）
│  ├─ docker-compose.gpu.yml          # 待实现：应用 + 推理容器（CUDA 12.1）
│  ├─ Dockerfile                      # 待实现：多阶段构建，非 root 运行
│  └─ volumes/                        # 容器数据（不入库；down -v 或 -Purge 才清）
├─ scripts/                           # docker_up/down.ps1、建库建表、导入 FAQ、阈值标定、压测、导出数据清单
├─ tests/                             # 单测、契约测试、降级演练
├─ datasets/                          # 项目方提供的语料（只读，不改不删）
├─ logs/                              # 运行日志（不入库）
└─ docs/
   ├─ EduRAG-Pro-1.41-企业级架构设计.md
   ├─ EduRAG-Pro-记忆系统设计.md          # 定稿 v1.0
   ├─ EduRAG-Pro-QAF与标签体系设计.md
   ├─ EduRAG-Pro-意图数据体检与训练方案.md
   ├─ qa_system-运行手册.md
   ├─ 数据清单-EduRAG-Pro.xlsx           # 全量数据清单（10 个 sheet，由 scripts/export_data_inventory.py 导出）
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
| `FAQ_HIGH_MARGIN` | 高置信档的 margin 下限（与 `FAQ_MARGIN` **同一空间**解释，默认 0.22）；设为等于 `FAQ_MARGIN` 即回退为单档（§7.1.9） |
| `FEATURE_FAQ_CANDIDATES` | **中置信** FAQ 的 top3 问答对是否作为**辅助上下文**进生成 prompt，默认 `true`；置 `false` = 中置信的 FAQ 原文一个字都不进生成器（§7.1.9 边界 2）。它是这条改动的回滚开关，也是一次干净的 A/B 分组 |
| `FAQ_CORPUS_CSV` | 无库模式：从该 CSV 装载 FAQ 语料；生产留空（MySQL 是真相源） |
| `MILVUS_URI` / `MILVUS_DATABASE` / `MILVUS_COLLECTION` / `MILVUS_DIM` | Milvus 连接串与集合名，**默认 `http://127.0.0.1:19541` / `edurag` / `edurag_chunk` / 1024**（2026-10-29 从 19531 迁到 19541：19531 被旧项目容器 `edu_agent_milvus` 占着，连错端口会看到一个"活着但空的库"） |
| `MINIO_ENDPOINT/ACCESS_KEY/SECRET_KEY/BUCKET` | 原始文档存储 |
| `LLM_API_KEY` / `LLM_BASE_URL` | 生成模型网关（不写死、不进镜像）；默认 OpenAI 兼容 `/chat/completions` |
| `LLM_QA_MODEL` / `LLM_FALLBACK_MODEL` | 生成主模型 / 兜底模型（空 = 不切模型）；路由模型 `LLM_ROUTER_MODEL` 留给 PR5 |
| `LLM_TIMEOUT_S` / `LLM_MAX_TOKENS` / `LLM_TEMPERATURE` | 单次生成超时、输出上限、温度（默认 60 / 1024 / 0.2） |
| `RAG_EMBED_MODEL_PATH` / `RAG_RERANKER_PATH` | 模型权重目录，**默认在本仓库内**：`rag_system/models/BAAI/bge-m3`、`rag_system/models/BAAI/bge-reranker-v2-m3`。相对路径按 `BASE_DIR` 解析（换工作目录不会找不到）。**指向仓库外（如 `../Ragpro/...`）会让 `build_embedder` 静默退化成 `HashEmbedder`**——检索质量断崖式下降而服务照常 200，所以重排器改为找不到就抛 `FileNotFoundError` |
| `EMBEDDING_DEVICE` / `RERANK_DEVICE` | `cuda:0` / `cpu` |
| `EMBEDDING_BATCH_SIZE` / `RERANK_BATCH_SIZE` | GPU 批大小（OOM 时下调） |
| `KB_VERSION` | 当前生效索引版本，进缓存 key，切索引必改 |
| `RETRIEVAL_K` / `RERANK_TOP_N` / `FINAL_PARENT_N` / `RERANK_MAX_TOKENS` | 检索与重排基线参数（见 §6.3） |
| `EVIDENCE_THRESHOLD` | 证据门阈值 τ |
| `TOTAL_BUDGET_MS` | 单请求总预算（默认 8000） |
| `MAX_REQUEST_BYTES` | 请求体字节上限（默认 65536 = 64 KB）；超限 **413**。按字节兜住整个 body，与下面那条按"多少个字"的拒绝原因分两层 |
| `CHAT_MAX_QUERY_CHARS` | 问句字符上限（默认 400）；超限 **400 `query_too_long`**。**不许把长文截断后照答**——那会把"用户问了一长段"变成"系统答了半句" |
| `JWT_SECRET` / `JWT_LEEWAY_S` | 接入层鉴权：HS256 密钥（**stdlib 实现，不引 pyjwt**）/ 时钟偏移容忍（默认 60 s）。`JWT_SECRET` 为空则关闭 JWT 分支，只走白名单环境的开发期请求头；生产为空等于所有 token 都 401（`self_check()` 告警） |
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
| `FEATURE_MEMORY` / `FEATURE_MEMORY_VECTOR` | 记忆总开关 / L2 向量召回开关（均默认 `true`）。后者置 `false` 或 RAG 未就绪或集合建不起来 → **退回字面 bigram 排序**，这是功能降级不是故障（§7.9.4） |
| `MILVUS_MEMORY_COLLECTION` | 原子记忆向量集合，默认 `edurag_memory`。**与 `MILVUS_COLLECTION` 分开**：知识库是可重建索引，记忆是用户数据，重建知识库不该 drop 记忆 |
| `FEATURE_INTENT` | 意图档位总开关，默认 `true`；置 `false` 时档位恒为 `BASELINE`（不收窄 `doc_type`），即 §7.8.4 的 A/B 基线 |
| `FEATURE_INTENT_LLM` | 意图走大模型，默认 `true`；置 `false` 退回规则分类器（§7.8.4 报的 F1 0.967 属于这个分类器） |
| `FEATURE_QUERY_REWRITE` | 查询改写走大模型，默认 `true`；与 `FEATURE_INTENT_LLM` 同时关掉时理解层零次大模型调用 |
| `FEATURE_GRADE_FILTER` | 年级/学段下推成 Milvus 过滤，默认 `true`；置 `false` 只丢年级，学科照旧（年级误抽的现场止损） |
| `UNDERSTANDING_TIMEOUT_MS` | 改写 + 意图这一次调用的硬上限，默认 800；**从首 token 预算里扣**，SLO 未重测（§6.1） |
| `LLM_ROUTER_MODEL` | 理解层（改写 + 意图）用的模型，默认 `deepseek-chat`；**与生成模型分两个变量**，理解要便宜快、生成要质量。留空则走网关默认模型 |
| `FEATURE_HYDE` | 短且抽象的问句先写一段假设性答案再检索（HyDE），**2026-11-01 默认 `true`**，生成侧走 DeepSeek（`LLM_QA_MODEL`）。**如实记录**：多付一次生成调用、压缩首 token 预算，且"召回是否变好"**没有黄金集对照报告**（§5.4 未跑）——按需求打开的实验性能力，置 `false` 即回滚为 `DIRECT` |
| `HYDE_BUDGET_MS` | 假设文生成的硬上限，默认 1500。**不能拿 `LLM_TIMEOUT_S`（60 s）当它**——那等于没有上限，一次慢调用会把 `TOTAL_BUDGET_MS`（8 s）吃穿；超时即退回 `DIRECT` 用原句检索（`hyde_failed`） |
| `FEATURE_COMPOUND_LLM` | 复合问句拆分走**本地小模型**（`LLM_LOCAL_*` 端点，不计费不出网），**2026-11-01 默认 `true`**；置 `false` = 纯规则拆分，与接模型前逐字节一致（回滚开关 / A/B 分组）。模型给的子问题必须通过字符多重集校验，编出来的词一律丢弃并退回规则 |
| `COMPOUND_TIMEOUT_MS` | 拆分调用的硬上限，默认 400。到点即退回规则（`compound_timeout`），不拖首 token |

> 启动自检：`REDIS_PASSWORD` 与容器不一致、Milvus 集合不存在，任一命中直接打印可执行的修复提示并退出——不要留到第一个用户请求才 401。
> `LLM_API_KEY` 为空则**不退出**：FAQ / RAG 不依赖 LLM，只把 chat 置为不可用（`/readyz` 暴露 `llm=false`）并打告警；`ALLOW_DEGRADED_START=false` 的生产环境才 fail-fast（见 §7.8）。

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

### 3.2 理解层（P0 意图/改写，P1 槽位；学科槽位先行，见 §7.1.8）

1. 意图 **K12 教学 6 类**（2026-10-26 定案，施工图见 §7.7）：知识点查询 / 题目讲解 / 教案素材 / 学习方法 / 备课辅助 / 闲聊。
2. 置信度 < 0.6 → **不猜**：`label="unknown"` + `method="fallback"`，走 `generic_clarify` 档（全库检索不收窄 + 通用模板 + 结尾追问），**不写共享缓存**。禁止低置信硬路由到某一专业档位——错误档位的答案会被写进缓存。
3. 意图**不写成下游的 `if/else`**：分类结果映射为**路由档位**（是否检索 / `doc_type` 过滤 / 策略先验 / 引用要求 / 模板 / 缓存资格），下游只读档位；标签字符串只进日志、指标与打标。
4. 会话改写：含指代词、长度 < 8 字、无实体时触发；规则优先，小模型兜底；输出 `rewritten` 与原 query 一并落库。
5. 敏感词与越界请求（涉政/涉黄/竞品诋毁）在前置节点拦截。
6. **验收**：黄金集多轮子集（50 题）改写后 Recall@10 提升 ≥ 15 个百分点；意图 F1 ≥ 0.90（`intent_test.jsonl` 1 020 条）；`unknown` 比例 ≤10%；6+1 档路由单测全覆盖；第一阶段真正的验收是**档位是否改善下游决策**（A/B 见 §7.7），F1 只是门槛。

**落地形态（2026-10-30，与上表逐条对应的实现口径）**

| 能力 | 谁来做 | 位置 | 备注 |
|---|---|---|---|
| 归一化（NFKC / 空白 / 重复标点折叠，**不删词**） | 确定性代码 | `rag_system/core/query_processor.normalize()` | 无条件执行，纯 CPU 微秒级；它是"改写失败"时唯一的兜底 |
| 查询改写 | **大模型** | `orchestration/chat.py::_understand()` | 接在 **FAQ 高置信命中之后**、检索之前 |
| 意图分类 | **大模型**（`FEATURE_INTENT_LLM`），回落规则分类器 | 同上，与改写**合并成一次调用** | 无 key / 超时 / 坏 JSON → 各自独立退回，互不牵连 |
| 槽位（学科 / 多学科 / 年级） | **过滤类都是规则**（`qa_system/res/qa_slots.py`），**不调模型**；章节名 / 知识点 / 任务参数允许模型补空（`understanding/slots.py`，任务 2.3） | `_new_state()` 里 `qa_slots.resolve()`；`_apply_slots()` 里校验+合并+按意图收尾 | 学科与年级下推成 Milvus 标量过滤，所以过滤类**永远**只能是规则结论 |

- **改写必须排在 FAQ 之后**：台账 763 题里 404 题（53%）由 FAQ 高置信直接答掉（§7.8.2），
  放在 FAQ 之前等于让这 53% 的请求白付一次 LLM 调用；而 FAQ 用原话做精确哈希、与改写无依赖，串行不加延迟。
- **意图与改写合并成一次调用**：两者的落点本来就是同一个位置（FAQ 未中之后、检索之前），
  分两次调用等于同一份上下文让模型读两遍。但**校验与降级仍是两条独立的线**——JSON 里 `query` 字段不合格
  只丢改写（退回归一化结果），`intent` 字段不合格只丢意图（退回规则分类器），
  任何一项失败都不许把另一项一起丢掉。`degraded` 里 `rewrite_rejected` / `intent_rejected` 分开记。
- **槽位用规则而不是大模型**：槽位直接决定检索过滤（收窄错了就是"用户拿不到本该有的答案"），
  它要的是**可复现、可单测、零延迟**；规则的错法（抽不到 → 全库）比模型的错法（抽错 → 收窄到另一门学科）安全。
- **指代消解已接线（2026-10-31）**：`_load_session()` 里 `understanding/rewrite.py::rewrite()` 把上一轮用户问句
  拼到当前问句前面（`rewrite_method="rule"`），并在 `has_reference()` 命中时用 `_history_subject_hint()` 把学科
  以 `history_hint` 交给 `qa_slots.resolve()` → `subject_source="context"` / 置信 0.6。
  解不开（历史为空 / 上一轮不是问句）时记 `coreference_unresolved` 并**不猜主语**（任务 2.5 第 4 条）。
  仍未做的只有"用模型做指代消解"：规则拼接覆盖不了的跨主题指代，目前直接降级，不调 LLM 补。

**三条越界红线**：

1. **意图不得碰学科过滤**：`IntentResult` 没有 `subject` 字段。学科只由 `SlotResult` 与检索路由决定（§7.1.8），意图只允许收窄 `doc_type`；`tenant_id` / `acl_tags` / `kb_version` 由接入层决定，同样不看意图。
2. **低置信不得触发高风险路由**：只能进 `generic_clarify`（全库检索不收窄 + 通用模板 + 结尾追问），不强制引用、不进生成型档位、不转人工、不写共享缓存。
3. **FAQ 不等意图**：FAQ 高置信命中立即返回，不设"等意图/记忆跑完"的屏障；被取消的分支不影响本次回答（§7.4 第 4–5 步）。

> **已定案（2026-10-26）**：采用 `datasets/06_intent` 与 `datasets/README.md` 的 **K12 教学 6 类**（训练 4 080 / 测试 1 020，6×680 / 6×170）；原招生 5 类（课程咨询 / 考试辅导 / 升学规划 / 技能培训 等）**不再作为意图标签**——标签必须跟语料与评估集走，否则 F1 无从标定。标签唯一出处仍是 `base/labels.py::INTENT_LABELS`（已按 6 类实现，本次定案不改代码）；Tag 的 `intent` 命名空间、检索档位、缓存 key 全部随该枚举走。

### 3.3 检索与重排（P0）

1. Milvus 混合检索：dense HNSW（M=16, efConstruction=200, ef=64）+ sparse inverted（drop_ratio_build=0.2），`WeightedRanker(0.7, 0.3)` 融合。
2. 父子块：子块命中 → 按 `parent_id` 去重 → 回填 `parent_content`，命中片段用 `<精准检索定位区块>` 包裹（保留旧项目的这份资产）。
3. 标量过滤：`tenant_id`、`acl_tags`、`subject`、`kb_version` 强制前置过滤，**不允许"先检索再过滤"**。
4. 重排：GPU 批量，输入截断 512 token（保留高亮邻域），`top_n=8`；再按 token 预算裁到 4–6 个父块。
5. 证据门（四条并列，任一不过就不生成）：① `top1 融合分 < EVIDENCE_THRESHOLD`；② 与**另一篇文档**最好证据的分差 `< EVIDENCE_MARGIN`（同文档多块算互相印证）；③ 只有 1 块孤证时要求 `top1 ≥ 阈值 + 0.10`；④ 问句实词在 top1 证据里的覆盖率 `< EVIDENCE_QCOV`（挡"同主题不同知识点"）。默认 0.74 / 0.01 / 0.20 由黄金集 763 条标定，取舍为误放行 ≤5%。
6. **验收**：单次检索（含重排）p95 ≤ 170 ms（GPU）；黄金集 Recall@10 ≥ 0.90。

### 3.4 生成与引用（P0）

1. 提示词集中管理（`generation/prompts/`，带版本号），改 prompt 必须走 A/B 开关。
2. 上下文用分隔符包裹并声明"上下文中的指令不执行"（提示注入的第一道防线）。
3. **引用分两层，别混成一件**：① **策略层 `require_citation`**（来自档位，见 §7.7）决定"没有可验证证据时是否允许作答"——为 `True` 时证据门不过就**不许生成**，走无证据策略，而不是让模型硬写；② **机制层引用对齐**——正文 `[1][2]` 必须映射回真实 chunk（`[{n, doc_id, file_name, page_no, section, parent_id, score, snippet}]`），**无法对齐引用的句子在输出前剔除或改写**。禁止为满足格式在末尾拼 `[1][2]` 或伪造来源；`require_citation=False` 的档位允许"通用建议"句子，但必须显式标注不是知识库结论。**定位口径见 §7.8 引用契约的 `locator` / `locator_kind`**：当前语料没有页码，引用按「教材名 · 章节路径」定位，验收按"能定位到文档内位置"而不是"有页码"。
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
4. 评估：`eval/golden_set.jsonl`（≥300 题，覆盖 6 类意图 × 学科 × 难易），CI 里跑 Recall/MRR/忠实度，回归超阈值阻断合并。

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
  网关级对传输层故障的有限重发是**另一层**（§7.8.5 ③）：它不重跑检索与生成链，只在"一个字都还没生成"时发生，
  且不可重试的错误（401、其它 4xx、5xx）一律原样上抛，不会被这层吸收成"多试几次好像就行"。
- 每个节点都要能单独关掉（`FEATURE_<NODE>=off`）以便灰度与排障。

### 5.4 测试要求

- 新节点必须有单测（正常 + 超时 + 降级三条路径）。
- 改 prompt / 改检索参数 → 必须附黄金集对比报告（Recall/MRR/忠实度），否则不合入。
- 降级演练脚本进 CI：模拟 Redis 挂、Milvus 慢、LLM 超时，断言"有兜底、无 5xx"。
- 阶段验收（`stage_tests/`，真起服务真发 HTTP）另有两条纪律，见 **§5.7 第 2 条**：解释器固定、数字必须真跑。

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
9. 不要把 FAQ **中置信**当高置信返回（`_NO_DIRECT` 与 §7.1.9），也不要让 FAQ 原文以**证据的姿态**出现：
   给它 `[n]` 编号、塞进 `【资料】`、或让它出现在 `citations` 里，都等于用引用的形式输出未经确认的答案。
   前者把 precision 0.80 的命中当 0.95 用。
   （中置信的问答对**可以**以辅助上下文进 prompt，但只能走 `prompts.render_faq_candidates()` 那条独立
   system 消息、用「候选N：」编号、不进【资料】——这三条隔离就是"参考"与"事实"的分界，见 §7.1.9 边界 2。）

### 5.7 AI 协作纪律（用户指定，2026-10-31）

> 这两条是**用户对本仓库 AI 编码助手的直接要求**，不是建议；优先级高于本书其余各节的默认做法。
> 与它们冲突时以本节为准，有疑问先停下来问，不要"按最合理的理解先做了再说"。

**第 1 条：没有用户明确要求，不要擅自新建项目文件。**

- 「新建文件」包括脚本、测试、文档、报告、配置、临时产物、目录占位——一律算。往 `docs/`、`tests/`、
  `scripts/` 里**顺手落一份**是最常见的越界方式，明确禁止。
- 允许直接做的只有三类：① 用户点名要改的文件；② 报错链路上必须改的文件；③ §5.5 要求的
  "改架构决策同步改 `CLAUDE.md`"。
- 确实需要新建文件时，**先说明"打算新建哪个文件、为什么非新建不可、不建会怎样"**，等用户点头再建。
- 跑批产物、调试脚本、一次性验证程序写到系统临时目录或用户指定的位置，不默认进仓库。
- 用户的常用约束同源：`stage_tests/` 不进版本控制（`.gitignore` 已忽略），训练脚本由用户自己跑。

**第 2 条：阶段测试必须跑在用户本机的真实环境里，不允许给出虚假测试。**

- **解释器固定** `F:/miniconda3/envs/ragpro/python.exe`（conda env `ragpro`）。不得改用系统 `python`、
  别的 conda 环境或任何未验证的解释器；报告中出现的每个数字都必须来自这个解释器这一次的真实输出。
- **没跑就写"未跑"**：不许估计、不许从上一轮报告复制、不许按"应该会返回 200"填写。
  同一份代码换了语料/环境后，旧数字即失效（§7.8.2 的教训：语料错一代就能把 FAQ 出口从 404 打到 29）。
- **断言前先确认环境真的就绪**：服务起着且 `/readyz` 返回 `ready`、连的是哪个 Milvus / MySQL / Redis、
  模型是否真的加载（`llm: true`）。环境没就绪就报"环境未就绪"，不报"通过"。
- **失败就是失败**：脚本报错要原样贴出来；不许改成"跳过该项"再报满绿，不许把 `xfail` 当通过，
  不许为了让报告好看而放宽断言。
- 阶段验收的报告口径见 `stage_tests/README.md`（真起服务、失败留现场、报告写实测值）；本节管的是
  **更前面的一条**：那些实测值必须是真的。

---

## 6. 架构约束与交付标准

### 6.1 单请求预算（逐节点扣减，超时即降级）

| 阶段 | 预算 | 降级动作 |
|---|---|---|
| 鉴权 + 限流 | 20 ms | 429 / 401 |
| 归一化（确定性） | 5 ms | 用原 query |
| **改写 + 意图（一次大模型调用）** | ≤ `UNDERSTANDING_TIMEOUT_MS`（默认 800） | 改写退归一化句、意图退规则分类器，各自独立 |
| **复合问句拆分（本地小模型）** | ≤ `COMPOUND_TIMEOUT_MS`（默认 400） | 退回规则拆分 |
| L1 精确缓存 | 10 ms | 视为未命中 |
| FAQ（单路，**不等意图**） | 60 ms | 只走 RAG |
| L2 语义缓存 | 60 ms | 视为未命中 |
| 策略规划 | 300 ms | 直接 DIRECT |
| 检索（含多路并行） | 200 ms | 减少路数 |
| **HyDE（假设文生成，只在短抽象问句上）** | ≤ `HYDE_BUDGET_MS`（默认 1500） | 退回 `DIRECT` 用原句检索 |
| 重排 | 150 ms | 用融合分排序 |
| 首 token | 600 ms | 切备用模型 |
| 生成 | 剩余预算 | 截断并提示"回答不完整" |
| **总计** | **8000 ms** | 兜底话术 + 转人工 |

> ⚠ **2026-10-30 的账要重算**：改写+意图的那 ≤800 ms 是从**首 token 预算**里扣的——
> 上面各行的 20/5/60/60/300/200/150 加起来 800 ms 是"确定性部分"的账，
> 接上大模型理解后，首 token p95 ≤1.2 s 这条 SLO（§1.4）**尚未在真实链路上重测**。
> 已知的缓解：FAQ 高置信命中（台账里 53%）根本不做理解，实测那部分端到端 p50 仍在 11 ms 量级（§7.8.2）。
> 反过来，走 RAG 的那部分现在要多付一次大模型往返——**这条缺口是"没量"，不是"没变"**，
> 上线前必须用 `scripts/eval_chat.py` 的 `latency_ms` 切片量出来；量超了先降 `UNDERSTANDING_TIMEOUT_MS`
> 或关 `FEATURE_QUERY_REWRITE`，而不是去改 τ。

> ⚠ **2026-11-01 又多了两项，账更要重算**：`FEATURE_HYDE=true` 与模型拆分默认打开后，**首 token 之前**
> 最多叠 `800（理解）+ 400（拆分）+ 1500（HyDE）= 2700 ms`，已超过 `TOTAL_BUDGET_MS` 的三分之一。
> `base/config.py::soft_warnings()` 会在默认配置下直接告警这一条（硬校验只拦"单个值为 0 / 超过总预算"这种
> 明显写错，三项相加是性能取舍不是配置错误，不该让进程起不来）。两点如实记：
> ① HyDE 只在 `len(query) < 10` 且抽象的短问句上触发，不是每条请求都付；模型拆分只对复合句发一次本地调用，
> 且本地端点不出网、几十毫秒量级；② **两项的真实耗时都还没量过**——首 token p95 ≤1.2 s 依旧只是账面上的数。
> 止损顺序：先关 `FEATURE_HYDE`（收益最不确定），再降 `HYDE_BUDGET_MS`，最后才动 `UNDERSTANDING_TIMEOUT_MS`。

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
| `FAQ_HIGH_MARGIN` | 0.22（`normalized` 空间，与 `FAQ_MARGIN` 同空间） | 高置信档下限；test 参考 0.957 / 0.291，设成 `FAQ_MARGIN` 即回退单档（§7.1.9） |
| `EVIDENCE_THRESHOLD / MARGIN / QCOV` | 0.74 / 0.01 / 0.20（已标定） | 证据门三判据（分数 / 跨文档分差 / 问题词覆盖） |
| 语义缓存阈值 | 0.95 | 低于此不开 |
| 缓存 TTL | 精确 24 h / 语义 24 h | 随 `kb_version` 失效 |
| 会话窗口 | 40 轮 / 28 000 token（先到先压） | Redis LIST + 压缩任务 |
| 记忆注入 | 画像 300 / 摘要 400 / 原子 300 | 合计 ≤1 000 token |
| 并发重排 batch | 8（GPU） | OOM 时下调 |

### 6.4 数据契约

- **Milvus 字段**：`id(child_id), text, dense_vector, sparse_vector, parent_id, parent_content, timestamp, doc_id, file_name, page_no, section, subject, tenant_id, acl_tags, kb_version, content_hash`。
- **MySQL 表**：`faq`、`chat_message`、`request_log`、`feedback`、`doc_meta`、`audit_log`（字段见架构设计书 §8.2）；记忆模块 `user_profile`、`session_summary`、`memory_atom`（含 `chain_root_id` / `valid_to` / `superseded_by` / `deleted_at`）、`memory_grant`、`memory_audit`；QAF 模块 `qa_pair`、`qa_version`、`miss_pool`、`tag`、`tag_synonym`、`qa_tag`、`doc_tag`。
- **Redis key**：`edurag:{env}:exact:{kb_version}:{tenant}:{md5}`、`:sem:`、`:session:{sid}`、`:faq:corpus:{subject}`、`:rl:`、`:lock:`、`:mem:hot:{user_id}`、`:mem:sum:{sid}`、`:tag:tree:{ns}`。
  - **FAQ 结论缓存（C1）的实际键**：`edurag:{env}:faq:t{tenant}:v{语料版本}:{检索范围}:{归一化问法 sha1 前 20}`。三段都是**必须**的：`tenant` 防跨租户共用（阶段 1 补，此前键里只有 `env` + 版本），`v{版本}` 让发布/下线后新语料自然路由到新键而不是遍历删除，`scope` 让"限定数学"与"全库"不互相污染。**`tenant` 段不允许为空**：空则落成 `t-`，而不是留下 `::` 这种"看起来没传、实际全租户共用一份"的键。归一化问法的原文**不进键空间**（只有 20 位 sha1）。
- **跨模块数据类**：`SlotResult`（学科/年级/来源/置信度，见 §7.1.8）与 `FAQHit / Chunk / Evidence / MemoryContext` 一样定义在 `base/contracts.py`，q/RAG/编排共享；学科词表 `SUBJECTS` 与 `SUBJECT_ALIASES` 的唯一出处是 `base/labels.py`。

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
      → ③ 按槽位状态选检索范围（§7.1.8）→ 分学科 BM25 top20 → 四条判定（见下）
      → 命中返回 / 未命中交棒（带 top3 候选）
```

**缓存层（2026-10-20 补，实现现状见 §7.1.6 的缓存行）**：
- key = `edurag:{env}:faq:v{语料版本}:{学科|*}:{归一化问法 sha1 前 20}`；带版本号是刻意的——发布/下线后新版本自然路由到新键，旧键靠 TTL 淘汰，不需要遍历删除。
- 正缓存 TTL 1 h（`FAQ_CACHE_TTL`），**绝对寿命** 6 h（`FAQ_CACHE_MAX_TTL`，`born` 字段判定），避免"一直命中一直续命"导致下线的答案永远不消失。
- **续命**：命中时若剩余寿命 ≤ `TTL × 0.5`（`FAQ_CACHE_SLIDE_RATIO`）才写一次 `EXPIRE`，省掉每次命中都写 Redis。实测 100 s → 3600 s。
- **负缓存** TTL 30 s（`FAQ_CACHE_NEGATIVE_TTL`）：未命中也要缓存（挡住重复脏查询反复打分），但**必须一并缓存 top3 候选**——候选是交棒 RAG 做 query 扩展的输入，丢了等于把下游线索省掉。
- 未命中时先看缓存指向的 `faq_id` 是否还在快照里；不在（被下线）则当未命中重算，记 `faq_cache_stale`。

- **学科是检索范围的第一层约束**：由 `SlotResult.subject_source` 决定"按学科过滤 / 多学科合并 / 全库"，
  见 §7.1.8；**禁止**把"学科非法或为空"静默当成全库检索（2026-10-26 修正，原实现即如此）。
- **分学科索引**：每个 `subject_name` 一个 `BM25Okapi`，避免全库 IDF 漂移（旧项目单索引，长尾学科分数被稀释）。
  代价是**原始分与相对分差只在同一学科索引内可比**，跨学科只能比归一化分，且要另立规则（§7.1.8 多学科行）。
- 归一化：全半角、大小写、去标点与语气词（"请问/一下/呢"）、数字统一（"三个月"↔"3个月"）、口语同义（咋→怎么、啥→什么）、别名整串映射。
- **数学运算符不参与去标点**（2026-10-26 补，`_PUNCT = string.punctuation` 会把 `+ - / = < >` 一并删掉）：
  仅当**两侧都是 ASCII 字母/数字**（或用 `)` `]` `}` / `(` `[` `{` 相邻）时保留，并统一写法
  `× → *`、`÷ → /`、`· → *`、`− → -`；一元正负号（`-5的绝对值`、`(-5)`、`|-5|`）在开头或左括号/绝对值符之后同样保留。
  其余位置的同类字符照旧删除——例如"数学-七年级"里的连字符仍被去掉，避免把分隔符当成算子。
  为什么：语料里存在**只差符号**的题目（`因式分解 x²-6x+9` / `x²+6x+9`、`2/3-1/6` / `2/3+1/6`），
  删符号会让它们在精确哈希里撞成同一条，后者永远命中不到自己（2026-10-26 `baseline` 实测 3 组、自检 1997/2000）；
  BM25 侧同样会丢掉数学题的关键信息。NFKC 会把 `x²` 压成 `x2`（视为同义，可接受），`× ÷ − ·` 则原样保留，故需上面这张映射表。
  **已知残余（不扩大改动面，等触发再修）**：括号仍按标点删除，`2(x+3) 展开` → `2x+3展开`、`(-3)² 和 -3² 区别` → `-32和32区别`。
  现行 2000 条语料含括号表达式 15 题但**无撞车**；若第二步生成的口语样本或第三步标定中出现 `2(x+3)` 与 `2x+3` 这类同串，
  再扩展"成对保留括号"的规则（成对判定，避免留下孤括号造成伪区分）。
- 分词：jieba + 领域词典（课程名、证书名、学科术语），加载时一次性建好，请求期不加载。
- **分数归一化**（旧项目直接用原始 BM25 分数对比 8.0，是最大缺陷）：`score_norm = bm25 / (bm25 + 12.0)`，落在 (0,1)。
  `FAQ_NORM_K=12` 是**刻度不是判据**：`score_norm ≥ t` 等价于 `bm25 ≥ 12t/(1−t)`，改 K 只换 threshold 的数字
  （实测 K∈[6,36] 判定一致率 95%–100%，见 `docs/analysis/norm_k_sensitivity.md`）；但 margin 的物理含义会随 K 变
  （高分段被压缩），所以 margin **不定义在归一化空间**。
- **命中判定四条**（按顺序；`decide()` 只负责①–③，④由路由层给结论）：

  | # | 判据 | 取值 | 说明 |
  |---|---|---|---|
  | ① | 绝对质量门槛 | `score_norm ≥ FAQ_THRESHOLD`（0.25 ⟺ 原始分 4.0） | 先绝对后相对，避免小分数把相对分差放大 |
  | ② | 候选数 ≥ 2 | 只有 1 条候选 → `single_candidate`，不命中 | 原实现此时 `margin = top1`，0.10 形同不存在 |
  | ③ | 相对领先幅度 | `(raw1 − raw2) / raw1 ≥ FAQ_MARGIN`（`FAQ_MARGIN_MODE=relative` 时） | **在学科之内**计算；默认值待 §7.1.7 第三步标定，不先写死 |
  | ④ | 学科状态 | 仅 `explicit` 允许直接答；`inferred/context/multi/conflict` 交 RAG 或追问 | 学科一致性不是 `decide()` 的职责，见 §7.1.8 |

  > 通过 ①②③ 只是"够格被考虑"，**不是"可以答"**：margin 落在 `[FAQ_MARGIN, FAQ_HIGH_MARGIN)` 的命中属**中置信**，
  > 不直接作答、只交 RAG 与追问，见 §7.1.9。`hit` 与 `tier=="high"` 恒等价。

- margin 三种候选，同一评估集上逐一重扫、在 precision ≥0.95 下比 recall 再定：
  ① 原始分差 `raw1 − raw2`（同索引内可解释，但随学科 IDF 尺度漂移）；
  ② **相对分差 `(raw1 − raw2)/raw1`（优先候选）**；③ 归一化分差 `score1 − score2`（现用，跨学科假装可比）。
  无论选哪种，都要单独报"错误直接命中率"与"错误过滤率"，否则分不清收益来自阈值还是检索范围。
  实现上由 `FAQ_MARGIN_MODE=relative|normalized|raw` 切换、`FAQ_MARGIN` 在该空间里解释；
  **重标定完成前默认 `normalized`**（与现有 0.25/0.10 基线一致），第三步出报告后再切 `relative`。

```python
@dataclass(frozen=True)
class FAQHit:
    hit: bool
    tier: str                            # high / mid / low；hit 由它派生（§7.1.9）
    faq_id: int | None
    answer: str | None
    score: float
    subject: str | None
    subject_state: SubjectState          # 实际检索范围：filtered / multi / full / empty
    degraded: tuple[str, ...]            # 软降级/异常来源，如 ("subject_invalid_fallback",)
    candidates: list[tuple[int, float]]   # 未命中时给 RAG 的 query 扩展候选

class FAQService:                                   # qa_system/res/qa_search.py
    # slots=None 等价于 subject_source="not_provided"（真的全库），不是任何一种"降级"
    async def search(self, query: str, *, slots: SlotResult | None = None) -> FAQHit: ...
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
| 缓存条目缺 `tier` | 旧版本写入，无法判断置信档 | 一律按 **low** 处理并覆盖写（记 `faq_cache_no_tier`）；**不许猜成 high**（§7.1.9） |
| 学科非法（`invalid`） | 参数不在 `SUBJECTS`/别名表内 | 公开 API 返回 400 `subject_invalid`（列出合法学科）；内部编排可全库兜底，但必须带 `degraded=["subject_invalid_fallback"]` |
| 学科合法但无数据（`empty`） | 该学科索引 0 条 | 返回 `subject_empty`，**不兜底成全库**，也不报 5xx |
| 学科冲突（`conflict`） | 规则与模型结论矛盾 | 不硬过滤：按 `unknown` 全库检索、标 `conflict`、记告警，结果交 RAG |

#### 7.1.5 验收

单次 `search` p95 ≤15 ms（内存索引，无 IO）；黄金 FAQ 集 200 题 top1 准确率 ≥0.95；误命中（答非所问）≤2%；发布到可查 ≤5 min（含索引重建）。

分档（§7.1.9）：高置信档精度 ≥0.95（且召回 ≥0.25）；`hit == (tier == "high")` 恒成立；`stats()["tiers"]` 三档计数齐备；`FAQ_HIGH_MARGIN = FAQ_MARGIN` 时必须精确复现单档行为（单测覆盖）。

#### 7.1.6 实现现状（PR2 v1，已跑通）

| 项 | 计划（§7.1.1–7.1.5） | 已实现 | 说明 |
|---|---|---|---|
| 索引 | 每学科一个 BM25Okapi | 自研 `BM25Index`（IDF = log(1+…)，保证非负） | 小语料下 rank_bm25 会出负分，且要控制空语料行为 |
| 别名 | BM25 命中 alias 回落 faq_id | `faq_alias` → 归一化后的**整串等价**映射 | 子串替换会产出"学费学费多少"，明确不做；近义走 BM25 |
| 热更新 | Redis Stream + 按学科重建 | 发布/下线后**全量重建 + 原子换快照**（160 条 ≈50 ms） | 语料 >5 万条再上事件流与增量 |
| 归一化 | 语气词/数字/别名 | 另加口语表（咋→怎么、啥→什么、甭→不用） | "勾股定理咋证明"因此能精确命中 |
| 归一化·算子 | 去标点（原实现连 `+ - /` 一起删） | 两侧为 ASCII 操作数时保留运算符，`× ÷ · −` 归一（见 §7.1.2） | 修掉 `x²-6x+9` / `x²+6x+9` 撞车：自检 1997→**2000/2000** |
| 表结构 | faq / faq_alias / faq_feedback | 同左，`faq` 自动补列（旧库可平滑迁移） | `--prune-legacy-subjects` 一次性下线非 K12 历史行 |
| 接口 | search/publish/deprecate/rebuild + healthz/readyz/metrics | 全部实现 | `X-Admin-Token` 保护写接口，未配置时仅本机调试 |
| 数据面 | compose 起 mysql+redis | 独立端口 3307/6380，`pull_policy: never`，offline save/load 脚本 | 见 `docs/docker-离线部署.md` |
| **Redis 二级缓存** | 可选缓存层（§5.2 缓存规范） | `RedisFAQCache` + `InMemoryFAQCache`（测试/无 Redis 时）+ `SingleFlight` | 正缓存 1 h/绝对 6 h、负缓存 30 s、命中续命（剩余 ≤50% 才写 EXPIRE）、负缓存带 top3 候选 |
| **可读日志** | structlog + 标准库 logging 单出口 | `base/logger.py`（行格式 + 轮转 + 落盘）、`base/log_messages.py`（事件 → 中文文案） | 控制台 `INFO` / 文件 `DEBUG`；检索行必须带 `[来源: Redis 缓存 / MySQL 索引 / RAG 检索]`；`faq_cache_write/faq_cache_slide/faq_db_query/faq_search_hit/faq_search_miss/faq_handoff_rag` |

代码位置：`base/{config,labels,contracts,logger,db}.py`（地基）、`qa_system/res/{qa_normalize,qa_corpus,qa_matcher,qa_db,qa_cache,qa_search,qa_csv,qa_slots}.py`（qa_system）、`qa_system/api.py` + `main.py`（HTTP）、`scripts/{init_db,import_faq,eval_faq,docker_up,docker_down}.py|ps1`、`tests/`（128 条用例）。

实测：**2000 条** K12 FAQ（`datasets/04_faq/edu_faq.csv` 已扩充）。其中 97 条的学科写作"道德与法治"，
而旧 `SUBJECTS` 只有"政治"，于是被归一成 None 计入 `unlabeled`——**不是真的缺学科**；
2026-10-26 补 `SUBJECT_ALIASES` 后 9 个学科全部归位、`unlabeled` 归零（§7.1.8）。
服务端 `search` 平均 0.3 ms；端到端（keep-alive）p50 5.7 ms / p95 9.1 ms（160 条语料时测得）。
`FAQ_THRESHOLD` 曾标定：`scripts/eval_faq.py` 用 deepseek-flash 为 **160 条** FAQ 生成 480 条同义问法（+80 条题库负样本），二维扫描后取 **threshold=0.25 + margin=0.10**（召回 0.637 / 精度 0.950）。**该结论已失效**；2026-10-26 在 2000 条语料上重标定两轮**均未通过验收**：第一轮（关键词式短句语料）test 精度上界 0.897；第二轮（项目方把 FAQ 问题改写成完整问句后）精度 0.95 只在召回 0.127 处可达，召回守 0.637 时精度上界 0.831。0.25/0.10 继续作为生产兜底值，数据与结论见 §7.1.7「当前状态」。

#### 7.1.7 FAQ 阈值重标定路线（六步，待执行）

> 为什么必须重标：0.25/0.10 是在 **160 条**语料上标的，语料已扩到 2000 条；`docs/analysis/faq_paraphrases.jsonl`
> 的 `faq_id` 记的是旧行号，2026-10-25 实测 **480/480 与当前正文不一致**——按现文件跑 `calibrate`
> 结果是"命中 128 条、全部判错、precision=0"，且脚本不报错。在重标完成前，0.25/0.10 只能当**保守兜底**，
> 不能当已验证结论引用。

| 阶段 | 任务 | 验收标准 |
|---|---|---|
| 第一步 | 用当前 2000 条 FAQ 重建 BM25 索引，评估与线上用同一版本语料 | `/v1/faq/stats` 的条数 == 评估装载条数；原句自检 2000/2000 命中自身；`built_at` / 语料指纹 / **归一化指纹**记录进报告（`scripts/eval_faq.py baseline` 一键产出；改归一化规则必须重跑，否则第二、三步的样本与分数不可比） |
| 第二步 | 重建更真实的正负样本集：口语、噪声、歧义（同题族）、无关查询各成一类 | 先归档旧 `faq_paraphrases.jsonl` 再 `gen`；正样本 ≥600 条覆盖全部学科，负样本 ≥300 条；每条 `original` 与当前正文逐条一致（脚本硬校验，不一致即中止） |
| 第三步 | 重新标定 `threshold` 与 `margin`（若动 `norm_k` 必须同时重扫 margin） | 在**独立**测试集上 precision ≥0.95、召回 ≥0.637（不低于现基线，目标 0.70）；`docs/analysis/faq_threshold_report.md` 更新并写入语料指纹与时间 |
| 第四步 | 增加三档路由：高置信直接答 / 中置信 FAQ 候选 + RAG 并给 / 低置信交棒 | **契约见 §7.1.9**：high 档 test 精度 ≥0.95 且召回 ≥0.25；mid 档不返回答案（`FAQHit.answer` 为 `None`，只给候选、追问与 prompt 辅助上下文）；`faq_route_high/mid/low_total` 三档计数齐备；`FAQ_HIGH_MARGIN=FAQ_MARGIN` 可精确回退到单档 |
| 第五步 | 记录线上反馈与人工标签（点赞/点踩/纠错必须带 `request_id`） | 能从 `faq_feedback` 反查到 query、档位、命中的 `faq_id`；每周产出误命中/漏命中清单 |
| 第六步 | 重新评估"题族策略"（同题族是否合并、margin 是否按题族放宽） | 有对照实验：相对基线的误命中下降 ≥30%，召回下降 ≤5 个百分点；做不到就回退，不保留只增复杂度的规则 |

六条前置约束（都是已经踩到或推演出来的坑）：

1. **第二步必须先归档旧样本**：`gen` 按 `faq_id` 跳过已存在条目，直接追加会把新旧标签混进同一个文件，而且不报错。
2. **`calibrate` 要加标签校验**：`original` 必须等于该 `faq_id` 的当前正文，否则中止并提示重新 `gen`（现在只会静默算出 0 精度）。
3. **`norm_k` 只是刻度，不是判据**（`docs/analysis/norm_k_sensitivity.md`）：threshold 经 `t = raw/(raw+K)` 等价于"原始分截断"，
   K 从 6 换到 36 判定一致率仍有 95%–100%；K 真正影响的是 margin 的物理含义——归一化在高分段压缩，
   raw 4 附近 0.10 ≈ 原始分差 2 分，raw 30 时要差 15 分才够。所以 margin 建议改为原始分差或相对差，K 只留给人读日志。
4. **BM25 是词袋模型：只改语序的"同义问法"不是样本**。`什么是X` / `X是什么`、`X有哪些` / `有哪些X` 的 token 多重集
   完全相同，分数一模一样，放进标定集只会灌水。**正样本必须改变词袋**（换词、增词、省词），这是判定一条变体
   是否有效的硬条件（与"归一化后不等于原句"并列，两个条件都过才算一条样本）。
5. **样本要标来源与类别，且分来源出结论**：`source=rule`（规则改写，离线可复现、无成本）与 `source=llm`
   （生成模型，更接近真实口语）的分布不同，混在一起算出的阈值不可解释。**`source=rule` 的结果只能算临时值**，
   不足以改写线上默认参数；改写默认值必须有一份 `source=llm` 的报告。
6. **调参与报数必须分家**：同一批样本既选阈值又报成绩，等于自己给自己判卷。按 `faq_id` 分桶切 dev/test
   （同一 FAQ 的变体只能落在同一侧，否则泄漏），dev 选点、test 只报数。

**第二步细则：样本契约（2026-10-26 落地，`scripts/eval_faq.py gen|sync`）**

| 项 | 规则 |
|---|---|
| 归档 | `gen` 启动即把旧文件改名为 `faq_paraphrases.<时间戳>.bak.jsonl`，**新文件从空开始**；有未归档旧文件时拒绝追加（除非显式 `--reuse`，且通过下面的标签校验） |
| 正样本类别 | `noise`（加称呼/口头语，保留全部实词：`老师，{q}` / `{q}，能说下吗` / `我想了解下{q}`）、`synonym`（同义替换，换词不换义）、`abbrev`（省略疑问尾巴，留下名词短语）、`llm`（有 key 时优先生成） |
| 有效性 | ① 归一化串 ≠ 原句；② **词袋 ≠ 原句**；两个条件都满足才落盘，被丢弃的计数写进报告 |
| 分层 | 按学科分层，每学科至少 60 条、按语料占比向上取整分配总量，固定种子抽样（同种子 ⇒ 同一批样本） |
| 负样本类别 | `unrelated`（`opensource_questions.jsonl` 中学科不在 9 科内的题）、`lexical_trap`（同 9 科内的考试题：**词面高度重合但问的不是同一件事**，专治"同主题不同知识点"）、`noise`（手写闲聊/垃圾串，如"在吗""asdfgh"） |
| 负样本硬过滤 | 归一化后与任一 FAQ 问题相同 → 剔除（那其实是正样本）；与该 FAQ 的 token 集合几乎包含（Jaccard ≥0.9）→ 剔除，避免把"其实该答"标成负样本，逼着阈值过高。题库整题先清洗（`_clean_stem`：去选项行、行内公式、填空下划线），只留 **8–60 字**题干——选项块会给长查询堆无意义 token，量出来的不是"该不该答" |
| 字段 | `faq_id / subject / original / paraphrase / kind / source / generated_at`；负样本为 `text / kind / source` |
| 校验 | `gen` 落盘前与 `calibrate` 装载时**各校验一次** `original == 该 faq_id 当前正文`；不一致直接 `SystemExit(2)` 并打印"请重新 gen"，不允许静默 0 精度 |

**第三步细则：标定契约（`scripts/eval_faq.py calibrate`）**

| 项 | 规则 |
|---|---|
| 分桶 | 正样本按 `faq_id % 10`（`<7` dev，`≥7` test）；负样本按 `sha1(text) % 10`。同一 FAQ 的全部变体落同一侧 |
| 扫描 | threshold ∈ `[0.10, 0.90]` 步长 0.01；三种 `margin_mode` 各扫自己的网格：`relative` 0–0.90 步 0.05、`normalized` 0–0.60 步 0.02、`raw` 0–40 步 1 |
| 判定 | 直接调 `qa_matcher.decide()`（与线上同一份代码），**不在脚本里复制判定逻辑**；`rank()` 只给归一化分，原始分按 `raw = K·s/(1−s)` 反解 |
| 选点 | dev 上 precision ≥0.95 的前提下取召回最大；平手取更宽松（margin 更小）的那个。**test 只报数，不参与选择** |
| 指标 | 召回；精度；**错误直接命中率**（负样本被答 ⇒ 答非所问）；**错误过滤率**（正样本被拒 ⇒ 用户拿不到本该有的答案）；命中但答错（top1 是另一条 FAQ）单列，它比"不答"更糟 |
| 报告 | `docs/analysis/faq_threshold_report.md` 必须写：语料指纹、**归一化指纹**、样本文件指纹与各类别条数、dev/test 条数、三套 margin 空间的对照表、选点、以及"样本来源是否 llm"。另须含三张诊断表：**现行参数 vs 选点**的对照、**阈值/margin 邻域**（判断哪根旋钮在起作用）、**按查询长度切片**（分开"阈值没调好"与"长查询靠词数堆分"），以及**全网格 test 精度上界**（判据是否可达本身就是结论） |

**第四步前置：诊断契约（`python scripts/eval_faq.py diagnose`，2026-10-26 定）**

> 为什么先诊断再改机制：两轮标定都卡在同一处——召回被 margin 压在 0.38，但"这段被拒的样本值不值得救"没人量过。
> 在分清"排序就没找对"和"找对了但 margin 太紧"之前上三档路由或题族标注，等于拿预算赌一个没量过的假设。
> 本诊断**只读**：不写库、不改线上参数、不改判定逻辑，产出 `docs/analysis/faq_diagnosis.md`。

| # | 问题 | 定义 | 决策规则 |
|---|---|---|---|
| D1 | 召回丢在哪一层 | 正样本按**绑定判据**唯一归类：`exact`（归一化串精确命中，线上直接返回）→ `hit`（过判据且 top1 正确）→ `wrong`（过判据但 top1 是另一条 FAQ，**最糟**）→ `reject_rank`（top1 就不是正确答案：排序/词面没找到）→ `reject_single`（候选数 <2）→ `reject_threshold` / `reject_margin` / `reject_both`（top1 正确却被门槛挡住，按"只差哪一根"归因）。负样本同构归类，看它们被哪根判据挡住 | `reject_rank` 占比高 → 该补**别名/语料**（数据动作）；`reject_margin` 占比高 → 该动**机制**（题族/三档路由），补别名无效 |
| D2 | 题族值不值得标 | 在 `reject_margin`（top1 正确、只差 margin）的正样本上，算 top2 与正确答案 question 的 token Jaccard 作题族代理——**这是 (top1, top2) 这一对 FAQ 的属性，与用户问法无关，可在建索引期预计算** | 代理同族（Jaccard ≥0.5）占比 **<20%** → 标注不值（拒得对，是两个不同问题）；**>40%** → 值得标（召回可从 0.38 回到 0.6 量级且精度不降）；20–40% 灰色 → 先做第四步三档路由，标注等真实 miss pool |

- 归因顺序写死为「阈值 → 候选数 → margin」，与 `decide()` 的判定顺序一致；同一份观测不能既算 `reject_threshold`
  又算 `reject_margin`，否则几张表加起来会超过总数。
- D2 顺带做一次**机制预演**（同族对放宽 margin 会怎样），dev/test 两个数都要报；它只是"值不值得做第六步"的估算，
  不作为采纳依据——采纳必须走第六步的对照实验。

**数据补齐契约（2026-10-26，第三轮标定前执行）**

| 项 | 动作 | 为什么 | 验收 |
|---|---|---|---|
| `unrelated` 负样本 | 换源到 `datasets/raw_opensource/ceval-exam` 的**非 K12 学科**（accountant / civil_servant / clinical_medicine / law / computer_network / metrology_engineer / fire_engineer / college_programming …），多学科轮转取满 300 条 | 第二轮该类别归零，精度只由同主题陷阱决定；学科外的真问题是"该不该答"的另一半 | 300 条、≥5 个学科、`nearest FAQ` Jaccard <0.9、归一化串不与任一 FAQ 相同 |
| 负样本归档 | `negatives` 与 `gen` 一样，落盘前先归档旧文件 | 换源后新旧类别混在一个文件里，报告里的"类别构成"就失去回溯依据 | 归档成功且新文件从空开始 |
| 题族标签 | **本轮不做** | 等 D2 出数：占比 <20% 就不做 | 以 D2 结论为准 |
| FAQ 别名扩张 | 暂缓，等 D1 | D1 若显示 `reject_rank` 占比高，别名才是有效动作；否则是给已经找对的 FAQ 加同义词，白做 | 以 D1 结论为准 |
| `source=llm` 正样本 | **本轮仍做不了**（无可用密钥） | — | 拿到密钥后补 |
| 教材重复 | 不删 `datasets/`（只读语料），在**摄入侧**做近重复检测 | `09_textbooks` 现存两份同一本教材（`人教版-数学-九年级上册.pdf` 与 `人教版数学九年级上册.pdf`，md5 不同、内容一致）：两份都入库会互为对手，跨文档 margin 直接判 `ambiguous` | 见 §7.2.6「未做」 |

- **第三轮的精度与第二轮不可比**：负样本 540 → 840，补进了学科外样本（更容易），精度会上升；与第一轮类别相同但来源不同
  （题库 → ceval），报告头必须写清。**召回不变是预期的**——本轮补的是负样本测量面与诊断，不是正样本口径。
- 任何"召回提升"若只来自换了一批样本，不算收益；只有同一样本集上的参数/机制对照才算。

**语料重复合并契约（2026-10-26，D1/D2 诊断的直接结论）**

> 诊断把"召回丢在 margin"这件事落到了语料上：`reject_margin` 的 top1/top2 **多数是同一知识点的两种问法**——
> 语料里 367 组同答案同科的问题、涉及 **1052 条（52.6%）**，其中 318 组是三胞胎
> （`什么是X？` / `请解释什么是X` / `X指什么？`）。BM25 给同义问法几乎相同的分数，margin 必然 ≈0，
> 于是**任何第四种问法都答不了**。这不是阈值没调好，是索引里同一事实并存三份。

| 项 | 规则 |
|---|---|
| 合并判据 | **同学科 + 答案逐字相同 + 去问句框架词后 Jaccard ≥ 0.8**。三条同时成立才合并；逐对判定（并查集），不整组一刀切 |
| 代表条目 | `min(faq_id)` 作 canonical；被合并者**登记为精确命中别名**（原话仍可答，指向 canonical）并**不进 BM25 索引** |
| 答案不一致 | 去框架 Jaccard ≥0.8 但答案不同 → **不合并**，记 `faq_near_dup_answer_conflict` 交教研（实测 6 对；例：`什么是复分解反应？发生条件是什么？` vs `复分解反应发生的条件是什么？`） |
| 框架词 | `QUESTION_FRAME_TOKENS`（`什么/是/的/指/请/请问/解释/介绍/一下/帮/讲讲/如何/怎样/怎么样/哪些/吗/呢/吧/呀/讲/说说/谈谈`）——唯一出处 `qa_system/res/qa_normalize.py`；**只用于相似度比较，不改请求路径的归一化与缓存 key** |
| 为什么不用包含度 | 短问句的包含度天生为 1：`什么是基因？` 的 token 全在 `基因的显隐性规律是什么？` 里，实测 **302 对包含度 1.0 却是不同问题**。包含度会把它们并成一条，Jaccard 不会 |
| 为什么不在 `datasets/` 删行 | 语料只读（§2.5 约定）；重复是**装载期**问题，合并必须可配置、可回滚、可计数 |
| 开关与观测 | `FAQ_DEDUPE_NEAR_DUP`（默认 `true`，置 `false` 只用于对照实验）；`/v1/faq/stats` 暴露 `docs`（载入条数）/ `indexed`（进索引条数）/ `merged_duplicates` |
| 验收 | ① 原句自检仍 **2000/2000**（合并后按 canonical 判定）；② `merged_duplicates` 与探针数一致；③ 同一样本集上 `reject_margin` 下降、`hit` 上升；④ 精度不降 |
| 单测 | `tests/test_service.py`：合并生效（`docs/indexed/merged_duplicates` + 原话仍答 + 第三种问法转 BM25 命中）、**答案不同不合并**（两条各答各的）、`dedupe_near_dup=False` 保持全量索引（对照实验口径） |
| 不做的事 | 不改阈值/margin 默认值、不删语料行、不动 `datasets/`；合并只影响"同一事实的多个问法"的竞争关系 |


**当前状态（2026-10-26，第二、三步已执行两轮，两轮都未通过第三步验收）**

**第一轮（语料指纹 `d9750ece…`，关键词式短句语料）**

- 正 **928 条**（`noise 788 / synonym 121 / abbrev 19`，9 学科全覆盖）＋负 **840 条**
  （`lexical_trap 500 / unrelated 300 / noise 40`）；dev 659/601、test 269/239；原句自检 2000/2000。
- dev 选出 `raw / 0.57 / 9.00` → test 精度 **0.808**、召回 **0.078**；按召回下限 0.637 反选，dev 最好精度 0.824（test 0.811）；
  全网格 test 精度上界 **0.897**（召回 0.130）。
- 结论：0.95 在该特征空间里不可达。

**第二轮（2026-10-26 22:26，语料指纹 `a4c72486…`；项目方把 788 条 FAQ 问题从关键词式短句改写成完整问句）**

- 旧样本 **928/928 全部与当前正文不一致** → 归档 `docs/analysis/archive/faq_paraphrases.20261026-222902.jsonl`，
  正负样本全部重生成；`baseline` 自检 2000/2000、重复 0 组、unlabeled 0。
- 正 **928 条**（`synonym 437 / noise 473 / abbrev 18`）＋负 **540 条**（`lexical_trap 500 / noise 40`）；
  dev 622/375、test 306/165。
- **负样本丢了一类**：题库 6186 条现在全部落在 9 学科内，`unrelated` 由 300 条降为 0。脚本改为**告警 + 报告头写明缺口**，
  不造替代数据；因此本轮精度只由同主题陷阱决定，**与第一轮的 0.807/0.808 不可直接比较**（负样本更难更窄）。
- dev 选出 `relative / 0.56 / 0.20` → test 精度 **0.879**、召回 **0.284**（dev 0.953 / 0.227）→ **未通过**。
  按召回下限 0.637 反选：test 精度 0.831。
- 全网格 test 精度上界变为 **1.000（召回 0.127）**：0.95 现在可达，但要牺牲 87% 的召回；
  召回目标 0.70 时最紧 margin 0.02 只能给出精度 0.702。
- 现行默认 `normalized / 0.25 / 0.10` 在 test 上：召回 **0.382**、精度 **0.824**、硬样本召回 0.356、
  错误直接命中率 0.152、错误过滤率 0.618。
- ⚠ **撤回上一版的说法**：不能再拿第一轮的 0.654 与第二轮的 0.382 比召回。两轮正样本的类别构成不同
  （第一轮 `noise 788 / synonym 121`，第二轮 `synonym 437 / noise 473`）——`noise` 变体保留全部实词（容易），
  `synonym` 换词（难），是**样本变难了**，不是"语料改写把召回砍掉 27 个百分点"。两轮之间可比的只有第二轮的内部结论
  （阈值在 0.10–0.35 不生效、margin 才是瓶颈）；跨轮只能比"同一批样本上的参数对照"。

**病因（第二轮按新数据重新推导；"FAQ 语料是关键词式短句"那条诊断已随语料改写作废）**

1. **分数分辨力不足**：正样本 p90 0.652 / max 0.733，负样本 p90 0.536 / max 0.752——负样本高分尾部伸进正样本主体，单一阈值只能二选一。
2. **长查询靠词数堆分**：`21–40` 字区间负样本误命中率 0.306，而正样本召回只有 0.267。BM25 原始分随命中词数增长，`score_norm` 只是单调压缩。
3. **阈值在 0.10–0.35 完全不生效**（召回/精度恒为 0.382/0.824）：真正卡住命中的是 margin 0.10，不是 threshold——"调阈值"这条路本来就没有空间。
   推断：完整问句共享大量问句框架词（"是什么/有哪些/怎么"），top1 与 top2 的词面重叠整体变大、间隔被压小，margin 因此拒掉了大量正确命中
   （margin 收到 0.02 时召回回到 0.748）——**语料改写治好了"词面漂移"，没治好"可分离性"**。
4. 判定里没有任何"查询形态"信号（概念题 / 选择题 / 计算题），也拿不到学科内同题族结构；只用（分数 + margin + 候选数）时，上面就是精度上界。

**结论：两轮都不改线上参数。** 0.95 只在召回 ~0.13 处可达，把召回砍到那个量级换精度不划算（"答错"比"不答"更伤，但近乎全不答更糟）。
预算转给机制：**第四步三档路由**（中置信命中交 RAG 并附 top3 候选，而不是直接答）才是消化这段"答不准"区间的正确位置；
要让 FAQ 自己扛精度，得先有查询形态信号或用学科内同题族做二次判别（第六步）。

**第三轮（2026-10-26 当晚，改语料装载而非参数；召回首次过线，精度仍未过）**

诊断把"召回丢在 margin"落到了语料上：被 margin 拒掉的 top1/top2 多数是**同一知识点的两种问法**，
在索引里各占一行、BM25 分数几乎相同，任何第三种问法都答不了。于是按本节的「语料重复合并契约」在**装载期**合并，语料文件不动。

- 规模：**2000 条 → 1374 条进索引，626 条并成精确命中别名**；答案不同的高相似对 6 对不合并、逐条告警交教研；原句自检 2000/2000。
- 负样本同时补回 `unrelated` **300 条**（ceval 学科外题库），负样本 540 → 840。
- **同一样本集 A/B**（样本/分桶/阈值全不变，只改索引；对照报告 `docs/analysis/archive/faq_*dedupe-off-20261026.md`）：

  | 指标 | 合并关闭 | 合并开启 |
  |---|---|---|
  | test 召回 / 精度 | 0.382 / 0.775 | **0.735 / 0.859** |
  | dev 召回 / 精度 | 0.384 / 0.771 | 0.706 / 0.831 |
  | test `reject_margin` / `reject_rank` | 144 / 45 | **71 / 10** |
  | D2 test 同族 ≥0.5 占比 | 81.2% | 56.3% |

- 精度不是靠"更谨慎"换来的：负样本误命中绝对数 34 → 37（基本不动），精度上升全部来自答对的多了 108 条。
- **召回目标 0.637 在现行默认参数下首次达到（0.735）**；precision ≥0.95 仍不成立（反向选点精度上界 test 0.870，合并前 0.831）。
- **仍不改线上参数**：本轮动的是装载期索引，`MatchPolicy` 与 `.env` 的 `0.25 / 0.10` 未动。
  0.95 闸门是为"FAQ 直接作答"设的；精度 0.859 更适合作为三档路由 high/mid 的分界，剩下的误命中交给第四步消化。

**仍未做**：① `source=llm` 的正式样本（环境里没有可用密钥：`LLM_API_KEY` / `DEEPSEEK_API_KEY` 均未设置，
环境中的 `OPENAI_API_KEY` 对 `api.deepseek.com` 返回 401）。拿到密钥后跑 `gen --source llm` + `calibrate`，
若仍不过则以该报告为"参数不可达"的正式结论；② `unrelated` 负样本**已补齐 300 条**（第三轮），精度口径不再只测同主题陷阱。

#### 7.1.8 槽位与检索路由（学科槽位先行，2026-10-26 定案）

**要解决的问题**：`subject` 现在只有"传了/没传"两种含义，于是三种情况全退化成全库检索——
拼错学科、学科合法但库里没数据、语料里叫"道德与法治"而标签表写"政治"（97 条被判 unlabeled）。
全库检索本身不是错，**错在调用方不知道发生了**。

**边界**：规则版槽位提取放 `qa_system/res/qa_slots.py`（纯函数、无 IO、**不调模型**）；
模型版放 `understanding/slots.py`（2026-10-31 已接线，任务 2.3），它**只校验/合并/收尾**，
不起第二个抽取器。编排层在 `orchestration/chat.py::_apply_slots()` 里把模型给的原样 JSON
校验后合并进规则结论，再按意图收尾；`qa_system` 只依赖契约，不 import `understanding`。

**契约**（`base/contracts.py`）：

```python
SubjectSource = Literal["explicit", "inferred", "context", "multi", "conflict",
                        "invalid", "empty", "unknown", "not_provided"]
# 本次实际检索范围；invalid/conflict/unknown 这些"为什么落到这个范围"记在 FAQHit.degraded
SubjectState = Literal["filtered", "multi", "full", "empty"]

@dataclass(frozen=True, slots=True)
class SlotResult:
    subject: str | None                     # 单学科（canonical 名，取自 SUBJECTS）
    subjects: tuple[str, ...] = ()          # 多学科候选；非空时 subject 取第一个
    grade: str | None = None                # 只记日志，暂不参与过滤（FAQ 表无年级字段）
    subject_source: SubjectSource = "not_provided"
    subject_confidence: float = 0.0         # 规则=1.0；推断=0.6；未知/非法=0；不采信模型自报分数
```

`not_provided`（上游没做槽位提取）与 `unknown`（做了但抽不出）必须分开——排障时是两条完全不同的线索。

**路由表**（谁决定检索范围、能不能直接答）：

| subject_source | 索引选择 | 允许直接命中 | 备注 |
|---|---|---|---|
| `explicit` | 该学科索引 | 是 | 唯一"硬过滤 + 直接答"的场景 |
| `inferred` / `context` | 该学科索引 + 全库各取 top-k | 否，交 RAG 或追问 | 置信度未验证，硬过滤会把正确答案挡在外面 |
| `multi` | 各学科分别检索后合并 | 条件命中：领先**同一学科内** top2，且与另一学科 best 的归一化分差 ≥ `CROSS_SUBJECT_DELTA` | 合并只在归一化分上做，跨学科不比原始分 |
| `conflict` | 全库 | 否 | 按 `unknown` 处理 + 告警，结果交 RAG |
| `invalid` | 无 | 否 | 公开 API 400；内部编排兜底必须带 `degraded` |
| `empty` | 该学科空索引 | 否 | 返回 `subject_empty`，**不兜底** |
| `unknown` / `not_provided` | 全库 | 是（仍受 §7.1.2 ①②③ 约束） | 用户没提学科 = 真的全库，不是降级 |

> **精确命中是例外**：`exact` / `alias_exact` 恒为高置信（§7.1.9），不受上表"允许直接命中=否"的封顶。
> 理由是归一化串完全相同属字面事实，且 `inferred/context` 的检索范围本来就走 `open_scan`（含全库），
> 与"学科是猜的"无关。现实现里精确命中确实绕过了 `_NO_DIRECT` 封顶——此处把它写成规则，而不是留成巧合。

**冲突的产出端（2026-10-31 补）**：`conflict` 此前只有消费端（`FAQService._resolve_scope` 走全库 +
`_NO_DIRECT` 封顶 mid），**没有任何代码产出它**——`resolve("数学的二次函数怎么求", param="化学")`
会静默返回 `explicit / 化学`，用错的学科做硬过滤，而响应体里看起来完全正常。
现在由 `qa_system/res/qa_slots.py::resolve()` 自己算，判据是**三条同时成立**：
① `param` 归一化后是合法学科；② 问句抽出的学科集合 `scope` 非空；③ `param` **不在**这个集合里。
命中即返回 `SlotResult(subject_source="conflict")`，学科清空（`subject=None` / `subjects=()` / `scope=()`，
于是下游不加学科过滤、走全库），**学科之外**的槽位（年级 / 章节 / 任务参数）照旧留给问句。

四种**刻意的非冲突**，改代码前先读这几条：

| 情形 | 结论 | 为什么不是冲突 |
|---|---|---|
| `param` 落在问句的多学科集合里（"数学和物理的题" + `param=数学`） | `explicit / 数学` | 用户只是没把话说全，收窄正是他要的；判成冲突等于把他点名的那门丢掉，两边都查一遍 |
| 问句没点名学科（`scope` 为空） | 照常 `explicit` | 没有可矛盾的对象，参数是唯一线索 |
| `param` 不在词表（"物理课"） | 仍是 `invalid` | 它要被 400 暴露；被问句或冲突态盖住，用户永远不知道参数传错了 |
| 上一轮聊数学、这一轮问化学 | 照常 `explicit / 化学` | 主题切换是正常对话，当前这一轮说的算这一轮的 |

`history_hint`（`context` 来源）也**不参与**冲突判定：拿历史去否决用户这一步写下的学科，
是"指代借用"那条错误过滤的另一种翻身方式。`/v1/faq/search` 只走 `from_param()`（参数单路），
所以冲突是 `/v1/chat` 独有现象——那里 `query` 与 `param` 才会被一起解析。

**学科词表（唯一出处 `base/labels.py`）**：
- `SUBJECTS` 是**权威配置**，不靠语料扫描自动扩展——否则一个错别字就成了合法学科；
- `SUBJECT_ALIASES` 负责写法归一：`政治 / 思想政治 / 思想品德 / 道法 / 品德 → 道德与法治`（现行语料 9 类里是"道德与法治"，
  旧标签表写"政治"导致那 97 条被判 unlabeled）；**2026-10-26 已按此表落地**，`SUBJECTS` 里"政治"换成"道德与法治"，
  语料 `unlabeled` 归零。口径若由项目方改回"政治"，**只改这两张表**，不碰检索代码；
- 启动与每次重建时扫描语料，**报告**（不自动加白）未识别学科名与条数，记 `faq_subject_unknown_total`。

**评估**（先有数，再改阈值）：
- 样本七类各 ≥20 条：明确 / 简称 / 口语 / 缺槽位 / 多学科 / 冲突 / 误导性关键词，人工标 `subject` 与 `grade`；
- 四个指标：槽位准确率、覆盖率、**错误过滤率**（因槽位判断导致正确 FAQ 没进候选）、兜底恢复率。
  错误过滤率优先看——它是唯一会让用户彻底拿不到正确答案的指标。

**暂缓项**：
- `question_type`：等学科路由跑通、且有评估数据证明能提升召回再加（词典、边界、置信度、回退策略都不便宜）；
- `grade` 硬过滤：FAQ 表没有年级字段，要做先补 `qa_tag` / `doc_meta`，不在检索路径里硬造。

**实现现状（2026-10-26，PR2 补丁，已跑通）**：

| 能力 | 代码位置 | 说明 |
|---|---|---|
| 契约 | `base/contracts.py` | `SlotResult` / `SubjectSource` / `SubjectState`；`FAQHit.subject_state` + `degraded` |
| 词表与别名 | `base/labels.py` | `SUBJECTS` 9 类（含道德与法治）、`SUBJECT_ALIASES`、`is_valid_subject` |
| 规则抽取 | `qa_system/res/qa_slots.py` | `from_param`（参数）/ `extract`（问句）/ `resolve`（参数→问句→历史提示），纯函数、不调模型 |
| 路由 | `qa_system/res/qa_search.py::_resolve_scope` | 7 行路由表落到一个函数；`_exact_in_scope` 让精确命中同样服从范围 |
| 判定 | `qa_system/res/qa_matcher.py` | 四条判据 + `decide_multi`（学科内判完再要求跨学科归一化分差） |
| 接口 | `qa_system/api.py` | `subject` 不在词表 → 400 `subject_invalid` + 合法学科列表 |
| 指标 | `FAQService.stats()["routes"]` | filtered/multi/full/empty 计数；重建时 `faq_subject_unknown` 报告未识别学科条数 |

实测（2000 条语料）：装载后 9 学科、`unlabeled` **0**（原 97）；`subject=政治` → 归一为道德与法治、`subject_state=filtered`；
`subject=化学` 查数学题 → `no_match`（不再串学科）；`invalid` → 全库 + `degraded=("subject_invalid_fallback",)`；
`inferred` → 分数够也不直接答（`subject_inferred_handoff`）。
判据 ②（候选数 ≥2）的影响面实测：480 条历史同义问法里只有 **6 条（1.2%）** 只剩单候选，不会明显压低命中率，但堵住了"单候选一律判命中"这个漏洞。
单测 **128 条全绿**（槽位 14 + HTTP 3 + 语料重复合并 3）。

---

#### 7.1.9 FAQ 置信三档（高 / 中 / 低，第四步施工图，2026-10-27 定案）

**要解决的问题**：`hit` 现在是"答/不答"二选一，三轮标定说明这个二分法必须拆开——
`normalized / 0.25 / 0.10` 在 test 上是召回 0.735 / 精度 0.859，**直接答出去的每 7 条里有 1 条是错的**；
而同一批特征把 margin 抬到 0.22 就能拿到精度 0.957，代价是召回降到 0.291。
"答错"比"不答"更伤信任，但把 70% 的命中丢掉又等于白建索引。三档的作用是**把"答不准"的那一段从"直接答"里拿出来**，
交给 RAG 与追问消化，而不是继续在两根已经贴着前沿的旋钮上做没有空间的事（§7.1.7 结论）。

**三条边界（违反即打回）**：

1. **分档只在 qa_system 里算**：score / margin / 候选数 / 学科状态只在它有齐。编排层只读 `FAQHit.tier` 决定动作，不再自己比阈值——两处各算一遍，日志与行为必然对不上。
2. **中置信不返回答案，但可以当辅助上下文**（2026-10-31 修订，原文是"禁止把 FAQ 原文塞进生成 prompt"）：
   - **不变的一条**：`FAQHit.answer` 仅在 `tier="high"` 时非空。中置信的答案不经接口交付，也不作为证据参与引用对齐——要答案仍必须走下面的用户确认通道。
   - **改掉的一条**：mid 的 top3 **问答对**可以作为**辅助上下文**进生成 prompt（`FEATURE_FAQ_CANDIDATES`，默认 `true`），由 `prompts.render_faq_candidates()` 渲染成**独立一条 system 消息**，用「候选N：」编号（**不是 `[n]`**）、**不进 `【资料】`**，并原样带上「仅供参考，不代表事实」的告知与四条回答要求（优先依据 RAG 证据 / 只辅助理解 / 冲突以知识库为准 / 无证据不得当事实）。
   - **为什么这次可以**：原顾虑是"模型照抄 → 一条错答案变成带引用的事实"。这个风险来自**编号与位置**，不是来自"文字进了上下文"。候选块与 `[n]` 编号、`【资料】` 段、`citations` 三者物理隔开之后，模型最多照抄出一句**没有引用**的话，而 `require_citation=True` 的档位会把它整句剔除或整段拒答（`citation.repair()`）。
   - **仍然禁止**：把 FAQ 原文**当成 `【资料】` 里的一条**、或给它 `[n]` 编号、或允许它出现在 `citations` 里。那不是"参考"，是伪造来源。
   - low 档的候选**不进** prompt：连阈值都没过的匹配，塞进去只是在一次本就证据不足的回答上再叠一层噪音。
3. **FAQ 置信档 ≠ 意图档位**：`FAQHit.tier` 只决定"这条 FAQ 结论怎么用"，`IntentProfile`（§7.7）只决定"检索与生成怎么做"，两者互不改变对方。FAQ 高置信命中仍然不等意图（§7.4 第 5 步）。

**判定**（在 `decide()` 之后，只加一根旋钮）：

| 档 | 条件 | 说明 |
|---|---|---|
| **high** | ⓐ `exact` / `alias_exact`；或 ⓑ 过 §7.1.2 ①②③ 且 `margin ≥ FAQ_HIGH_MARGIN` | **唯一允许直接作答、且跳过检索的一档** |
| **mid** | ⓐ 过 ①②③ 但 `FAQ_MARGIN ≤ margin < FAQ_HIGH_MARGIN`；或 ⓑ 分数够但学科来源是 `inferred/context/conflict/invalid`（§7.1.8 封顶） | 跑 RAG + 候选；RAG 不过门则**追问**，不硬答 |
| **low** | 其余：`below_threshold` / `ambiguous`（margin < `FAQ_MARGIN`）/ `single_candidate` / `no_match` / `empty_tokens` / `subject_empty` / `not_loaded` / `empty_query` | 交棒 RAG（候选照旧做扩展）；不过门走兜底 + 转人工 |

- **只加 `FAQ_HIGH_MARGIN` 一根旋钮**（与 `FAQ_MARGIN` 同一 `FAQ_MARGIN_MODE` 空间解释）：阈值在 0.10–0.45 实测不生效（§7.1.7 阈值邻域），再加第二根阈值只扩大选点面、不增精度；margin 是唯一随精度单调变化的信号。
- **`multi` 封顶 mid**：跨学科只在归一化分上比，`FAQ_CROSS_SUBJECT_DELTA=0.05` 没经过精度标定，不能让这条路径直接作答。封顶只作用于 BM25 路径，精确命中见下条。
- **精确命中恒为 high**（含 `subject_source ∈ {inferred, context, conflict, invalid}` 与多学科）：归一化串完全相同是字面事实，且这些来源的检索范围本来都含全库（`_resolve_scope` 的 `open_scan`），与"学科是猜的"无关。这条顺便补齐 §7.1.8 的一处缺口——现实现里精确命中绕过了 `_NO_DIRECT` 封顶，这里不是继续绕，而是**明确写下来**。
- **档位只降不升**：`decide()` 给基准档，`qa_search` 只能按学科状态（multi / `_NO_DIRECT`）把它**下调**；任何路径都不许把 low 抬成 high。
- **回滚开关**：`FAQ_HIGH_MARGIN = FAQ_MARGIN` 精确复现今天的行为（所有命中都是 high）。单测必须覆盖这条等价性——它是灰度与回退的唯一手段，不能靠改代码。实现上 `MatchPolicy.high_margin=None` 就是"与 `margin` 相同"，**这是 matcher 的缺省值**：分档由 `.env` / `Settings` 显式打开，手工构造 `MatchPolicy` 的地方（脚本、单测）默认仍是单档。
- **`hit` 语义收紧**：`hit == (tier == "high")`，不提供第三种组合。这是对 `/v1/faq/search` 的**行为变更**（今天 `hit=true` 的中置信命中会变成 `hit=false` + `tier="mid"`），发布说明里必须写明。

```python
@dataclass(frozen=True, slots=True)
class FAQHit:
    ...
    tier: Literal["high", "mid", "low"] = "low"   # 新增；hit 由它派生，不再独立赋值

class _Stats:                                     # qa_system/res/qa_search.py
    tiers: dict[str, int]                         # {"high": n, "mid": n, "low": n}
```

**三档的动作**（编排层唯一读法）：

| 档 | 检索 | 生成 prompt | 返回 | RAG 不过证据门时 |
|---|---|---|---|---|
| high | **跳过**（省 §6.1 的检索 + 重排 350 ms） | 不生成 | FAQ 答案逐字返回 + `faq_id` 来源；`require_citation` 不适用（FAQ 是人工审核过的事实） | — |
| mid | 跑；候选 top3 进 query 扩展 | **top3 问答对进 prompt 当辅助上下文**（`FEATURE_FAQ_CANDIDATES`，独立 system 消息、候选N 编号、不进【资料】） | RAG 答案 + 引用；附 `faq_suggestion = {faq_id, question}`（**不含 answer**，点开走 `GET /v1/faq/{id}`） | **追问**："你是不是想问：<top1 question>？"+ 候选列表；不返回 FAQ 答案 |
| low | 跑；候选照旧 | 候选**不进** prompt | RAG 答案 + 引用 | 兜底话术 + 转人工（§3.4） |

mid 与 low 的区别就两条：mid 多一个**可信的追问对象**（top1 候选）和一张 FAQ 卡片。这两样放到 low 档是噪音——候选连阈值都没过，追问只会更离谱；同理，low 的候选也不进 prompt。

> **"能参考"与"能交付"是两件事**：mid 的 FAQ 文本进 prompt，是为了让模型听懂用户在问什么、必要时换个说法，而不是让它把那条答案端出去。落地上的三条隔离——独立 system 消息、`候选N` 而非 `[n]`、不进 `【资料】`——保证了它**进不了 `citations`**；`FAQHit.answer` 在 mid 档仍是 `None`，接口层也没有任何一条路径能把候选答案直接回给用户。

**确认通道**：用户点"是，就是这个"后，客户端带 `selected_faq_id` 重新提交；编排层用 `canonical_id()` 归一后直接返回该条答案（逐字），`tier="high"`、`reason="user_confirmed"`。

- 它不是"绕过精度闸门"：决定权交给用户，而不是让分类器猜。
- 参数无效（id 不存在 / 已下线 / 不在快照）→ 忽略参数走正常链路，`degraded=("faq_selected_invalid",)`，**不返回 4xx**（用户点的是旧卡片，不是参数错误）。
- 它同时是**在线标定信号**：追问采纳率 = `confirmed / clarify_shown`。

**缓存**：

- `CacheEntry` 增 `tier`：高置信写正缓存（1 h / 绝对 6 h）；中/低写负缓存（30 s），带 `tier` + top3 候选。
- 读回**按 `entry.tier` 复原，不重新判定**——同一 query 不能因为 margin 抖动在两档之间来回跳。
- 缺 `tier` 的旧条目（滚动发布期间的存量）按 **low** 处理并覆盖写，记 `faq_cache_no_tier`。**不许猜成 high**：猜错等于把没验过的答案直接返回。
- 候选必须随缓存一起走（与"负缓存必须带候选"同一条教训）：丢了候选，mid 就没有追问对象，这一档等于退化成 low。

**降级**：

| 故障 | 行为 |
|---|---|
| RAG 未就绪（`rag=false` / Milvus 未 load） | mid **不得升级为 high**：走追问 + `degraded=("rag_unavailable",)`。RAG 挂掉不等于"FAQ 就变可信了"——这正是精度闸门要挡的那种"退而求其次" |
| 缓存无 `tier` | 按 low 覆盖写（见上） |
| `selected_faq_id` 无效 | 忽略参数 + `faq_selected_invalid` |
| 语料未加载 | 全部 low（`not_loaded`），编排层只剩 RAG 与兜底 |
| 意图 / 记忆慢 | 与本档无关：high 命中立即返回（§7.4 第 5 步），不设屏障 |

**观测**：

- `FAQHit.tier` 进 `as_dict()`、`faq_search_hit` / `faq_search_miss` 日志行、`request_log.tier`。
- `stats()["tiers"]`；`/metrics` 增 `edurag_faq_route_high_total` / `_mid_total` / `_low_total`。与 `routes` **分开计数**：一个是"查了多大范围"，一个是"结论有多可信"，混在一起就再也分不清命中率变化是谁造成的。
- 第五步要闭环的三个数：**追问采纳率**、mid 档 RAG 证据门通过率、high 档点踩率。

**验收**（离线：第三轮同一批 928 正 / 840 负、同一 dev/test 分桶，只加这一根旋钮）：

1. high 档 test 精度 ≥0.95，且 high 档 test 召回 ≥0.25。参考值（由 §7.1.7 的 test margin 邻域反推）：0.22 → **0.957 / 0.291**，0.20 → 0.945 / 0.340；最终默认值由 dev 选点、test 只报数。
2. 报告必须写 **mid 档"若直接答"的精度**：由同一张邻域表反推 ≈ **0.80**（tp 136 / 169），远低于 0.95——这是"它不许直答"的量化依据，不是拍的。
3. `0.25 / 0.10` 一字不动；`FAQ_HIGH_MARGIN = FAQ_MARGIN` 精确复现现行行为（单测覆盖）。
4. 三档计数在 `/v1/faq/stats` 与 `/metrics` 同时可见。

**在线验收（P1，随第五步闭环）**：high 档点踩率**显著低于** mid 档；追问采纳率 > 50%（< 30% 说明 mid 把好东西挡在门外，应收紧 mid 边界）。若 high 档点踩率与 mid 档无差异，说明 margin 不是好信号 → 回退单档，不保留只增复杂度的三档（与 §7.7 同一条原则）。

**实施现状：已实现（2026-10-27，七步全部落地）**。落地顺序与实际落点：

| # | 落点 | 关键约束 |
|---|---|---|
| 1 | `base/config.py` + `.env.example` | `FAQ_HIGH_MARGIN` 默认 0.22（注释写明"与 `FAQ_MARGIN` 同空间；等于它即回退"），`self_check()` 校验 `FAQ_HIGH_MARGIN ≥ FAQ_MARGIN` |
| 2 | `base/contracts.py` | `FAQHit.tier` + `as_dict()`；`hit` 由 tier 派生（不设独立字段）；`Tier = Literal["high","mid","low"]` |
| 3 | `qa_system/res/qa_matcher.py` | `MatchPolicy.high_margin: float \| None = None` + `resolve_high_margin()`（`None` ⇒ `margin`，即单档）+ `MatchResult.tier`；`decide_multi()` 的跨学科打平降到 mid |
| 4 | `qa_system/res/qa_search.py` | `_finish(tier, ...)` 唯一入参；`_Stats.tiers`（三档恒齐备）；`CacheEntry.tier` 读写 + 缺字段按 low 且 `overwrite` 覆盖写；`_NO_DIRECT`/`multi` 封顶（exact 例外）；mid 的 `answer=None` 且负缓存同寿 |
| 5 | `qa_system/api.py` + `main.py` | `/v1/faq/stats` 暴露 `tiers`；`/metrics` 增 `edurag_faq_route_{high,mid,low}_total`；`selected_faq_id` 进 `/v1/faq/search` |
| 6 | `scripts/eval_faq.py` | `calibrate` 增 `tier_of/tier_stats`（同一 `obs` 上把 margin 切成两段）与「三档路由」「高置信下限邻域」两节；`diagnose` 的 D1 桶拆成 `hit_high/hit_mid/wrong_high/wrong_mid/false_hit_high/false_hit_mid`，且**不再与 `reject_*` 混算** |
| 7 | `tests/test_tiers.py`（17 条）+ `tests/test_api.py`（1 条） | 高/中/低各一条、`exact` 恒 high（含 `inferred` 与 `multi` 两种来源）、`inferred`/`multi` 封顶 mid、确认通道（含无效 id 不算 4xx）、缓存复现 tier、缺 tier 覆盖写、tier 决定缓存 TTL、三档计数、`high_margin=None` 与 `== margin` 逐字段等价 |

两处与施工图的偏差（都是落地时才发现，已按"档位只降不升"的原则处理）：

- **`invalid` 在 `_NO_DIRECT` 里，`test_service.py` 那条必须改**。原文写"现有用例一条都不用改"，实际只有一条不成立：
  `test_unknown_subject_falls_back_to_all`（断言 `subject="天文学"` 仍直接命中 faq 4）已改名为
  `test_unknown_subject_falls_back_to_full_scan_not_direct_answer`——`invalid` 兜底成全库后**只到 mid**。
  这不是回退，正是本节"学科来源被封顶"那条在起作用；且用例的查询从精确命中改成 BM25 命中，
  因为**精确命中恒为 high**（字面事实），本来就不受封顶管辖——原用例正好落在例外里，测不到封顶。
- **`hits` 的语义随之收紧**：`_Stats.hits` 只在 `tier=="high"` 时 +1，`misses` 吃 mid+low，
  `/metrics` 的 `edurag_faq_hit_rate` 因此是"可直接作答率"，与 `routes`（查了多大范围）分开看。

离线验证（`python scripts/eval_faq.py calibrate`，928 正 / 840 负）：高置信下限 0.22 在独立 test 上
**精度 0.957 / 召回 0.291 → 通过** §7.1.9 验收（≥0.95 且 ≥0.25）；mid 档「若直接答」精度 **0.805**，
即"中置信不许直答"的量化依据；`FAQ_HIGH_MARGIN=FAQ_MARGIN` 时高置信自动退化为 0.735/0.859（单档）。
报告见 `docs/analysis/faq_threshold_report.md` 的「三档路由」与「高置信下限邻域」两节。

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
| 无 CUDA（`RAG_RERANK_ENABLED=auto`） | 启动即跳过重排，按融合分排序；CPU 上 40 候选实测 115 s，不可上线 |
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
| 近重复检测 | `document_loaders/dedup.py` + `ingest.py` | 归一化文件名 + 正文前 8000 字的字符二元组 simhash（汉明 ≤3）；命中报 `doc_duplicate` 并拒绝入库；指纹落 `data/doc_fingerprints.json` |
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

**未做**：黄金集 Recall/MRR 基线；索引版本化与 alias 切换；GPU 下的重排延迟基线；
题库入库（`03_question_bank` 的 answer 多数只有一个选项字母、analysis 多为空，不能当 FAQ 用；
作为 `doc_type=question_bank` 的 RAG 证据也需先过滤掉无解析的条目）。

**近重复文档检测（2026-10-28 已落地）**：`09_textbooks` 里 `人教版-数学-九年级上册.pdf` 与 `人教版数学九年级上册.pdf`
是同一本教材的两次下载，md5 不同、内容一致，`content_hash` 去重抓不到——两份都入库会让同一段证据以"两篇文档"的身份
互为对手，证据门的跨文档 margin 直接判 `ambiguous`。`datasets/` 只读，所以只在**摄入侧**拒绝：

| 项 | 实现 |
|---|---|
| 判据①（便宜） | `normalize_name`：去扩展名 + 去 `空格 - _ · 、 ， . ( ) 【 】 《 》` 后转小写。**这一条就抓住了上面那对真文件** |
| 判据②（兜底） | 正文前 `RAG_DEDUP_PREFIX_CHARS`（默认 8000）字符的**字符二元组 simhash**（blake2b 加权 64 位），汉明距离 ≤ `RAG_DEDUP_DISTANCE`（默认 3）视为同一份。中文无空格，word 级 simhash 不适用 |
| 只留先到的 | 后到者拒绝，入库顺序由文件列表决定，可复现；"新的覆盖旧的"会让重跑随 mtime 漂移 |
| 自己不算重复 | 同一 `source` 重跑入库（改了内容）必须放过，否则第二次就入库不了 |
| 被拒者不登记 | 否则第三种命名会认错"先到者" |
| 持久化 | `data/doc_fingerprints.json`（`IngestPipeline.fingerprint_path`），增量入库能拦住上一批进过的那份；文件损坏当空表，**宁可漏拦不能让入库失败** |
| 开关与报告 | `RAG_DEDUP_ENABLED`（默认 `true`）、CLI `--allow-duplicate`（对照实验用）；`IngestReport.duplicates` + `summary()["skipped_duplicates"]` 列明"谁被拦了、跟谁重复"，单行日志事件 `ingest_duplicate` |

实测（`RAG_PDF_PARSER=local`，两份真文件 + README）：`docs_ok=2 / duplicates=1`，被拒那份报
`doc_duplicate: 与 …人教版-数学-九年级上册.pdf 近重复（name）`，379 个子块只入库一份。
**诚实结论**：这对真文件的 simhash 距离是 **26**（两次解析出的正文长度 114 878 vs 86 241 字符，差异来自解析路径而非内容），
所以 simhash 只是兜底，挡住这对文件的是文件名判据；simhash 覆盖的是"另存为/重命名、正文几乎一致"的那类副本。
单测 `tests/test_dedup.py` 15 条（判据、自我排除、被拒不登记、持久化往返、损坏文件、误杀防线、流水线接入），全量 **265 条全绿**（含 `tests/test_prompts.py` 5 条高亮用例）。

**证据门改为跨文档比较（2026-10-20）**：原先 `margin` 取 top1−top2，但 top2 常是同一份教案的相邻父块，分数天然接近（实测 0.0006–0.02），top1 已经 0.99 仍判 `ambiguous`。现在 `rag_system/core/evidence_gate.py::evaluate` 只在"另一篇文档"里找竞争对手（doc key 取自 `parent_id` 的 `{doc_id}:p{n}` 前缀），同一文档多块视为互相印证；`n_docs` 统计的是文档数。

**证据门标定完成（2026-10-20，commit 40918c2）**：`datasets/07_eval/ragas_eval.jsonl` 是 **1000 条**（不是预估的 157），`scripts/eval_rag.py` 建 763 条可判定样本（去掉跨学科冲突与空证据），跑检索并解出两条判据：① 分数门槛做不了"同题族"区分——可答与不可答的 top1 中位数只差 0.04（0.782 vs 0.744），且 58% 不可答题分数高于 0.70；② **问题词覆盖率（qcov）** 是更强的独立信号（可答中位 0.6 vs 不可答 0.2）。据此给 `evaluate()` 加第四条判据 `min_qcov` 并新增 `query_coverage()`，标定取「误放行 ≤5%」的取舍点 **threshold=0.74 / margin=0.01 / qcov=0.20 → 可答放行 34%、不可答误放 3%**（放松到 0.72 则 50%/9%）。根因不在门槛而在语料：黄金集覆盖 7–9 年级，库里只有 1 本九年级数学 + 6 份短教材，可答题里只有 20% 能召回到正确文档。报告 `docs/analysis/rag_threshold_report.md`，原始跑批 `docs/analysis/rag_eval_runs.jsonl`（可复跑重标定）（763 行，含每题 top-10 证据与 qcov）。

**加载器矩阵（第二批，2026-10-20）**：`.md/.txt/.markdown/.srt/.docx/.pptx/.pdf/.jpg/.jpeg/.png/.bmp/.webp` 全覆盖；PDF 解析器由 `RAG_PDF_PARSER=auto|mineru|local` 决定，`auto` 下云解析任何异常都退本地并打 `pdf_parser_fallback`（同时在 `meta` 里留 `parser_fallback`/`mineru_error`，不静默）；两条都失败时把两个原因一起抛。加载器单测 16 条（含 docx 标题+表格、pptx 形状顺序、本地 PDF 页标记与空 PDF、工厂分派与双失败信息）；全量 **100 passed**。

#### 7.2.7 完整检索链路的施工现状（2026-10-29，PR3-A…PR3-D 已落地）

> 施工计划书见项目交付方提供的 `CLAUDE.md` §7.2.7；本节记录**代码里现在到底有什么**，
> 与计划逐项对照。原则同 §7.2.6：设计写了不等于实现了，没跑过的指标不许写成实测值。

**模块落点**（计划里的文件 ↔ 仓库里的文件）

| 计划的文件 | 实际落点 | 状态 |
|---|---|---|
| `base/contracts.py` 的 DTO/枚举 | `QueryContext` / `RetrievalFilters` / `QueryTask` / `RetrievalPlan` / `RetrievalResult` / `Evidence` / `EvidenceDecision` / `RetryAction` / `Strategy` / `UnderstandingResult` / `ChatResponse` | ✅ |
| `understanding/query_rewriter.py` | 职责在 `understanding/rewrite.py`（原 `memory` 侧接口不动） | ✅ 不另造同名文件 |
| `understanding/slot_extractor.py` | 规则版在 `qa_system/res/qa_slots.py`，模型版在 `understanding/slots.py` | ✅ |
| `understanding/intent_classifier.py` | `understanding/intent.py` + 档位表 `understanding/profiles.py` | ✅ |
| `understanding/compound_query.py` | 规则拆分（distribute → connective），`_verified()` 用**字符多重集**保证子问题里的字原句都出现过；`_build_understanding()` 产出 `sub_queries`，planner 按 `len(subs) >= 2` 走 `SUB_QUERY` 一任务一路（2026-10-31 接线完成） | ✅ |
| `rag_system/core/planner.py` | 同 | ✅ 规则优先、无 LLM |
| `rag_system/core/query_processor.py` | 同（归一化 + 入参校验） | ✅ 只做这两件事 |
| `rag_system/core/fusion.py` | 同（`max(score × weight)` 融合，**不用 RRF**） | ✅ |
| `rag_system/core/evidence_builder.py` | 同（父块去重 / `child_hits` 合并 / token 预算 / 冲突检测） | ✅ |
| `rag_system/core/rag_pipeline.py` | 同（**唯一持有重试决策的地方**） | ✅ |
| `rag_system/core/evidence_gate.py` | 在既有四条判据上改输出为结构化 `status/reasons/missing_tasks/retry_action` | ✅ |
| `rag_system/core/retriever.py` | 保留 Milvus/父子块/reranker，输出统一成 `RetrievalResult` + `stages` | ✅ |
| `orchestration/{state,chat,api}.py` | 同；`/v1/chat` 非流式 + `/v1/chat/stream` 四帧 | ✅ |
| `rag_system/api.py` | 保留 `/v1/rag/search` 诊断接口，内部走**同一个** `RagPipeline` | ✅ |

**证据门输出契约**（`EvidenceDecision`）

```python
status: "sufficient" | "insufficient" | "ambiguous" | "conflicting"
reasons: tuple[str, ...]        # 多条并列，不再只有一个字符串
missing_tasks: tuple[str, ...]  # SUB_QUERY 时：计划里有、但一条证据都没产出的那几路
retry_action: RetryAction       # 门只"建议"，不执行
retry_count: int                # 由 pipeline 写回，门自己不改
top_score / margin / n_docs / qcov
```

判据顺序写死（与 `reasons` 的归因顺序一致）：空结果（区分是否有业务过滤）→ `below_threshold` →
`single_weak_evidence` → `conflicting` → `ambiguous` → `low_query_coverage` → `sufficient`。
**顺序不能乱**：同一份观测既算 `below_threshold` 又算"孤证"，两张诊断表加起来就会超过总数。

`reason`（单字符串）保留为旧响应体形状，**只给"只想显示一句原因"的调用方**，不许从它反推判据。

**原因 → 修复动作**（`RagPipeline._repair_plan`，`_REPAIRABLE` 之外的直接上交编排层）

| 判据 | 动作 | 实际做了什么 |
|---|---|---|
| `no_candidates`（无业务过滤） | `retry_direct` | 用原问句、`DIRECT` 再检一次 |
| `no_candidates_with_filter` | `relax_business_filter` | 只清四项业务过滤 —— `subjects` / `doc_types` / `stages` / `grades`（`RetrievalFilters.relaxed()`） |
| `below_threshold` | `retry_original_query` | 同上，忽略策略重检 |
| `low_query_coverage` | `rewrite_query` | 没有改写模型时**如实退化为原问句**，记 `retry:rewrite_to_original`，不假装改写过 |
| `single_weak_evidence` | `expand_neighborhood` | `top_k × 2` 扩大同文档邻域 |
| `ambiguous` | `ask_disambiguate` | **不修**，交编排层追问 |
| `conflicting` | `return_conflict` | **不修**，并列返回或转人工 |
| `sufficient` | `none` | — |

三条硬约束（都有单测守着）：

1. **至多一次**：`max_repairs=1`。第二次仍不过门 → `retry_action` 收敛成 `fallback` + `degraded=("retry_exhausted",)`。
2. **修完必须重新过门**：不存在"改过过滤就当过了"。另：**一次都没试**（`max_repairs=0` 或动作不可修）不许报 `retry_exhausted`——
   那会把报告里的"重试收益"做成假数。
3. **安全过滤不可放宽**：`relaxed()` 只清业务过滤；`tenant_id / acl_tags / kb_version` 在每一次 retriever 调用上逐字节相同
   （`tests/test_rag_pipeline.py::test_security_filters_survive_relaxation`）。

**分阶段耗时（如实记录，缺的不编）**

| 层 | 键名 |
|---|---|
| `RetrievalResult.stages`（单次检索内部） | `query_embed / recall / parent_expand / fusion / rerank / total` |
| `PipelineOutcome.stages`（含修复） | 上面这些 + `retry`（只在真修过时出现）+ 自己的 `total` |
| `ChatState.latency_ms`（端到端） | `faq / retrieve / generate / session / memory / total` |

- `fusion` 恒为 **0**：Milvus 在**一次** `hybrid_search` 里完成稠密+稀疏融合，本层没有独立阶段，如实透传不虚填。
- 并行任务（SUB_QUERY）的各阶段取 **max 而不是 sum**：相加会把并行三路算成串行三倍。
- 计划书建议的 `recall_dense / recall_sparse` 被合并成 `recall`，原因同上（Milvus 内部融合，本层拿不到拆分点）。

**失败隔离与错误码**：检索全线失败 → `RetrievalUnavailable`（对外 `retrieval_unavailable`）；
部分路失败 → `degraded=("partial_recall_failed",)` 继续，不整体失败。
`base/errors.py` 补齐六类对外错误码 `invalid_request / forbidden / retrieval_unavailable / no_evidence / generation_unavailable / citation_invalid`，
`ChatResponse.error_code` 与 `ChatState.error_code` 都从这里出——**"检索坏了"与"知识库没有"在响应体和日志里永远分得开**。

**测试（§7.2.7-E 的确定性部分）**

| 文件 | 条数 | 盯的是什么 |
|---|---|---|
| `tests/test_rag_pipeline.py` | 15 | 每种状态迁移：放宽后**重新过门**、只修一次、不可修动作原样上交、`max_repairs=0` 不谎报耗尽、预算耗尽不修、BACKTRACK 无历史不检索、SUB_QUERY 一任务一路、全失败 `retrieval_unavailable`、部分失败降级、跨任务父块去重、阶段耗时为实测值 |
| `tests/test_planner.py` | 14 | 策略优先级与 reason、SUB_QUERY ≤3、HyDE 无模型时诚实降级、档位先验只认 DIRECT/SUB_QUERY、**意图只收窄 `doc_type`、学科只从槽位来**、安全三项只从 `QueryContext` 来 |
| `tests/test_query_processor.py` | 13 | NFKC（含 `²`→`2` 的既定行为）、空白/重复标点折叠、**不删词**、空问句 → `invalid_request`、超 400 字截断而非报错、幂等 |
| `tests/test_fusion.py` | 7 | 同分空间 `max(score × weight)`、父块去重并合并 `child_hits`、主证据换分高的那条但不吞掉已合并片段、同分时多路支持优先、权重数量不匹配退化为不加权 |
| `tests/test_evidence_gate_table.py` | 25 | 九条判据落点逐条断言（含 `reason/action`）、`missing_tasks`、`with_retry` 不改原对象、`as_dict` 同时给 `reason` 与 `reasons`、`relaxed()` **只清业务过滤** |

新增 **71 条**，全量 **336 passed**（2026-10-29 晚补齐第一轮缺口后为 **415 passed**，见 7.2.7-E）。

**DoD 逐项（计划书的九条）**

| # | 条目 | 现状 |
|---|---|---|
| 1 | `/v1/chat` 统一编排覆盖 FAQ hit/miss、RAG 四态、依赖异常 | ✅ `tests/test_chat_api.py` 19 条、`test_chat_stream.py` 18 条 |
| 2 | DIRECT 跑完 normalize → plan → recall → fusion → parent expansion → rerank → evidence build → gate → generate/fallback → citation | ✅ |
| 3 | 安全过滤检索前生效，retry/放宽都无法移除 | ✅ 单测逐字节断言 |
| 4 | 门的原因与动作结构化，重试次数与总预算硬上限 | ✅ |
| 5 | 引用编号映射到真实 evidence，无支撑内容不静默保留 | ✅ `tests/test_citation.py` |
| 6 | 降级可预测，`retrieval_unavailable` 与 `no_evidence` 明确区分 | ✅ |
| 7 | 单测/契约/集成/API 全绿，mock 即可覆盖所有状态迁移 | ✅ 336 |
| 8 | 日志能按 `request_id` 还原链路与阶段耗时 | ✅ 字段齐（`request_id/trace_id/strategy/gate/degraded/stages/latency_ms`）；问句本体受 `LOG_QUERY_TEXT` 控制 |
| 9 | 真实数据质量指标单独报告，不伪造 | ⚠️ **未跑**，见下 |

**未做（写在这里，避免下次误以为已完成）**

1. ~~**HyDE 未接线**~~ **已接（2026-11-01）**：`main.py::Container._hyde_gateway()` 用 DeepSeek 生成假设性答案
   （显式点名 `LLM_QA_MODEL`，不是网关默认模型），预算走 `HYDE_BUDGET_MS`（1500），并同时套在
   `RagPipeline(hyde=..., hyde_budget_ms=...)` 的外层 `wait_for` 上；planner 由此拿到真实的
   `hyde_available`。三条护栏没变：HyDE 文本**只用于检索**（`_drop_hyde_echo` 保证它永不进证据与引用）、
   失败退回 `DIRECT + hyde_failed`、`FEATURE_HYDE=false` 即整体回滚。
   **诚实缺口**：打开它违反了 §5.4「改检索参数要附黄金集对照报告」——那份报告**没跑**，所以这是
   "按需求打开的实验性能力"，不是"已验证的召回优化"。回滚开关就是 `FEATURE_HYDE`。
2. ~~**复合问句拆分未接**~~ **已接（2026-10-31）**：`_build_understanding()` 用 `compound_query.split(state.query)`
   产出 `sub_queries`，planner 据此一任务一路并行检索（`tests/test_chat_understanding.py` 断言规格原例产生
   **三次** `retriever.search`）。拆不出来时退回单路，不猜。
   **2026-11-01 起拆分交给本地小模型**（`orchestration/chat.py::_split_compound` → `compound_query.split_with_model`），
   排在**意图之后**（拆几路取决于已判定的档位）、槽位收尾之后（会追问早退时先拆只是白付一次本地调用）。
   模型给的子问题必须通过字符多重集校验，编出来的词丢弃并退回规则；超时 / 异常 / 坏 JSON / 全部被拒
   各自记 `compound_timeout` / `compound_failed` / `compound_bad_json` / `compound_rejected`。
   `state.compound` 是一条唯一产物（`_build_understanding()` 只读它，不再自己 `split()` 一次）。
3. **黄金集未复跑**：`scripts/eval_rag.py` 的 763 条基线是**旧管线**（无 plan / 无融合层 / 无修复）跑出来的。
   新链路把 `fusion` 换成同分空间的 max×weight，理论上与旧 Milvus 融合分同尺度，但**没有实测证据**。
   Recall@10 / MRR / 门放行率 / 延迟基线必须在黄金集上重跑后才能改 §7.2.5 的数字——**这里的空白不是"没变"，是"没量"**。
   现在还要加上 HyDE 与模型拆分两个变量：同一份报告必须能区分"是链路变了"还是"这两步改变了口气"。
4. **Milvus 集合需重建**：`tenant_id / acl_tags / kb_version` 已进入 schema 与标量过滤表达式，
   旧集合没有这三列，`expr` 里引用会直接查不出东西。切到本版本前先 `python scripts/init_milvus.py --recreate`（或建 `edurag_v{N+1}` 再切 alias），
   并重新入库；`data/ingest_manifest.json` 与 `data/doc_fingerprints.json` 可留着，近重复检测照常生效。

#### 7.2.7-E 第一轮缺口补齐（2026-10-29 晚）

对照实施手册复查第一轮，补了五项此前**契约写了、代码没落地**的东西。判断依据一律是手册，不是"看起来缺"。

| # | 缺口 | 落点 | 为什么必须补 |
|---|---|---|---|
| 1 | `QueryTask.query_role` | `base/contracts.py` 的 `QueryRole = Literal["original","rewrite","hyde","sub"]`；`planner.build_plan()` 按策略标注（HYDE 路 → `hyde`，SUB_QUERY 拆成 ≥2 路 → `sub`，其余沿用调用方角色）；`RagPipeline.run()` 自动判定 `rewrite`——**与 `ctx.original_query` 比，不新增入参** | 手册 §2.3/§4.2 要求计划里能分清"这一路送进检索的是原话、改写句、假设性答案还是子问题"。否则接上 HyDE / 复合拆分之后，日志与报告里所有检索都长得像"用的原话"，A/B 无法归因。多一个参数就多一处能忘传，所以从上下文推导而不是加形参 |
| 2 | `doc_type_too_narrow` / `subject_filter_empty` 两条 reason | `evidence_gate._empty_reason(filters)`（空结果时唯一的分诊点） | 原来只回一个 `no_candidates_with_filter`，"学科过滤把答案挡在外面"与"doc_type 收窄太狠"分不开，§7.7 硬规则 6（`doc_type` 收窄 0 命中要放宽一次）与 §7.1.8 的错误过滤率统计都缺输入。两条都走 `relax_business_filter`，**安全三项一次不动** |
| 3 | 安全三项的空值守卫 | `RetrievalFilters.__post_init__`：`tenant_id` / `kb_version` 为空（含全空白）即抛 `PolicyViolation`（400） | 空租户进 Milvus 表达式会退化成 `tenant_id == ""`；更危险的是"不传就不加过滤"的实现会变成**跨租户检索**。与其在每个调用侧记得校验，不如让构造本身失败——漏传的地方立刻炸在单测里 |
| 4 | 身份 → 检索过滤的完整通路 | `X-Tenant-Id` / `X-ACL-Tags` → `_identity()` → `ChatState.tenant_id / acl_tags` → `QueryContext.filters()` → `RetrievalFilters` → 检索器实参 | 手册要求 `tenant_id / acl_tags / kb_version` **由接入层决定、不从外部透传**。此前 `QueryContext` 只有 `settings.tenant_id` 一个来源，多租户部署时同一进程没法同时服务两个学校 |
| 5 | 引用定位降级 | `Evidence.locator_kind` / `Evidence.locator`（`base/contracts.py`）+ `citation._citation()` 的 `title/locator/locator_kind` | `page` 在当前语料里 **3256/3256 全为 0**，按页码定位根本不可达（见 §7.8 引用契约）。降到 `page > section > title > file_name` 逐级，`locator_kind` 如实告诉前端降到了哪一级 |

**顺带修掉的一个真缺陷**：`_repair_plan("retry_original_query")` 原先用**传进 `run()` 的 query** 重检，
而那个 query 可能已经是会话改写后的——等于把同一句话再检一遍（分数不变、白花一次预算），
与手册 §5.3「`below_threshold` → 对原 query 再召回」不符。现在改用 `ctx.original_query`；
修复后的计划也改为用**它自己的 query** 执行（`retry_query = plan2.tasks[0].query`），
否则 `RetrievalResult.query` 与实际检索文本不一致，证据门的 `qcov` 会拿错的问句去比。

**`_trust_headers` 改成白名单**：只有 `dev / development / local / test / testing` 才信任客户端自报的租户与 ACL。
原实现是"排除 `prod` / `production`"，**staging 因此被算成安全的**——预发环境连的常常是生产数据，
这是最容易被忽略的一处跨租户口子。将来 `APP_ENV` 多出一个没见过的取值时，白名单让它默认安全。

**测试**：新增 `tests/test_permissions_contract.py` **26 条**（安全三项空值即抛、`QueryContext` 无租户不能建过滤、
`relaxed()` 逐字段保留安全三项、空 ACL 不是通配、`_identity` 信任/不信任两种形态、无 `user_id` 不发 `session_id`、
`_trust_headers` 白名单逐环境、**HTTP 头 → 检索器实参**的完整链路在 dev 生效 / 在 prod 被丢弃 / 缺省回落到部署配置）；
`tests/test_citation.py` 新增 4 条 locator 降级用例；`tests/test_evidence_gate_table.py` 的空结果参数表由 11 行扩到 14 行。
全量 **415 passed**。

#### 7.2.7-F 重排仲裁与检索时间预算（2026-10-29 晚，一次真实故障的收口）

**故障现象**：`GET /v1/rag/search?q=勾股定理的逆定理是什么？` 返回 `cost_ms=231974`
（`rerank 227443` + `query_embed 3414`），`degraded=["budget_exceeded"]`。

**两个都是真 bug，且都不是"语料没有这一章"造成的**：

| # | 缺陷 | 为什么没被发现 | 修法 |
|---|---|---|---|
| 1 | **重排绕过配置**：`planner.build_plan(rerank: bool = True)` 的默认值让 `RagPipeline.run()` 不传 rerank 时计划里恒为 `True`；而 `retriever.search()` 只在 `rerank is None` 时才读 `RAG_RERANK_ENABLED`，于是 `auto`（本机无 CUDA → 本该跳过）与 `.env` 同时失效。`rag_system/api.py` 还有第二处 `rerank if rerank is not None else True` | `/v1/rag/stats` 读的是**启动配置**（`rerank_enabled: false`），日志里的 `reranked` 读的是**实际执行**（`True`）。两个数各自都对，坏的是中间那条仲裁链——除非把两处并排看，否则看不出矛盾 | 三层统一为 `bool \| None = None`（`RetrievalPlan.rerank` / `build_plan` / `_plan`），`api.py` 去掉 `else True`。**`None` 的语义是"不表态"**，由 `RAG_RERANK_ENABLED` 独裁；调试接口传 `rerank=true` 仍能强制开 |
| 2 | **单次检索没有时间上限**：`rag_pipeline.py` / `retriever.py` 里 `wait_for` / `timeout` 一个都没有，`budget` 只在"要不要再修复一次"时被读一次。§6.1 给的检索 200 ms 只写在文档里 | 只有慢到分钟级才会暴露；`budget_exceeded` 是**事后**补的标记，看起来像"已处理"，实则一分钟内没有任何东西在拦 | `_recall()` 用 `asyncio.wait_for` 卡住**剩余预算**；超时先关重排重检一次（`retrieval_timeout`），仍超时才抛 `RetrievalUnavailable` |

三条口径写进代码注释与单测：

- **重排是降级项，不是必需品**：超时后砍掉它重检，比"整条链超时 → 一条证据都没有"有用得多（CPU 上关掉重排同一批候选几十毫秒）。`plan.rerank is False` 时不再重检——没有可降的层级了，直接判 `retrieval_unavailable`。
- **修复那次失败不能推翻第一次的结果**：`_execute` 在修复轮抛 `RetrievalUnavailable` 时保留首轮 `outcome`，记 `retry_failed` 后 `break`。否则"多试一次没打成"会把本来能答的请求降成服务异常。
- **预算按剩余给，不按总额给**：两次召回共用同一个 8 s（`budget - 已耗时`），各给一份总额等于把"至多修复一次"变成"至多跑两倍预算"。

新增单测 5 条：`test_rerank_default_is_unset_not_true`（计划默认 `None`）、
`test_plan_rerank_reaches_the_retriever_as_unset`（`None` 一路到底不被填）、
`test_recall_timeout_drops_rerank_and_retries_instead_of_hanging`（超时 → `rerank=False` 重检 → 照答）、
`test_recall_timeout_with_rerank_already_off_is_retrieval_unavailable`（没有可降层级 → 检索不可用，且不白跑第二次）、
`test_failed_repair_keeps_the_first_round_result`（修复超时 → 保留首轮结论 + `retry_failed`）。全量 **453 passed**。

> **这次故障里不属于代码的部分**（不改代码，也不能靠调参解决）：
> ① `勾股` 全库（16 476 块）只有 **4 块**命中，且没有一块是讲这一章的——`初中/数学/八年级/教材.pdf` 只有
> **八上**（第十一~十五章：三角形 … 分式），**八下（第十七章 勾股定理）不在语料里**；七年级只有一~四章、
> 九年级只有二十一~二十五章，同样是"每册只有上册"。命中那 3 块来自**高中数学 高一**的「拓广探索」
> （`设 a,b,c 分别是 △ABC 的三条边`，正是勾股逆定理的典型习题），是**相邻知识点**而不是本章正文。
> ② `page` 字段在初中三册 **1622/1622 全为 0**，引用只能靠「教材名 · 章节路径」定位（与 §7.2.8 一致）。
> 证据门说 `below_threshold` 是**正确结论**，`top_score 0.0492` 也是真实的低分，不是向量坏了
> （同一批块自搜索 `1.0000`、同章 0.72/0.69，`embedder` 全为 `bge-m3:cpu`）；重排关掉后同一问句的
> **融合分 top1 = 0.7252**，只比 τ=0.74 低 0.015——语料补齐后这一条会自然过门。
>
> 语料入库仍在进行（用户明确"暂时不要管"）：`高中数学 高一` 在故障当时尚未入库，写这份记录时已经进了
> **1327 块**，所以本节只断言"勾股那一章不在库里"，不断言"哪份文件在不在库"——后者会随时失效。

#### 7.2.7-G 标量过滤从未生效：`hybrid_search` 没有 `filter` 参数（2026-10-29）

**故障现象**：`GET /v1/rag/search?q=新加坡是一个什么样的国家&subject=地理` 返回的 6 条证据里，
1 条地理、4 条道德与法治、1 条历史；**去掉 `subject` 参数再查一次，6 条逐条相同、分数也相同**。

**根因不在我们的代码里，在 Milvus 客户端 API 的形状上**：

| 事实 | 实测 |
|---|---|
| `MilvusClient.hybrid_search(...)` 的签名 | `(collection_name, reqs, ranker, limit=10, output_fields=None, timeout=None, partition_names=None, **kwargs)` —— **没有 `filter`**（本机 pymilvus 3.0.2 实测） |
| 原实现 | `hybrid_search(..., filter=expr or "")` — 落进 `**kwargs`，由 `_search` 吞掉，不报错不告警 |
| 反证 | `filter='subject == "不存在的学科"'` **照样返回 10 条结果** |
| 唯一生效入口 | `AnnSearchRequest(data, anns_field, param, limit, expr=...)`，**dense 与 sparse 两路各持一份** |

**影响面远不止学科**：`tenant_id` / `acl_tags` / `kb_version` / `subject` / `doc_type` / `stage` / `grade`
全部走同一个 `_build_expr()` → 同一个被丢弃的 kwarg。也就是说在这套部署上
**租户隔离、ACL、索引版本隔离同时失效**——这违反 §3.3.3「标量过滤强制前置」与 §5.6 第 5 条，
只是此前没人把两个租户的数据放进同一个集合，所以没被撞见。
真正有效的只有 `partition_names`（物理分区），这也是"`doc_type` 收窄看起来管用"的原因：
它顺带被物理隔离了，掩盖了同一份表达式在标量维度上完全没起作用。

**修法**（`rag_system/store/milvus_store.py::ChunkStore.search`）：把 `expr` 挂到**两路** `AnnSearchRequest` 上，
并从 `hybrid_search` 调用里删掉 `filter=`。只挂一路不行——另一路仍全库召回，0.3 的权重足够把脏块带进 top-k。

**验收（连真库，同一问句）**：

| 查询 | 命中学科 | top1 |
|---|---|---|
| `subject=地理` | 地理 ×6 | **0.7393** |
| `filters(subjects=("地理",))`（线上路径） | 地理 ×6 | 0.7393 |
| `subjects=("不存在的学科",)` | **0 条** | — |
| 无过滤（旧行为基线） | 道德与法治 ×4 / 地理 ×1 / 历史 ×1 | 0.6939 |

> 顺带说明：`0.7393` 与旧的 `0.6939` 是两个不同的第一名。过滤生效前，全库里的道德与法治块把真正的地理块挤到了后面；
> 过滤生效后地理学科内部的 top1 才露出来。**这意味着此前所有检索类指标都在"没过滤"的语料上量的**，
> §7.2.5 的 Recall@10 / MRR 需要在修复后重跑（与"语料换了一代"是同一件事，见 §7.2.8 未做第 1 条）。

**为什么没被测试抓到（覆盖缺口）**：`tests/test_evidence_gate_table.py` 只断言 `_build_expr()` 的**字符串内容**，
从没断言它**到达请求对象**。新增 `tests/test_rag_store_filter.py` 7 条补上这一环——用假 client 拦住
`hybrid_search`，断言 `[req.expr for req in reqs]` 两路都含 `subject == "地理"`、安全三项也在，且
**`"filter" not in kwargs`**（这条是防线本身：谁把它改回顶层 kwarg，测试立刻红）。全量 **453 → 460 passed**。

**运维提醒**：`python main.py` 起的进程要**重启**才会用上新代码；重启后 `/v1/rag/search` 的诊断响应里
仍然不回显 `filters.subjects`，判断是否生效看证据的 `subject` 分布是否唯一。

#### 7.2.7-H 年级/学段下推接线（2026-10-30）

`-G` 修好的是"表达式有没有到达请求对象"，这一条修的是"**表达式里有没有年级/学段**"：
`milvus_store._build_expr()` 早就能把 `stage` / `grade` 拼进 `expr`（`_enum_conds` 生成
`stage in ["初中"]`、`grade in ["八年级"]`，与 `doc_type in [...]` 同一套），
但从 `QueryContext` 到那里中间断了一环——`filters()` 不接受这两个参数，上游永远传空。

| 项 | 落点 | 约束 |
|---|---|---|
| 契约 | `base/contracts.py::QueryContext` 增 `stages` / `grades`，`filters()` 增同名关键字参数（空则回落实例字段） | 只在 `QueryContext` 上改一处，`planner.py` 取 `ctx.filters(...)` 的路径**自动生效**，DIRECT / SUB_QUERY / BACKTRACK 三种策略不用各改一遍 |
| 取值 | `orchestration/chat.py::_context()`：`grades = (slots.grade,)`，`stages = (labels.GRADE_STAGE[grade],)` | 学段由 `GRADE_STAGE` 推出，**不单独抽**；`FEATURE_GRADE_FILTER=false` 时两者都给空元组（现场止损） |
| 放宽 | `RetrievalFilters.relaxed()` 现在清**四项**业务过滤（`subjects/doc_types/stages/grades`），安全三项原样 | 年级误抽把候选清空时，`_empty_reason()` → `no_candidates_with_filter` → `relax_business_filter` 能自动救回；救不回来是"真的没有"，不是"过滤太紧" |
| 单测 | `tests/test_slots_wiring.py` 9 条 + `tests/test_rag_store_filter.py` 新增 2 条 | 断言的是"字段抵达执行点"：两路 `AnnSearchRequest.expr` 都含 `stage in [...]` 与 `grade in [...]`，`relaxed()` 之后只剩安全三项 |

**安全边界不变**：`tenant_id` / `acl_tags` / `kb_version` 只从 `QueryContext` 来（接入层给），
意图与槽位都不得覆写；放宽也只放宽业务项。

#### 7.2.8 语料换代：初高中教材 / 教案 / 试卷接入（2026-10-29）

项目方把语料换成了**按学段 × 学科 × 文档类型**组织的完整教科书语料，`document_loaders` 与入库配置随之调整。

**语料结构**（`datasets/` 只读，不改不删）：

```
datasets/
├─ 01_textbook/      {初中|高中}/{学科}/教材.pdf     19 份（18 本教材 + 1 份资源说明页）
├─ 02_lesson_plans/  {初中|高中}/{学科}/教案.html|docx 17 份
├─ 10_exam_papers/   {初中|高中}/{学科}/试卷.html|pdf  18 份
└─ 08_subtitles/     *.srt                          4 份（旧语料，保留）
```

教材是完整教科书：82–270 页、6–66 MB；`高中/数学/教材.pdf` 最长（270 页 / 63 MB）。

**三个新特点，决定了必须按路径判语义，不能只看文件名**：

1. 文件名是**通用词**（`教材.pdf` / `教案.html` / `试卷.html`），学科与学段在目录上；
2. 同一学科跨学段存在同名文件（`初中/数学/教材.pdf` 与 `高中/数学/教材.pdf`）；
3. 大半是 **HTML**（31 份），旧工厂只认 md/txt/docx/pptx/pdf/srt/图片。

| 项 | 落点 | 说明 |
|---|---|---|
| HTML 加载器 | `rag_system/document_loaders/loader_html.py`（新） | 标准库 `HTMLParser` → Markdown，不引第三方依赖：`h1..h6`→`#…`、`li`→`- `、`p/div/tr/br`→换行、`td/th`→`\| ` 且 `</tr>` 补收尾竖线；`script/style/svg/noscript/head` 整段丢弃（MathJax 与样式表留着就是脏 chunk）；行内 LaTeX（`$x^2$`）原样保留；空正文**抛错**而不返回空串 |
| 空元素踩坑 | 同上 `_SKIP_TAGS` | `meta`/`link` 是空元素，放进"整段吃掉"的集合里自增不自减，会把后面正文全吞掉（第一次跑出来就是空文档）；`<title>` 单独走 `_in_title`，只做标题不当正文 |
| 文档类型 | `thy_factory._DOC_TYPE_BY_DIR` | 目录名**决定** `doc_type`：`01_textbook→textbook`、`02_lesson_plans→lesson_plan`、`10_exam_papers→exam`；从文件所在目录往上走，先命中者胜；无目录线索才退回文件名关键词 |
| 标题 | `thy_factory._title()` | stem 是通用词（`_GENERIC_STEMS = {教材,教案,试卷,讲义,教学设计}`）时拼 `学段+学科+类型`（`初中数学教材`），否则用 `<title>`/文件名；否则 18 本书在引用卡片里全叫"教材" |
| 学科 | `base.py::guess_subject` + `SUBJECT_ALIASES` | 按路径切分后逐段匹配，别名归一：`生物学→生物`、`思想政治/思想品德/道法/品德→道德与法治`；认不出返回 `None`（**不猜**）。否则 `高中/思想政治/教材.pdf` 与 `初中/生物学/教材.pdf` 会掉进 `unlabeled` |
| 导航页 | `thy_factory._SKIP_NAMES` | `索引.html` / `资源说明.html` / `index.html` / `readme.html` 不入库（正文是链接列表，进库就是噪音） |
| **近重复判据修复** | `dedup.py::normalize_name` | **见下条**，这是本次唯一的严重 bug |

**`normalize_name` 的假重复事故**：原实现只取文件名（stem）做归一标识，于是 18 本教材全部归一成 `教材` 一个 key，
第一次全量 dry-run 的结果是 `files: 52, docs_ok: 3, duplicates: 49`——整库只剩一本书，其余 17 本以
`doc_duplicate: 与 …\初中\化学\教材.pdf 近重复（name）` 被拒。判据本意是拦"同一本书的两次下载"
（§7.2.6 的 `人教版-数学-九年级上册.pdf` / `人教版数学九年级上册.pdf`），却把"不同学科的同名文件"当成重复。
改为 **最后两级目录 + 文件名** 一起归一后两个方向都对：同目录的两次下载仍相等，不同学科的同名 `教材.pdf` 不再撞车。
这条与 §7.2.6 的"宁可漏拦不能误杀"同源——**多拦一份比漏拦一份糟得多**，所以宁可让 key 更细。

**入库范围**（`.env` / `.env.example` 的 `RAG_DOCS_DIR`）：

```
RAG_DOCS_DIR=datasets/01_textbook,datasets/02_lesson_plans,datasets/10_exam_papers,datasets/08_subtitles
```

- **加入 `10_exam_papers`**：试卷是"题目讲解"最直接的证据。
- **不含 `09_textbooks`**：它存的正是 `01_textbook` 里同一本九年级数学的另一次下载（§7.2.6 已记过）。
  两份都入库，同一段证据会以"两篇文档"的身份互为对手，证据门的**跨文档 margin** 直接判 `ambiguous`——
  这不是靠 dedup 每次都拦得住的事（两份的 simhash 距离是 26，拦它们的是文件名判据，换一批文件名就漏）。

**意图档位**：`understanding/profiles.py::KB_STEPWISE`（题目讲解）的 `doc_types` 由
`(textbook, lesson_plan)` 改为 **`(exam, textbook, lesson_plan)`**。不改的话结果更糟：
**加之前没有任何档位包含 `exam`**，而收窄了 `doc_type` 的检索会把它下推成 Milvus 前缀过滤，
试卷落了库也永远召不回来。§7.7 的档位表同步。

**教材语料暴露的两个硬限制（都已修，且都不是"只在新语料上才出现"的问题）**

1. **切分：`_pack` 只在"句"之间装箱，一句超长就整句进块。**
   MinerU 解出来的表格行、公式行、没有句号的标题串，几千字一个换行都没有。
   实测（`data/mineru/*/content.md`，子块目标 300 字）：修复前最大子块 **12 140 字**（40 倍于目标），
   `parent_text` 同样超限。修复为 `_hard_split()`——单句按 `size` 硬切、相邻片段保留 `overlap` 重叠、
   尾片整段落在上一片重叠区时丢弃（否则同一段文字被喂给 embedding 两遍）。
   修复后同一批缓存 markdown 的最大子块 **331 字**、最大父块 **1 300 字**。
   **在切分层兜住而不是在写库层静默截断**：截断丢掉的正是引用要定位的那半句，
   "写进去了但对不上原文"比"写不进去"更难查。
2. **写库：Milvus 的 `max_length` 按 UTF-8 字节校验，不是字符。**
   `milvus_store.py` 原注释写着"按字符算"，于是按字符截断对中文毫无保护——
   一个 1 401 汉字的子块 = **4 203 字节 > `max_length=4096`**，`upsert_rows` 直接报
   `invalid length string`，整份教材入库失败（第一轮 18 本教材全部卡在这里）。
   改为 `_clip(text, max_bytes)` 按字节截断，全部写库字段（含 `source/title/heading/tenant_id/kb_version/acl_tags`）
   统一走它。

**单测**：`tests/test_rag_loaders.py` 新增 11 条（HTML 结构→Markdown、`script/style` 丢弃、`<title>` 与行内公式保留、
学科从路径推断、目录决定 `doc_type`、通用 stem 拼标题、导航页被跳过、学科别名归一），
`tests/test_dedup.py` 新增 2 条（`normalize_name` 保留上级目录、不同学科的同名文件不是重复），
`tests/test_rag_split.py` 新增 3 条（无标点长串被硬切且父子块都不超长、`_hard_split` 的长度与重叠、
`overlap ≥ size` 时不打转），`tests/test_rag_store_clip.py` 新增 3 条（字节而非字符、不切半个字符、空值安全）。
全量 **355 passed**。

**未做（本轮仍是空白，别当成已完成）**：

1. **黄金集未复跑**：`scripts/eval_rag.py` 的 763 条基线是旧语料（6 份短文档 + 1 本教材 / 622 块）上量的；
   语料换成 18 本教材后 Recall@10 / MRR / 门放行率/阈值 τ=0.74 全部**不可比**，必须重跑才能改 §7.2.5 的数字。
2. **证据门阈值未重标**：`0.74 / 0.01 / 0.20` 是旧语料标定的。教材进来了，可答题的证据分整体会变。
3. **档位 A/B 未复跑**：`KB_STEPWISE` 加了 `exam`，`intent_ab_report.md` 的结论是在不含试卷的语料上得出的。
4. **题库仍未入库**：`datasets/03_question_bank` 的 answer 多数只有一个选项字母、analysis 为空，不是可用语料。
5. **重排仍是 CPU 跳过态**：本机无 CUDA，`RAG_RERANK_ENABLED=auto` 实际不重排，按融合分排序。

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

> **下表是 2026-10-30 之后的实际顺序**，与 10-27 定稿版有两处不同，两处都是实现时发现"设计顺序做不到"：
> ① 原第 2 步的"会话改写"与第 4 步的"意图"被合并成一次大模型调用，并**挪到 FAQ 之后**（原设计理解为与 FAQ 并行）；
> ② 原第 4 步的 `FAQ ∥ 意图 ∥ 记忆召回` 三路并行拆开：FAQ 串行最前，记忆召回与检索并行。
> 理由见 §7.7 开头的偏离说明与 §6.1 的预算块。

| # | 阶段 | 模块 | 输出 | 预算 | 失败降级 |
|---|---|---|---|---|---|
| 1 | 鉴权/限流 | 接入层 | user_id/tenant_id | 20 ms | 401/429 |
| 2 | **归一化（确定性）** | rag 侧 `query_processor` | normalized | 5 ms | 用原 query |
| 3 | L1 缓存*（未接线） | infra | answer | 10 ms | 未命中 |
| 4 | **FAQ（单路，串行最前）** | qa | FAQHit | 60 ms | 未命中即继续，不报错 |
| 5 | FAQ 哪个档？ | qa | **high 一到就返回（本次请求零次大模型调用）**；mid/low 继续并带 top3 候选（§7.1.9） | — | 落空继续 |
| 6 | **改写 + 意图（一次大模型调用）** | understanding | rewritten / IntentResult | ≤ `UNDERSTANDING_TIMEOUT_MS`（默认 800） | 改写退回归一化句、意图退规则分类器，**各自独立** |
| 6.5 | **复合问句拆分（本地小模型）** | understanding | `sub_queries` | ≤ `COMPOUND_TIMEOUT_MS`（默认 400） | 规则兜底：`compound_timeout` / `compound_failed` / `compound_bad_json` / `compound_rejected` |
| 7 | L2 语义缓存*（未接线） | infra | answer | 60 ms | 未命中 |
| 8 | 记忆召回 **∥** 检索+重排（含 HyDE） | memory ‖ rag | MemoryContext / chunks | 召回 ≤120 ms；检索 350 ms；HyDE ≤ `HYDE_BUDGET_MS`（1500） | 召回超时只丢背景（不抛）；检索降级排序；HyDE 失败退回 `DIRECT + hyde_failed` |
| 9 | 证据门 | rag | pass/fail | 5 ms | 兜底 / 追问 |
| 10 | 生成+引用 | rag + memory(上下文) | 流式文本 | 剩余预算 | 切模型/兜底 |
| 11 | 回写 | memory + infra | 历史/记忆/指标 | 异步 | 不阻塞 |

\* L1/L2 共享缓存尚未接入编排层（§7.9「已知未做」第 3 条），保留在表里是因为它决定了 `cacheable` 这条规则的落点；当前这两行不产生耗时，也不产生缓存命中率。

**关键约束**：FAQ 在第 4 步**串行**跑在最前面，因此"FAQ 不等意图"不是靠抢跑实现的，而是**结构上不可能等**（§7.7 硬规则 5）。这条对延迟的意义比原设计更大：占台账 53% 的高置信命中连一次大模型调用都不付（`tests/test_chat_rewrite.py` 断言 `understanding_calls == 0`）。

第 8 步的并行是**单向的**：记忆召回慢不影响检索（`recall()` 自带 `wait_for` + fail-open，join 不会抛）；反过来检索慢也不影响记忆写入——写入走 `_persist_later`（`asyncio.create_task`），在应答之后。

### 7.5 落地顺序（按 PR 切，每个 PR 可独立验收）

| PR | 内容 | 完成定义（DoD） |
|---|---|---|
| **PR1 地基** | `infra/`（config/db/cache/obs/inference）、DDL、契约数据类、`/healthz` `/readyz`、启动预加载+预热 | 空壳服务能起、readyz 三模型就绪、DDL 可重复执行 |
| **PR2 qa_system** | repository/corpus/matcher/service + 热更新 + import_faq | 200 题黄金 FAQ top1 ≥0.95、p95 ≤15 ms |
| **PR3 rag_system** | 摄入流水线 + 检索/重排/证据门 + 生成/引用 + 流式 | 黄金集 Recall@10 ≥0.90、引用可定位 100% |
| **PR4 记忆** | 读写链路 + 记忆页 API + 链删除 + 召回降级 | MemRecall@5 ≥0.85、删除 2 s 生效 |
| **PR5 编排+前端** | 理解层（意图分类 / 会话改写 / 槽位）+ LangGraph 12 节点、WS 四类帧、前端三件套、反馈闭环 | SLO 表逐项达标、降级演练全绿；意图 F1 ≥0.90 只是门槛，**DoD 是档位 A/B 证明下游增益**（Recall@10 降幅 ≤1pp、错误过滤率不升、转人工率不升） |

顺序不可乱：qa_system 先做，因为它**不依赖 GPU 与 LLM**，可以最快验证"链路 + 观测 + 降级"这套骨架是否成立。

### 7.6 契约速查（实现时以此为准）

```python
SlotExtractor.extract(q, history) -> SlotResult          # 规则在 qa_system/res/qa_slots.py
IntentClassifier.classify(q, history) -> IntentResult    # 6 类 + unknown，档位见 §7.7
FAQService.search(q, slots) -> FAQHit                    # slots=None 即 not_provided（真的全库）
Retriever.search(q, k, filters, strategy) -> list[Chunk]
Reranker.rerank(q, docs, top_n) -> list[Chunk]
EvidenceGate.check(chunks, intent) -> Evidence(ok: bool, reason: str, top_score: float)
LLMGateway.stream(msgs, budget_ms, model) -> AsyncIterator[str]
MemoryService.recall(user_id, sid, q, budget_ms) -> MemoryContext
MemoryWriter.on_turn(user_id, sid, turn) -> None
```

五个数据类 `SlotResult / FAQHit / Chunk / Evidence / MemoryContext` 定义在 `base/contracts.py`，**三模块只依赖它，不互相依赖**；理解层的 `IntentResult / IntentProfile` 同样定义在这里（见 §7.7）。

### 7.7 理解层：意图分类与下游路由（PR5，2026-10-30 已接线）

**目标**：把 query 稳定归到 6 类之一，并把它翻译成**一个路由档位**；下游节点只读档位，不读意图字符串。

**边界**：理解层不检索、不生成、不写库；输入 `query + history`，输出 `IntentResult` + `SlotResult`（学科槽位规则版在 `qa_system/res/qa_slots.py`，见 §7.1.8）。理解层失败**不允许**影响 FAQ 命中。

> **与 §7.4 第 4 步的偏离（2026-10-30，有意的）**：设计写的是"FAQ ∥ 意图 ∥ 记忆召回三者并行"，
> 实现是 **FAQ 先跑、命中即返回，未中才开始理解**。理由两条：① FAQ 高置信命中占台账的 53%（§7.8.2），
> 并行意味着这一半请求白付一次大模型调用；② 串行在"命中"时**比并行更快**（省掉调用本身），
> 在"未命中"时只多出不存在的启动开销——理解本来就要在检索之前完成。
> §7.7 硬规则 5（FAQ 不等意图）因此由"靠 `as_completed` 抢跑"变成"结构上不可能等"，
> 这比并行版本**更强**，不是妥协。

`IntentResult` **没有 `subject` 字段**：学科过滤是槽位与检索路由的职责（§7.1.8），意图只允许收窄 `doc_type`。
分类器输出中文标签或内部 id 都可以，但**进编排层第一步就映射成 `IntentProfile`**，之后下游只读档位对象——`label` 只进日志、指标与打标，不参与任何分支。

**文件**：

```
understanding/
├─ llm_understanding.py # **大模型唯一入口**：analyze() 一次调用同时产出「改写句 + 意图」（2026-10-30 新增）
├─ intent.py            # 规则分类器 classify()（关键词加权打分），大模型不可用时兜底
├─ profiles.py          # label → IntentProfile 的**数据表**（不是 if/else）+ profile_for() + resolve_profile()
├─ rewrite.py           # 会话拼接改写 rewrite() + needs_rewrite()（指代，仍未接线）
└─ slots.py             # 模型版槽位（2026-10-31 已接线，任务 2.3）：**只校验/合并/收尾**，
                        # 学科/年级/教材版本/学期由 qa_system/res/qa_slots.py 的规则版给
```

> 这里很容易写错：**`intent.py` 里没有 `understand()`**，`rewrite.py` 里也没有 `llm_rewrite()`。
> 两者合并在 `llm_understanding.py::analyze()`——正是下面"意图与改写合并成一次调用"那张口径的落地形态。
>
> `slots.py` 也不是"第二个抽取器"：它**不抽**，只做三件事——`parse_model()` 校验模型给的 JSON
> （丢弃过滤类字段、词表外值、越界数字）、`merge()` 规则优先/模型补空、`finalize()` 把意图映射成
> `TaskSlots.kind` 并把坐标投影进任务槽位。抽取仍然只有一处：`qa_slots.py`（规则）。
> 模型槽位与改写/意图挤在**同一次** `analyze()` 调用里（`want_slots`），不额外付一次往返。

**三项能力的分工（2026-10-30 定案，用户拍板）**——三者不是"都用大模型"，也不是"都用规则"：

| 能力 | 谁来做 | 为什么 |
|---|---|---|
| **意图分类** | **大模型**（`understanding/llm_understanding.py::analyze()` 的 `intent` 字段），失败退规则分类器（`understanding/intent.py::classify()`） | 意图只影响"检索与生成怎么做"，分错最多是多查一轮；大模型对口语、生僻表述的覆盖比关键词表好 |
| **槽位（学科 / 多学科 / 年级 / 教材版本 / 学期）** | **规则**（`qa_system/res/qa_slots.py`），**不调模型** | 槽位会直接变成 Milvus 标量过滤，**抽错就是把正确答案挡在门外**。规则失败是"抽不到 → 全库检索"（安全失效），模型失败是"抽错 → 只查另一门学科"（危险失效）。这条不对称决定了必须用规则 |
| **槽位·非过滤类（章节名 / 知识点 / 题型·题量·目标考试等任务参数）** | **大模型补空**（`understanding/slots.py`，2026-10-31 / 任务 2.3），规则优先、模型只填 `None` | 这些字段不进检索过滤，写错最多是"多答一点"，与上面那条的危险程度不同级。`ModelSlots` 在**类型上**就没有过滤类字段，模型即使返回 `subject` 也被丢弃并记 `slot_model_filter_dropped` |
| **查询改写** | **大模型**（同一个 `analyze()` 的 `query` 字段），接在 **FAQ 之后** | 改写只改"用哪句话去检索"，改坏了还有出口校验兜底；接在 FAQ 之后是因为 FAQ 高置信命中占 53%（§7.8.2），先跑 FAQ 能让一半请求完全不付这次调用 |

**意图与改写合并成一次调用**：两者都在 FAQ 未命中之后、检索之前，位置相同，所以 `analyze()`
一次返回 `{"query": 改写句, "intent": 标签, "confidence": 分数}`，省掉一次往返。
但**校验与降级是分开的**：改写失败记 `rewrite_rejected` / 用归一化句，意图失败记 `intent_rejected` / 用规则分类器，
互不牵连——合并的是网络往返，不是两者的成败（`tests/test_chat_rewrite.py` 逐条守着）。
调用点唯一：`orchestration/chat.py::_understand()`，位置在 `_load_session()` **之后**、`_classify()` **之前**；
`want_rewrite` / `want_intent` 由开关决定，两个都关掉时**一次调用都不发**。

**开关**（全部默认 `true`，任一置 `false` 都是独立回滚）：

| 开关 | 关掉之后 |
|---|---|
| `FEATURE_INTENT` | 档位恒为 `BASELINE`（不收窄 `doc_type`、必检索、必引用），即 §7.8.4 的 A/B 基线 |
| `FEATURE_INTENT_LLM` | 意图退回规则分类器（§7.8.4 报的 F1 就是它的数） |
| `FEATURE_QUERY_REWRITE` | 不调改写，用归一化句检索；**两个都关掉时理解层零次大模型调用** |
| `FEATURE_GRADE_FILTER` | 年级/学段不下推（学科照旧），用于"年级误抽"的现场止损 |
| `UNDERSTANDING_TIMEOUT_MS`（默认 800） | 超时只丢改写与模型意图，检索照常用归一化句 |

> **诚实标注（别把旧报告当现状）**：§7.8.4 的 macro F1 **0.967** / `unknown` **5.59%** 与整份
> `intent_ab_report.md` 都是在**规则分类器**上量的。`FEATURE_INTENT_LLM=true` 是默认值，
> 也就是说**线上跑的不是那份报告里的分类器**——大模型路径目前只有接线单测（"结论到了档位""词表外标签回退规则"
> "低置信不猜"），**没有在 `intent_test.jsonl` 上评过 F1**。要引用分类质量，先关掉 `FEATURE_INTENT_LLM`
> 跑一遍，或重新在 1 020 条上评大模型版；在此之前任何"意图 F1 0.967"的说法都只描述已被替代的那条路径。

**契约**（`base/contracts.py`）：

```python
@dataclass(frozen=True, slots=True)
class IntentResult:                 # 只存"分类结果"
    label: str      # INTENT_LABELS 之一，或 "unknown"；**不含 subject**
    confidence: float
    method: Literal["model", "rule", "fallback"]
    profile: str    # 档位名，见下表；下游只认这个

@dataclass(frozen=True, slots=True)
class IntentProfile:                # 只存"策略配置"
    name: str
    retrieval: bool                 # 是否走检索
    doc_types: tuple[str, ...]      # 唯一可被意图影响的过滤条件，空元组 = 不限
    strategy: str                   # 策略先验 DIRECT / HyDE / SUB_QUERY / BACKTRACK
    require_citation: bool          # 策略：答案必须有可验证证据，不是"末尾拼 [n]"
    allow_general: bool             # 无证据时给"通用说明"还是转人工
    prompt_id: str                  # prompts/ 下的模板 id（带版本）
    cacheable: bool                 # 是否允许写共享缓存
```

**6 类 → 档位表**（`understanding/profiles.py`，改行为只改这张表）：

| label | 档位 | 检索 | `doc_type` 过滤 | 策略先验 | 引用 | 无证据时 | 共享缓存 |
|---|---|---|---|---|---|---|---|
| 知识点查询 | `kb_strict` | 是 | textbook, lesson_plan | DIRECT（短且抽象 → HyDE） | 必须 | 转人工 | 是 |
| 题目讲解 | `kb_stepwise` | 是 | **exam**, textbook, lesson_plan（2026-10-29 起含 `exam`，见 §7.2.8；题库接入后加 question_bank 并置首） | DIRECT；多小问 → SUB_QUERY | 必须 | 转人工 | 是 |
| 教案素材 | `doc_lookup` | 是 | lesson_plan, textbook | DIRECT | 必须 | 转人工 | 是 |
| 学习方法 | `general_advisory` | 是 | 不限（方法类证据散落，收窄会挡答案） | DIRECT | 有则标 | 通用建议 + 声明"非教材原文" | 是 |
| 备课辅助 | `generate_assist` | 是 | lesson_plan | DIRECT；含"设计 / 生成 / 出题" → SUB_QUERY | 参考来源必须标 | 允许生成，标注哪些是模型补充 | 是 |
| 闲聊 | `no_retrieval` | 否 | — | — | 不要求 | 直接短答（≤80 字） | 否 |
| unknown（<0.6 / 模型失败） | `generic_clarify` | 是 | 不限 | DIRECT | 有则标 | 通用回答 + 结尾追问 | 否 |

**意图对下游的 8 个作用点**（其余地方一律不看意图）：

| # | 下游 | 影响 | 落点 |
|---|---|---|---|
| 1 | 检索 | `retrieval=False` 跳过整条检索链（省 350 ms） | 编排层 |
| 2 | Milvus 过滤 | 只把 `doc_types` 加进 `expr`；`subjects` / `grades` / `stages` **只由 `SlotResult` 给**（`FEATURE_GRADE_FILTER` 管后两项），`tenant_id/acl_tags/kb_version` 由接入层给，意图一律不得覆写 | `Retriever.search(filters)` |
| 3 | 策略规划 | 档位只给**先验**；硬规则（指代 → BACKTRACK、连接词 → SUB_QUERY）优先，LLM 仅兜底 | 策略节点 |
| 4 | 证据门 | **首版不改 τ**（0.74）；档位只决定不过门之后的动作：`allow_general` → 通用说明，否则转人工 | 证据门 + 兜底节点 |
| 5 | 生成 | `prompt_id` 选模板：讲解要步骤、教案要四段式、闲聊要短 | `generation/prompts/` |
| 6 | 引用对齐 | `require_citation=False` 时允许无引用句，但必须标注"通用建议" | 引用节点 |
| 7 | 缓存 | `cacheable=False` 不写共享缓存；**记忆参与过的答案一律不写共享缓存**（key 无 `user_id`，会跨用户串味） | 缓存节点 |
| 8 | 观测 | `request_log.intent` + 分意图的转人工率 / 证据通过率 / 低置信比例 | 指标 |

**六条硬规则**：

1. **不把意图写成 `if/else`**：下游只读 `IntentProfile`。加档位 = 改表，加标签 = 改 `INTENT_LABELS`，两者都不碰下游代码。
2. **低置信不落 `闲聊`**：`label="unknown"`、`method="fallback"`、走 `generic_clarify` 且不缓存。把"没听懂"记成"闲聊"会让指标失真，还会把错误答案写进缓存。实现上 `INTENT_FALLBACK="闲聊"` 只是过渡值，PR5 接真实分类器时改为 `"unknown"`，`INTENT_NO_RETRIEVAL` 保持只有 `闲聊`。
3. **低置信不得触发高风险路由**：`unknown` 只能进 `generic_clarify`——不强制引用、不进生成型档位（`generate_assist`）、不转人工、不写共享缓存。转人工是证据门与兜底的结论，不是分类器给的。
4. **意图不越权碰过滤条件**：只允许收窄 `doc_type`。学科 / 年级 / 学段由 `SlotResult` 决定（§7.1.8、§7.9.1），租户 / ACL / `kb_version` 由接入层决定，三者都不看意图。
5. **FAQ 不等意图**：FAQ 高置信命中立即返回，不设"等意图分类完成"的屏障。**实现上比"并行抢跑"更强**——2026-10-30 起 FAQ 是**串行跑在最前**的，理解在它未命中之后才开始，所以"等"这件事在结构上不可能发生（见本节开头的偏离说明）。FAQ 是人工审核过的事实，不该被一次大模型调用拖住，更不该被它否决。
6. **`doc_type` 收窄导致 0 命中要放宽一次**：去掉 `doc_type` 重检，记 `degraded=("doc_type_relaxed",)`；仍空才走兜底。与 §7.1.8"别让过滤把正确答案挡在外面"是同一条教训。

> **两个"档"不是一回事，别串起来**：`IntentProfile`（意图档位，本节）决定**检索与生成怎么做**；
> `FAQHit.tier`（FAQ 置信档，§7.1.9）决定**这条 FAQ 结论怎么用**。意图不得改 FAQ 档位，FAQ 档位也不得改意图档位；
> high 命中仍然在意图分类完成前就返回（硬规则 5）。

**引用是两件事，别混成一件**：

| | `require_citation`（策略） | 引用对齐（机制） |
|---|---|---|
| 回答什么 | "这条答案**必须**有可验证的证据吗" | "正文写的 `[n]` 是否真对上了 chunk" |
| 在哪 | 档位字段（`IntentProfile`） | 生成后处理（§3.4.3 / `citation.py`） |
| 失败时 | 证据门不过 → 走无证据策略（`allow_general` ? 通用说明 : 转人工） | 剔除 / 改写无证据支撑的句子 |
| 禁止 | 为凑格式拼 `[1][2]`、伪造来源 | 把不相关片段硬塞进引用卡片 |

`require_citation=True` 的含义是"**没有可验证引用就不许答**"，不是"答案末尾要有方括号"。证据不足时只能走无证据策略，不能反过来给模型施压让它编引用。

**降级**：

| 故障 | 行为 |
|---|---|
| 意图模型未加载 | 规则分类（关键词表）；明显命中给 `method="rule"`、置信 1.0，否则 `unknown` |
| 推理超 30 ms | 放弃本次结果，用 `unknown`，记 `intent_timeout_total`（不拖 FAQ 的 15 ms） |
| 模型输出不在标签集 | `validate_intent` 归 `unknown` + 告警，不抛异常 |
| 大模型意图不可用 / 超时 / JSON 坏 | 记 `intent_rejected` / `understanding_*`，**退回规则分类器**，检索照常 |
| 改写不可用 / 超时 / 五道出口校验不过 | 记 `rewrite_rejected` / `understanding_*`，**退回归一化句**，意图照常生效 |
| FAQ 慢 | 与本节无关：FAQ 在理解之前，它慢只会推迟理解开始，不会被理解拖慢（§7.9 边界 4） |

**验收（两段，顺序不能反）**：

1. **前置门槛**：`datasets/06_intent/intent_test.jsonl`（1 020 条）F1 ≥0.90；`unknown` 比例 ≤10%；6+1 档路由单测全覆盖。
2. **真正的验收——档位是否改善下游决策**：黄金集上做 **A/B：档位路由 vs 全量 DIRECT 基线**，按意图切片比较：
   - `Recall@10` 降幅 ≤1 个百分点（分档不许伤召回）；
   - **错误过滤率不升**（因 `doc_type` 收窄导致正确证据没进候选的比例，必须单独统计，不能只看总 Recall）；
   - 证据通过率不降、转人工率不升；
   - `闲聊` / `no_retrieval` 档的检索延迟下降 ≥30%（这是唯一确定的收益）。

   判据：**如果分档把正确证据挡在外面，这套分类就是"昂贵的装饰品"**——回退全量 DIRECT，不保留只增复杂度的档位。F1 高但下游指标不改善，同样不予上线。

**暂缓**：① 按意图调证据门阈值（需分切片标定，先不改）；② 一句话多诉求的多标签（先取 top1）；③ 意图回压 FAQ（FAQ 结果优先，人工审核过的事实不该被分类器否决）。

---

### 7.8 生成层与编排层（PR3 收尾 + PR5 前置，2026-10-27 定案）

**要解决的问题**：检索链路已经能给出带页码的证据，但证据没有出口——没有生成、没有引用对齐、没有兜底话术。`/v1/rag/search` 只回答"该不该答"，不回答用户的问题。本节把"两级检索系统"补成"可用的问答系统"的最小闭环，七项按依赖顺序推进：

| # | 任务 | 依赖 | 完成定义 |
|---|---|---|---|
| 1 | LLM 网关（`llm_gateway.py`） | — | 统一模型调用 / 超时 / 异常映射 / 配置；无 key 时 chat 不可用但 FAQ、RAG 照常 |
| 2 | 引用对齐（`citation.py`） | 1 | `[n]` 只能映射到本次检索的 chunk；越界引用剔除；无支撑句标记；整体无引用时拒答 |
| 3 | 同步 `/v1/chat` | 1,2 | FAQ high 直返；mid/low 走 RAG；证据门不过走追问 / 兜底 |
| 4 | 端到端评估 | 3 | 答案正确性、引用正确率、拒答表现、延迟与召回 |
| 5 | SSE 流式 | 3 | `meta/delta/citations/done` 四类帧 |
| 6 | IntentProfile 接入 | 3 | 档位 A/B：召回降幅 ≤1pp、错误过滤率不升 |
| 7 | 会话与记忆 | 3 | 多轮指代、摘要、长期记忆召回 / 删除 |

**四条边界（违反即打回）**：

1. 生成只读本次检索的 `Evidence`；记忆（PR4）只提供背景，**不能作为引用来源**。
2. `orchestration/` 是唯一同时 import `qa_system` 与 `rag_system` 的包；两个业务包仍然互不 import（§5.1）。
3. 引用必须能映射回真实 chunk；**禁止**为凑格式在末尾拼 `[1][2]`，无法对齐的引用在输出前剔除。
4. 失败必须有出口：LLM 不可用 → 兜底话术 + 转人工；证据门不过 → 不生成（mid 追问、low 兜底）。

**目录**：

```
orchestration/
├─ state.py        # ChatState（节点间唯一通信载体）
├─ chat.py         # ChatPipeline（同步最小闭环）
└─ api.py          # POST /v1/chat；GET /v1/chat/stream（SSE，第 5 项）
rag_system/core/
├─ llm_gateway.py  # OpenAI 兼容 /chat/completions（httpx）
├─ prompts.py      # 模板集中管理，带版本号
└─ citation.py     # 引用对齐与校验
base/
└─ errors.py       # BudgetExceeded / UpstreamUnavailable / EmptyRetrieval / PolicyViolation / InternalError
```

**为什么是 SSE 而不是先做 WebSocket**：§3.1 原定 WS 双通道，但流式只需服务端单向推送，SSE 在普通 HTTP 上就能做（浏览器自动重连、断线即取消生成），少一层握手与心跳；帧协议保持一致（`meta/delta/citations/done`），WS 版本留到需要双向中断 / 心跳时再加。**同步接口先于流式**：生成、引用、兜底的问题在同步接口上定位成本最低。

**LLM 网关契约**：

```python
class LLMClient(Protocol):
    async def complete(self, messages: list[Msg], *, budget_ms: int,
                       model: str | None = None) -> LLMResult: ...
    def stream(self, messages: list[Msg], *, budget_ms: int,
               model: str | None = None) -> AsyncIterator[str]: ...

@dataclass(slots=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
```

- 走 **OpenAI 兼容协议**（`POST {LLM_BASE_URL}/chat/completions`），不引第三方 SDK：换供应商只改 `base_url` / `model`，不改代码。
- 超时 = `LLM_TIMEOUT_S` 与剩余预算取小；**生成节点不重试**（§5.3）。首 token / 连接失败时**换模型**算降级不算重试：主模型 `LLM_QA_MODEL`，兜底 `LLM_FALLBACK_MODEL`（可为空）。
- 异常映射：未配置 → `LLMNotConfigured`；连接 / 5xx / 超时 → `UpstreamUnavailable`；超出剩余预算 → `BudgetExceeded`。
- **启动策略**：`LLM_API_KEY` 为空时**不退出**（FAQ / RAG 不依赖 LLM），把 chat 置为不可用并在 `/readyz` 暴露 `llm=false`；仅 `ALLOW_DEGRADED_START=false`（生产）时 fail-fast。这是对 §2.5「LLM_API_KEY 为空直接退出」的修正——那条会让没有 key 的开发机连 FAQ 都起不来。

**引用契约（`citation.py`）**：

```json
{"n": 1, "doc_id": "...", "file_name": "人教版-数学-九年级上册.pdf",
 "page_no": 12, "section": "22.1 二次函数", "parent_id": "...:p3",
 "score": 0.81, "snippet": "顶点式 y=a(x-h)²+k ...",
 "title": "人教版-数学-九年级上册", "locator": "第12页 · 22.1 二次函数", "locator_kind": "page"}
```

- **`locator` / `locator_kind` 是"点开定位"的唯一口径**，前端不要自己拼（`Evidence.locator` 与 `Evidence.citation()` 共用同一实现，诊断接口与用户引用不会各说一套）。取值按 `page > section(heading) > title > file_name` 逐级降级，`locator_kind` 如实告诉前端降到了哪一级。
- **当前语料里 `page` 恒为 0**（实测 3256/3256 行），所以线上的实际形态是 `locator_kind="section"`、`locator="<教材名> · <章节路径>"`，例如
  `初中数学教材 · 第二十二章 二次函数 > 22.1 二次函数的图象和性质`。§3.4「引用可点开定位率 100%」按这个口径验收：**章节 + 标题**能定位到文档内的位置，不要求页码；等到解析链路能把页码写进 `page`，`locator` 会自动回到 `page` 形态，调用方不用改。

- 编号在本轮上下文里从 1 开始；`n` 必须落在 `[1, len(evidences)]`，否则视为模型编造，**从正文中剔除**并记 `citation_out_of_range`。
- 支持度：句级切分（`。！？；!?;\n`）后，用 jieba 实词与所引证据做覆盖度（去停用词），低于 `CITATION_MIN_SUPPORT`（默认 0.30）记为"无支撑句"。v1 **不做自动改写**（改写要二次调用 LLM，延迟与成本翻倍），只做标记 + 整体拒答；embedding 版（§7.2.2 的 0.55）等有评估数据再上。
- 只有**正文真正出现的**引用才进 `citations`；`require_citation=True` 且一条都没有 → 不返回模型答案，走兜底。

**ChatState**：

```python
@dataclass(slots=True)
class ChatState:
    request_id: str
    query: str
    slots: SlotResult
    faq: FAQHit | None = None
    retrieval: RetrievalResult | None = None
    gate: GateDecision | None = None
    answer: str = ""
    citations: list[dict] = field(default_factory=list)
    source: Literal["faq", "rag", "clarify", "fallback"] = "fallback"
    degraded: list[str] = field(default_factory=list)
    latency_ms: dict[str, int] = field(default_factory=dict)
    usage: dict[str, int] = field(default_factory=dict)
```

**`POST /v1/chat` 协议**：

```json
// 请求
{"query": "二次函数的顶点式怎么求", "subject": null, "selected_faq_id": null, "session_id": null}

// 响应
{"request_id": "...", "source": "rag",
 "answer": "顶点式可以通过配方法得到[1]。",
 "citations": [{"n": 1, "file_name": "...", "page_no": 12, "snippet": "顶点式 y=a(x-h)²+k ..."}],
 "faq_suggestion": {"faq_id": 180, "question": "顶点式怎么写？"},
 "clarify": null,
 "degraded": [], "latency_ms": {"faq": 1, "retrieve": 120, "generate": 900, "total": 1050},
 "usage": {"prompt_tokens": 1800, "completion_tokens": 120}}
```

- `source` 四值：`faq`（逐字返回，带 `faq_id`）、`rag`（生成 + 引用）、`clarify`（mid 档追问，附候选）、`fallback`（无证据 / 系统异常，转人工）。
- **一律 200**：兜底话术是正常产品行为，不是错误；参数错误（学科非法）仍 400，内部未预期异常 500 且带 `request_id`。

**三档动作（编排层唯一读法）**：

| FAQ 档 | 动作 | 返回 |
|---|---|---|
| high | **跳过检索**（省约 350 ms） | 逐字 FAQ 答案 + `faq_id` |
| mid | 跑 RAG；不过门 → 追问 top1 候选（不答 FAQ 原文） | `clarify` + `faq_suggestion`（不含 answer） |
| low | 跑 RAG；不过门 → 兜底 + 转人工 | `fallback` |

**降级**：

| 故障 | 行为 |
|---|---|
| LLM 未配置 / 不可用 | 有证据也不生成：`source=fallback`，`degraded=["llm_unavailable"]`，话术带客服电话 |
| RAG 未就绪 + FAQ mid | **不升级为 high**：`clarify` + `degraded=["rag_unavailable"]` |
| 引用全部越界 | 丢弃模型答案，`fallback`，记 `citation_invalid` |
| 答案无任何引用（`require_citation`） | 同上 |
| 生成超预算 | 截断并提示"回答不完整"，记 `budget_exceeded` |

**验收**：

1. 单测覆盖：FAQ high 直返（不调用 LLM）、mid 追问、low 兜底、证据通过 → 生成 + 引用、LLM 不可用兜底、越界引用剔除。
2. 无 key 环境能起服务：`/healthz` 200、`/readyz` 200 且 `llm=false`、FAQ 与 RAG 接口正常。
3. 有 key 时：黄金集抽 50 题，引用可点开定位率 100%，编造率 ≤2%（人工抽检）。

#### 7.8.1 落地现状（2026-10-27，七项全部实现）

| # | 落点 | 验证 |
|---|---|---|
| 1 LLM 网关 | `rag_system/core/llm_gateway.py`（httpx，OpenAI 兼容，流式走 `text/event-stream`） | `tests/test_llm_gateway.py`；无 key → `/readyz` 的 `llm=false`，FAQ / RAG 照常 |
| 2 引用对齐 | `rag_system/core/citation.py`；上下文高亮在 `rag_system/core/prompts.py::highlight` | `tests/test_citation.py`、`tests/test_prompts.py`；越界引用剔除、无支撑句标记、整体无引用拒答；**命中子块用 `<精准检索定位区块>` 括起来**（§3.3.2 资产），子块找不到就原样返回不猜位置 |
| 3 同步 `/v1/chat` | `orchestration/{state,chat,api}.py` | `tests/test_chat_api.py`；FAQ high 直返**不调用 LLM** |
| 4 端到端评估 | `scripts/eval_chat.py` | `docs/analysis/chat_eval_report.md`（763 题） |
| 5 SSE 流式 | `GET /v1/chat/stream` | `tests/test_chat_stream.py`；四类帧齐全 |
| 6 IntentProfile | `understanding/{intent,profiles}.py` + 编排层接入 | `docs/analysis/intent_report.md`、`intent_ab_report.md` |
| 7 会话与记忆 | `memory/`（service/session/store/extractor/summary/api） | `tests/{test_session,test_rewrite,test_memory}.py`，契约见 §7.9 |

#### 7.8.2 第 4 项：端到端评估（2026-10-27 首跑 / 2026-10-28 换正确语料后重跑，**都不能给答案正确性结论**）

样本 763 题（可答 46 / 不可答 717），向量库 622 块，`embedder=bge-m3:cpu`、`rerank=off`、
生成模型 **stub**（本机无 `LLM_API_KEY`）。报告 `docs/analysis/chat_eval_report.md`。

**首跑（10-27 23:14）用的是作废的 FAQ 语料**：MySQL 里还留着 2026-10-26 改写前那一代（2000 条关键词式短句，
未走去重合并，`indexed=1998`），重导入时按 `content_hash` 判重**新增**了第二代、老一代仍 `published`。
修法与验证见变更记录（`--deprecate-missing`）。下表两组数只差"语料对不对"，其余口径完全一致：

| 观测 | 首跑（作废语料，索引 1998） | 重跑（现行语料，索引 1374） | 读法 |
|---|---|---|---|
| Recall@1 / @3 / @6 / @10 | 0.0275 / 0.0485 / 0.0603 / 0.0603 | 同左（检索侧未改） | 全体召回天然低（黄金集覆盖 7–9 年级，语料只有几份教材 + 教案） |
| 可答子集 Recall@10 | **1.0**（46） | 同左 | 语料里有的，检索基本都找到了 |
| MRR@10 / nDCG@10 | 0.0388 / 0.0433 | 同左 | — |
| 出口分布 | faq 29 / rag 32 / clarify 97 / fallback 605 | faq **404** / rag 22 / clarify 28 / fallback 309 | 差别全在 FAQ：旧语料问法是关键词式短句，黄金集的完整问句匹配不上 |
| FAQ 档位 | low 622 / high 29 / mid 112 | low 329 / **high 404** / mid 30 | — |
| 证据门 | ok 57 / below / amb 10 / qcov 10 | ok 25 / below 324 / amb 2 / qcov 7 | 被 FAQ 提前接走，门的绝对量随之下降 |
| 拒答 | 可答被答 30.4%、不可答被拒 **93.4%** | 可答被答 **73.9%**、不可答被拒 45.3% | 下降不是"变松了"：375 条"不可答"走的是 FAQ 高置信（见下条） |
| 引用 | 可定位 32/32 = 100%、相关 4/32 = 12.5% | 可定位 100%、相关 3/14 = 21.4% | 对齐环节始终正确；**证据选错是瓶颈** |
| 延迟 p50/p95 | 端到端 621/1072 ms | 端到端 **15/803** ms | 404 题走 FAQ 直返（p50 11 ms），没走检索 |

**404 条 FAQ 高置信命中是有效答案，不是乱命中**（2026-10-28 专门核过）：把 763 条黄金集逐条过
`FAQService.search()`，high 档 404 条中命中问题与原题的字符 bigram Jaccard **中位 1.000、≥0.5 占 98.3%、
<0.2 仅 1 条**（唯一那条是"有没有关于生物的遗传和变异的教案"→"什么是遗传和变异？"，属档位边界而非错答）。
即黄金集里有大量题与 FAQ 语料是同一句话——FAQ 先接走它们既正确也省掉整条检索链。

结论四条：① 引用可定位率 100% 说明机制层没问题；② 引用相关率低说明**要补语料 / 改切分 / 上 GPU 重排**，
不是去调提示词；③ 「答案正确性 / 编造率」这两项在 stub 下无意义，等 `LLM_API_KEY` 到位跑 `--llm live` 再补；
④ **评估数字必须绑定语料指纹**——同一份代码、同一批样本，语料错一代就能把 FAQ 出口从 404 打到 29。

#### 7.8.3 第 5 项：SSE 帧契约（`GET /v1/chat/stream`）

四类帧，字段与 §7.8 的 `ChatState` 一一对应：

| 帧 | 何时发 | 关键字段 |
|---|---|---|
| `meta` | 生成前一次 | `request_id` / `source` / `strategy` / `cached` / `session_id` |
| `delta` | 每句生成后 | `text`（增量） |
| `citations` | 结束前 | `citations`（可多次，编号 `n` 稳定） |
| `done` | 最后 | `answer`（**权威全文**）/ `source` / `citations` / `latency_ms` / `usage` / `degraded` |

**`done` 是权威结果**：流式发出去的字节收不回来，所以引用不合格时不做"回溯删除"，而是
「逐句剔除（不合格的句子根本不发）+ `done.answer` 覆盖」；客户端规则是 `done.source != "rag"` 时丢弃此前 delta、
只显示 `done.answer`。早期出口（`faq` / `clarify` / `fallback`）同样走 `meta → delta → citations → done` 四帧，
**每条出口都带 `session_id`**，客户端不需要区分路径。

#### 7.8.4 第 6 项：IntentProfile 接入（门槛 + A/B 均通过）

分类器是**规则版**（`understanding/intent.py`，关键词加权打分）：仓库里没有训好的 BERT 权重、本机 torch 为 CPU 版，
规则版先接进编排层，把 §7.7 真正要验的"档位是否改善下游决策"量出来；模型版替换同一个 `classify()` 签名即可。

**第一段（门槛）**：`intent_test.jsonl` 1 020 条 → macro F1 **0.967**、accuracy 0.9412、`unknown` **5.59%**（门槛 F1 ≥0.90、unknown ≤10%）；
`intent_train.jsonl` 4 080 条交叉检查 0.9637 / 0.9328 / 6.59%。**诚实标注**：关键词表是看着 test 错例补的，
F1 是乐观估计，两片同源只能说明没过拟合到个别句子；上线依据是第二段。

**第二段（档位 A/B）**：763 题里 378 题被档位收窄了 `doc_type`（前缀过滤**下推到 Milvus**，不是查后过滤），
`no_retrieval` 档**按线上行为真的不检索**（证据记空，跳过检索的代价算进去），档位判定调
`understanding/profiles.py::resolve_profile` 与编排层同一份代码。

| 口径 | 基线 Recall@10 | 档位 Recall@10 | 降幅 | 错误过滤率 |
|---|---|---|---|---|
| 主口径（覆盖 ≥0.5） | 0.0603（46/763） | 0.0603 | **0.0 pp** | **0.0**（基线 0） |
| 放宽口径（覆盖 ≥0.2） | 0.2792（213/763） | 0.2792 | **0.0 pp** | 0.0 |

判据（召回降幅 ≤1pp、错误过滤率不升）**通过**。档位分布：`generic_clarify` 323 / `kb_strict` 249 /
`doc_lookup` 106 / `general_advisory` 61 / `kb_stepwise` 22 / `generate_assist` 1 / `no_retrieval` 1。
**如实记一笔**：黄金集全是学业问句，只有 1 题（0.1%）真的跳过检索，所以"闲聊档省 350 ms"这个收益
**在这份样本上量不出来**（要在真实闲聊流量上量）；反过来 42% 落到 `generic_clarify`（全库不收窄）意味着
在这份样本上档位的收益是"**没有伤害**"而不是"召回变好"——这正是 §7.7 验收要的口径，别把它读成"意图没用"。

**两处实现层面的偏离（有意，写在代码注释里）**：

1. `GENERIC_CLARIFY.require_citation=True`、`allow_general=False`（§7.7 表格原写"有则标"）。
   理由："不知道用户问的是什么"恰恰是最该守住引用闸门的档位；放宽等于在最不确定处取消闸门，
   与 §3.4「无证据 → 知识库暂无 + 转人工」冲突。只影响无证据情形，不会因此多答一条没依据的结论。
2. **疑问形态安全阀**（`looks_like_question` + `intent_no_retrieval_guarded`）：带疑问词的输入哪怕被分成"闲聊"，
   也**不许**因此跳过检索，改走 `generic_clarify`。`no_retrieval` 是唯一跳过检索的档位，也就唯一有资格把正确证据挡在外面；
   高风险路由要往"多查一轮"倒，不往"省一次检索"倒。规则放在 `profiles.py` 而不是 `intent.py`——这是**路由安全**，不是分类。
3. `FEATURE_INTENT=false` 时用 `BASELINE` 档位对象（不收窄、必检索、必引用），它**就是** A/B 的基线，
   灰度与对照实验共用同一对象，免得"基线"在脚本里再实现一遍、与线上不一致。

#### 7.8.5 第二轮收口（2026-10-29）：可问答闭环

> 依据实施手册 §12「第二轮：完成可回答闭环」（PR3-E / PR3-F）。这一轮**不加新能力**，
> 只做四件"让闭环真的闭合"的事：出口统一、引用可修、重试有界、阶段可读。

**① 统一响应体（`base/contracts.py::ChatResponse`）**

`ChatState.to_response()` 是**唯一出口**，`to_dict()` 委托给它。23 个字段，`/v1/chat` 与
异常 500 共用同一形状，前端只写一份渲染逻辑：

| 组 | 字段 |
|---|---|
| 身份 | `request_id` / `session_id` / `faq_id` |
| 结论 | `status` / `source` / `answer` / `citations` / `clarify` / `faq_suggestion` |
| 诊断 | `error_code` / `gate` / `degraded`（机器码）/ `degraded_reasons`（中文）/ `profile` / `intent` / `intent_method` / `intent_confidence` / `strategy` / `rewritten` / `memory_tokens` |
| 性能 | `stages` / `latency_ms` / `usage` |

`status` 五值 `ok / clarify / no_evidence / degraded / error`，由 `ChatState.resolve_status()`
**推导**而非各分支各自赋值——出口十几条，靠人记着设 `status` 必漏，而漏掉的恰是线上唯一一次兜底。
判定顺序写死：`clarify`（追问是正常产品行为）→ `fallback` 按 `error_code` 分 `error` / `no_evidence`
→ 无答案记 `error` → 命中 `_DEGRADED_ANSWER` 记 `degraded`。

两套口径必须分开，否则"知识库没有"会被当成故障告警：

| 口径 | 含义 | 谁 | `status` |
|---|---|---|---|
| `_HARD_ERRORS` | 系统失败（`retrieval_unavailable` / `generation_unavailable` / `citation_invalid` / `internal_error`） | 该修服务 | `error` |
| 其余 `fallback` | 知识库/证据不足 | 该补语料 | `no_evidence` |

`_DEGRADED_ANSWER` 的判据是"**用户读到的这句话是否因此变弱**"，不是"这次请求有没有出过岔子"：
`faq_unavailable`（人工审核过的逐字答案变成生成答案）、`citation_repaired`（剔过句子）在内；
`memory_degraded` / `session_unavailable` / `citation_out_of_range`（已剔除、不影响发出去的正文）不在内。

**未预期异常也回同一结构**（`orchestration/api.py::_internal_error`）：HTTP 500 而不是 200——
这不是"知识库没有"那种产品行为，监控要能直接按 5xx 报警；但 body 仍是 23 个字段，
`status="error"` + `error_code="internal_error"` + `request_id` 一定有值（没有中间件时自己生成），
因为它此刻是**唯一**能把用户截图和日志对上的线索。兜底话术与编排内生成失败共用
`ChatPipeline.system_error()` 一处出处——同一个故障不该因为恰好在哪一层被抓住而换一种说法。

**② 引用受控修复（`citation.repair()`）**

`align()` 之后若有 `unsupported` 句，**整句剔除**再验一遍（至多一次）：修过且有引用记
`citation_repaired`（`status=degraded`）；剔完一句不剩由编排层拒答 `citation_invalid`。
三条不做：**不调模型改写**（等于让模型给自己补依据）、**不改来源元数据迁就模型**
（`[n]` 对上了但指向的不是原文）、**不重新分配编号**（`[3]` 删掉就断了，不许改挂到 `[1]` 上）。
`repair()` 返回 `(alignment, 是否真的删过)`——"没删过"与"删完还能答"是两种结果，混成一句会让引用修复率变成假数。
越界编号在第一遍就被删光、第二遍看不见，所以 `out_of_range` 从第一遍带过来。

**③ 网关有限重试（两层"不重试"要分清）**

| 层 | 规则 | 位置 |
|---|---|---|
| 节点级 | 生成节点**绝不重试**（§5.3），失败即兜底 | `orchestration/chat.py` |
| 网关级 | 一个字都还没生成前，对**传输层故障**（超时 / 连接中断 / 408 / 429）重发同一次请求，`LLM_MAX_RETRIES` 默认 1 | `llm_gateway._complete_with_retry` |
| 换模型 | 主模型失败换 `LLM_FALLBACK_MODEL`，是**降级不是重试**，且两个模型只换一次 | `llm_gateway.complete` |

401/403 抛 `LLMNotConfigured`（**配置问题**，不可重试），其它 4xx 与 5xx 明确 `retryable=False`：
前者重发一万次结果相同，后者交回 `complete()` 换模型更划算。重试前看一眼剩余预算，
不足 `0.5 s` 就带**原始错误**上去，不制造第二个错误类型。重试次数是运维信号（日志 `llm_retry`），
**不进** `degraded`：最后拿到的就是正常答案，不该把 `status` 推到 `degraded`。

**顺带修掉一处泄漏**：上游把 key 回显在错误体里是常态（`Incorrect API key provided: sk-xxx`），
原实现把正文前 200 字原样塞进异常 → `log.warning(..., error=...)` → **双出口两份落盘**。
`_raise_for_status` 现在先 `_redact()` 再截断，位置保留、内容抹成 `***`。

**④ 阶段遥测补齐（§11.4）**

`stages` 现在覆盖 `plan / recall / parent_expand / fusion / rerank / evidence_gate / retry / total`，
`latency_ms` 覆盖 `faq / generate / citation / total`。两条口径：并行任务取 **max**（同一件事并行做了三遍不是三倍耗时）；
同一阶段的**先后重复执行取累加**（`evidence_gate` 修完要再过一次门，两次都是真花的）。`fusion` 恒为 0 如实透传
（Milvus 在一次 `hybrid_search` 内完成融合，本层没有独立阶段）。BACKTRACK 早退不写 `evidence_gate`——门没跑，缺的不编。

**验证**：新增 `tests/test_chat_round2.py` 19 条（§7.5 十三条逐条 + 统一结构 + `degraded_reasons` 中文 +
`stages` 含 `plan`/`evidence_gate` + HTTP 未预期异常 → 500 同形状）、`test_llm_gateway.py` +7（重试一次 /
400·401·503 各不重试 / 429 重试 / 备模型也挂 / key 不泄漏）、`test_citation.py` +5（`repair()` 五种情形）。
全量 **415 → 446 passed**。

**仍未做（第三轮）**：SSE 行为改动、`planner` 的 HyDE 接线与复合问句拆分、黄金集在新链路上复跑。

---

### 7.9 会话与记忆（§7.8 第 7 项，2026-10-27 落地）

**要解决的问题**：多轮对话里"它的公式是什么"离开上一轮读不懂；同一学员跨会话说过的"我上初三""数学不太好"
在新会话里全丢。本节补 L0 窗口 / L1 摘要 / L2 原子记忆 / L3 画像四层（§3.7 / §7.3），
并且**只做读侧与写入侧的最小闭环**：记忆是背景，不是依据，永不出现在引用里（§7.8 边界 1）。

**四条边界（违反即打回）**：

1. **记忆不参与事实性回答**：`MemoryContext` 只进 prompt 的"记忆"段，**不能**成为引用来源；渲染带
   `【学员画像】` / `【会话摘要】` / `【历史记忆】` 段落标记，让模型能分清"背景"与"依据"。
2. **记忆模块不 import `qa_system` / `rag_system`**（§7.0 硬边界 1）。`memory/` 只依赖 `base/` 与自己的 `store`。
3. **无主会话不落库**：没有 `X-User-Id` 就**不签发** `session_id`（`orchestration/api.py::_identity`）。
   没有归属的会话写进 MySQL 就再也删不掉（记忆页按 `user_id` 清），一律不给 id。
4. **FAQ 不等记忆**（§7.7 硬规则 5）：`_prepare` 的顺序是 **FAQ 命中 → 才加载会话窗口/改写 → 分类 → 检索**。
   记忆召回与检索**并行**（`asyncio.create_task` 先起，检索 await 之后再 join），且 `recall()` 自带
   `wait_for` + fail-open，所以 join 不会抛。

**文件**：

```
memory/
├─ service.py     # MemoryService：recall / remember / list_atoms / patch / forget / chain / clear_all / export
├─ session.py     # SessionStore：L0 窗口（Redis 热 + MySQL 真相源）、双上限压缩、摘要滚动
├─ summary.py     # 摘要：有 LLM 走模型，没有走抽取式 digest()（同一签名）
├─ extractor.py   # 门控 + 规则抽取（五类事实，不调模型）
├─ store.py       # Atom 数据类 + AtomRepo 协议 + InMemory / MySQL 两实现 + ChatLogRepo
└─ api.py         # /v1/memory/*（记忆页）
```

**契约**：

```python
class MemoryService:
    async def recall(self, user_id: str, query: str, *, sid: str | None = None,
                     budget_ms: int | None = None) -> MemoryContext
    async def remember(self, user_id: str, sid: str, query: str, *, turn_index: int = 0) -> list[Atom]
    async def list_atoms(self, user_id: str, *, fact_type=None, q=None,
                         page=1, page_size=20, include_dead=False) -> dict
    async def patch(self, user_id: str, atom_id: str, *, content: str | None = None,
                    confidence: float | None = None) -> Atom | None
    async def forget(self, user_id: str, atom_id: str, mode: Literal["revive", "purge_chain"]) -> dict
    async def chain(self, user_id: str, atom_id: str) -> dict
    async def clear_all(self, user_id: str) -> int
    async def export(self, user_id: str) -> dict

class SessionStore:
    async def append(self, sid, role, content, *, user_id: str = "") -> SessionState
    async def window(self, sid) -> list[Turn];  async def summary(self, sid) -> str
    async def set_summary(self, sid, text) -> None;  async def clear(self, sid) -> None
```

**四层怎么落的（含与 §7.3 的偏离）**：

| 层 | 计划（§7.3） | 实际实现 | 为什么 |
|---|---|---|---|
| L0 窗口 | Redis LIST，40 轮 / 28 K token 先到先压 | `SessionStore`：Redis 存 JSON 快照（无 Redis 则进程内 dict），压缩后同步写 `chat_message` | 快照整取整存比 LIST 少一次读改写；压缩要看到全窗口才能算摘要 |
| L1 摘要 | 每 10 轮 / 3 000 token 滚动，≤400 token | `_compress` 按 `mem_compress_batch_turns` 压最旧一批，`digest()` 抽取式滚动（topic + 未完成项），压过才写 `session_summary` | 没有可用 LLM key；`digest` 与模型版同签名，换实现不改调用方 |
| L2 原子 | Milvus 向量 top5，p95 ≤120 ms | MySQL `memory_atom` 是真相源；**向量召回**走 `MilvusAtomIndex`（bge-m3 稠密 + 集合 `MILVUS_MEMORY_COLLECTION`），失败/关闭时退回**字面 bigram 排序**（`_rank`）。整段 `asyncio.to_thread` + `wait_for` | 见 §7.9.4：索引是副本，召回顺序退化可以，记忆一条不少 |
| L3 画像 | 稳定属性常驻 300 token | `user_profile` 表 + `ProfileRepo`（内存/MySQL 两实现）：**人工写过的（`manual`）落库并优先**，没写过的由身份类原子记忆现算（`derived`，不落库），裁到 `MEM_PROFILE_MAX_TOKENS` | 见 §7.9.4：派生画像随时可重算，同步一张派生表只会多一处不一致 |

**写入六步（§7.3.2）的实现口径**：门控 → 抽取（`extractor`：第一人称 + 陈述语气，五类事实，
**不按逗号切**就无法一句话抽两条）→ 去重（同 `fact_key` 且 `_similar ≥ MEM_DEDUPE_SIM` → 合并、
`mention_count+1`）→ 冲突（同 `fact_key` 新值胜，旧值 `valid_to=now` + `superseded_by` + `chain_root_id`）
→ 摘要滚动 → 落库。**写入走 `_persist_later`（`asyncio.create_task` + 强引用集合），不阻塞回答**；
`user_id` 为空时不写。抽取规则版**只抽用户原话，不抽助手回答**——答案来自知识库，不是用户属性。

**数据面**：`scripts/init_db.py` 新增四张表（已执行，`--drop` 按 `DROP_ORDER` 逆序）：
`memory_atom`（`valid_from/valid_to/superseded_by/chain_root_id/deleted_at` + `idx_user_alive(tenant_id,user_id,valid_to,deleted_at)` + `idx_chain` + `idx_user_key`）、
`memory_audit`、`chat_message`、`session_summary`。**召回可见条件三件套**：
`valid_to is null and deleted_at is null and confidence >= MEM_MIN_CONFIDENCE`——少一个就会出现"已删除记忆还在回答"。

**记忆页接口**（`/v1/memory`，全部要求 `X-User-Id`，空值 401 而**不**退化成共享的 `anonymous`）：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/atoms` | 分页列表；`fact_type` / `q` 过滤；`include_dead` 下钻历史。返回 `{total,page,page_size,items}` |
| PATCH | `/atoms/{id}` | 改 `content` / `confidence`，写 `memory_audit` |
| DELETE | `/atoms/{id}?mode=revive\|purge_chain` | `mode` **必填**：默认任何一边都会让另一边变成事故 |
| GET | `/chains/{id}` | 链视图（同一条事实的全部历史版本） |
| POST | `/export` | 导出该用户全部记忆 |
| DELETE | `/all` | 清空（写审计） |

**身份**：`X-User-Id` 请求头是**开发期身份**——接入层做鉴权后由网关注入、**不从外部透传**。上生产前必须接真鉴权，
`memory/api.py` 的模块注释与本节都写明这一点。**没有拿到授权的人一律当"不存在"（404）**——区分 404 与 403
等于告诉攻击者这条记忆存在；只有**已经拿到读授权**的顾问去写，才是 403（他本来就知道这条记忆存在，
403 只是把"只能看"这条边界说出来）。完整口径见 §7.9.3。

**连接方式（`main.py::Container.build_memory`）**：记忆用 MySQL **当且仅当** FAQ 从 MySQL 装载
（`self.backend == "mysql"`），否则退 `InMemoryAtomRepo` 并 `log.warning("memory_backend_in_memory")`——
避免"FAQ 在内存、记忆在库"的半持久化状态。`FEATURE_MEMORY=off` 时全模块空转（`remember` 返回 `[]`、
`recall` 返回空 `MemoryContext`）。`/readyz` 暴露 `memory` 与 `memory_backend`。

**降级**：召回超 `MEM_RECALL_BUDGET_MS`（默认 120 ms）→ `MemoryContext(degraded=True, extra={"reason":"timeout"})`，
主链路继续检索（**超时是正常路径的一种，不抛错**：记忆缺一段，模型只靠知识库也能答）；召回异常同处理（`reason="error"`）；
写库失败只记日志（fail-open）；Redis 挂了用进程内窗口，不阻塞请求。

**验收（`tests/`，28 条）**：

| 文件 | 条数 | 盯的是什么 |
|---|---|---|
| `test_session.py` | 7 | 轮数上限 / **token 上限先触发** / 只有真压过才写摘要 / 无主会话不落库 / 写库失败 fail-open / clear 清窗口 / 缓存值损坏降级为空窗口 |
| `test_rewrite.py` | 8 | 指代接上一轮 / **助手回答永不作为上文** / 完整问句不动 / 短问句补上文 / 只取上一轮第一句 / 寒暄跳过 / 无历史原样返回 / 改写结果截到 120 字 |
| `test_memory.py` | 32 | 知识问句不产生记忆 / 抽取+召回+段落标记 / 去重合并（`mention_count`）/ supersede 成链 / `revive` 复活 / `purge_chain` 整链删除 / 跨用户不可见 / 低置信不进 prompt 但记忆页看得见 / 召回超时与异常降级 / token 预算裁剪 / 上限拒绝新增而非淘汰 / 开关关闭；**L2 向量召回与 L3 画像的 19 条见 §7.9.4** |

全量单测 **792 passed**（`F:/miniconda3/envs/ragpro/python.exe -m pytest`，2026-10-31 实测）。

**已知未做**：① ~~L2 的 Milvus 向量召回~~（§7.9.4 已落地，字面排序退为兜底）；② ~~`ChatLogRepo` 的 30 天物理清理任务~~（§7.9.2 已用 `scripts/purge_memory.py` 补上）；
③ 记忆参与过的答案不写共享缓存（§7.7 作用点 7）——当前共享缓存尚未接入编排层，规则先记在这里；
④ `selected_faq_id` 确认通道的在线采纳率统计（需要前端）。

#### 7.9.1 结构化坐标（M0 坐标收口，2026-10-29 落地）

**要解决的问题**：语料换代后，Milvus 的 `subject` / `grade` 存的是 `base/labels` 的**规范名**
（`九年级`，不是 `初三`），而记忆侧的年级只活在 `fact_key` 字符串里、学科塞在 `"subject_weak:数学"` 里。
两套写法并存的结果不是报错，是**永远零命中**——按年级筛记忆筛不出来，按年级召回匹配不上，
画像与引用卡片各说一套。M0 只做"把坐标统一成一套写法"，**不加任何能力**。

| 项 | 落点 | 约束 |
|---|---|---|
| 槽位出口 | `qa_system/res/qa_slots.py::_extract_grade` 过 `normalize_grade` | 抽出来的 `SlotResult.grade` 与 Milvus `grade`、`Atom.grade`、引用卡片同一套写法；认得出学段认不出年级（"初中"）返回 `None`，不硬塞不在词表里的值 |
| 记忆坐标 | `Candidate.subject/.grade` → `Atom.subject/.grade` → `memory_atom.subject/grade` | 学科**按句子**取（`subject_weak:数学` 那条才带学科，年级事实不该被标成数学）；年级**按轮**取（它是说话人的属性，`我上初三，数学不太好` 两条事实都该带九年级） |
| 持久化一致性 | `store.py` 的 `_COLS` / `insert` / `update` / `_row_to_atom` 四处同时带两列 | 只加 `_COLS` 不加 `insert` 就是每次都 Unknown column。老库由 `scripts/init_db.py::migrate_memory()` 幂等补列 + 补 `idx_user_subject_grade`——**升级后必须跑一次** |
| 合并语义 | `service.py::_merge` 用 `old.x or cand.x` | 同一槽位的两次说法合并时任一侧缺的坐标由另一侧补，不能被空值冲掉。**不做冲突判定**（"我上初三" vs "我上高一" 走 supersede） |
| 记忆页筛选 | `GET /v1/memory/atoms?subject=&grade=` + `api.py::_coordinate()` | 认不出**就 400**，不做"归一不上就原样查"（原样查会零命中且不报错）。**空串是合法查询**（筛未归类的），所以 `list_atoms` 的判据是 `is not None` 而不是 `if subject` |
| 段落标记 | `MemoryContext.render()` 与 `prompts.py::_memory_turns` 逐字对齐本节边界 1 | `【学员画像】` / `【会话摘要】` / `【历史记忆】`。这是模型区分"背景"与"依据"的唯一线索，说一套发一套等于没说 |

**`prompts.py` 的改动是改名不是改行为**：只换了段落名，指令语义一字未动，故不触发 §5.4 的提示词 A/B 要求；
将来若改的是角色/禁令/字数，照样要附报告。

**测试**：新增 7 条（均在 `tests/test_memory.py` 的「结构化坐标」组；`tests/test_slots.py` 的年级断言同步改写），
全量 **467 passed**。其中 `test_mysql_repo_sql_carries_coordinates` 断言的是"坐标真的到达了 SQL 参数"——
§7.2.7-G 的教训：SQL 字符串看着对、参数列表却是另一件事，只断言 SQL 文本的测试拦不住。

后续阶段（M1 向量召回 / M3 读侧联动 / M5 评估）见
`docs/EduRAG-Pro-记忆模块实现.md` §5 与 §11.3。

#### 7.9.2 物理清理（M2，2026-10-29 落地）

**要解决的问题**：§3.7 第 5 条写了"软删 30 天后物理清理"，代码里只有软删——
`forget` 写 `deleted_at`、`clear_all` 写 `deleted_at`，然后这些行**永远留在库里**。
`chat_message` 更糟：它是唯一会无限增长的表，而它的内容是最敏感的（学生原话）。
没有清理任务，"用户要求删除"就只删了可见性，没删数据。

| 项 | 落点 | 约束 |
|---|---|---|
| 清理脚本 | `scripts/purge_memory.py` | `--dry-run`（只出条数）/ `--days` / `--batch` / `--table`；每日定时或维护窗口手动跑，**不在请求路径上** |
| 三张表三种口径 | `TARGETS` 常量表 | `memory_atom` 按 **`deleted_at`**、`chat_message` 按 `created_at`、`session_summary` 按 `updated_at`。合成一条 SQL 是错的：它们是三种不同的时间语义 |
| `valid_to` 永不参与清理 | 同上的 WHERE 子句 + 单测 | `valid_to` 是 supersede 留下的**历史链**，链视图 `GET /v1/memory/chains/{id}` 要能看。按它清理等于抹掉"这条事实以前是什么"——这跟"用户要求删除"是两件事 |
| `memory_audit` 永不清 | `TARGETS` 里没有它 + 单测 | 审计的意义就是比数据活得久。删掉"谁在什么时候删了这条记忆"的记录，等于没有审计 |
| 配置 | `MEM_PURGE_AFTER_DAYS`（默认 30）、`MEM_PURGE_BATCH`（默认 2000） | **只有一个窗口管三张表**。拆成三个变量之后"到底哪一个才是留存策略"没人说得清；真要分开，连 `.env.example` 与本节一起改 |
| 分批删 | `purge()` 的 `DELETE ... LIMIT n` 循环 | 一次删十万行会长时间持锁，把同库上的 FAQ 读与记忆写一起卡住。`LIMIT` 用 `int()` 校验后拼接（pymysql 的 `LIMIT %s` 会渲染成 `LIMIT '2000'`，严格模式下是语法错误），窗口天数仍走占位符 |
| 参数守卫 | `main()` 里 `days <= 0` / `batch <= 0` 直接返回 2，**在连库之前** | `--days 0` 删的是"包括今天在内的全部"。那不是参数写错，是数据事故 |
| 失败隔离 | 单表异常记 `memory_purge_failed` 后继续其余表，退出码 1 | 一张表报错不该让另外两张半途而废，但 cron 必须能发现 |

**如实记录（三条）**：

1. 本次只跑过 `--dry-run`（`memory_atom 0 / chat_message 0 / session_summary 0`），**没有在开发库里真删过任何一行**。
2. **默认就是真删，`--dry-run` 才预演**（与实施手册 §8.4 一致）。定时任务请显式写 `--days`，别依赖默认值随 `.env` 漂移。
3. `chat_message` 上按 `created_at` 单列比较走不到现有索引前缀（`(session_id, id)` / `(tenant_id, user_id, created_at)`），
   大表上是全表扫。现在表很小，百万行量级再补一次 DDL——**这是已知的、不是漏的**。

**测试**：新增 12 条（`tests/test_purge_memory.py`），全量 **467 → 479 passed**。三条盯的是"错了会悄无声息"的地方：
`valid_to` 不出现在任何到期条件里、`memory_audit` 不在清理范围内、`--dry-run` 的每条 SQL 都必须以 `SELECT COUNT(*)` 开头
（预演写成真删是这类脚本最典型的翻车）。另有一条断言**窗口天数真的到达了 SQL 参数**——§7.2.7-G 的同一条教训。

#### 7.9.3 权限与合规（M4，2026-10-29 落地）

**要解决的问题**：§3.7 第 7 条要求"顾问经 `memory_grant` 授权后只读；写审计；管理员只看聚合"，
而此前 `memory_grant` 表不存在、`memory_audit` 只在写入路径上零散几行、
`X-User-Id` 在**任何环境都被无条件信任**——也就是说，生产上改个请求头就能读别人的记忆。

| 项 | 落点 | 约束 |
|---|---|---|
| 授权表 | `scripts/init_db.py` → `memory_grant` | `scope ENUM('read')`（现在只有读）、`UNIQUE KEY uk_owner_grantee(tenant_id,owner_id,grantee_id)` 让"重新授权"是一次**覆盖**而不是堆行 |
| 仓储 | `memory/store.py` → `Grant` / `GrantRepo` / `InMemoryGrantRepo` / `MySQLGrantRepo` | `Grant.active_at(now)` 是唯一的判活口径；撤销是**改状态**不是删行（删了"曾经授权过给谁"就查不到） |
| 服务 | `memory/service.py` → `grant_read` / `revoke_read` / `list_grants` / `authorize_read` + `_require_read` / `_guard_write` / `_note_advisor_read` | 读写各一个闸门，**不散在每个方法里**；`grants=None` 是"不提供授权"（默认拒绝），不是"人人可读" |
| 接口 | `memory/api.py` → `/v1/memory/grants`（POST/DELETE/GET）+ `_user()` / `_owner()` / `_self_only()` | 身份与目标拆成两个头，授权管理入口只许本人 |
| 注入 | `main.py::build_memory` | `MySQLGrantRepo` / `InMemoryGrantRepo` 按 FAQ 装载来源选；`FEATURE_MEMORY=off` 分支也注入**空的**仓库而不是 `None` |
| 日志 | `base/log_messages.py` | `memory_audit_failed` / `memory_grant_read` / `memory_grant_revoke` / `memory_grant_denied` / `memory_advisor_read` / `memory_export` 六个事件 |

**三个状态码不是随手选的**：

| 场景 | 对外 | 为什么不是另一个 |
|---|---|---|
| 无 grant 的跨用户读/写 | **404** | 403 等于承认"这条记忆存在，只是你没权看"；"某学员有这条记忆"本身就是隐私 |
| 有读 grant 的顾问写 | **403** | 对方已拿到读权限，记忆存在他本来就知道；403 只把"只能看"这条边界说出来 |
| 没有可用的调用方身份 | **401** | 绝不退化成共享的 `anonymous`——那会让所有未登录用户共用一份长期记忆 |

**两个头的分工是本模块最容易写错的地方**：`X-User-Id` 是**调用方身份**（白名单外只认网关注入的
`request.state.user_id`），`X-Memory-Owner-Id` 是**目标资源 id**、**不携带任何权限**——能不能读只由
`memory_grant` 决定。混成一个头等于让任何人自报"我是张三"就读张三的记忆。
白名单本体在 `base/config.py::is_trusted_env`（`dev/development/local/test/testing`），
`memory/api.py` 只调用不复制：`memory/` 不许 import `orchestration/`（§7.0 硬边界 1），
两份白名单迟早只改一处、然后两个接口在同一环境上分叉。

**如实记录（五条）**：

1. **白名单外只认 `request.state.user_id`，不回落请求头**。生产目前没人注入这个字段，
   所以记忆页在生产配置下会稳定 401——这是**有意的 fail-closed**：与其指望上线前有人记得删掉那段透传代码，
   不如让它默认不生效。M4 的这条 DoD 原文就叫"`X-User-Id` 生产前置条件写死"。
2. **授权管理入口只许本人**（`_self_only`）：顾问不能给自己发授权，管理员也不行。本版本靠"表里只有
   owner→grantee 一个方向"保证 §3.7 第 7 条的"管理员默认不看原文"，**不加 `role` 字段**——
   加了就必须有角色管理，而那是 v2 的团队记忆（M4 明确不做）。
3. **撤销改状态、重新授权复用同一个 `id`**：前端不会看到主键冲突变 500，审计也留得住"曾经授权给过谁"。
4. **过期判定放 `Grant.active_at()`，且 `expires_at == now` 即失效**（不给"整秒"留后门）。
   这条写错是那种"上线半年后某个顾问还能看到上一届学员记忆"的错，所以它必须能单独测。
5. **`InMemoryAtomRepo.audit()` 故意不吞异常**（`MySQLAtomRepo.audit()` 则吞并记 `memory_audit_failed`）：
   内存版没有网络、没有约束，写不进去说明代码错了。它的 `.audits` 属性存在的唯一理由，
   是让"所有写操作落 `memory_audit`"这条 DoD **不连库也测得出来**。

**顺带修掉的漂移**：`memory/store.py` 原先打 `chat_message_write_failed` / `session_summary_write_failed`，
两个都没在 `base/log_messages.py` 注册，而 `memory/session.py` 已经在用注册过的 `chat_message_persist_failed`。
按"一个失败一个事件名"改回注册名——再登记两个同义事件等于把一处漂移变成三处。

**测试**：新增 27 条（`tests/test_memory_permissions.py`），全量 **479 → 506 passed**。
六条专门断言"措辞以外的实质"：过期 grant 不放行、撤销后立即失效、顾问写入必须是 403 而不是 404、
生产配置下 `X-User-Id` 头被忽略、**被拒的读与写都留下审计行**、`grants=None` 时跨用户默认拒绝。

后续阶段（M1 向量召回 / M3 读侧联动 / M5 评估）见
`docs/EduRAG-Pro-记忆模块实现.md` §5 与 §11.3。

#### 7.9.4 L2 向量召回与 L3 画像落库（M1 前半 + L3，2026-10-31 落地）

**要解决的问题**：四层里两层还是"半层"——L2 的排序是字面 bigram（"我数学不好"与"数学怎么提高"
只靠共用两个字排在一起，换个说法就召不回），L3 画像根本没有存储，每次从身份类原子记忆现算，
于是**运营改过的画像存不下来、下次读又被派生值盖回去**。

| 层 | 落点 | 约束 |
|---|---|---|
| L2 向量召回 | `memory/store.py::MilvusAtomIndex` + `memory/service.py::_recall_order` | 集合 `MILVUS_MEMORY_COLLECTION`（默认 `edurag_memory`）**与知识库集合分开**：知识库是可重建索引，记忆是用户数据，重建知识库不该 drop 记忆；两边过滤条件也完全不同 |
| 注入而非自建 | `main.py::Container._atom_index()` | `memory/` **不许** import `rag_system`、**不许**自己 `MilvusClient(...)`（§7.0 硬边界）。组合根把 RAG 已在用的 client 与 bge-m3 递进来，顺带避免第二次装载 2.5 GB 权重。`ensure()` 失败 / `FEATURE_MEMORY_VECTOR=false` / `index=None` → 记 `memory_vector_unavailable` / `memory_vector_disabled`，字面排序照常 |
| 时间列 | `valid_to` / `deleted_at` 存 INT64 epoch，`NULL` → `0` | Milvus 没有 NULL。可见条件因此是 `valid_to == 0 and deleted_at == 0 and confidence >= MEM_MIN_CONFIDENCE`，与 MySQL 侧"三件套"同义——**两套存储的可见条件必须逐字等价**，否则会出现"记忆页看得到、召回看不到"这类只有对照两侧才发现的偏差 |
| 表达式转义 | `_expr_str()` 转义 `\` 与 `"` | Milvus 过滤表达式是**字符串拼接、没有参数化**，`user_id` 里一个未转义的引号就能拼出 `or` 子句。这条与 §7.9.3 的"身份头不可信"是同一个攻击面 |
| 过滤抵达性 | `search()` 用 `client.search(..., filter=self._visible(user_id), output_fields=["id"])` | `MilvusClient.search` **确实接受 `filter`**（与 §7.2.7-G 的 `hybrid_search` 相反）。教训同 §7.2.7-E：只断言"表达式字符串长什么样"拦不住回归，单测断言的是**它出现在捕获到的 `filter` 实参里**（`tenant_id` / `user_id` / `valid_to == 0` / `deleted_at == 0` / `confidence >= 0.6` 五段都要在） |
| 召回顺序 | `_recall_order`：索引命中的 id 作**前缀**，其余按字面 `_rank` 的顺序**补齐** | 索引滞后（写穿失败、集合重建中）意味着"这次没排上"，**绝不等于"这条记忆不存在"**。只返回索引结果会让一次抖动变成"用户记忆凭空少一半" |
| 删改传播 | `_sync_index`：活着的 `upsert`，失效/删除的**从索引里删掉** | 不写 `valid_to=now`：Milvus 侧一旦哪次过滤写错，"已删除的记忆"就会被重新召回。删除是幂等且可重建的动作（MySQL 是真相源，重建索引能补回来） |
| fail-open 两层 | `MilvusAtomIndex` 自己吞异常返 `None`；`MemoryService._index_search` / `_sync_index` 再各兜一层 | 第一层是具体实现的自觉，第二层是 Protocol 契约：换一个会抛异常的实现，也只该退化成字面排序（召回）或只记日志（写入），不能把整次召回连坐成空记忆、不能把一次成功的写入变成失败 |
| L3 画像 | `user_profile` 表（`PRIMARY KEY (tenant_id, user_id)`）+ `ProfileRepo` / `InMemoryProfileRepo` / `MySQLProfileRepo`，`ON DUPLICATE KEY UPDATE` | 只有**人工写过的**才落库（`source='manual'`）；`derived` 版每次现算、**不落库**——它随时可重算，同步一张派生表只会多一处不一致 |
| 读侧优先级 | `_recall_inner` 先取 `_stored_profile`，有则用它，无则 `_profile_text` 现算 | 运营改过的画像必须压过派生值，否则"改了就变回去" |
| 画像接口 | `GET / PATCH / DELETE /v1/memory/profile`（`memory/api.py`） | 与记忆页其余接口同一套身份口径：`X-User-Id` 是调用方、`X-Memory-Owner-Id` 只是资源 id；写走 `_guard_write`（本人可写、只读顾问 403） |
| 写审计 | `set_profile` / `clear_profile` 各写一行 `memory_audit`（`profile_set` / `profile_clear`） | 画像是最"像人"的一条数据，改了什么必须能追溯；`clear_all`（清空用户全部记忆）同时删画像——"清空"的意思是这个人在这里没留下东西 |
| 长度上限 | `set_profile` 裁到 `MEM_PROFILE_MAX_TOKENS` 对应的字符数，`ProfileRequest.content` `max_length=4000` | 两张闸门都要有：接口层挡超大请求体，服务层挡"合法但装不进预算"的内容（画像每次都要进 prompt，超长是持续成本） |

**契约（最容易写错的一条）**：`AtomIndex.search()` 的返回值里 **`None` 与 `[]` 不是一回事**——
`None` = 索引不可用（退化到字面排序），`[]` = 索引健康但这条查询没有匹配（**就用字面排序的补集**）。
把两者混成一个，会让"索引挂了"看起来像"这个用户没有记忆"，而且**永远不会有告警**：
`degraded` 不会置位、日志里只有一次 warning。单测在服务层与 Milvus 层各钉一条。

**测试**（`tests/test_memory.py` 新增 19 条，文件累计 32 条）：

| 组 | 盯的是什么 |
|---|---|
| 召回顺序 | 索引命中的排前面且**剩余的仍按字面补回来**；索引返 `None` 退回字面序；索引**抛异常**也只退化成排序（`degraded` 仍为 `False`、记忆一条不少） |
| 写穿 | 新增/更新推 `upsert`、失效/删除推 `delete`；**索引抛异常时写入仍成功** |
| Milvus 适配 | 五段过滤条件真的出现在 `search` 收到的 `filter` 实参里；空问句直接 `[]` 不查库；`search` 异常返回 `None` |
| L3 画像 | 存过的画像压过派生值；超长内容被裁且留审计行；清除后回落到派生值（原子记忆未被牵连）；`clear_all` 一并清画像；`user_profile` DDL 与 `DROP_ORDER` 已登记 |

**未做（本轮仍是空白，别当成已完成）**：

1. **向量召回的召回质量没量过**：`MemRecall@5 ≥0.85`（§3.7 第 9 条）**未跑**——
   `edurag_memory` 集合里没有真实记忆语料，也没有 100 题记忆回归集。当前只有"过滤条件抵达请求对象"
   与"失败退化"两类结构性断言，**召回率一个数字都不能引用**。
2. **索引重建脚本没有**：`_sync_index` 只在写路径上补，历史行（本功能上线前写入的原子记忆）不会自动进集合。
   真要上线，需要一次性回填 + 定期校对（`count()` 已留好，但没有人调它）。
3. **p95 ≤120 ms 未测**：bge-m3 编码在 CPU 上是秒级，`MEM_RECALL_BUDGET_MS` 默认 120 ms 时
   向量召回几乎必然超时并退化成字面排序——这条要等 GPU 环境量，**在 CPU 机器上"通过"没有意义**。

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
| 2026-10-20 | **证据门改跨文档比较**：`margin` 只在"另一篇文档"里比（同一文档多块算互相印证，原先 0.99 分也会被判 `ambiguous`）；`RAG_RERANK_ENABLED` 默认改 `auto`（CPU 上重排 40 条候选实测 **115 s**，无 CUDA 即跳过并打日志）；运行环境改用 conda **`ragpro`**；真实入库 15 文件 / 622 块，`/readyz` 返回 `rag=true rag_docs=622`；单测 101 条全绿 | 项目组 |
| 2026-10-20 | **PR3 真 BGE-M3 端到端跑通（CPU）**：离线分段拼装 torch 2.5.1+cpu；锁定 transformers 4.44.2 / sentencepiece **0.2.0**（0.2.2 在 Windows cp312 上 import 即崩）/ datasets 2.21.0 / huggingface-hub 0.36.2 / peft 0.12.0 / pymilvus 2.4.9 + `setuptools<81` + `marshmallow 3.22`；14 份文档 → 110 子块入库 82 s；`/v1/rag/search` top1 0.9794 命中原文、evidence 通过；实测重排 40 候选 CPU 约 4.5 s（`rerank=false` 599 ms）；`requirements.txt` 补 RAG 段、`.gitignore` 忽略 `rag_system/models/` | 项目组 |
| 2026-10-20 | **PR3 第三种：加载器矩阵补齐**（参考旧项目 `Ragpro/rag_system/document_loaders`）：新增 `ocr.py`（rapidocr 进程内单例，paddle 优先、onnxruntime 兜底）、`loader_office.py`（docx/pptx 按文档流顺序、标题样式转 Markdown、表格转 Markdown 表格、内嵌图 OCR 可选）、`loader_image.py`（图片 OCR，失败抛错不返回空串）、`loader_pdf_local.py`（PyMuPDF 逐页 + 大图/扫描页 OCR + `<!-- page: n -->` 页标记）；工厂支持 12 种后缀，PDF 增 `RAG_PDF_PARSER=auto|mineru|local` 与云→本地自动降级（打 `pdf_parser_fallback`、meta 留痕）；新增 16 条加载器单测，全量 100 passed | 项目组 |

| 2026-10-20 | **RAG 证据门标定完成**：`scripts/eval_rag.py` 用 `datasets/07_eval` 1000 条黄金集建 763 条可判定样本，解出阈值/margin/qcov 三条判据的取舍点；`evidence_gate.py` 新增 `query_coverage()` 第四条判据（问句实词覆盖，挡"同主题不同知识点"）；默认值改 **0.74 / 0.01 / 0.20**（可答放行 34%、不可答误放 3%，误放 ≤5% 口径）；763 题跑批 9.6 min；单测 104 条全绿 | 项目组 |
| 2026-10-25 | **新增 §7.1.7 FAQ 阈值重标定路线（六步）**：实测旧标定数据已失效（2000 条语料下 480/480 标签与正文不一致，按现文件跑 `calibrate` precision=0 且不报错）；明确 `norm_k` 只是刻度不是判据，margin 建议改原始分差；§7.1.6 中"已标定"改为"曾标定" | 项目组 |
| 2026-10-26 | **新增 §7.1.8 槽位与检索路由**并落地四项决策：① `SUBJECTS` 为权威配置 + `SUBJECT_ALIASES` 归一 + 语料扫描只报告不加白；② margin 三选一先做对比实验，相对分差为优先候选；③ `subject_invalid` 对外 400、内部兜底必须带 `degraded`，且与 `subject_empty` 严格区分；④ 首版只做学科/多学科/年级三槽位，`question_type` 暂缓。命中判定由三条件改四条（新增候选数 ≥2，解决单候选必过），`FAQHit` 增 `subject_state`/`degraded` | 项目组 |
| 2026-10-26 | **§7.1.8 代码落地**：`base/contracts.py` 增 `SlotResult`/`SubjectSource`/`SubjectState` 与 `FAQHit.subject_state|degraded`；`base/labels.py` 的 `SUBJECTS` 把"政治"改为"道德与法治"并新增 `SUBJECT_ALIASES`（政治/思想政治/思想品德/道法/品德）、`is_valid_subject`；新增 `qa_system/res/qa_slots.py`（纯规则抽取）；`qa_search.py` 以 `_resolve_scope` 实现路由表、精确命中加范围校验、缓存 key 改带检索范围；`qa_matcher.py` 实现四条判据 + `decide_multi` + `FAQ_MARGIN_MODE`（relative/normalized/raw）；`api.py` 非法学科 400；配置增 `FAQ_MARGIN_MODE`/`FAQ_MIN_CANDIDATES`/`FAQ_CROSS_SUBJECT_DELTA`。实测 2000 条语料 `unlabeled` 97→0，单测 121 条全绿 | 项目组 |
| 2026-10-26 | **意图口径定案为 K12 教学 6 类**（知识点查询 / 题目讲解 / 教案素材 / 学习方法 / 备课辅助 / 闲聊），招生 5 类废弃；`INTENT_LABELS` 已是该口径，本次不改代码。同步 §1.3 / §2.2 / §3.2 / §3.6 / §7.5，并新增 **§7.7 理解层施工图**：`IntentResult`/`IntentProfile` 契约、6+1 档路由表（`kb_strict`/`kb_stepwise`/`doc_lookup`/`general_advisory`/`generate_assist`/`no_retrieval`/`generic_clarify`）、8 个下游作用点、三条硬规则（不写 if/else、低置信不落闲聊、doc_type 放宽重试）、降级与验收 | 项目组 |
| 2026-10-26 | **新增全量数据清单**：`scripts/export_data_inventory.py` 导出 `docs/数据清单-EduRAG-Pro.xlsx`（10 sheet：总览 / 原始语料 / FAQ 规范 / MySQL 20 张表 / Milvus schema / Redis 键族 / 评估与标定 / 模型与服务 / 配置基线 / 数据缺口）。表中把"已建"与"规划中"分开标注，并列出 15 项数据缺口（P0：FAQ 标定样本失效、槽位评估集缺失、教材覆盖不足） | 项目组 |
| 2026-10-26 | **§7.7 硬规则由三条扩到六条 + 引用概念拆分 + 验收改两段**：① 意图**不得碰学科/租户/ACL 过滤**（`IntentResult` 不含 `subject`，只允许收窄 `doc_type`；学科只由 `SlotResult` 给）；② **FAQ 不等意图**（`asyncio.as_completed`，命中即返回并取消其余分支，§7.4 第 4–5 步同步）；③ **低置信只进 `generic_clarify`**，不强制引用、不进生成型档位、不转人工、不写缓存；④ `IntentResult` 只存分类结果、`IntentProfile` 只存策略，编排层入口先映射后由统一策略执行；⑤ **"引用"拆成两个概念**——`require_citation` 是策略（无可验证证据就不许答，走无证据策略，禁止拼 `[1][2]`/伪造来源），引用对齐是机制（`[n]`→chunk 映射、无支撑句剔除或改写），§3.4.3 同步改写；⑥ **验收两段**——F1 ≥0.90 只是门槛，DoD 是"档位路由 vs 全量 DIRECT"的 A/B（Recall@10 降幅 ≤1pp、错误过滤率不升、证据通过率不降、转人工率不升、`no_retrieval` 延迟降 ≥30%），伤召回即回退，不保留"昂贵的装饰品" | 项目组 |
| 2026-10-26 | **§7.1.7 第一步完成 + 归一化算子修复**：新增 `scripts/eval_faq.py baseline`（一跑即产出 `docs/analysis/faq_corpus_baseline.{md,json}`，含语料 sha256 / `built_at` / 归一化指纹，三项任一不达标 exit=2）。首跑抓到真问题：`_PUNCT = string.punctuation` 把 `+ - /` 当标点删掉，导致 `因式分解 x²-6x+9` 与 `x²+6x+9`、`2/3-1/6` 与 `2/3+1/6` 在精确哈希里撞成同一条，**后一行永远命中不到自己**（自检 1997/2000、3 组重复）。按 §7.1.2 新规则修复：仅当两侧为 ASCII 操作数时保留 `+ - * / = < >`，并归一 `× → *`、`÷ → /`、`· → *`、`− → -`，一元正负号（`-5的绝对值`、`(-5)`、`|-5|`）保留；连字符/加号夹在中文之间仍按分隔符删除（`数学-七年级` → `数学七年级`）。语料指纹 + **归一化行为指纹**（12 条探针）进报告，供第二、三步断言语料与分数可比。实测自检 **2000/2000**、重复 0 组、9 学科、unlabeled 0；单测 125 条全绿 | 项目组 |
| 2026-10-26 | **§7.1.7 第二、三步执行完毕，第三步未通过验收（诚实记录）**：旧 480 条样本归档作废；重建 **正 928 条**（`noise 788 / synonym 121 / abbrev 19`，9 学科）＋**负 840 条**（`lexical_trap 500 / unrelated 300 / noise 40`），负样本改走 `_clean_stem`（去选项行/公式/下划线，只留 8–60 字题干）——门槛不变而误命中 8.3%→17.6%，说明变真的是负样本、不是参数退化。dev 选点 `raw / 0.57 / 9.00` 在独立 test 上精度 **0.808**、召回 **0.078**（dev 0.951/0.117），反选（召回 ≥0.637）最好精度 test 0.811；**全网格 test 精度上界 0.897（召回 0.130）**，证明 0.95 在该特征空间不可达。**不改线上参数**：现行 `normalized / 0.25 / 0.10` test 召回 0.654 / 精度 0.807，已在精度-召回前沿上；margin 收到 0.18 可换精度 0.855 但召回掉到 0.264。病因记在特征：误命中集中在 11–40 字查询（28/140 vs ≤10 字 2/31）、FAQ 语料是关键词式短句、判定无查询形态信号。下一步转第四、六步（三档路由 / 查询形态信号 / FAQ 问题改写为完整问句）；`source=llm` 样本因无可用密钥仍未做 | 项目组 |
| 2026-10-26 | **数据更新后第二次重标定（覆盖上一行的结论，仍未通过验收）**：`edu_faq.csv` 被项目方改写——788 条 FAQ 问题从关键词式短句改成完整问句（语料指纹 `d9750ece…` → `a4c72486…`），旧 928 条样本与当前正文 **928/928 全部不一致**，归档为 `archive/faq_paraphrases.20261026-222902.jsonl` 后重生成。新样本：正 **928**（`synonym 437 / noise 473 / abbrev 18`）＋负 **540**（`lexical_trap 500 / noise 40`），dev 622/375、test 306/165；**`unrelated` 类归零**（题库 6186 条全部落在 9 学科内），脚本改为告警 + 报告头写缺口，不造替代数据，并声明本轮精度与上一轮不可比。判据从"不可达"变为"仅在极低召回处可达"：全网格 test 精度上界 **1.000（召回 0.127）**；dev 选点 `relative / 0.56 / 0.20` → test **0.879 / 0.284**，召回下限 0.637 时精度上界 **0.831**；现行 `normalized / 0.25 / 0.10` test **召回 0.382 / 精度 0.824 / 错误直接命中率 0.152**（同参数上一轮是 0.654/0.807——**改写把召回砍掉 27pp**）。病因按新数据重推：分数分布重叠（负 max 0.752 > 正 max 0.733）、长查询靠词数堆分（21–40 字区间误命中 0.306 而正召回 0.267）、阈值在 0.10–0.35 完全不生效（真正卡住的是 margin）。**仍不改线上参数**，预算转第四步三档路由。报告新增：三张诊断表统一按现行参数计算、负样本缺口自动声明、病因由数据推导而非写死 | 项目组 |
| 2026-10-26 | **§7.1.7 第三轮：语料重复合并（装载期）+ 补回 `unrelated` 负样本，召回首次过线**。诊断证实被 margin 拒掉的 top1/top2 多是同一知识点的两种问法（合并关闭时 D2 里 8 条 **Jaccard 1.00 / margin 0.000**），据此在**装载期**按「同学科 + 答案逐字相同 + 去问句框架词后 Jaccard ≥0.8」合并：**2000 条 → 1374 条进索引，626 条并成精确命中别名**，答案不同的 6 对不合并只告警，原句自检仍 2000/2000；负样本补回 `unrelated` 300 条（ceval 学科外题库），540 → 840。**同一样本集 A/B**（只改索引）：test 召回 **0.382 → 0.735**、精度 **0.775 → 0.859**，dev 0.384/0.771 → 0.706/0.831，test `reject_margin` 144→71、`reject_rank` 45→10，D2 同族 ≥0.5 占比 81.2%→56.3%；精度上升全部来自答对数 +108，负样本误命中绝对数 34→37 基本不动。**召回 0.637 首次达到**，precision ≥0.95 仍不成立（反选上界 test 0.870），**线上 `0.25 / 0.10` 一字未动**。新增 `QUESTION_FRAME_TOKENS`/`content_tokens`（仅用于相似度，不进请求路径归一化）、`FAQ_DEDUPE_NEAR_DUP` 开关、`/v1/faq/stats` 的 `docs/indexed/merged_duplicates`、`FAQService.canonical_id()`；对照报告归档 `docs/analysis/archive/faq_{threshold_report,diagnosis}.dedupe-off-20261026.md` | 项目组 |
| 2026-10-27 | **语料重复合并补单测**（§7.1.7 合并契约的验收项①③④此前只有跑批证据）：`tests/test_service.py` 新增 3 条——合并生效（`docs=4/indexed=3/merged_duplicates=1`、被合并者的原话仍走精确命中答代表条目、第三种问法转 BM25 也能答）、去框架 Jaccard ≈0.86 但**答案不同不合并**（两条各答各的，交教研）、`dedupe_near_dup=False` 保持全量索引（对照实验口径）。全量单测 **128 条全绿** | 项目组 |
| 2026-10-27 | **新增 §7.1.9 FAQ 置信三档（第四步契约，代码未动）**：high = `exact/alias_exact` 或过 §7.1.2 ①②③ 且 `margin ≥ FAQ_HIGH_MARGIN`（唯一直接答、跳过检索的一档）；mid = margin 落在 `[FAQ_MARGIN, HIGH)` 或学科来源被封顶（`inferred/context/conflict/invalid`，多学科另封顶）→ 跑 RAG + 候选，不过门则**追问**而非硬答；low = 其余 → 交棒 RAG。只加**一根旋钮**（阈值在 0.10–0.45 实测不生效，margin 才是单调信号），`FAQ_HIGH_MARGIN` 与 `FAQ_MARGIN` 同空间解释，**等于它即精确回退单档**（灰度与回滚的唯一手段）。三条边界写死：分档只在 qa_system 算、**中置信不返回答案且禁止把 FAQ 原文喂给生成器**、FAQ 置信档 ≠ 意图档位。配套：`hit == (tier=="high")` 语义收紧、`CacheEntry.tier`（缺字段按 low 且不许猜成 high）、RAG 未就绪时 mid **不升级为 high**、`stats()["tiers"]` + `faq_route_{high,mid,low}_total`、确认通道 `selected_faq_id`（追问采纳率作为在线标定信号）。验收：high 档 test 精度 ≥0.95 且召回 ≥0.25（参考 0.22 → 0.957/0.291），报告须列 mid"若直接答"精度（反推 ≈0.80，即其不许直答的依据）；在线以 high 点踩率是否显著低于 mid 作为"margin 是不是好信号"的最终判据，无差异则回退单档。**`0.25 / 0.10` 一字未动** | 项目组 |
| 2026-10-27 | **§7.1.9 三档路由七步落地（代码已实现，此前只有契约）**：① `base/config.py` + `.env.example` 增 `FAQ_HIGH_MARGIN`（默认 0.22，`self_check()` 校验 ≥ `FAQ_MARGIN`）；② `base/contracts.py` 增 `Tier` 与 `FAQHit.tier`（`hit` 派生）；③ `qa_matcher.py` 增 `MatchPolicy.high_margin: float \| None = None` + `resolve_high_margin()`（`None` ⇒ 单档）+ `MatchResult.tier`，`decide_multi()` 跨学科打平降 mid；④ `qa_search.py` 的 `_finish` 改以 tier 为唯一入参、`_Stats.tiers`、`CacheEntry.tier` 读写（缺字段按 low 且 `overwrite` 覆盖写）、`_NO_DIRECT`/`multi` 封顶（exact 例外）、mid `answer=None` 且走负缓存 TTL、确认通道 `selected_faq_id`（无效 id 只记 `faq_selected_invalid`，不返回 4xx）；⑤ `/v1/faq/stats` 暴露 `tiers`、`/metrics` 增 `edurag_faq_route_{high,mid,low}_total`；⑥ `scripts/eval_faq.py` 的 `calibrate` 增三档模拟与「三档路由」「高置信下限邻域」两节、`diagnose` 的 D1 桶拆成 `hit_high/hit_mid/wrong_high/wrong_mid/false_hit_high/false_hit_mid`（不再与 `reject_*` 混算）；⑦ 新增 `tests/test_tiers.py`（17 条）+ `test_api.py` 确认通道 1 条，全量单测 **128 → 146 全绿**。**两处与施工图的偏差**：`invalid` 在 `_NO_DIRECT` 里，`test_service.py::test_unknown_subject_falls_back_to_all` 必须改名并改断言（`invalid` 兜底全库后只到 mid，且原用例落在"精确命中恒 high"的例外里，测不到封顶）；`_Stats.hits` 只在 `tier=="high"` 时 +1，`/metrics` 的命中率因此是"可直接作答率"。**离线验收通过**：`calibrate`（928 正 / 840 负）在独立 test 上 `FAQ_HIGH_MARGIN=0.22` → 高置信精度 **0.957** / 召回 **0.291**（≥0.95 且 ≥0.25），mid「若直接答」精度 **0.805**；`FAQ_HIGH_MARGIN=FAQ_MARGIN` 时退化为 0.735/0.859（单档）。**线上参数 `0.25 / 0.10` 仍未改动**，第五、六步（反馈闭环、查询形态/同题族二次判别）待做 | 项目组 |
| 2026-10-27 | **§7.8 第 1–3 项落地（生成层与编排层的最小闭环）**：`rag_system/core/llm_gateway.py`（httpx + OpenAI 兼容，流式走 `text/event-stream`，无 key 只置 `llm=false` 不退出）、`citation.py`（`[n]` 越界剔除 + 无支撑句标记 + 整体无引用拒答）、`orchestration/{state,chat,api}.py`（FAQ high 直返不调 LLM、mid 追问、low 兜底、证据门不过不生成） | 项目组 |
| 2026-10-27 | **§7.8 第 4 项：端到端评估跑完，答案正确性因无 key 不能评**（报告 `docs/analysis/chat_eval_report.md`，763 题 / 可答 46 / 语料 FAQ 2000 + 向量库 622 块 / 生成模型 stub）：Recall@10 全体 0.0603、**可答子集 1.0**、MRR@10 0.0388；出口 faq 29 / rag 32 / clarify 97 / fallback 605，证据门 ok 57 / below_threshold 657 / ambiguous 10 / low_query_coverage 10；不可答被拒 **93.4%**、可答被答 30.4%；**引用可定位 100%（32/32）、引用相关仅 12.5%**——对齐机制正确，瓶颈在"证据选错"，处置是补语料 / 改切分 / 上 GPU 重排，不是调提示词；延迟 p50/p95：FAQ 13/24、检索 597/1049、生成 0/14、端到端 621/**1072** ms | 项目组 |
| 2026-10-27 | **§7.8 第 5 项：SSE 流式落地**（`GET /v1/chat/stream`，`tests/test_chat_stream.py`）：`meta`（request_id/source/strategy/cached/session_id）/ `delta` / `citations` / `done` 四类帧；**`done` 是权威结果**（`done.answer` 覆盖已发内容，客户端在 `done.source != "rag"` 时丢弃 delta）——流式收不回已发字节，所以引用不合格走"逐句剔除 + done 兜底"而不是回溯删除；早期出口（faq/clarify/fallback）同样四帧齐全且带 `session_id` | 项目组 |
| 2026-10-27 | **§7.8 第 6 项：IntentProfile 接入，门槛与 A/B 双通过**。分类器为**规则版**（无 BERT 权重、torch 为 CPU 版，模型版替换同一 `classify()` 签名）：`intent_test.jsonl` 1 020 条 macro F1 **0.967** / acc 0.9412 / `unknown` **5.59%**（train 4 080 条交叉检查 0.9637 / 0.9328 / 6.59%，并如实标注"关键词表看着 test 错例补出来、F1 是乐观估计"）。**档位 A/B**（763 题、378 题被收窄 `doc_type`、过滤下推 Milvus、`no_retrieval` 真的不检索）：Recall@10 **0.0603 → 0.0603（降幅 0.0 pp）**、错误过滤率 **0.0**（基线 0），放宽口径 0.2792 → 0.2792，判据通过。档位分布 `generic_clarify` 323 / `kb_strict` 249 / `doc_lookup` 106 / `general_advisory` 61 / `kb_stepwise` 22 / `generate_assist` 1 / `no_retrieval` 1（真实跳过检索仅 1 题，故"闲聊省 350 ms"这份样本量不出来；42% 落到 `generic_clarify` 意味着档位在这份样本上的收益是"无伤害"而非"召回变好"）。**三处有意偏离**：`GENERIC_CLARIFY.require_citation=True` + `allow_general=False`（最不确定处更该守引用闸门）；疑问形态安全阀 `looks_like_question` → `intent_no_retrieval_guarded`（带疑问词的输入不许因"闲聊"跳过检索，放在 `profiles.py` 因为那是路由安全不是分类）；`FEATURE_INTENT=false` 用 `BASELINE` 档位对象，灰度与 A/B 共用同一基线 | 项目组 |
| 2026-10-27 | **新增 §7.9 会话与记忆（§7.8 第 7 项）并落地**：`memory/{service,session,summary,extractor,store,api}.py`——L0 窗口（Redis 快照 + MySQL `chat_message`，双上限先到先压）、L1 摘要（`digest()` 抽取式滚动，带段落标记）、L2 原子记忆（MySQL + bigram 排序，`to_thread` + `wait_for` 120 ms，超时/异常返回 `degraded` 不抛）、L3 画像（≤300 token）；写入六步（门控→规则抽取→去重→supersede 成链→摘要滚动→异步落库，`_persist_later` 不阻塞回答）。**四条边界**：记忆不作引用来源、`memory/` 不 import 业务包、**无 `X-User-Id` 就不签发 `session_id`**（无主会话写进库就删不掉）、FAQ 命中不等记忆/会话（`_prepare` 顺序 FAQ → 窗口改写 → 分类 → 检索，召回与检索并行）。`scripts/init_db.py` 新增 `memory_atom` / `memory_audit` / `chat_message` / `session_summary` 四表（已执行，`--drop` 走 `DROP_ORDER`）；`/v1/memory/*` 六接口（`X-User-Id` 空值 401，**不**退化成共享 anonymous；删除 `mode` 必填）；`main.py::build_memory` 按 FAQ 装载来源选 MySQL 或内存仓库，`FEATURE_MEMORY` 可整体关停，`/readyz` 增 `memory` / `memory_backend`；`base/config.py` + `.env.example` 补 `MEM_RECALL_BUDGET_MS` 与整块记忆配置。**新增 28 条单测**（session 7 / rewrite 8 / memory 13），全量 **242 passed** | 项目组 |
| 2026-10-28 | **生成上下文补"命中片段高亮"（§3.3.2 资产）**：`rag_system/core/prompts.py` 新增 `HIGHLIGHT_OPEN/CLOSE` 与 `highlight(parent, children)`，`build_messages` 对每条证据输出 `highlight(ev.text[:max_chars], ev.child_hits)`——父块里真正被检索命中的那一句用 `<精准检索定位区块>` 括起来，模型知道"依据是这一句"而不是整段，前端也能只高亮这一段。**只标第一处命中**（重复出现标第二处没有定位价值，标满整段等于没标）；子块在父块里找不到（切分动过空白 / 被 `max_chars` 截断在外）就原样返回，**不猜位置**——标错位置会让模型引用一句它没读到的话。`SYSTEM_PROMPT` 一字未动（§5.4 改提示词要附 A/B 报告，高亮属于上下文装配不属于提示词模板）。新增 `tests/test_prompts.py` 5 条（括号只包一次、去掉括号与原文逐字相同、找不到/空子块是 no-op、第一个命中的子块胜出、注入防线文案仍在） | 项目组 |
| 2026-10-28 | **§7.2.6「未做」项落地：近重复文档检测**（`rag_system/document_loaders/dedup.py` + `IngestPipeline`）：两份同一教材（`人教版-数学-九年级上册.pdf` / `人教版数学九年级上册.pdf`）会让同一段证据以"两篇文档"身份互为对手，证据门跨文档 margin 直接判 `ambiguous`；`datasets/` 只读，所以只在**摄入侧拒绝**。判据①归一化文件名（去扩展名与 `- _ · 、 . ( ) 【 】 《 》` 后比较）——**真文件对由这一条抓住**；判据②正文前 8000 字的**字符二元组 simhash**（blake2b 加权 64 位，汉明 ≤3）兜住"另存为/重命名、正文几乎一致"的副本。只留先到的那份（顺序可复现）；同一 `source` 重跑不算自我重复；被拒者不登记（否则第三种命名会认错先到者）；指纹落 `data/doc_fingerprints.json`（增量入库能拦住上一批），文件损坏当空表——宁可漏拦不能让入库失败。新增 `RAG_DEDUP_ENABLED/DISTANCE/PREFIX_CHARS` 三个配置、CLI `--allow-duplicate`、`IngestReport.duplicates` 与 `summary()["skipped_duplicates"]`、日志事件 `ingest_duplicate`。**实测**（`RAG_PDF_PARSER=local`，真文件 + README）：`docs_ok=2 / duplicates=1`，379 子块只入库一份，被拒报 `doc_duplicate: 与 …人教版-数学-九年级上册.pdf 近重复（name）`；**诚实记录**：这对真文件的 simhash 距离是 **26**（两次解析 114 878 vs 86 241 字符，差异来自解析路径不是内容），simhash 只是兜底。`tests/test_dedup.py` 15 条（含**误杀防线**：同学科不同教材都保留、重跑不自我拦截），全量 **265 passed** | 项目组 |
| 2026-10-28 | **修数据：MySQL 里的 FAQ 是作废的那一代**。2026-10-26 语料改写后 `content_hash` 全变，`import_faq.py --publish` 按 hash 判重只会**新增**，于是库里同时躺着两代 `published`（2000 → 4000 行，`/v1/faq/stats` 的 `indexed` 也从 1374 变回 1998、`unlabeled` 从 0 变回 97），服务对外答的是改写前的关键词式短句——与 CLAUDE.md 记录的每一个数字都对不上。修法：`scripts/import_faq.py` 新增 **`--deprecate-missing`**（CSV 里没有的已发布行置 `deprecated`，分批 500），执行后 `published=2000 / deprecated=2000`、9 学科齐全、`/v1/faq/stats` 回到 `docs=2000 / indexed=1374 / merged_duplicates=626 / unlabeled=0`。开关是**显式**的：局部导入（只导数学）时默认下线其余学科会让整库只剩数学；日志改用 `faq_deprecated_missing`（`legacy_subjects_deprecated` 留给 `init_db.py`），默认路径不再出现"已下线 N 行"这种没发生的事 | 项目组 |
| 2026-10-28 | **§7.8 第 4 项按正确语料重跑并核验 FAQ 命中有效性**（报告 `docs/analysis/chat_eval_report.md` 已刷新为 2026-10-28 01:18）：出口 faq **29 → 404**、rag 32 → 22、clarify 97 → 28、fallback 605 → 309，端到端 p50/p95 **621/1072 → 15/803 ms**（404 题走 FAQ 直返，不进检索）；检索侧指标一字未变（Recall@10 0.0603、可答子集 1.0、MRR 0.0388）。"不可答被拒 93.4% → 45.3%"不是变松：**专门核过 404 条 FAQ 高置信命中**——逐条与黄金集比字符 bigram Jaccard，**中位 1.000、≥0.5 占 98.3%、<0.2 仅 1 条**（唯一那条是"有没有关于生物的遗传和变异的教案"→"什么是遗传和变异？"，属档位边界）。即黄金集有大量题与 FAQ 语料是同一句话，FAQ 先接走既正确又省掉整条检索链。教训写进 §7.8.2：**评估数字必须绑定语料指纹**，同一份代码 + 同一批样本，语料错一代就能把 FAQ 出口从 404 打到 29 | 项目组 |
| 2026-10-29 | **完整检索链路闭环（§7.2.7 施工，PR3-A…PR3-D）并新增 §7.2.7 现状节**：① 契约收口到 `base/contracts.py`——`QueryContext` / `RetrievalFilters`（含 `relaxed()`）/ `QueryTask` / `RetrievalPlan` / `RetrievalResult`（含 `stages`）/ `EvidenceDecision` / `RetryAction` / `Strategy`；② 新增 `rag_system/core/{query_processor,planner,fusion,evidence_builder,rag_pipeline}.py`，`retriever` 输出统一成 `RetrievalResult`，`evidence_gate` 在原有四条判据上改输出**结构化** `status(Sufficient/Insufficient/Ambiguous/Conflicting) + reasons + missing_tasks + retry_action + retry_count`；③ **重试决策只在 `RagPipeline`**：至多 1 次修复、修完**必须重新过门**、`no_candidates`→`retry_direct` / `no_candidates_with_filter`→`relax_business_filter` / `below_threshold`→`retry_original_query` / `low_query_coverage`→`rewrite_query` / `single_weak_evidence`→`expand_neighborhood`、`ambiguous` 与 `conflicting` **不修**直接上交；`retry_exhausted` 只在**真修过**时上报（没试过不许谎报）；④ **安全过滤不可放宽**：`relaxed()` 只清 `subjects`/`doc_types`，`tenant_id`/`acl_tags`/`kb_version` 在每次 retriever 调用上逐字节相同（单测守着）；⑤ 分阶段耗时 `query_embed/recall/parent_expand/fusion(=0)/rerank/total` + `retry` + 端到端 `faq/retrieve/generate/session/memory/total`，`fusion` 恒 0 如实记录（Milvus 单次 `hybrid_search` 内融合，本层无独立阶段）；⑥ `base/errors.py` 六类对外错误码，`retrieval_unavailable` 与 `no_evidence` 严格区分；⑦ 新增 **71 条**确定性单测（`test_rag_pipeline` 15 / `test_planner` 14 / `test_query_processor` 13 / `test_fusion` 7 / `test_evidence_gate_table` 22），全量 **336 passed**；⑧ **清掉一处假 telemetry**：`orchestration/chat.py` 原先按"LLM 可用"传 `hyde_available=True`，但没有任何节点生成假设性答案，会让 `plan.strategy` 声称跑过 HyDE——改为 `False`，planner 如实降级 `DIRECT + hyde_unavailable`。**诚实标注未做**：HyDE 未接线、复合问句拆分（`sub_queries` 无人产出）未接、黄金集 Recall@10/MRR **未在新管线上复跑**（旧基线属旧管线，空白是"没量"不是"没变"）、Milvus 集合因新增 `tenant_id/acl_tags/kb_version` 三列**需 `--recreate` 后重新入库** | 项目组 |
| 2026-10-29 | **第一轮缺口补齐（新增 §7.2.7-E）**：① `QueryTask.query_role`（`original/rewrite/hyde/sub`）落进契约与 planner，`RagPipeline.run()` 用 `ctx.original_query` 自动判定 `rewrite`（不加形参，少一处能忘传）；② 证据门空结果按过滤条件分诊出 `doc_type_too_narrow` / `subject_filter_empty`（`_empty_reason()` 单点），两条都走 `relax_business_filter`；③ `RetrievalFilters.__post_init__` 对空 `tenant_id` / `kb_version` 直接抛 `PolicyViolation`，不再靠调用侧记得校验；④ `X-Tenant-Id` / `X-ACL-Tags` → `ChatState` → `QueryContext.filters()` 的完整通路打通（多租户同进程），`_trust_headers` 由“排除 prod”改为**白名单**（staging 也默认不信任）；⑤ 引用定位降级：`Evidence.locator_kind/locator` + `citation` 输出 `title/locator/locator_kind`，按 `page > section > title > file_name` 逐级（`page` 实测 3256/3256 全为 0，线上形态是「教材名 · 章节路径」）；⑥ 修掉 `retry_original_query` 用改写句重检的真缺陷（改为 `ctx.original_query`，修复后计划用自己的 query 执行，否则 `qcov` 拿错问句）；⑦ 模型路径从 `../Ragpro/...` 迁回本仓库 `rag_system/models/BAAI/...`，相对路径按 `BASE_DIR` 解析，重排器找不到权重直接抛错而不是静默退化。新增 `tests/test_permissions_contract.py` 26 条 + citation locator 4 条 + gate 参数表 3 行，全量 **415 passed** | 项目组 |
| 2026-10-29 | **第二轮（可问答闭环，PR3-E / PR3-F）收口并新增 §7.8.5**：① **统一响应体** `base/contracts.py::ChatResponse`（23 字段）成为唯一出口，`ChatState.to_response()`，`to_dict()` 委托；`status` 五值由 `ChatState.resolve_status()` **推导**（clarify → fallback 按 `error_code` 分 `error`/`no_evidence` → 无答案记 error → 命中 `_DEGRADED_ANSWER` 记 degraded），判据是"用户读到的这句话是否因此变弱"而非"这次请求有没有出过岔子"；`_HARD_ERRORS`（该修服务）与 `no_evidence`（该补语料）两套口径分开。② **未预期异常也回同一结构**：`orchestration/api.py::_internal_error` 回 **500**（不是 200，监控要能按 5xx 报警）+ 23 字段齐全 + `request_id` 必有值（无中间件时自生成）；兜底话术唯一出处 `ChatPipeline.system_error()`。③ **引用受控修复** `citation.repair()`：无支撑句整句剔除后**重新验证**（至多一次），修过且有引用记 `citation_repaired`（degraded），剔空则 `citation_invalid` 拒答；三条不做——不调模型改写、不改来源元数据迁就模型、不重新分配编号；返回 `(alignment, 是否真删过)`，越界观测从第一遍带回。④ **网关有限重试**与 §5.3「生成节点不重试」分层写明：网关级只对传输层故障（超时/连接/408/429）重发同一次请求（`LLM_MAX_RETRIES` 默认 1），401/403 → `LLMNotConfigured` 不可重试、其它 4xx 与 5xx 明确 `retryable=False`（交回 `complete()` 换模型更划算）；重试次数是运维信号，**不进** `degraded`。⑤ **修一处密钥泄漏**：上游回显 key 的错误体原先被原样塞进异常并落双出口日志，`_raise_for_status` 现先 `_redact()` 再截断（位置保留、内容抹成 `***`）。⑥ **阶段遥测补齐** `plan` / `evidence_gate`：并行取 max、同阶段先后重复执行取累加、`fusion` 恒 0 如实透传、BACKTRACK 早退不写门（没跑的不编）。⑦ 新增 **31 条**测试（`test_chat_round2.py` 19 条覆盖手册 §7.5 十三条 + 统一结构 + HTTP 500 同形状；`test_llm_gateway.py` +7 重试矩阵与不泄漏；`test_citation.py` +5 覆盖 `repair()`），全量 **415 → 446 passed**。**未做（第三轮）**：SSE 行为改动、HyDE 接线、复合问句拆分、黄金集在新链路上复跑 | 项目组 |
| 2026-10-29 | **新增 §7.2.7-G：标量过滤从未生效（一次比"学科过滤失效"更严重的故障）**。`GET /v1/rag/search?q=新加坡是一个什么样的国家&subject=地理` 返回 6 条证据里只有 1 条地理、4 条道德与法治、1 条历史，且**去掉 `subject` 再查一次逐条相同、分数也相同**。根因：`MilvusClient.hybrid_search` **没有 `filter` 参数**（本机 pymilvus 3.0.2 实测；`requirements.txt` 钉的 2.4.9 也未核对过有这个参数，两版都不该赌），`filter=expr` 落进 `**kwargs` 被静默丢弃——反证是 `subject == "不存在的学科"` 照样返回 10 条。唯一生效入口是 `AnnSearchRequest(..., expr=...)`，dense 与 sparse **两路各持同一份**。影响面远超学科：`tenant_id` / `acl_tags` / `kb_version` / `doc_type` / `stage` / `grade` 全走同一个被丢弃的 kwarg，**租户隔离与 ACL 在这套部署上也是失效的**（只有 `partition_names` 物理分区真在起作用，这正是"`doc_type` 收窄看起来管用"的原因）。修法：`ChunkStore.search()` 把 expr 挂到两路请求、删掉 `filter=`。连真库验收：`subject=地理` → 6/6 地理、top1 **0.7393**；`subjects=("不存在的学科",)` → **0 条**；无过滤基线仍是 道德与法治×4/地理×1/历史×1（top1 0.6939）——过滤生效后地理学科内部的真 top1 才露出来，意味着**此前所有检索指标都在"没过滤"的语料上量的**，Recall@10 / MRR 需重跑。覆盖缺口：原测试只断言 `_build_expr()` 的字符串内容、从没断言它到达请求对象；新增 `tests/test_rag_store_filter.py` 7 条（假 client 拦 `hybrid_search`，断言两路 `req.expr` 都含学科与安全三项，且 `"filter" not in kwargs` 作为防线），全量 **453 → 460 passed**。**运维**：`python main.py` 需重启才生效 | 项目组 |
| 2026-10-29 | **新增 §7.2.7-F：重排仲裁与检索时间预算（一次真实故障的收口）**。`GET /v1/rag/search?q=勾股定理的逆定理是什么？` 返回 `cost_ms=231974`（`rerank 227443`）。① **重排绕过配置**：`build_plan(rerank: bool = True)` 的默认值 + `rag_system/api.py` 的 `rerank if rerank is not None else True` 让 `RAG_RERANK_ENABLED=auto`（本机无 CUDA）与 `.env` 同时失效，40 条候选在 CPU 上重排 227 s；`/v1/rag/stats` 的 `rerank_enabled: false` 与日志 `reranked: True` 因此各说一套。三层统一改 `bool \| None = None`（`RetrievalPlan.rerank` / `build_plan` / `_plan` / `api.py`），`None` = 不表态、由配置独裁，调试接口仍可强制开。② **单次检索无时间上限**：`asyncio.wait_for` 包住 `_recall()`（超时 = **剩余预算**，两次召回共用同一个 8 s）；超时先关重排重检一次并记 `retrieval_timeout`，仍超时才 `RetrievalUnavailable`；`plan.rerank is False` 时不再重检。③ 修复轮超时**不推翻首轮结果**（保留 `outcome` + `retry_failed` 后 break），否则"多试一次没打成"会把能答的请求降成服务异常。④ `base/log_messages.py` 登记 `rag_recall_timeout`、`retrieval_timeout`、`retry_failed`。⑤ 新增 5 条单测，全量 **453 passed**。**故障中不属于代码的部分**：`勾股` 全库仅 4 块且无一是该章——初中数学三册只有**上册**（七上 1–4 章 / 八上 11–15 章 / 九上 21–25 章），**八下第十七章 勾股定理不在语料里**，命中的 3 块来自高中数学高一「拓广探索」的勾股逆定理习题；`page` 在初中三册 1622/1622 全为 0；`top_score 0.0492` 是真实低分（同批块自搜索 1.0000、同章 0.72/0.69，`embedder` 全 `bge-m3:cpu`），关掉重排后同一问句融合分 top1 = **0.7252**（τ=0.74），证据门判 `below_threshold` 正确。语料入库仍在进行（高中高一数学在故障当时未入库、记录时已进 1327 块），本节只断言章节缺口 | 项目组 |
| 2026-10-29 | **记忆模块 M0「坐标收口」落地，新增 §7.9.1**（实施手册 `docs/EduRAG-Pro-记忆模块实现.md` §4 的 D1/D2/D3 全部修复，零新增能力、零 CPU）：① **D1 年级三套写法收口**——`qa_slots.py::_extract_grade` 出口过 `normalize_grade`，抽出来的 `SlotResult.grade` 与 Milvus `grade`、`Atom.grade`、引用卡片同一套写法（`初二`→`八年级`），认得出学段认不出年级（"初中"）返回 `None` 而不硬塞。原实现的后果不是报错而是**永远零命中**。② **D2 原子记忆增结构化坐标**——`Candidate`/`Atom` 增 `subject`/`grade`，`memory_atom` DDL 增两列 + `idx_user_subject_grade`，`_COLS`/`insert`/`update`/`_row_to_atom` **四处同时带**（只加 `_COLS` 就是每次都 Unknown column）；老库由新增的 `scripts/init_db.py::migrate_memory()` 幂等补列，**升级后必须跑一次**。取法有意不对称：学科**按句子**（`subject_weak:数学` 那条才带学科，年级事实不该被标成数学）、年级**按轮**（说话人的属性，`我上初三，数学不太好` 两条事实都带九年级——顺带修掉一处"注释说两条都带、代码只给第一句"的实现缺陷）。`_merge` 用 `old.x or cand.x`，合并时任一侧缺的坐标由另一侧补。③ **记忆页可按坐标筛**：`GET /v1/memory/atoms?subject=&grade=`，`api.py::_coordinate()` 认不出**就 400**——原样查会零命中且不报错，正是本阶段要消灭的静默不一致；**空串是合法查询**（筛未归类），故 `list_atoms` 判据是 `is not None` 而非 `if subject`。④ **D3 段落标记对齐**：`MemoryContext.render()` 与 `prompts.py::_memory_turns` 统一为 `【学员画像】`/`【会话摘要】`/`【历史记忆】`；**这是改名不是改行为**，故不触发 §5.4 的提示词 A/B 要求（将来改角色/禁令/字数照样要报）。⑤ 新增 7 条单测（`test_memory.py` 「结构化坐标」组；`test_slots.py` 的年级断言同步改写），**全量 460 → 467 passed**；其中 `test_mysql_repo_sql_carries_coordinates` 断言"坐标真的到达了 SQL 参数"——§7.2.7-G 的教训（SQL 文本对、参数列表是另一件事）。后续 M1 向量召回 / M2 30 天清理 / M3 读侧联动 / M4 记忆授权 / M5 评估仍未做，顺序见实施手册 §11 | 项目组 |
| 2026-10-29 | **记忆模块 M2「写侧补强」落地：30 天物理清理，新增 §7.9.2**（`docs/EduRAG-Pro-记忆模块实现.md` §8.4）。此前只有软删：`forget` / `clear_all` 写 `deleted_at` 之后这些行**永远留在库里**，`chat_message`（学生原话，唯一会无限增长的表）更是只增不减——"用户要求删除"只删了可见性，没删数据。新增 `scripts/purge_memory.py`：① **三张表三种到期口径**，不合成一条 SQL——`memory_atom` 按 `deleted_at`、`chat_message` 按 `created_at`、`session_summary` 按 `updated_at`；② **`valid_to` 永不参与清理**（它是 supersede 的历史链，链视图要能看，按它删等于抹掉"这条事实以前是什么"，与"用户要求删除"是两件事）；③ **`memory_audit` 永不清**（审计的意义就是比数据活得久）；④ 配置只加 **一个窗口** + **一个批大小**（`MEM_PURGE_AFTER_DAYS=30` / `MEM_PURGE_BATCH=2000`）——拆成三个窗口变量之后"到底哪个才是留存策略"没人说得清；⑤ `DELETE ... LIMIT n` **分批**（一次删十万行会把同库 FAQ 读与记忆写一起卡住），`LIMIT` 用 `int()` 校验后拼接（pymysql 的 `LIMIT %s` 渲染成 `LIMIT '2000'`，严格模式下语法错误），窗口天数仍走占位符；⑥ `days<=0` / `batch<=0` **在连库之前**拒绝并 exit 2（`--days 0` 删的是"包括今天在内的全部"，那不是参数写错是数据事故）；⑦ 单表失败不拦其余表但退出码 1（cron 要能发现）；⑧ `base/log_messages.py` 登记 `memory_purge_planned/done/failed` 三个事件，预演与真删用不同事件名，运维一眼能分清删没删。**新增 12 条单测**（`tests/test_purge_memory.py`，含"`valid_to` 不出现在任何到期条件"、"`memory_audit` 不在清理范围"、"--dry-run 每条 SQL 都以 `SELECT COUNT(*)` 开头"、"窗口天数真的到达 SQL 参数"），全量 **467 → 479 passed**；真实 `--dry-run` 输出 `[dry-run] 窗口=30天 待删: memory_atom 0, chat_message 0, session_summary 0`。**如实记录**：本次只跑预演，**没有在开发库里真删过任何一行**；`chat_message` 按 `created_at` 单列比较走不到现有索引前缀，百万行量级需补 DDL（已知未做）。下一步按 §11 顺序是 **M4 权限与合规**（零 CPU），M1 / M5 仍等 ingest 跑完 | 项目组 |
| 2026-10-29 | **记忆权限与合规（M4）落地，新增 §7.9.3**。此前 `memory_grant` 表不存在、`memory_audit` 只有写入路径上零散几行、`X-User-Id` 在**任何环境都被无条件信任**——生产上改个请求头就能读别人的记忆。① `scripts/init_db.py` 新增 `memory_grant`（`scope ENUM('read')`、`status ENUM('active','revoked')`、`UNIQUE KEY uk_owner_grantee(tenant_id,owner_id,grantee_id)` 让重新授权是一次覆盖、`KEY idx_grantee`），`DROP_ORDER` 补位；② `memory/store.py` 新增 `Grant`（判活口径 `active_at(now)`，`expires_at == now` 即失效）/ `GrantRepo` 协议 / `InMemoryGrantRepo` / `MySQLGrantRepo`，`InMemoryAtomRepo` 增 `audit()` + `.audits`；③ `memory/service.py` 新增 `grant_read` / `revoke_read` / `list_grants` / `authorize_read` 与两个闸门 `_require_read`（→ `MemoryAccessDenied`）/ `_guard_write`（→ 404 或 `MemoryReadOnly`）+ `_note_advisor_read`；④ `memory/api.py` 把**身份与目标拆成 `_user()` / `_owner()`**（`X-User-Id` 带权限、`X-Memory-Owner-Id` 只是一个资源 id、不携带任何权限），新增 `POST/DELETE/GET /v1/memory/grants` 且只许本人操作；⑤ `main.py::build_memory` 注入 grant 仓库；⑥ `base/log_messages.py` 登记 6 个事件，并修掉 `store.py` 两个**未注册的事件名**（`chat_message_write_failed` / `session_summary_write_failed` → 已注册的 `chat_message_persist_failed` / `session_summary_persist_failed`）。**状态码语义**：无 grant 的跨用户读写一律 **404**（403 会承认"这条记忆存在"）；有读 grant 的顾问写 **403**（对方已知存在，只需说明只读）；无身份 **401** 且**绝不**退化成共享 `anonymous`。**身份规则**：白名单在 `base/config.py::is_trusted_env`（`memory/` 不许 import `orchestration/`，所以不放 `memory/api.py`），白名单外**只认**网关注入的 `request.state.user_id`、**不回落**请求头——于是生产配置下没接网关就是稳定 401，这是有意的 fail-closed。**不做**：团队记忆（v2）、管理员原文读取（靠"表里只有 owner→grantee 一个方向"保证，不加 `role` 字段）。**新增 27 条单测**（`tests/test_memory_permissions.py`：授权语义 14 + HTTP 状态码与身份头 13，含"过期 grant 不放行""撤销后立即失效""顾问写入是 403 而非 404""生产下忽略 `X-User-Id`""被拒的读与写都留审计行""`grants=None` 时跨用户默认拒绝"），全量 **479 → 506 passed**。下一步 M1 向量召回 / M5 评估仍等 ingest 跑完 | 项目组 |
| 2026-10-30 | **理解层接线（PR5 落地）：意图改大模型 + 槽位/归一化/年级下推三条断线接上，新增 §7.2.7-H 并重写 §7.4**。此前三件"理解"能力代码都已写好但**生产路径一次都没调用过**：`qa_slots.extract()/resolve()` 只有单测在调、`query_processor.normalize()` 唯一 import 是它自己的单测、`QueryContext.filters()` 不接受 `stages/grades`（于是 `_build_expr` 的年级/学段条件永远传空）。后果是"**数学**勾股定理怎么证明"仍按全库检索、`inferred/context/multi/conflict` 四条路由在 `/v1/chat` 上不可达、全角与换行原样进 embedding。① **意图改大模型**：新增 `understanding/llm_understanding.py::analyze()`，一次调用同时产出「改写句 + 意图」——两者落在流水线同一位置（FAQ 未高置信命中之后、检索之前），拆两次等于同一份预算花两遍；**校验与降级仍是两条独立的线**（`rewrite_rejected` / `intent_rejected` 分开记，任一失败不影响另一项），词表外标签与置信度 <0.6 一律回退（后者落 `unknown` → `generic_clarify`，不猜）。规则分类器 `understanding/intent.py::classify()` 保留为回落，`FEATURE_INTENT_LLM=false` 即退回它。② **槽位接线**：`orchestration/chat.py` 的 `qa_slots.from_param(subject)` → `qa_slots.resolve(query, param=subject)`，规则版 `extract()` 首次在 `/v1/chat` 可达（仍**不调模型**：槽位直接决定 Milvus 过滤，"抽不到→全库"比"抽错→只查另一门学科"安全）。**行为变化须知晓**：问句里点名学科后 `subject_source` 由 `not_provided` 变 `explicit`，FAQ 走该学科硬过滤，可能由 `high` 变 `subject_empty`/`ambiguous`——这正是要的效果。③ **归一化接线**：`_understand()` 里过 `query_processor.normalize()`（NFKC + 空白/重复标点折叠，**不删词**），只改 `state.rewritten`，`state.query` 保持用户原话（FAQ 精确哈希与日志原话都依赖它）。④ **年级/学段下推**：`QueryContext` 增 `stages/grades`、`filters()` 透传，`planner` 取 `ctx.filters()` 的路径自动生效（三种策略不用各改一遍）；`relaxed()` 现在清**四项**业务过滤（`subjects/doc_types/stages/grades`）、安全三项原样，年级误抽在空结果时由 `relax_business_filter` 自动救回。⑤ **修一处既有口径缺陷**：证据门的 `qcov` 原先读"实际检索用的那句"，接上改写后分母变小 → 覆盖度虚高 → 门变松，而 τ=0.74 / qcov=0.20 是按原问句标定的；改为对 `ctx.original_query` 计算（`RagPipeline._evaluate_into(..., original_query=...)`，只传给门、不改结果对象）。⑥ **新配置**：`FEATURE_INTENT_LLM` / `FEATURE_QUERY_REWRITE` / `FEATURE_GRADE_FILTER` / `UNDERSTANDING_TIMEOUT_MS`(800) / `LLM_ROUTER_MODEL`，`self_check()` 校验超时在 `(0, TOTAL_BUDGET_MS)` 内且不超总预算一半、开关打开但无 key 时只告警不退出。⑦ **新增单测 24 条**（`test_chat_rewrite.py` 13：FAQ high 时 `llm.complete` 调用 0 次 / 无 key / 超时 / 五种出口校验 / 全角折叠 / 开关关掉不调；`test_slots_wiring.py` 9：`explicit` 收窄到检索 / 未提学科仍全库 / 年级学段抵达 `filters()` / `FEATURE_GRADE_FILTER` 回滚 / 多学科可达；`test_rag_store_filter.py` +2：两路 `expr` 都含 `stage in [...]`/`grade in [...]`，`relaxed()` 后只剩安全三项），全量 **506 → 530 passed**。⑧ **文档**：§3.2 补「落地形态」表（含"改写必须排在 FAQ 之后"与"槽位用规则不用模型"的理由）、§6.1 补预算账（理解层 ≤800 ms 从**首 token 预算**里扣，SLO 未重测）、§7.4 整节重写为 11 步实际时序、§7.7 重写（文件树 + 分工表 + 开关表 + 诚实标注）、新增 §7.2.7-H。**诚实缺口（都不是"变了"而是"没量"）**：指代消解仍未做（`rewrite()` 的会话拼接路径一行未动，`history_hint` 未传）；首 token p95 ≤1.2 s 与新链路未重测；黄金集 Recall@10 / MRR、τ=0.74、qcov=0.20、`FAQ_HIGH_MARGIN=0.22` 全是在旧链路上标的；**大模型意图从未在 `intent_test.jsonl` 1 020 条上评过 F1**（§7.8.4 的 0.967 属于规则分类器，而线上默认跑的不是它）。另：本次 key 只写进 `.env`（`.gitignore` 已忽略），`.env.example` 留空值 | 项目组 |
| 2026-10-31 | **阶段 2 任务 2.4（复杂问题与指代关系）与 2.5（统一理解结果）落地并验收**。① **复合问句真正拆开了**：`_build_understanding()` 用 `compound_query.split(state.query)` 产出 `sub_queries`，planner 按 `len(subs) >= 2` 走 `SUB_QUERY`，**一任务一次 `retriever.search()`**——规格原例「比较一次函数和二次函数的区别，并分别给我出三道练习题。」拆成 `distribute` 三路（区别 / 一次函数三道 / 二次函数三道），测试断言**恰好三次检索调用**而不是只看 `plan.strategy`（纯规则下 `looks_multi()` 也会把整句标成 `sub_query`，那时只有一路，策略名分辨不出"真拆了"）。子问题来源是**结构性保证**：`_verified()` 用字符多重集校验，子问题里的每个字原句都出现过（次数不超过），编出来的候选直接进 `dropped` 而不是被修好。② **指代从"代码存在但不可达"变成可达**：`_load_session()` 里 `rewrite()` 把上一轮用户问句拼在前面（`rewrite_method="rule"`），`has_reference()` 命中时 `_history_subject_hint()` 经 `history_hint` 交给 `qa_slots.resolve()` → `subject="数学"` / `subject_source="context"` / 置信 0.6（按 §7.1.8 路由表**不许直接作答**，只能交 RAG）。解不开时记 `coreference_unresolved` 且**不猜主语**；换一句自成一句的问句则**不许借用上一轮的学科**（这是最容易犯的错误过滤）。③ **统一理解结果**：`UnderstandingResult` 的五条规则逐条可观测——原话 `query` 一个字不动（归一化只进 `rewritten`）／没改写时 `rewritten=""` 而不是复制原话（否则"改写生效没有"看不出来）／`retrieval_query` 的 `rewritten → normalized → query` 是**属性回退链**不是分支／`success` 的边界写在类型上：`understanding_bad_json` 判 `False`、`intent_rejected` 仍 `True`（合并的是网络往返，不是成败）。④ **新增 28 条测试**：`tests/test_compound_query.py` 12 条（16 个用例，含 5 条参数化不拆样本）+ `tests/test_chat_understanding.py` 12 条（接线用替身，三个开关默认关掉以保证断言失败时指向接线而不是模型输出）。用自己的解释器实测：理解与问答九套件 **136 passed**，全量 **700 passed / 8 failed**——8 条全部是 `tests/test_acceptance_stage1.py` 的既有红线（阶段 1 未做的那批），与本次无关；基线 672 passed，本次 +28 全部通过 | 项目组 |
| 2026-10-31 | **补上槽位冲突的产出端，阶段 1 / 阶段 2 口径对齐**。`conflict` 此前只有消费端没有生产端：`FAQService._resolve_scope` 早就为它实现了全库扫描 + `_NO_DIRECT` 封顶 mid，但**库里没有任何一条路径能产出 `subject_source="conflict"`**——`qa_slots.resolve("数学的二次函数怎么求", param="化学")` 静默返回 `explicit / 化学`，用错的学科做了硬过滤，而响应体看起来完全正常（`explicit` 是个"正常"取值），排障时看不出任何异常。① **`qa_slots.resolve()` 自己算冲突**，判据三条同时成立：参数归一化后合法 + 问句抽出的学科集合非空 + 参数不在该集合里；命中即返回 `SlotResult(subject_source="conflict")`，学科清空（`scope==()` → 下游不加过滤、走全库）。新增 `_query_fields()` 复用问句侧的非学科槽位（年级 / 章节 / 任务参数），冲突只清学科、其余照旧留给问句，同时避免字段清单被抄第四遍。② **四种刻意的非冲突**写进代码注释与 §7.1.8：参数落在问句的多学科集合里（用户没把话说全，收窄正是他要的）/ 问句没点名学科（没有可矛盾的对象）/ 参数不在词表仍判 `invalid`（要被 400 暴露，不许被冲突态盖住）/ 历史提示与问句不一致（主题切换是正常对话，当前这一轮算数）。`history_hint` 不参与冲突判定——拿历史否决用户这一步写下的学科，是"指代借用"那条错误过滤的另一种翻身方式。③ **阶段 1 口径对齐**（`docs/EduRAG-Pro-阶段1-请求上下文与入口设计.md` §3.4）：`admit()` 的规则 4 只判"参数写没写错"（不在词表 → 400 `subject_invalid`），**不判"参数与问句是否矛盾"**——后者是 `conflict`，正常走全库 + 追问。阶段 2（`docs/EduRAG-Pro-阶段2-问题理解设计.md` §3.8 第 6 条）写明冲突只由槽位层产出、意图层不参与。④ **消费端零改动**（`_NO_DIRECT` 早已含 `conflict`；`understanding/slots.py::merge/finalize` 走 `dataclasses.replace` 保住 `subject_source`），因此本次只改产出端。⑤ **新增 8 条测试**：`test_slots.py` 6 条（冲突置位与清空 / 非学科槽位保留 / 多学科问句也判冲突 / 参数落在多学科集合内不算冲突 / 历史提示永不与问句冲突 / 参数非法不被冲突掩盖）+ `test_acceptance_stage2.py` 1 条**产出端→消费端接线**（`resolve()` 判 conflict → `FAQService.search()` 给 `subject_state="full"` / `degraded=("subject_conflict_full_scan",)` / `tier="mid"` / `answer is None`），并把此前 `xfail(strict=True)` 的那条验收项翻成正式通过（标记已删）。实测 `tests/test_slots.py tests/test_acceptance_stage2.py` **40 passed**，全量 **726 passed / 8 failed**——8 条全部是 `tests/test_acceptance_stage1.py` 的既有红线（阶段 1 准入层未做），本次未引入新失败、`xfailed` 由 1 归 0。**未做**：阶段 1 准入层的 8 条红线仍未实现（`conflict` 的 `/v1/chat` 路径目前不经过 `admit()` 的长度/会话校验），那是另一批工作，等确认再做 | 项目组 |
| 2026-10-31 | **阶段 1 准入层落地，8 条红线全绿（`tests/test_acceptance_stage1.py` 20 passed；全量 726 passed / 8 failed → 734 passed / 0 failed）**。`/v1/chat` 与 `/v1/chat/stream` 统一走 `orchestration/api.py → admit() → RequestContext → ChatPipeline.ask()`，此后"非法请求不进后续流程"由**结构**保证而不是靠每个端点各自记得校验。① **八条红线逐条**：空问句 400 `empty_query` 且 `faq.calls==0`（不进 FAQ）/ 401 字 400 `query_too_long`（**正好 400 字必须放行**）/ 400 响应体是完整 `ChatResponse` 形状 + `detail`（前端只写一份解析）/ 非法学科 400 `subject_invalid` 带合法列表 / 请求体未知字段（`"qurey"`）400（`extra="forbid"`）/ 生产环境忽略 `X-User-Id` 因而不签发 `session_id` / 白名单环境认它并签发、且**带回来必须沿用**（不另起一个）/ 请求体 `"role":"ADMIN"` 永不成为响应 `role` / 他人 `session_id` 400 `invalid_session`。② **身份三级先严后宽**：JWT Bearer（畸形 → 401 `invalid_token`，**不回落匿名**）→ 白名单环境的开发期请求头 → 匿名；`role` 只从 token 或服务端上下文来，请求体里的 `role` 连解析都进不去。③ **校验顺序固定**：401 优先于一切；然后 空 → 超长 → 无可识别字符 → 学科非法 → sid 格式 → 归属他人 → 有 sid 无 `user_id`（规则 7 先于 6，否则"没身份带 sid"会被误报成越权）。`owner_of` 抛异常**fail-open** + `degraded=("session_owner_unchecked",)`——Redis 抖动不该让人问不了问题。④ **两处路由层守卫（有意偏离设计 §5，已写进设计文档 §9）**：body 64 KB → 413 与 `ValidationError` → 400 放在 router 而不是 `main.py` 中间件，因为验收用例用裸 `FastAPI()` 只挂 `chat_router`，中间件根本不在链上；`POST` 处理函数**故意不声明 body 形参**，这样体积能在 pydantic 解析之前被拦住（解析本身就会把超长 body 读进内存）。⑤ **`pyjwt` 不装**（设计 §5 原文写"增 pyjwt"）：`admission.py::decode_token` 用 stdlib `hmac` 实现 HS256，`alg` 必须**严格等于** `HS256`（`none` 与算法混淆直接拒绝）。⑥ **FAQ 缓存键补 `tenant` 段**（§6.4）：`edurag:{env}:faq:t{tenant}:v{版本}:{范围}:{sha1}`，空租户落成 `t-` 而不是空段；`FAQService` 构造期收 `tenant_id=`（别名 `tenant=`），`main.py` / `qa_system/main.py` 都已接线。⑦ **两处既有断言按新契约改写**：`test_chat_api.py` 与 `test_chat_stream.py` 的空问句 422 → 400（422 的语义是"字段类型/长度不合规"，400 是"参数不对"；前端不该为同一类产品口径分两条分支）。⑧ **未做（明确记录，避免误以为已完成）**：`query_processor.py` 的 400 字截断分支**保留**为入口之下的兜底（准入层已先拒，实际不可达；改它要动 `test_query_processor.py` 的既有断言，收益为零）；`qa_cache.cache_key()` 尚未成为 C1 的唯一构造出口（`FAQService._cache_key` 仍自己拼，形状与 §6.4 一致）；`llm_gateway` 尚未带 `X-Request-Id` / `X-Trace-Id`；`main.py` 的 `trace_middleware` 仍不解析 `traceparent`（`make_trace_id` 在 `admit()` 里做）；设计 §6 的四个新测试文件未单独建（现有 `test_acceptance_stage1.py` / `test_permissions_contract.py` / `test_trace_middleware.py` / `test_chat_api.py` / `test_chat_stream.py` 已覆盖同一批断言） | 项目组 |
| 2026-10-31 | **中置信 FAQ 候选作为辅助上下文进生成 prompt（§7.1.9 边界 2 修订）**。原口径是"中置信禁止把 FAQ 原文塞进生成 prompt"，本轮改为"**可以进，但只能当参考、不能当证据**"。① `base/contracts.py::FAQCandidate` 增 `answer: str = ""`（默认值保住测试里现有的三参数构造点；空答案的含义是"这条候选没有答案"，整条在渲染时跳过而不是渲染出空「答案：」）；`qa_search.py` 的两处候选构造点（正缓存回填 / 负缓存回填）从 `snapshot.records[fid].answer` 取答案——**缓存 schema 未动**（`CacheEntry.candidates` 仍只存 `(faq_id, score)`，快照与语料同版本，不需要把正文也塞进 Redis）。② `rag_system/core/prompts.py` 新增 `FAQ_CANDIDATE_NOTICE` / `FAQ_CANDIDATE_RULES` / `FAQ_CANDIDATE_MAX_CHARS=500` / `render_faq_candidates()`，`build_messages(..., faq_candidates=())` 把候选渲染成**独立一条 system 消息**，装配顺序固定为 记忆 → FAQ 候选 → 历史 → 资料 → 问题；**GENERAL 分支（无证据的通用模板）不收候选**——那个模板禁止给具体事实，半确信的答案塞进去只会自相矛盾。500 字上限是为了让三条候选最坏 ≈750 token，挤不掉 6 000 的知识预算。③ `orchestration/chat.py` 新增 `_faq_prompt_candidates()`（只 mid 档、只 top3、空答案整条丢、注入条数进日志），同步与流式两个 `build_messages` 调用点都已接线。④ 新开关 `FEATURE_FAQ_CANDIDATES`（默认 `true`）是**回滚开关兼 A/B 分组**；**没有升 `PROMPT_VERSION`**——`understanding/profiles.py` 有 8 处硬编码 `prompt_id="edurag-qa-v1"`，升版本号会把"改了一个开关"扩散成一整轮模板治理。⑤ **三条隔离保证它进不了 `citations`**：独立 system 消息、「候选N：」而非 `[n]`、不进 `【资料】`——`citation.align()` 的对齐表里根本没有它的位置；模型最坏也只会写出一句没有引用的话，而 `require_citation=True` 的档位会由 `citation.repair()` 整句剔除或整体拒答。**明确没有变的三条**：`FAQHit.answer` 在 mid 档仍是 `None`；`faq_suggestion` 仍不含 `answer`（卡片点开走 `GET /v1/faq/{id}`）；候选永远不进 `citations`。⑥ `base/log_messages.py` 登记 `faq_candidates_injected`。⑦ 新增 `tests/test_faq_candidates_prompt.py` **19 条**（渲染层：编号形态与免责声明逐字、空答案跳过、超长截断、自定义上限；装配层：只在一条 system 里、不改知识编号、顺序固定、GENERAL 分支不收、无候选不加；编排层：mid 注入且只调一次模型、mid 响应仍无 FAQ 答案、`citations` 只来自【资料】、low 永不进、high 零次生成、开关关掉回旧行为、全空答案整块不加、最多 3 条），全量 **734 → 753 passed**。⑧ 文档同步：§7.1.9 边界 2、§7.8 三档动作表、§5.6 第 9 条、§2.5 环境变量表、`.env.example` | 项目组 |
| 2026-10-31 | **新增 §5.7「AI 协作纪律」（用户指定，两条硬约束）**：① **没有用户明确要求不得新建项目文件**——脚本/测试/文档/报告/临时产物一律算，允许直接做的只有"用户点名的文件、报错链路必须改的文件、§5.5 的架构决策同步"，确需新建先说清"建哪个、为什么非建不可、不建会怎样"再等点头；② **阶段测试必须跑在用户本机真实环境里、不许给虚假结果**——解释器固定 `F:/miniconda3/envs/ragpro/python.exe`（conda env `ragpro`），报告里每个数字必须是这一次的真实输出，没跑就写"未跑"，断言前先确认 `/readyz` 与所连的 Milvus/MySQL/Redis，脚本报错原样贴出、不许 `xfail` 当通过或放宽断言凑绿。§5.4 加了一条指向本节第 2 条的交叉引用 | 用户 |
| 2026-10-31 | **新增 `auth/` 认证模块（登录登出闭环）**：`/api/v1/auth/{register,login,refresh,logout,me}` 五个接口，与聊天入口的 `{code,message,data}` 信封**分开**（用户决定 4）。① **角色只认库**：`register` 恒发 `STUDENT`，`role` / `tenant_id` / `user_id` / `permissions` / `acl_tags` 出现在请求体即 400（`_Body` 的 `extra="forbid"`），`login` 的角色取自 `sys_user.role` 而非令牌或请求体；令牌 claim 只有 `sub/role/tenant/acl`，**刻意不含 `kb_version`**（写进令牌会让"切索引后旧令牌查旧库"成为可能）。② **口令**：Argon2id（`argon2-cffi==25.1.0`，导入名 `argon2`），哈希/校验全部走 `asyncio.to_thread`；`login` **先验口令再看 `status`**，因此"口令错"与"查无此人"同为 401 `invalid_credentials` 同句文案（用户不存在时先 `burn_equivalent_time` 抹平耗时差），"账号停用"必须口令正确才暴露 403 `user_disabled`。③ **令牌**：`access` 有效期 **600 s**（用户决定 2，覆盖原 spec 的 3600），登出只作废 **refresh**——所以"登出后 access 仍能调 `/me` 直到过期"是契约而非缺陷，已写成测试钉住；`refresh` 轮换即作废旧 `jti`，`typ` 不可互换（refresh 当 access 用 → 401）；`logout` 作废该用户**全部** refresh 且**永不报错**（空 body / 垃圾串 → 200 + `revoked: 0`）。④ **实现**：`auth/tokens.py` 是 HS256 的**唯一**实现（stdlib `hmac`，不引 `pyjwt`；`orchestration/admission.py` 只是薄包装），`auth/passwords.py` / `auth/store.py`（`User`/`TokenRecord`/Protocol + 内存与 MySQL 两实现）/ `auth/service.py` / `auth/api.py`；`scripts/init_db.py` 增 `sys_user`（`UNIQUE (tenant_id, username)`，`role`/`status` 枚举，`password_hash VARCHAR(255)` 存 Argon2id 全串）与 `auth_token`（只记 refresh，`jti` 主键），`DROP_ORDER` 同步。⑤ **降级**：`JWT_SECRET` 为空**不阻断启动**（`main.py` 记 `auth_not_configured` 告警，四个写接口回 503），`app.state.auth` 缺失回 503 `auth_not_ready`（部署问题，不是 500），未预期异常回 500 信封且不回显异常正文。⑥ **未做（明确记录）**：`captcha` 字段收下但**不校验**（无图形验证码服务）、登录限流未接（`admission` 的 `:rl:` key 族留给后续）、`sys_user` 的 `TEACHER`/`ADMIN` 目前没有任何接口能创建（只能手工入库）。⑦ **验证**：新增 `tests/test_auth.py` **23 条**（注册/登录/me/刷新/登出/超时/降级七组，含"请求体注入角色被拒且一条不落库""过期令牌在 `JWT_LEEWAY_S=60` 下仍接受"两条），`F:/miniconda3/envs/ragpro/python.exe -m pytest` 实测 **23 passed**，全量 **778 passed / 0 failed**。这是 §5.4「新节点必须有单测」的配套文件，不是顺手新建 | 项目组 |
| 2026-10-31 | **记忆模块 L2 向量召回与 L3 画像落库落地（新增 §7.9.4），四层架构补齐**。此前 L2 的排序是字面 bigram（换同义说法就召不回）、L3 画像**没有存储**（每次从身份类原子记忆现算，运营改过的存不下来）。① **L2**：`memory/store.py::MilvusAtomIndex` 用 bge-m3 稠密向量 + 独立集合 `MILVUS_MEMORY_COLLECTION`（默认 `edurag_memory`，**与知识库集合分开**：知识库可重建、记忆是用户数据）；client 与 embedder **由组合根注入**（`main.py::Container._atom_index()` 把 RAG 已在用的那份递进来，避免第二次装载 2.5 GB 权重），守住 §7.0「`memory/` 不 import `rag_system`、不自建 `MilvusClient`」。时间列 `valid_to`/`deleted_at` 存 INT64 epoch、`NULL→0`（Milvus 无 NULL），可见条件 `valid_to == 0 and deleted_at == 0 and confidence >= MEM_MIN_CONFIDENCE` 与 MySQL 侧三件套逐字等价；`_expr_str()` 转义 `\` 与 `"`（表达式是字符串拼接，一个未转义引号就能拼出 `or` 子句）。② **召回顺序与删改传播**：`_recall_order` 把索引命中作**前缀**、其余按字面 `_rank` **补齐**（索引滞后只该意味着"这次没排上"，绝不等于"这条记忆不存在"）；`_sync_index` 活着的 `upsert`、失效/删除的**从索引删掉**而不是写 `valid_to=now`（滤错一次就会把已删除的记忆重新召回）。③ **fail-open 两层**：`MilvusAtomIndex` 自己吞异常返 `None`，`MemoryService._index_search` / `_sync_index` 再各兜一层——换一个会抛异常的实现也只该退化成字面排序或只记日志，不能把召回连坐成空记忆、不能把成功的写入变失败。**契约：`search()` 的 `None`（索引不可用）与 `[]`（健康但没匹配）不是一回事**，混成一个会让"索引挂了"看起来像"这个用户没有记忆"且永不告警，服务层与 Milvus 层各钉一条单测。④ **L3**：`user_profile` 表（`PRIMARY KEY (tenant_id, user_id)`）+ `ProfileRepo` / `InMemoryProfileRepo` / `MySQLProfileRepo`（`ON DUPLICATE KEY UPDATE`），**只有人工写过的才落库**（`source='manual'`），`derived` 版每次现算、不落库（随时可重算，同步派生表只会多一处不一致）；`_recall_inner` 里**存过的画像压过派生值**；`GET/PATCH/DELETE /v1/memory/profile` 走同一套身份与 `_guard_write` 口径（本人可写、只读顾问 403），改与清各写一行 `memory_audit`（`profile_set` / `profile_clear`），`clear_all` 一并删画像。⑤ 配置：`FEATURE_MEMORY_VECTOR`（默认 `true`，关掉即字面排序）、`MILVUS_MEMORY_COLLECTION`，`.env.example` 与 §2.5 同步；`scripts/init_db.py` 的 `user_profile` DDL 与 `DROP_ORDER` 已登记。⑥ **验证**：`tests/test_memory.py` 新增 19 条（召回顺序三态 / 写穿与删除传播 / Milvus 过滤条件抵达 `search` 的实参 / `user_profile` DDL 与 `DROP_ORDER`），该文件 32 条，`F:/miniconda3/envs/ragpro/python.exe -m pytest` 实测 **792 passed / 0 failed**（本次共连续跑 4 次全绿；其中最早两次全量运行曾打印若干 FAILED，隔离复跑单文件即全绿、后续四次全量均未复现，**未能定位原因**，如实记录不下结论）。⑦ **未做（明确记录）**：`MemRecall@5 ≥0.85` **未跑**（`edurag_memory` 无真实语料、无 100 题记忆回归集，召回率一个数字都不能引用）；索引**重建/回填脚本没有**（历史行不会自动进集合，`count()` 留好了但没人调）；`p95 ≤120 ms` **未测**——CPU 上 bge-m3 编码是秒级，默认 120 ms 预算下向量召回几乎必超时退化为字面排序，要等 GPU 环境量，在 CPU 机器上报"通过"没有意义 | 项目组 |
| 2026-11-01 | **复合问句拆分改由本地小模型完成、HyDE 交给 DeepSeek（用户指定）**。① **拆分**：`understanding/compound_query.py` 新增 `split_with_model()`——**结构预筛在前**（单诉求句根本不发本地调用）、超时 `COMPOUND_TIMEOUT_MS`（默认 400）、模型给的子问题必须过既有的**字符多重集校验** `_verified()`，编出来的字一律丢弃。四条失败码各自独立：`compound_timeout` / `compound_failed` / `compound_bad_json` / `compound_rejected`，**任何一条都退回规则拆分、永不抛异常**；模型回空列表**不覆盖**规则已经拆对的结果（0.8B 偶尔把三诉求句当一件事，信它一次用户就少拿两件事的证据，而退回规则最坏只是多查两路，代价不对称）。编排侧 `orchestration/chat.py::_split_compound()` 排在**意图之后**（拆几路取决于已判定的档位）、槽位收尾之后（会追问早退时先拆只是白付一次本地调用）；`state.compound` 是唯一产物，`_build_understanding()` 只读它。`FEATURE_COMPOUND_LLM=false` = 纯规则、与接模型之前逐字节一致（回滚开关兼 A/B 分组）。**没配本地端点不记 degraded**：那是部署形态不是本次请求的岔子，记了会让开发机每条复合问句都挂一个降级码、把线上降级率永久污染。② **HyDE**：`main.py::Container._hyde_gateway()` 用 DeepSeek 生成假设性答案，**显式点名 `LLM_QA_MODEL`**（不跟网关默认模型走，否则换网关时 HyDE 与正式答案会静默分叉），预算走 `HYDE_BUDGET_MS`（1500）并**同时**套在生成调用与 `RagPipeline` 外层 `wait_for` 上——修掉了原先传 `int(llm_timeout_s*1000)`=60 000 ms 的缺陷（那等于没有上限，一次慢调用就把 `TOTAL_BUDGET_MS` 吃穿）。三条护栏不变：HyDE 文本只用于检索（`_drop_hyde_echo`）、失败退回 `DIRECT + hyde_failed`、`FEATURE_HYDE=false` 整体回滚。③ **配置与文案**：新增 `FEATURE_HYDE` / `HYDE_BUDGET_MS` / `FEATURE_COMPOUND_LLM` / `COMPOUND_TIMEOUT_MS`，`base/config.py::soft_warnings()` 在"理解 800 + 拆分 400 + HyDE 1500 = 2700 ms"叠起来时告警（**故意不进** `self_check()`：硬校验只拦"单个值为 0 / 超总预算"那种明显写错，三项相加是性能取舍，`ALLOW_DEGRADED_START=false` 时不该让进程起不来）；`base/log_messages.py` 登记四个 `compound_*` 码与 `compound_split` 事件，**不进** `_DEGRADED_ANSWER`（退回规则拆不弱化用户读到的那句话，与"记忆缺一段"同一条判据）。④ **测试**：`tests/test_compound_query.py` 24 条（+8：无端点不降级 / 单句不发调用 / 模型拆分过校验且预算与模型名抵达网关 / 编词被拒回退 / 超时 / 网关异常 / 坏 JSON / 模型回空列表不覆盖规则结果）、`tests/test_chat_understanding.py` 14 条（+2：本地模型拆分**真的抵达检索器**——断的是 `retriever.queries` 而不只是 `state.compound`，"拆分器拆对了"与"这份拆分真的驱动了检索"是两件事；开关关闭时本地模型一次都不被调用）。`F:/miniconda3/envs/ragpro/python.exe -m pytest` 实测 **802 passed / 0 failed**。⑤ **诚实缺口**：打开 HyDE 违反 §5.4「改检索参数要附黄金集对照报告」——**该报告没跑**，所以这是"按需求打开的实验性能力"，不是"已验证的召回优化"；打开后默认配置在首 token 之前最多叠 2700 ms，**首 token p95 ≤1.2 s 仍未重测**（§6.1 已记）。⑥ §2.5 / §7.2.7 / §7.4 / §6.1 同步 | 用户 |
