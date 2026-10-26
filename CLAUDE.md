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
├─ understanding/                     # 理解层（PR5 待建）：意图分类 / 会话改写 / 模型版槽位
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

### 3.2 理解层（P0 意图/改写，P1 槽位；学科槽位先行，见 §7.1.8）

1. 意图 **K12 教学 6 类**（2026-10-26 定案，施工图见 §7.7）：知识点查询 / 题目讲解 / 教案素材 / 学习方法 / 备课辅助 / 闲聊。
2. 置信度 < 0.6 → **不猜**：`label="unknown"` + `method="fallback"`，走 `generic_clarify` 档（全库检索不收窄 + 通用模板 + 结尾追问），**不写共享缓存**。禁止低置信硬路由到某一专业档位——错误档位的答案会被写进缓存。
3. 意图**不写成下游的 `if/else`**：分类结果映射为**路由档位**（是否检索 / `doc_type` 过滤 / 策略先验 / 引用要求 / 模板 / 缓存资格），下游只读档位；标签字符串只进日志、指标与打标。
4. 会话改写：含指代词、长度 < 8 字、无实体时触发；规则优先，小模型兜底；输出 `rewritten` 与原 query 一并落库。
5. 敏感词与越界请求（涉政/涉黄/竞品诋毁）在前置节点拦截。
6. **验收**：黄金集多轮子集（50 题）改写后 Recall@10 提升 ≥ 15 个百分点；意图 F1 ≥ 0.90（`intent_test.jsonl` 1 020 条）；`unknown` 比例 ≤10%；6+1 档路由单测全覆盖；第一阶段真正的验收是**档位是否改善下游决策**（A/B 见 §7.7），F1 只是门槛。

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
3. **引用分两层，别混成一件**：① **策略层 `require_citation`**（来自档位，见 §7.7）决定"没有可验证证据时是否允许作答"——为 `True` 时证据门不过就**不许生成**，走无证据策略，而不是让模型硬写；② **机制层引用对齐**——正文 `[1][2]` 必须映射回真实 chunk（`[{n, doc_id, file_name, page_no, section, parent_id, score, snippet}]`），**无法对齐引用的句子在输出前剔除或改写**。禁止为满足格式在末尾拼 `[1][2]` 或伪造来源；`require_citation=False` 的档位允许"通用建议"句子，但必须显式标注不是知识库结论。
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
9. 不要把 FAQ **中置信**当高置信返回，也不要把 FAQ 原文塞进生成 prompt 让它"改写"（`_NO_DIRECT` 与 §7.1.9）。
   前者把 precision 0.80 的命中当 0.95 用；后者模型会整段照抄，等于用引用的形式输出未经确认的答案。

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
| 第四步 | 增加三档路由：高置信直接答 / 中置信 FAQ 候选 + RAG 并给 / 低置信交棒 | **契约见 §7.1.9**：high 档 test 精度 ≥0.95 且召回 ≥0.25；mid 档不返回答案（只给候选与追问）；`faq_route_high/mid/low_total` 三档计数齐备；`FAQ_HIGH_MARGIN=FAQ_MARGIN` 可精确回退到单档 |
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
模型版放 `understanding/slots.py`（PR5），由编排层算好后**通过参数**把 `SlotResult` 传进来。
`qa_system` 只依赖契约，不 import `understanding`。

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
2. **中置信不返回答案**：`FAQHit.answer` 仅在 `tier="high"` 时非空。中置信只给候选**问题**（做 query 扩展与追问），**禁止**把 FAQ 原文塞进生成 prompt——模型会照抄，一条错答案立刻变成带引用的"事实"。要答案必须走下面的用户确认。
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

| 档 | 检索 | 返回 | RAG 不过证据门时 |
|---|---|---|---|
| high | **跳过**（省 §6.1 的检索 + 重排 350 ms） | FAQ 答案逐字返回 + `faq_id` 来源；`require_citation` 不适用（FAQ 是人工审核过的事实） | — |
| mid | 跑；候选 top3 进 query 扩展 | RAG 答案 + 引用；附 `faq_suggestion = {faq_id, question}`（**不含 answer**，点开走 `GET /v1/faq/{id}`） | **追问**："你是不是想问：<top1 question>？"+ 候选列表；不返回 FAQ 答案 |
| low | 跑；候选照旧 | RAG 答案 + 引用 | 兜底话术 + 转人工（§3.4） |

mid 与 low 的区别就两条：mid 多一个**可信的追问对象**（top1 候选）和一张 FAQ 卡片。这两样放到 low 档是噪音——候选连阈值都没过，追问只会更离谱。

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
**近重复文档检测**（`09_textbooks` 里 `人教版-数学-九年级上册.pdf` 与 `人教版数学九年级上册.pdf` 是同一本教材的两次下载，
md5 不同、内容一致，`content_hash` 去重抓不到——两份都入库会让同一段证据以"两篇文档"的身份互为对手，
证据门的跨文档 margin 直接判 `ambiguous`。`datasets/` 只读，处理方式只能是摄入侧加近重复检测：
先按 `file_name` 归一化查重，再对前 N 页文本做 simhash/MinHash，命中即拒绝入库并报 `doc_duplicate`）；
题库入库（`03_question_bank` 的 answer 多数只有一个选项字母、analysis 多为空，不能当 FAQ 用；
作为 `doc_type=question_bank` 的 RAG 证据也需先过滤掉无解析的条目）。

**证据门改为跨文档比较（2026-10-20）**：原先 `margin` 取 top1−top2，但 top2 常是同一份教案的相邻父块，分数天然接近（实测 0.0006–0.02），top1 已经 0.99 仍判 `ambiguous`。现在 `rag_system/core/evidence_gate.py::evaluate` 只在"另一篇文档"里找竞争对手（doc key 取自 `parent_id` 的 `{doc_id}:p{n}` 前缀），同一文档多块视为互相印证；`n_docs` 统计的是文档数。

**证据门标定完成（2026-10-20，commit 40918c2）**：`datasets/07_eval/ragas_eval.jsonl` 是 **1000 条**（不是预估的 157），`scripts/eval_rag.py` 建 763 条可判定样本（去掉跨学科冲突与空证据），跑检索并解出两条判据：① 分数门槛做不了"同题族"区分——可答与不可答的 top1 中位数只差 0.04（0.782 vs 0.744），且 58% 不可答题分数高于 0.70；② **问题词覆盖率（qcov）** 是更强的独立信号（可答中位 0.6 vs 不可答 0.2）。据此给 `evaluate()` 加第四条判据 `min_qcov` 并新增 `query_coverage()`，标定取「误放行 ≤5%」的取舍点 **threshold=0.74 / margin=0.01 / qcov=0.20 → 可答放行 34%、不可答误放 3%**（放松到 0.72 则 50%/9%）。根因不在门槛而在语料：黄金集覆盖 7–9 年级，库里只有 1 本九年级数学 + 6 份短教材，可答题里只有 20% 能召回到正确文档。报告 `docs/analysis/rag_threshold_report.md`，原始跑批 `docs/analysis/rag_eval_runs.jsonl`（可复跑重标定）（763 行，含每题 top-10 证据与 qcov）。

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
| 4 | **FAQ ∥ 意图 ∥ 记忆召回** | qa + understanding + memory | FAQHit / intent / MemoryContext | 120 ms（并行，各自独立超时） | 任一失败则该路为空 |
| 5 | FAQ 哪个档？ | qa | **high 一到就返回**（不等意图/记忆）；mid/low 继续并带 top3 候选（§7.1.9） | — | 落空继续 |
| 6 | L2 语义缓存 | infra | answer | 60 ms | 未命中 |
| 7 | 检索+重排 | rag | chunks | 350 ms | 降级排序 |
| 8 | 证据门 | rag | pass/fail | 5 ms | 兜底转人工 |
| 9 | 生成+引用 | rag + memory(上下文) | 流式文本 | 剩余预算 | 切模型/兜底 |
| 10 | 回写 | memory + infra | 历史/缓存/指标 | 异步 | 不阻塞 |

**关键约束**：第 4 步三条并行分支必须**各自独立超时**（`asyncio.wait` + `timeout`），不能因为记忆召回慢而拖住 FAQ 命中——FAQ 本该 15 ms 返回。

第 4 步是**并发，不是屏障**：用 `asyncio.as_completed` 拿第一个有用结果，FAQ 高置信命中就立即返回并取消其余分支；意图/记忆只用于 FAQ 落空之后的路径。FAQ 是人工审核过的事实，不该被 60 ms 的分类器拖住，更不该被它的标签否决（§7.7 硬规则 5）。取消的分支不影响回写：历史与记忆写入走异步链路（第 10 步）。

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

### 7.7 理解层：意图分类与下游路由（PR5，待建）

**目标**：把 query 稳定归到 6 类之一，并把它翻译成**一个路由档位**；下游节点只读档位，不读意图字符串。

**边界**：理解层不检索、不生成、不写库；输入 `query + history`，输出 `IntentResult` + `SlotResult`（学科槽位规则版在 `qa_system/res/qa_slots.py`，见 §7.1.8）。理解层失败**不允许**影响 FAQ 命中——三者并行、各自超时（§7.4 第 4 步）。

`IntentResult` **没有 `subject` 字段**：学科过滤是槽位与检索路由的职责（§7.1.8），意图只允许收窄 `doc_type`。
分类器输出中文标签或内部 id 都可以，但**进编排层第一步就映射成 `IntentProfile`**，之后下游只读档位对象——`label` 只进日志、指标与打标，不参与任何分支。

**文件**：

```
understanding/
├─ intent.py            # ONNX 推理 + 规则快路径 + 置信度
├─ profiles.py          # label → IntentProfile 的**数据表**（不是 if/else）
├─ rewrite.py           # 会话改写（规则优先，小模型兜底）
└─ slots.py             # 模型版槽位（规则版在 qa_system/res/qa_slots.py）
```

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
| 题目讲解 | `kb_stepwise` | 是 | textbook, lesson_plan（题库接入后加 question_bank 并置首） | DIRECT；多小问 → SUB_QUERY | 必须 | 转人工 | 是 |
| 教案素材 | `doc_lookup` | 是 | lesson_plan, textbook | DIRECT | 必须 | 转人工 | 是 |
| 学习方法 | `general_advisory` | 是 | 不限（方法类证据散落，收窄会挡答案） | DIRECT | 有则标 | 通用建议 + 声明"非教材原文" | 是 |
| 备课辅助 | `generate_assist` | 是 | lesson_plan | DIRECT；含"设计 / 生成 / 出题" → SUB_QUERY | 参考来源必须标 | 允许生成，标注哪些是模型补充 | 是 |
| 闲聊 | `no_retrieval` | 否 | — | — | 不要求 | 直接短答（≤80 字） | 否 |
| unknown（<0.6 / 模型失败） | `generic_clarify` | 是 | 不限 | DIRECT | 有则标 | 通用回答 + 结尾追问 | 否 |

**意图对下游的 8 个作用点**（其余地方一律不看意图）：

| # | 下游 | 影响 | 落点 |
|---|---|---|---|
| 1 | 检索 | `retrieval=False` 跳过整条检索链（省 350 ms） | 编排层 |
| 2 | Milvus 过滤 | 只把 `doc_types` 加进 `expr`；`subject` **只由 `SlotResult` 给**，`tenant_id/acl_tags/kb_version` 由接入层给，意图一律不得覆写 | `Retriever.search(filters)` |
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
4. **意图不越权碰过滤条件**：只允许收窄 `doc_type`。学科由 `SlotResult` 决定（§7.1.8），租户 / ACL / `kb_version` 由接入层决定，三者都不看意图。
5. **FAQ 不等意图**：FAQ 高置信命中立即返回，不设"等意图分类完成"的屏障（`asyncio.as_completed` + 取消其余分支）。FAQ 是人工审核过的事实，不该被 60 ms 的分类器拖住，更不该被它否决。
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
| 记忆 / FAQ 慢 | 三者并行独立超时，意图慢不影响 FAQ 命中 |

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
