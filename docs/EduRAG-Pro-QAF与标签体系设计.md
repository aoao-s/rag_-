# EduRAG-Pro QAF 与标签（Tag）体系设计（v1.0）

> 术语约定：**QAF = Question–Answer–Feedback**，即"问答对 + 用户反馈"的闭环数据集（若你指的是别的含义，告诉我，我改）。
> 定位：QAF 是系统的**低成本快路径与质量燃料**；Tag 是贯穿**路由、检索、评测、权限、记忆**的横向元数据。
> 原则：不做独立复杂系统——QAF 与 Tag 都挂在现有 MySQL + Milvus + Redis 上，用最少的表解决 80% 的问题。

---

## 1. 整体架构

```
                    ┌─────────────────── QAF 生命周期 ───────────────────┐
  CSV 导入 ──┐      │ draft ─► pending ─► published ─► deprecated     │
  未命中挖掘 ─┼─► QAF│        ▲              │              │            │
  人工创建 ──┘      │        └── review ◄───┘  定时下线 ◄───┘            │
                    └───────┬──────────────────────────┬───────────────┘
                            │ 打标                      │ 版本/审计
                            ▼                          ▼
                    ┌───────────────┐          ┌──────────────┐
                    │ Tag 体系      │          │ qa_version   │
                    │ 9 个命名空间   │          └──────────────┘
                    └───────┬───────┘
        ┌───────────────────┼────────────────────┬──────────────────┐
        ▼                   ▼                    ▼                  ▼
   意图路由            检索过滤           评测切片/权限        记忆对齐
   (intent)        (subject/acl)        (scenario/level)   (subject)
```

三个消费面：**在线问答**（FAQ 快路径）、**运营**（审核/挖掘/反馈）、**质量**（评测切片、指标）。

---

## 2. QAF 数据模型

```sql
-- 问答对主表
CREATE TABLE qa_pair (
  id            BIGINT PRIMARY KEY AUTO_INCREMENT,
  tenant_id     VARCHAR(32) NOT NULL DEFAULT 't_01',
  question      VARCHAR(500) NOT NULL,
  question_norm VARCHAR(500) NOT NULL,        -- 归一化：去标点/全半角/同义替换，用于去重
  answer        TEXT NOT NULL,
  status        ENUM('draft','pending','published','deprecated') NOT NULL DEFAULT 'draft',
  source        ENUM('import','mined','manual','feedback') NOT NULL,
  subject_id    BIGINT,                        -- 冗余主学科标签，加速过滤
  hit_count     INT DEFAULT 0,                 -- 命中次数（热度）
  up_count      INT DEFAULT 0, down_count INT DEFAULT 0,
  quality_score DECIMAL(3,2) DEFAULT 0.70,     -- 反馈 + 人工评分合成
  created_by    VARCHAR(32), updated_by VARCHAR(32),
  created_at    DATETIME(3), updated_at DATETIME(3),
  UNIQUE KEY uk_norm (tenant_id, question_norm),
  KEY idx_status_subject (status, subject_id)
);

-- 版本（每次编辑留档，支持回滚）
CREATE TABLE qa_version (
  id BIGINT PRIMARY KEY AUTO_INCREMENT, qa_id BIGINT NOT NULL, version INT NOT NULL,
  question VARCHAR(500), answer TEXT, reason VARCHAR(200),
  changed_by VARCHAR(32), created_at DATETIME(3),
  UNIQUE KEY uk_qa_ver (qa_id, version)
);

-- 反馈（三元组：谁、对哪次请求、什么评价）
CREATE TABLE feedback (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  request_id VARCHAR(40) NOT NULL,             -- 关联 request_log
  qa_id BIGINT,                                 -- 命中的 FAQ（可空：RAG 答案）
  user_id VARCHAR(32), tenant_id VARCHAR(32),
  rating TINYINT NOT NULL,                      -- 1 赞 / -1 踩 / 0 未表态但有评论
  tags JSON,                                    -- ["答非所问","信息过时","太啰嗦"]
  comment VARCHAR(1000),
  created_at DATETIME(3),
  KEY idx_qa_time (qa_id, created_at), KEY idx_req (request_id)
);

-- 未命中池（问答没答好 → 未来的 FAQ）
CREATE TABLE miss_pool (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  query VARCHAR(500) NOT NULL, query_norm VARCHAR(500),
  intent VARCHAR(20), cluster_id BIGINT,        -- 聚类后的同义问题归一组
  freq INT DEFAULT 1, last_seen DATETIME,
  status ENUM('new','reviewing','converted','ignored') DEFAULT 'new',
  created_at DATETIME(3), UNIQUE KEY uk_norm (query_norm)
);
```

---

## 3. QAF 生命周期

```
draft ──提交──► pending ──审核──► published ──定时/人工──► deprecated
  ▲                 │                  │
  └────驳回─────────┘                  └──差评超阈值──► pending（进入修订）
```

| 状态 | 可见性 | 触发动作 |
|---|---|---|
| `draft` | 仅创建者 | 编辑中 |
| `pending` | 运营 | 审核队列（人工 30 秒/条） |
| `published` | **在线命中** | 写入 BM25 语料 + Milvus 向量（异步，5 分钟内生效） |
| `deprecated` | 不命中 | 保留历史，可复活 |

**发布即建索引**：`published` 的 QA 同时进两路——BM25 内存语料（热更新）与 Milvus `qa` collection（BGE-M3 编码），双路召回后 RRF 融合，再用"归一化分数 + top1/top2 间隔"判命中（替代老项目硬编码的 8.0 阈值）。

---

## 4. 三个来源

| 来源 | 技术 | 频率 |
|---|---|---|
| CSV 批量导入 | pandas 读取 → 清洗 → `question_norm` 去重 → 直接 `published`（可配 `pending`） | 冷启动一次 |
| 未命中挖掘 | 每周跑批：`miss_pool` 按 embedding 聚类（cosine ≥ 0.90）→ 组内 freq 降序取 Top20 → `deepseek-flash` 生成候选答案 → `pending` | 每周 |
| 反馈驱动修订 | `down_count ≥ 3` 且 `up/(up+down) < 0.6` → 生成修订任务 → 运营处理 | 每日 |
| 人工创建 | 运营后台表单 + 标签选择 | 随时 |

---

## 5. 在线检索链路（FAQ 快路径）

```
query → 归一化 → ① Redis 精确缓存 → ② 并行：BM25(top20) ∥ 向量(top20) + 标签过滤
      → ③ RRF 融合 → ④ 命中判定 → ⑤ 命中：返回 answer + qa_id + 标签来源 / 未命中：进 RAG
```

| 环节 | 技术 | 关键点 |
|---|---|---|
| 归一化 | 自研规则 + `question_norm` | 去标点、全半角、同义词（"学费/费用/多少钱"） |
| BM25 | `rank_bm25`（内存语料，发布时热更新） | 中文先用 jieba 分词；阈值改为自适应 |
| 向量 | Milvus `qa` collection（BGE-M3，复用知识库同一模型） | `subject` 标量过滤前置 |
| 融合 | RRF（k=60） | 不做分数加权（两路分数量纲不同） |
| 命中判定 | `norm_score ≥ 0.62` **且** `top1 - top2 ≥ 0.05` **且** 学科一致 | 三条件避免"擦边命中"（老项目 7.16 vs 8.0 的教训） |
| 兜底 | 未命中 → RAG（携带 `top3` 候选作为提示） | 不丢弃候选，帮助 RAG 理解用户意图 |

### 5.1 两条链路的分工：BM25 只服务 FAQ，知识库走 dense + sparse

不要把 BM25 和 Milvus 稀疏向量当成二选一的"同一个东西"，它们在不同链路上：

| 链路 | 双路构成 | 为什么 |
|---|---|---|
| **知识库**（文档切块） | BGE-M3 **稠密向量 + 稀疏向量**（Milvus `hybrid_search`）+ Reranker | 稀疏向量同一次编码产出，与稠密同源、支持增量、免维护分词语料；Milvus 原生加权（0.7/0.3）或 RRF |
| **FAQ**（问答对） | **BM25 + 稠密向量**（量小时可只用 BM25） | 语料只有几千到几万条，BM25 毫秒级、零 GPU、可解释；短问句（"多少钱""包住宿吗"）关键词命中比向量更稳 |

判据：**语料是否需要"词面精确"且规模小 → BM25；规模大、增删频繁、要语义泛化 → 稀疏向量**。FAQ 同时具备"小 + 词面精确"，故 BM25 更合适；知识库正好相反。

代价说清：如果强行把 BM25 塞进知识库链路，就要额外维护分词语料并随文档增删重建，收益只有一个"关键词召回"，而这个能力稀疏向量已经覆盖——所以知识库不用 BM25。

### 5.2 门控与意图分类都不用大模型

- **意图识别**：升级 `bert-intent-ft`（现在只有"通用知识/专业咨询"两类）为 5 类 + 置信度（temperature scaling / margin），批量化 ONNX 推理，单条 < 5 ms。冷启动标注数据用 LLM 打标 + 人工抽检（蒸馏），不要直接拿 LLM 上线跑在线分类。
- **门控**（记忆模块）是规则打分，见记忆设计 §5.1；两者都把 LLM 留给"抽取/摘要/生成"这类真正需要的环节。

---

## 6. 反馈闭环

```
用户点踩 + 选标签 → feedback 落库 → 日聚合（feedback_agg 视图）→ 触发规则 → 任务
                                                    │
              ┌─────────────────────────────────────┼──────────────────────────┐
              ▼                                     ▼                          ▼
      QA 修订（差评率高）                    未命中池（无答案）           文档缺口（RAG 引用为空）
      运营改答案 → 新 version            聚类 → 候选 QA              反查知识库 → 补文档
```

- 反馈标签固定枚举（便于统计）：`答非所问 / 信息过时 / 不完整 / 太啰嗦 / 有错别字 / 引用不对 / 其它`。
- **反馈必须能定位到 `request_id`**，否则无法复盘（老项目缺这一层，导致"用户说不好用"无法归因）。

---

## 7. Tag 体系

### 7.1 九个命名空间

| 命名空间 | 取值示例 | 谁用 | 存在形式 |
|---|---|---|---|
| `intent` | 课程咨询 / 考试辅导 / 学习方法 / 升学规划 / 技能培训 / other | 路由、评测 | 枚举（与意图分类共用） |
| `subject` | AI 大模型 / Java / 大数据 / 前端 / 测试 / 运维 | 检索过滤、记忆对齐 | 树（学科 → 方向） |
| `course` | AI 大模型 7.5 / Java 就业班 / 大数据 6.0 | 检索、答案生成 | 树（学科下挂课程） |
| `level` | 零基础 / 有基础 / 进阶 | 检索、评测切片 | 枚举 |
| `scenario` | 就业 / 转行 / 提升 / 兴趣 / 考证 | 路由、推荐 | 枚举 |
| `audience` | 学员 / 家长 / 企业客户 | 权限、话术 | 枚举 |
| `doc_type` | faq / 大纲 / 课件 / 价目表 / 合同 / 政策 | 检索、引用展示 | 枚举 |
| `risk` | 价格 / 承诺 / 就业率 / 退款 | 输出过滤、合规审查 | 枚举（命中即走校验） |
| `acl` | public / internal / vip | 权限过滤（Milvus 标量） | 枚举 |

### 7.2 表结构（邻接表 + 同义词，够用且简单）

```sql
CREATE TABLE tag (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  namespace VARCHAR(20) NOT NULL,
  name VARCHAR(64) NOT NULL,
  parent_id BIGINT DEFAULT NULL,
  weight DECIMAL(3,2) DEFAULT 1.00,     -- 用于检索加权（如"就业"比"兴趣"权重高）
  status ENUM('active','deprecated') DEFAULT 'active',
  UNIQUE KEY uk_ns_name (namespace, name), KEY idx_parent (parent_id)
);

CREATE TABLE tag_synonym (
  id BIGINT PRIMARY KEY AUTO_INCREMENT, tag_id BIGINT NOT NULL,
  alias VARCHAR(64) NOT NULL, UNIQUE KEY uk_alias (alias)
);

CREATE TABLE qa_tag (
  qa_id BIGINT NOT NULL, tag_id BIGINT NOT NULL,
  source ENUM('rule','llm','human') NOT NULL, confidence DECIMAL(3,2) DEFAULT 1.00,
  PRIMARY KEY (qa_id, tag_id)
);

CREATE TABLE doc_tag (          -- 知识库文档复用同一套标签
  doc_id VARCHAR(64) NOT NULL, tag_id BIGINT NOT NULL,
  source ENUM('rule','llm','human'), confidence DECIMAL(3,2),
  PRIMARY KEY (doc_id, tag_id)
);
```

### 7.3 打标流水线（每部分用什么技术）

| 步 | 技术 | 说明 |
|---|---|---|
| ① 词典匹配 | `pyahocorasick`（AC 自动机）+ `tag_synonym` 全量加载 | 毫秒级，覆盖学科/课程/城市等确定性标签 |
| ② 规则匹配 | 正则 + 句式（"多少钱"→`risk:价格`；"包就业"→`risk:承诺`） | 可解释、零成本 |
| ③ 模型打标 | `deepseek-flash` 零样本 + JSON Schema 强约束（只允许多选题库里的标签） | 覆盖语义模糊场景，置信度 <0.6 转人工 |
| ④ 冲突合并 | 同命名空间内按 `source` 优先级（human > rule > llm）+ 置信度加权 | 避免一个 QA 挂 5 个学科 |
| ⑤ 人工复核 | 运营后台"待确认标签"队列 | 只复核 llm 来源且置信度低的 |
| ⑥ 落库与缓存 | MySQL 明细 + Redis 热点（`tag:tree:{ns}` 全量树，TTL 1h） | 检索时读缓存，不打库 |

### 7.4 标签的四个消费点

| 消费点 | 用法 |
|---|---|
| 意图路由 | `intent` 标签与 BERT 分类结果互相校验；不一致时降置信、走澄清 |
| 检索过滤 | Milvus `subject`/`doc_type`/`acl` 标量过滤前置（**先过滤后检索**） |
| 评测切片 | 黄金集按 `scenario × level` 切片看 Recall，定位"哪类问题答得差" |
| 记忆对齐 | 原子记忆的 `subject` 必须来自 `subject` 命名空间，保证同一套词汇 |

---

## 8. 每部分技术栈总览

| 部分 | 技术选型 | 备注 |
|---|---|---|
| QAF 存储 | MySQL 8.0（qa_pair/qa_version/feedback/miss_pool） | 事务 + 版本留档 |
| QAF 检索 | `rank_bm25` + Milvus（`qa` collection, BGE-M3）+ RRF | 复用现有模型，零新增 |
| QAF 发布 | 异步任务：写 MySQL → 更新 BM25 内存语料 → 写 Milvus | 5 分钟内生效 |
| 反馈收集 | FastAPI `POST /v1/feedback` + 前端按钮 | 必带 `request_id` |
| 反馈聚合 | MySQL 视图 + 每日定时任务（APScheduler） | 触发修订任务 |
| 挖掘聚类 | BGE-M3 向量 + 余弦 ≥0.90 单链聚类 | 每周跑批，产出 Top20 |
| 候选生成 | `deepseek-flash`（结构化输出） | 人工审核后发布 |
| Tag 存储 | MySQL 邻接表 + Redis 树缓存 | 不引入图数据库 |
| Tag 打标 | `pyahocorasick` + 正则 + `deepseek-flash` | 规则优先，模型兜底 |
| Tag 检索 | Milvus 标量过滤 + MySQL 子树查询（`parent_id` 递归一次） | 过滤前置 |
| 审核后台 | 现有 Web（Vue3）+ 权限 `role=operator` | 不单独建系统 |

---

## 9. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/v1/faq/search?q=&subject=` | 在线 FAQ 检索（调试用，线上走编排内调用） |
| `POST` | `/v1/feedback` | `{request_id, qa_id?, rating, tags[], comment}` |
| `GET/POST/PATCH` | `/v1/admin/qa` | QA 列表 / 创建 / 编辑（含状态流转） |
| `POST` | `/v1/admin/qa/{id}/publish` | 发布并建索引 |
| `POST` | `/v1/admin/qa/import` | CSV 批量导入 |
| `GET` | `/v1/admin/miss-pool?status=new` | 未命中池 |
| `POST` | `/v1/admin/miss-pool/{id}/convert` | 转成候选 QA |
| `GET` | `/v1/admin/tags?namespace=` | 标签树 |
| `POST` | `/v1/admin/tags/review` | 人工复核标签 |

---

## 10. 指标

| 指标 | 目标 | 说明 |
|---|---|---|
| FAQ 命中率 | 35–50% | 太低=知识库缺口；太高=可能在答错（配合采纳率看） |
| 答案采纳率（未点踩） | ≥ 92% | 反馈口径 |
| 反馈率 | ≥ 3% | 低于 1% 说明按钮没引导好 |
| 差评响应时效 | ≤ 24 h | 从差评到修订发布 |
| 未命中池转化率 | ≥ 20%/月 | 挖掘是否有效 |
| 标签覆盖率（published QA 至少 1 个 subject） | 100% | 缺标签的 QA 不发布 |
| 标签准确率（抽检 100 条） | ≥ 95% | 人工复核 |
| FAQ 检索 p95 | ≤ 60 ms | 双路 + RRF |

---

## 11. 复杂度控制（适当减法）

| 不做 | 原因 |
|---|---|
| 独立 FAQ 服务/独立向量库 | 与知识库共用 Milvus，独立 collection 即可 |
| 标签图谱/本体推理 | 邻接表 + 同义词已覆盖教育场景 |
| 自动答案生成直接发布 | 必须人工审核，避免错误答案进快路径（快路径错一次影响面最大） |
| 多级反馈（星级+维度打分） | 赞/踩 + 标签 + 自由文本足够归因 |
| 实时聚类 | 每周跑批足够，实时聚类成本高且噪声大 |

**保留的三件事**：`qa_version`（可回滚）、`request_id` 关联（可归因）、发布前必须打 `subject` 标签（保证检索过滤可用）。
