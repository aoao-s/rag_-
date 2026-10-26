# EduRAG-Pro 分阶段技术方案

> 版本：v1.0（2026-10-22）　|　适用仓库：`F:\heima\RAG\RAGpro_智能教育系统`
> 本文回答三个问题：**每个阶段用什么技术、实现了什么功能、怎么验收**；同时把 GPT 的 7 条意见与本仓真实代码状态逐条对齐。
> 阅读顺序建议：先看 §1（全局）与 §2（真实完成度），再按阶段读 §4；§5 是与 GPT 意见的对照，§8 是仍需你拍板的决策。

---

## 0. 结论先行

| 判断 | 结论 | 依据 |
|---|---|---|
| 技术路线 | **可行** | FastAPI + Milvus(dense+sparse) + BGE-M3 + cross-encoder + MySQL/Redis 的组合已被本仓代码验证到"检索出证据"这一层 |
| 最大风险 | **不是选型，而是"设计完整、闭环未通"** | `main.py` 只有 `/v1/faq/*`、`/v1/rag/*`、`/healthz`、`/readyz`、`/metrics`；没有 `/v1/chat`，没有生成、没有引用、没有记忆、没有编排 |
| 第二风险 | **证据门阈值与分数尺度混用** | 四判据 0.74/0.01/0.10/0.20 是在 `rerank=False` 的融合分上标定的，开重排后整组数字失效 |
| 第三风险 | **语料覆盖不足被误读成阈值问题** | 763 条可判定样本里，可答题只有约 20% 能召回正确文档；放行率 34–38% 的主因是"库里没有"，不是"门太严" |
| 本方案的骨架 | S0 口径 → S1 摄入与基线 → S2 规则路由 + 最小 E2E → S3 检索与证据 → S4 生成与引用 → S5 编排与记忆 → S6 运营交付 | 每个阶段都有**可运行的出口**和**可测量的验收**，不再出现"连续四个阶段测不了答对率"的真空 |

一句话概括：**先把"能跑通、能测量"的闭环做出来，再把每个环节做深；架构图上不新增模块，只把已有模块接上电。**

---

## 1. 全局架构与一次请求的时序

### 1.1 三层 + 三模块

![图1 目标架构](diagrams/06_edurag_pro_architecture.png)

```mermaid
flowchart TB
    subgraph L1["接入层 FastAPI"]
        A1["鉴权 / 限流 / request_id"]
        A2["REST /v1/chat + WS /v1/ws/chat"]
    end
    subgraph L2["编排层 Orchestration（状态机）"]
        O1["归一化 + 会话改写"]
        O2["FAQ ∥ 意图 ∥ 记忆召回"]
        O3["策略选择 / 预算扣减 / 降级"]
        O4["证据门 → 生成 → 引用 → 回写"]
    end
    subgraph L3["业务模块（平级，互不 import）"]
        Q["qa_system<br/>BM25 + 别名 + 二级缓存"]
        R["rag_system<br/>混合检索 + 重排 + 证据门"]
        M["memory<br/>L0 逐字 / L1 摘要"]
    end
    subgraph L4["基础设施"]
        DB[("MySQL 8.0")]
        RD[("Redis 7+")]
        MV[("Milvus 2.4")]
        MO["BGE-M3 / Reranker / LLM 网关"]
    end
    L1 --> L2
    L2 --> Q & R & M
    Q & R & M --> L4
```

三条硬边界（来自 `CLAUDE.md` §7.0，本方案不改）：

1. `qa_system` 不调用 LLM；`rag_system` 不读 FAQ 表；`memory` 不作为事实性回答的引用来源。
2. 三个模块都不自己建 Milvus/MySQL 连接，只拿 `base/` 注入的句柄。
3. 编排层只依赖 `base/contracts.py` 的数据类与三模块的公开方法。

### 1.2 一次请求的时序（含预算）

```mermaid
sequenceDiagram
    autonumber
    participant C as 客户端
    participant API as 接入层
    participant ORC as 编排层
    participant FAQ as qa_system
    participant RAG as rag_system
    participant MEM as memory
    participant LLM as LLM 网关

    C->>API: POST /v1/chat（或 WS 帧）
    API->>API: 鉴权/限流/生成 request_id（20ms）
    API->>ORC: ChatState
    ORC->>ORC: 归一化 + 会话改写（80ms）
    ORC->>FAQ: 并行：FAQ 检索（60ms）
    ORC->>MEM: 并行：记忆召回（120ms，超时降级）
    FAQ-->>ORC: FAQHit（命中即返回）
    alt FAQ 命中
        ORC-->>C: 审核答案 + faq_id 来源
    else FAQ 未命中
        ORC->>RAG: 混合检索 + 重排 + 证据门（350ms）
        RAG-->>ORC: Evidence + Chunks
        alt 证据门通过
            ORC->>LLM: 流式生成（剩余预算）
            LLM-->>C: meta / delta / citations / done
        else 证据不足
            ORC-->>C: 兜底话术 + 转人工（不调 LLM）
        end
    end
    ORC->>ORC: 异步写历史 / 缓存 / 指标
```

预算表（`CLAUDE.md` §6.1）不是装饰：每个节点进入时 `ctx.check_budget(min_left_ms=...)`，不够就抛 `BudgetExceeded` 走降级，**不允许"最后超时了再兜底"**。

| 阶段 | 预算 | 降级动作 |
|---|---|---|
| 鉴权 + 限流 | 20 ms | 429 / 401 |
| 归一化 + 会话改写 | 80 ms | 跳过改写，用原 query |
| L1 精确缓存 | 10 ms | 视为未命中 |
| FAQ ∥ 意图 ∥ 记忆 | 120 ms | 各自独立超时，失败即空 |
| 策略规划 | 300 ms | 直接 DIRECT |
| 检索（含多路） | 200 ms | 减少路数 |
| 重排 | 150 ms | 用融合分排序 |
| 首 token | 600 ms | 切备用模型 |
| 生成 | 剩余 | 截断并提示"回答不完整" |

### 1.3 统一状态对象

节点之间只通过 `ChatState` 通信，任何模块不得直接读另一个模块的返回值。

```python
# 目标位置：base/contracts.py（新增 TypedDict，现有数据类不动）
class ChatState(TypedDict, total=False):
    request_id: str; tenant_id: str; user_id: str; session_id: str
    original_query: str; rewritten_query: str
    intent: str; slots: dict[str, Any]
    faq_hit: bool; faq_answer: str | None
    chunks: list[Chunk]; evidence: Evidence
    answer: str; citations: list[Citation]
    degrade: list[str]          # 本请求触发了哪些降级，进日志与指标
    error: str | None
```

`degrade` 字段是刻意加的：排障时"为什么这次答得差"必须能一眼看出是重排降级、记忆超时还是 LLM 切模型。

---

## 2. 真实完成度盘点（先对齐事实，再谈阶段）

> 这一节是本方案的地基。GPT 的意见基于 `CLAUDE.md` 与架构图，本表基于**代码实际存在的东西**。

| 能力 | 状态 | 代码位置 | 缺口 |
|---|---|---|---|
| FAQ 检索（BM25 + 别名 + 三条件） | ✅ 已实现并标定 | `qa_system/res/qa_search.py`、`qa_corpus.py`、`qa_matcher.py`、`qa_normalize.py` | 无语义缓存；热更新是全量重建 |
| FAQ 缓存（正/负/续命/单飞） | ✅ 已实现 | `qa_system/res/qa_cache.py` | 负缓存候选未接 RAG 扩展 |
| FAQ HTTP 接口 | ✅ 6 个 | `qa_system/api.py` | — |
| 文档摄入（loader/parse/split/embed/入库） | ✅ 已实现 | `rag_system/ingest.py`、`document_loaders/*`、`text_splitters/parent_child.py`、`embedding/bge_m3.py`、`store/milvus_store.py` | 无索引版本化与 alias 切换；编码成本无预案 |
| 混合检索 + 父块合并 | ✅ 已实现 | `rag_system/core/retriever.py` | 无查询扩展；`doc_types` 过滤未接意图 |
| 重排 | ✅ 代码在，⚠️ 性能未达标 | `rag_system/core/reranker.py` | CPU 40 候选约 4.5 s；GPU 未验证 |
| 证据门 | ⚠️ 可用但尺度混用 | `rag_system/core/evidence_gate.py` | 四判据在融合分下标定；`evaluate()` 默认 0.35/0.03 与配置 0.74/0.01 不一致 |
| RAG HTTP 接口 | ✅ 2 个 | `rag_system/api.py` | 只回证据，不生成 |
| 意图分类 | ❌ 未接线 | 无 `understanding/` | 唯一权重是旧项目 2 分类 `bert-intent-ft`，与 6 类 `INTENT_LABELS` 不匹配 |
| 槽位抽取 | ❌ 未实现 | — | 无规则实现、无标注 |
| 生成 / 流式 / 引用对齐 | ❌ 未实现 | 无 `llm_gateway.py`、无 `/v1/chat` | 四类帧协议只在文档里 |
| 编排层 | ❌ 未实现 | `main.py` 只有路由挂载 | 无 LangGraph、无状态机、无预算 |
| 记忆 | ❌ 未实现 | `memory/__init__.py` 空 | 四层设计全在文档 |
| 索引版本化 / 灰度 / 回滚 | ❌ 未实现 | — | 单集合写死 |
| GPU 基线 / 压测 | ❌ 未做 | — | 本机 `torch 2.5.1+cpu`、`cuda False` |

**这张表决定了阶段顺序**：FAQ 与摄入是资产，检索是半成品，生成/编排/记忆是空地。所以顺序必须是"补资产的测量能力 → 补闭环 → 再补深度"，而不是按架构图的模块顺序平推。

```mermaid
flowchart LR
    subgraph 现有资产["现有资产（可复用）"]
        F["FAQ 全链路<br/>p95 9.1ms"]
        I["摄入管线<br/>14 文档→110 块"]
        RT["混合检索<br/>父块回填"]
    end
    subgraph 半成品["半成品（需返工）"]
        G["证据门<br/>尺度混用"]
        RR["重排<br/>CPU 4.5s"]
    end
    subgraph 空地["空地（需新建）"]
        U["理解/意图"]
        GEN["生成/引用"]
        ORC["编排"]
        MEM["记忆"]
        OPS["运营/交付"]
    end
    现有资产 --> 半成品 --> 空地
```

---

## 3. 阶段划分总览

### 3.1 七个阶段与出口

```mermaid
flowchart LR
    S0["S0 口径与评估设计<br/>指标字典 + 100 题标注"] --> S1["S1 摄入与可测基线<br/>索引 v1 + Recall/MRR 基线"]
    S1 --> S2["S2 规则路由 + 最小 E2E<br/>FAQ 直返 / 未命中给证据卡片"]
    S2 --> S3["S3 检索与证据<br/>重排决策 + 四判据重标定"]
    S3 --> S4["S4 生成与引用<br/>流式 + 四类帧 + 引用对齐"]
    S4 --> S5["S5 编排与记忆<br/>/v1/chat + L0/L1"]
    S5 --> S6["S6 运营与交付<br/>指标/压测/灰度/回滚"]
```

| 阶段 | 一句话目标 | 主技术 | 出口（可运行/可测量） | 预估 |
|---|---|---|---|---|
| S0 | 把"好"定义清楚 | 标注规范、指标字典、语料盘点 | 指标字典 + 100 题 v0 标注集 + 缺口清单 | 2–3 天 |
| S1 | 让检索第一次有可信基线 | loader/parser/父子块/BGE-M3/Milvus HNSW+SPARSE/alias | 索引 `edurag_v1` + Recall@10/MRR 基线报告 | 5–7 天 |
| S2 | 让系统第一次端到端可答 | 归一化/BM25/Redis 缓存/规则二分器/模板回答 | `/v1/answer` 可跑；路由准确率、答对率首次可测 | 4–6 天 |
| S3 | 让"检索到"变成"够回答" | WeightedRanker/重排/证据门重标定/aspect coverage/冲突 shadow | 可答放行率↑、误放行 ≤5%、引用可定位 | 6–9 天 |
| S4 | 让答案可流式、可溯源、不编造 | LLM 网关/四段 prompt/引用对齐/忠实度抽检 | WS 四类帧 + 引用支持率报告 | 5–7 天 |
| S5 | 让所有能力走一个入口 | 显式状态机（LangGraph 待定）/预算/并发/记忆 L0/L1 | `/v1/chat` 单入口 + 多轮可用 | 7–10 天 |
| S6 | 让系统可交付、可回滚 | 指标/日志/压测/Docker/灰度/GPU 基线 | 验收记录 + 压测报告 + 回滚演练 | 5–8 天 |

> 时间是"一个人、有 GPU 可用"的粗估，不含语料补齐的等待时间。

### 3.2 为什么顺序是这样（三条来自审查的修正）

1. **基线不能冻结在 S0，必须并入 S1 出口。** 黄金集现在测出的"20% 正确文档召回"是残缺管线的产物；S1 补语料、建索引之后基线必然漂移。所以 S0 只交付**定义**（指标字典、标注规范），S1 出口才交付**数值**。
2. **最小端到端必须前移到 S2，不能等到 S5。** 否则 S0–S4 五个阶段都只能用代理指标（Recall/MRR/放行率），"答对率"与"引用是否支持"永远测不到，配置不一致的问题也暴露不出来。
3. **意图不能拖到 S5，但也不能在 S2 承诺 6 分类。** S2 只做**规则二分器**（FAQ 分数 + 关键词规则决定走 FAQ 还是 RAG），6 类意图与槽位降级为"日志埋点 + 数据积累"，等训练数据与标签口径拍板后单开子阶段。

---

## 4. 逐阶段详解

每个阶段统一六段：**解决什么问题 → 用什么技术 → 怎么工作 → 落到哪个文件 → 怎么验收 → 常见坑**。

---

### S0 口径与评估设计

**解决什么问题**：现在"好不好"没有共同语言——Recall、放行率、忠实度各说各话，且阈值是在不同分数尺度上拍的。先把定义钉死，后面每个阶段才有验收基准。

**用什么技术**
- 标注规范（JSONL schema）：`query / rewritten / intent / slots / gold_doc_ids / gold_parent_ids / answerable / must_have_facts`。
- 指标字典：Recall@10、MRR、正确文档命中率、可答放行率、不可答误放行率、引用支持率、忠实度、首 token p95、端到端 p95。
- 失败归因枚举：`retrieval_miss / knowledge_gap / parse_loss / split_loss / rerank_loss / gate_false_reject / gate_false_pass / generation_unsupported`。
- 语料盘点脚本：对比黄金集 `gold_doc_ids` 与 `doc_meta`，直接算出"库里到底缺哪些文档"。

**怎么工作**
1. 从 `datasets/07_eval/ragas_eval.jsonl`（1000 条）中抽 100 条做人工核验，补齐 `gold_parent_ids` 与 `answerable`。
2. 逐条标注 `must_have_facts`（如"三个阶段""每阶段 3 个月"），这是后面 aspect coverage 的输入。
3. 语料盘点：输出"缺失文档清单"，交给机构补 PDF，而不是让模型硬答。
4. 指标字典写进 `docs/`，后续报告只引用这里定义的口径。

**落到哪个文件**
- 新增 `eval/schema.md`、`eval/golden_v0.jsonl`、`scripts/audit_corpus.py`。
- 指标定义并入 `docs/EduRAG-Pro-1.41-企业级架构设计.md` 的评估章节。

**怎么验收**
- 100 题 v0 集人工复核一致率 ≥ 95%（两人标同一批，看分歧）。
- 语料缺口清单可执行：每条缺口能指到"哪份文档、哪个年级/学科"。
- 指标字典里每个指标都有：定义、计算公式、数据来源、目标值、告警线。

**常见坑**
- 把"拒答率低"当目标。教育知识库正确拒答比错误作答有价值，优化目标应是"可答放行率↑ 且 误放行率≤5%"。
- 黄金集与知识库同源（用教材出题、又只喂这本教材），指标虚高。要混入真实学员问法。

---

### S1 摄入与可测基线

**解决什么问题**：把"文档 → 可检索证据"的链路做成可重复、可测量、可回滚的工程，并第一次给出可信的 Recall/MRR 基线。

**用什么技术**
- 加载分发：`thy_factory.py` 按后缀选 loader（`.md/.txt/.srt/.docx/.pptx/.pdf/.jpg/...`）。
- 解析：PDF 双解析器（MinerU 云 + PyMuPDF 本地兜底），扫描页/大图走 OCR（`rapidocr_paddle` → `rapidocr_onnxruntime`）。
- 清洗：去页眉页脚、断行合并、front-matter 剥离。
- 切分：父块 1200/重叠 100，子块 300/重叠 20，父块按标题树聚合，子块继承标题并携带 `parent_text`。
- 向量化：BGE-M3 一次出 dense(1024) + sparse；批量 ≥8；无权重时退 `HashEmbedder`（只用于冒烟，不进交付）。
- 入库：Milvus `edurag_v{N}`，dense HNSW（M=16, efConstruction=200, ef=64）+ sparse inverted（drop_ratio_build=0.2）；`delete_by_source` 先删后写保证幂等。
- 版本化：`edurag_v{N+1}` 建好后校验，再切 alias `edurag_current`；`KB_VERSION` 变更让缓存自然失效。

```mermaid
flowchart LR
    A["原始文档"] --> B["loader 分发"]
    B --> C["parser<br/>MinerU / PyMuPDF / OCR"]
    C --> D["cleaner<br/>页眉页脚 / 断行"]
    D --> E["父子块切分<br/>1200 / 300"]
    E --> F["BGE-M3<br/>dense + sparse"]
    F --> G["Milvus edurag_vN"]
    G --> H["质检<br/>长度/重复/乱码"]
    H -->|通过| I["alias 切换"]
    H -->|不通过| J["拒绝切 alias"]
```

**怎么工作**
1. `python scripts/ingest.py --src datasets --kb-version v1`，落 `data/ingest_manifest.json`（每文件：块数、耗时、失败原因）。
2. 质检四项：块长分布、块间 Jaccard>0.9 报警、乱码率、表格/代码块断裂率；不达标不切 alias。
3. **编码成本预案**（蓝队点名）：先测"单块编码耗时 × 块数"的线性基线；全量重建超过 30 min 就上三件事之一——ONNX INT8（约 2–3×）、只对变更文档重编码（`content_hash` 未变跳过）、夜间批处理 + 双索引。
4. 基线报告：`Recall@10`、`MRR`、`正确文档命中率`，并按"有/无正确文档"切片，明确区分 `retrieval_miss` 与 `knowledge_gap`。

**落到哪个文件**
- 已有：`rag_system/ingest.py`、`document_loaders/*`、`text_splitters/parent_child.py`、`embedding/bge_m3.py`、`store/milvus_store.py`。
- 新增：`rag_system/store/index_version.py`（建/校验/切/回滚）、`scripts/eval_retrieval.py`（只测检索，不测生成）。

**怎么验收**
- 同一批文档重复摄入不产生重复块（按 `doc_id` 计数稳定）。
- 基线报告可复跑：同参数两次 Recall 差异 < 0.5 个百分点。
- 索引切换/回滚 ≤ 30 s，切换期间查询不 5xx。
- 单文档更新 5 min 内可查。

**常见坑**
- 只做 upsert 不做 delete：文档改版后旧块残留，检索命中过期内容。
- 切分把标题与正文拆散：子块失去章节上下文，重排分数虚高。
- 编码在请求路径上做：第一次查询 10 s+。编码只能离线。

---

### S2 规则路由 + 最小端到端

**解决什么问题**：让系统第一次"从输入到答案"跑通，并且让"答对率"从这一阶段起就能持续测量；同时用规则解决 FAQ/RAG 的路由，不依赖还不存在的意图模型。

**用什么技术**
- 归一化：全半角/大小写/标点/语气词/数字统一/口语同义/别名整串映射（`qa_normalize.py`）。
- FAQ：分学科 BM25 + 三条件（`score_norm≥0.25`、`top1−top2≥0.10`、学科不冲突）+ 二级缓存（正 1 h/绝对 6 h、负 30 s 带 top3、剩余寿命 ≤50% 才续命）。
- 路由二分器（新增，纯规则）：
  - 强信号：FAQ 命中（分数 + margin 同时过）→ FAQ 直返。
  - 弱信号：FAQ top1 在 0.15–0.25 且 margin < 0.10 → 带 top3 候选进 RAG 做查询扩展。
  - 其他 → 直接 RAG。
- 最小 E2E：FAQ 命中返回审核答案；未命中时返回"证据卡片 + 模板化回答"（把 top 父块按固定句式拼出来，不调 LLM），保证链路可演示、可测答对率。
- 意图埋点：只记录 query 特征与路由结果，不训练、不承诺分类准确率。

**怎么工作**
1. 请求进来 → 归一化 → FAQ → 命中直返；未命中 → 检索 top-k → 证据卡片。
2. 模板回答的句式固定（"根据《文件》第 n 页：……"），引用 ID 与父块一一对应。
3. 每次请求写 `route_reason`（`faq_hit / faq_weak / no_faq`），供 S5 换真分类器时对比。

**落到哪个文件**
- 新增 `orchestration/router.py`（规则二分器）、`orchestration/templates.py`（模板回答）。
- 复用 `qa_system/res/*`、`rag_system/core/retriever.py`。
- 新增 `main.py` 的 `POST /v1/answer`（注意：这不是最终 `/v1/chat`，只是过渡出口）。

**怎么验收**
- 100 题 v0 集上：路由准确率 ≥ 90%（FAQ 该走的走 FAQ，该进 RAG 的进 RAG）。
- FAQ 命中路径 p95 ≤ 15 ms；未命中路径（不含重排）p95 ≤ 300 ms。
- 模板回答的引用可定位率 100%（点开能到文件页码）。
- 答对率（人工判"是否回答了问题"）首次有数，作为 S3 的对照基线。

**常见坑**
- 用 FAQ 阈值当"答案质量"阈值：FAQ 只判断"是不是同一个标准问题"，不判断"答案对不对"。
- 模板回答直接上线给学员：它不会组织语言，只用于验证链路与测量，S4 必须替换。

---

### S3 检索与证据（本方案的技术核心）

**解决什么问题**：把"检索到相关内容"变成"证据足以回答"，并且让证据门的每个分数都有明确来源、可分别标定。

#### 4.3.1 检索链路

```mermaid
flowchart LR
    Q["query"] --> N["归一化"]
    N --> D["dense ANN k=40"]
    N --> S["sparse ANN k=40"]
    N --> X["FAQ top3 候选<br/>查询扩展"]
    D --> W["WeightedRanker 0.7/0.3"]
    S --> W
    X --> W
    W --> P["按 parent_id 去重<br/>回填 parent_content"]
    P --> T["截断 512 token"]
    T --> RR["cross-encoder 重排 top8"]
    RR --> G["证据门"]
    G -->|通过| OUT["4–6 父块 → 生成"]
    G -->|不通过| FB["兜底 / 澄清 / 转人工"]
```

#### 4.3.2 分数尺度必须分离（GPT 问题 2）

现状是把 `retrieval_score`（WeightedRanker 融合分）、`rerank_score`（sigmoid）和 `qcov`（词面覆盖）混在一个 `threshold=0.74` 下比较。改造为显式输入契约：

```python
# rag_system/core/evidence_gate.py
@dataclass(frozen=True)
class GateInput:
    retrieval_score: float        # WeightedRanker 融合分，0–1
    rerank_score: float | None    # cross-encoder sigmoid，0–1；None = 未重排
    qcov: float                   # 问题实词覆盖率，0–1
    n_docs: int                   # 覆盖的文档数
    aspects: list[Aspect]         # 问题拆出的语义方面
    evidence: list[Evidence]      # 父块级证据
```

规则：
- `mode = "rerank" if rerank_score is not None else "fusion"`。
- 两套阈值分别配置：`RAG_GATE_FUSION_*` 与 `RAG_GATE_RERANK_*`，**禁止跨模式复用**。
- 重排失败降级到融合分时，阈值也一起切回 fusion 那套，并在 `degrade` 里记 `rerank_fallback`。

#### 4.3.3 四判据的处置

| 判据 | 现状 | 处置 | 理由 |
|---|---|---|---|
| 绝对分 `threshold` | 0.74（融合分） | 保留，但按 mode 各标一套 | 绝对分仍能挡"整体不相关" |
| 跨文档 `margin` | 0.01 | **废弃**（默认 0.0） | 互补证据会被误判成歧义；标定扫描的最优点本来就是 0.0 |
| 孤证 `+0.10` | 0.10 | 保留 | 单块证据确实更易错 |
| `qcov` | 0.20 | 保留，但只作辅助 | 它是词面覆盖，挡不住"主题对、知识点错" |

#### 4.3.4 用 aspect coverage 替代 margin（用户 m00176 的批评 + 我们的修正）

关键区分：**互补 ≠ 矛盾，重复 ≠ 独立印证**。

```mermaid
flowchart TB
    A["问题拆解"] --> B{"方面类型"}
    B -->|列举| C["需要 ≥N 项，可跨块合并"]
    B -->|数值| D["必须单块覆盖，不能拼接"]
    B -->|步骤| E["需要有序过程文本"]
    B -->|定义| F["需要定义句式"]
    B -->|实体| G["需要实体+属性同现"]
    C & D & E & F & G --> H["coverage = 已覆盖方面 / 总方面"]
    H --> I{"coverage ≥ 0.8 ?"}
    I -->|是| J["通过"]
    I -->|否| K["拒答 / 澄清"]
```

- 方面抽取先用规则：疑问词类型（几个/哪些 → 列举；多少/几天 → 数值；怎么/如何 → 步骤；是什么 → 定义）+ 实体槽（学科/年级/知识点）。
- **硬方面**（数值、单点事实）必须由同一个父块覆盖；**软方面**（列举、步骤）允许跨块合并。
- 用户举的例子必须通过：Q = "课程分为哪几个阶段？第三阶段学什么？"，文档 A 给三阶段列表、文档 B 详述第三阶段 → 方面 1 由 A 覆盖、方面 2 由 B 覆盖，coverage = 2/2 → **放行**；旧 margin 会因两文档分数接近而判 `ambiguous`，这是必须修掉的假拒答。

#### 4.3.5 冲突检测只处理"真矛盾"

```python
@dataclass(frozen=True)
class Claim:
    slot: str        # price / duration / date / ratio / audience
    value: str       # 归一化后的值，如 "12800" / "3个月" / "2026-09"
    scope: str       # subject|grade|region|tenant 组合，用于判断适用范围
    doc_id: str
```

- 触发条件：`slot` 相同、`scope` 重叠、归一化 `value` 不同。
- 消解顺序：`effective_date` 新者胜 → `authority` 高者胜（官方文件 > 教材 > 教案 > 字幕）→ 仍判不出则双引用并声明"不同来源存在差异"。
- **先 shadow**：只记 `conflict_shadow_total`，不改变回答；积累两周真实样本后再决定是否 block。
- 纯正则 + 词典，不调 LLM；只有"数值相同但语义相反"这种极少数情况才允许一次 LLM 判定（超时 300 ms，失败即放过）。

#### 4.3.6 元数据前提

`CLAUDE.md` §6.4 的 Milvus schema 没有 `effective_date / grade / authority`。本方案**不改 schema**，新增 MySQL 表：

```sql
CREATE TABLE doc_meta (
  doc_id        VARCHAR(40) PRIMARY KEY,
  doc_type      VARCHAR(32),   -- official / textbook / lesson_plan / subtitle
  subject       VARCHAR(32),
  grade         VARCHAR(16),
  effective_date DATE,
  authority     TINYINT,       -- 3 官方 / 2 教材 / 1 教案 / 0 其他
  status        VARCHAR(16),   -- active / deprecated
  KEY idx_subject_grade (subject, grade)
);
```

检索结果按 `doc_id` join（一次批量查询，别在循环里查库）。这也顺带解决"文档相同不代表内容正确"：过期/失效文档在生成前就被标注或剔除。

#### 4.3.7 重排策略决策（蓝队点名）

| 方案 | 质量 | 延迟 | 适用 |
|---|---|---|---|
| GPU cross-encoder 全量 top40 | 最好 | ~50–80 ms（待实测） | 有卡且并发可控 |
| GPU 只重排 top20 | 略降 | ~30–40 ms | 有卡、高并发 |
| CPU 重排 | 同质量 | 4.5 s/40 候选 | **仅离线评估**，不上线 |
| 关闭重排 | 降级 | 0 | 无卡时的默认 |

决策规则：`RAG_RERANK_ENABLED=auto`，无 CUDA 直接关；有 CUDA 时按压测结果选 top20 或 top40。**没有 GPU 就不承诺"重排后的指标"**，只承诺融合分模式下的指标。

**怎么工作（标定流程）**
1. 用 763 条可判定样本，分别跑 `rerank=False` 与 `rerank=True`，各自存 top-10 证据、qcov、aspects。
2. 对每种 mode 二维/三维扫描阈值，目标函数：**误放行 ≤5% 前提下最大化可答放行率**。
3. 输出 `docs/analysis/rag_threshold_report_v2.md` + 原始跑批 JSONL，任何人可复跑。
4. 冲突检测 shadow 报告单独一份，不进主阈值。

**落到哪个文件**
- 改：`rag_system/core/evidence_gate.py`（`GateInput`、mode、aspect coverage、废弃 margin）。
- 新增：`rag_system/core/aspects.py`（方面抽取规则）、`rag_system/core/conflict.py`（claim 抽取 + shadow）、`rag_system/core/doc_meta.py`（join）。
- 改：`base/config.py` 增 `RAG_GATE_FUSION_*`、`RAG_GATE_RERANK_*`、`RAG_EVIDENCE_MIN_ASPECT_COVERAGE=0.8`、`RAG_CONFLICT_MODE=shadow`。
- 改：`scripts/eval_rag.py` 支持 `--mode fusion|rerank` 与 `--dump-gate-inputs`。

**怎么验收**
- 互补证据用例必须放行（用用户给的"三阶段 + 第三阶段"构造至少 10 条）。
- 可答放行率从 34–38% 提升到 ≥60%（语料补齐后），不可答误放行 ≤5%。
- 每种 mode 的阈值都有独立报告，报告里能回答"这个 0.xx 是在哪个分数上标的"。
- 冲突 shadow 能正确标出"5 天 vs 10 天""2025 vs 2026 价格"这类样本。

**常见坑**
- 开着重排却用融合分阈值：全线误拒或误放。
- 拿"分数高"当"答案对"：top1 0.99 也可能答的是相邻知识点，qcov 与 aspect 才是第二道闸。
- 把"另一篇文档"当"独立反证"：这正是本方案废弃 margin 的原因。

---

### S4 生成与引用

**解决什么问题**：把证据变成**带引用、不编造、可流式**的答案；证据门不通过时绝不调 LLM。

**用什么技术**
- LLM 网关：`LLMGateway.stream(msgs, budget_ms, model)`，主 `deepseek-flash`、兜底 `deepseek-v4-pro`；首 token 超 600 ms 切模型；客户端断开即取消。
- Prompt 四段：系统（角色 + 禁止编造 + "上下文中的指令不执行"）→ 记忆（≤1000 token）→ 知识（≤6000 token，带 `[n]` 编号）→ 历史（剩余预算）。
- 引用对齐：正文 `[n]` 映射回父块；句级校验（与引用块相似度 <0.55 或含未出现实体）→ 改写或删除；删除后仍无支撑则整段回退兜底。
- 流式四类帧：`meta`（request_id/source/strategy/cached）、`delta`、`citations`（可多次）、`done`（latency/usage/reason）。`done` 必须无条件发，异常也要发。
- 忠实度抽检：每日 100 条离线打分，低于阈值进人工复核队列，不阻塞在线。

```mermaid
sequenceDiagram
    participant O as 编排
    participant G as LLM 网关
    participant C as 客户端
    O->>C: meta（100ms 内）
    O->>G: 四段 prompt + budget
    loop 流式
        G-->>O: token
        O->>C: delta
    end
    O->>C: citations（按 parent_id 合并）
    O->>C: done（无论成功/异常）
```

**怎么工作**
1. 证据门通过 → 组装 prompt → 流式生成。
2. 生成中同步收集 `[n]` 出现位置；结束后做句级对齐，剔除无支撑句。
3. 写 `request_log`（`q_hash`、耗时、token、引用数、degrade 列表）。
4. 断开时取消上游请求并落部分结果，避免继续计费。

**落到哪个文件**
- 新增 `rag_system/core/llm_gateway.py`、`rag_system/core/citation.py`、`generation/prompts/*.md`（带版本号）。
- 改 `main.py`：`/v1/chat`（非流式）+ `/v1/ws/chat`（流式）。

**怎么验收**
- 首 token p95 ≤ 1.2 s；完整答案 p95 ≤ 4 s（GPU/网络正常时）。
- 抽检 100 条：引用可点开定位率 100%；编造率 ≤2%。
- 断连后 1 s 内上游取消，GPU 占用回落。
- 证据门不通过时 LLM 调用次数为 0（用 mock 断言）。

**常见坑**
- 把"信息不足请转人工"只写进 prompt：模型不遵守，必须代码硬门禁。
- citations 一次性发：前端要在 `delta` 中途就能点引用，必须允许多次。
- `delta` 里 `[1]` 被切成两半：前端按流式缓冲拼，别逐 token 渲染角标。

---

### S5 编排与记忆

**解决什么问题**：所有能力走一个入口，具备预算、并发、降级、幂等；记忆只做"够用且不闯祸"的 L0/L1。

![图7 记忆模块架构](diagrams/07_memory_architecture.png)

**用什么技术**
- 单入口：`POST /v1/chat` + `GET /v1/ws/chat`，同一 `ChatState`。
- 编排实现：**先显式状态机**（`orchestration/graph.py`：节点函数 + 条件边 + 预算上下文），LangGraph 作为可替换后端（见 §8 决策 D2）。
- 并发：`asyncio.gather` + 每个分支独立 `wait_for` 超时；记忆慢不能拖住 FAQ。
- 幂等：`request_id` 贯穿缓存、历史、工单。
- 记忆 L0/L1：
  - L0 逐字历史：Redis LIST 热存 + MySQL 真相源；窗口 40 轮 / 28000 token 先到先压。
  - L1 会话摘要：每 10 轮或 3000 token 滚动，≤400 token。
  - 写入异步：Redis Stream 消费，p95 ≤1.5 s，失败重试 + 死信。
  - L2 原子记忆 / L3 画像：**本版不做**，只留表结构与开关。

**怎么工作**
1. 请求进入 → 预算上下文 → 归一化/改写 → `gather(FAQ, memory.recall)`。
2. FAQ 命中即返回（记忆召回结果丢弃，省 token）。
3. 未命中 → 检索/证据门/生成 → 流式返回 → 异步写历史与摘要。
4. 每节点失败只影响本节点，`degrade` 记录后继续。

**落到哪个文件**
- 新增 `orchestration/{graph,budget,nodes}.py`。
- 新增 `memory/{service,writer,summarizer,store}.py`。
- 改 `main.py` 的 lifespan：加载模型 → 预热三条链路 → `/readyz` 门控。

**怎么验收**
- 单入口跑通 FAQ 命中、RAG 通过、证据不足兜底、记忆超时降级四条路径。
- 记忆召回 p95 ≤120 ms；删除 2 s 内对召回生效。
- 同一 `request_id` 重发不重复计费、不重复写历史。
- 优雅停机：SIGTERM 后 `/readyz` 先 503，在途 ≤10 s 完成。

**常见坑**
- 并行分支共用一个超时：FAQ 被记忆拖慢，15 ms 变 300 ms。
- 记忆写同步阻塞回答：必须 Stream 异步。
- 状态机里直接调数据库：违反分层，测试无法 mock。

---

### S6 运营与交付

**解决什么问题**：让系统"可观测、可压测、可灰度、可回滚"，把 SLO 从文档变成验收记录。

**用什么技术**
- 指标：QPS、分阶段耗时、缓存命中率、错误分类、LLM token/成本、转人工率、`degrade_total`、`gate_reason_total`。
- 日志：JSON 结构化 + `trace_id`，双出口（stdout + `logs/app.log` 轮转 50 MB×10）。
- 压测：`scripts/loadtest.py --qps 20 --duration 120s`，记录 p50/p95/p99、显存峰值、OOM 降级行为。
- 灰度：按 `KB_VERSION` 与实例比例切流；回滚 = 切回 alias + 改 `KB_VERSION`（缓存自然失效）。
- 部署：Docker 多阶段构建，非 root，`/readyz` 门控接流量。

**怎么验收**
- SLO 表逐项打勾（首 token、端到端、错误率、Recall、忠实度、缓存命中、转人工）。
- 压测报告含"并发 20 时显存峰值"与"OOM 后是否降级而非 5xx"。
- 回滚演练 ≤30 s，回滚后 5 min 内指标恢复。
- 故障演练：杀 Redis、断 Milvus、LLM 超时，均"有兜底、无 5xx"。

**常见坑**
- 只写 stdout 不落盘：线上没有现场。
- 压测用 FAQ 命中路径：掩盖 RAG 真实延迟。
- 灰度只看错误率不看质量：新索引可能"能答但答错"。

---

## 5. 与 GPT 意见的逐条对照

| # | GPT 的问题 | 是否成立 | 本方案怎么落 | 落在 |
|---|---|---|---|---|
| 1 | 检索到 ≠ 答对；可答题仅约 20% 能召回正确文档 | ✅ 成立，且是最优先 | S0 语料盘点 + S1 补语料与基线；失败归因区分 `retrieval_miss` / `knowledge_gap` | S0/S1 |
| 2 | 融合分、重排分、证据门阈值可能混用 | ✅ 成立，代码里确实混 | `GateInput` 显式字段 + `fusion/rerank` 两套阈值，禁止跨模式复用 | S3 |
| 3 | qcov 只是词面覆盖，挡不住"主题对、知识点错" | ✅ 成立 | 保留 qcov 作辅助，新增 aspect coverage（硬方面单块覆盖） | S3 |
| 4 | Agent 容易变成昂贵串行链 | ✅ 成立 | S2 规则二分器 + FAQ 短路；HyDE/SUB_QUERY 只在多实体/多条件时启用 | S2/S5 |
| 5 | 记忆一次性做完不现实 | ✅ 成立 | S5 只做 L0/L1，L2/L3 留表与开关；先测"删除生效、隔离、冲突" | S5 |
| 6 | 技术栈重、部署复杂 | ✅ 成立 | S6 才做全栈交付；S1 起就固定 CUDA/torch/模型版本，避免后期返工 | S1/S6 |
| 7 | 配置与完成标准不一致（意图 5 类 vs 6 类等） | ✅ 成立 | S0 拍板意图口径；`evaluate()` 默认值与配置对齐；文档滞后视为缺陷 | S0/S3 |
| 8 | 建议用 LangGraph 管理状态 | ⚠️ 部分采纳 | 需要状态机，但先自研显式状态机；LangGraph 作为可替换后端，引入前先改 `CLAUDE.md` | S5 |
| 9 | 课程价格/学制走结构化查询而非向量检索 | ✅ 值得采纳 | S5 预留 `get_course_info` 工具；价格类问题优先查 MySQL | S5 |
| 10 | 不要把拒答率低当目标 | ✅ 成立 | 验收口径固定为"放行率↑ 且 误放行≤5%" | S0/S3 |

**我们不同意/需要修正 GPT 的地方**（诚实标注）：

- GPT 说"证据门只放行 34% 可答题"是阈值问题——不完全是。主因是语料覆盖（只有一本九年级教材 + 6 份短教材对 7–9 年级黄金集）。**先补语料再调阈值**，否则调阈值只是把误放行换上来。
- GPT 建议"跨文档比较合理"——本方案**废弃 margin**，因为互补证据会被误杀；改为 aspect coverage + 只在真矛盾时触发冲突检测（见 §4.3.4/4.3.5）。

---

## 6. 蓝队审查记录与采纳

| 蓝队缺陷 | 采纳情况 | 具体改动 |
|---|---|---|
| S0 冻结基线会作废 | ✅ 采纳 | S0 只交付定义；基线测量并入 S1 出口 |
| S2 承诺 6 类意图 + 槽位无法验收 | ✅ 采纳 | S2 只做规则二分器；意图/槽位改为埋点，单开子阶段 |
| S3 之前没有端到端出口，答对率测不到 | ✅ 采纳 | 最小 E2E 前移 S2（模板回答 + 证据卡片） |
| S3 不能当已完成品 | ✅ 采纳 | 重排策略决策 + 四判据重标定列为 S3 独立验收项 |
| S1 编码成本无预案 | ✅ 采纳 | 编码耗时基线 + ONNX INT8 / 增量编码 / 夜间批处理三选一 |
| 意图路由不该拖到 S5 | ✅ 采纳 | 前移为 S2 规则路由（用 0.25/0.10 当二分器） |

---

## 7. 验收红线（每阶段门禁）

| 阶段 | 门禁（不过就不进下一阶段） |
|---|---|
| S0 | 100 题人工复核一致率 ≥95%；缺口清单可执行 |
| S1 | Recall@10 基线可复跑；索引切换/回滚 ≤30 s；重复摄入无重复块 |
| S2 | 路由准确率 ≥90%；模板回答引用可定位 100%；答对率基线有数 |
| S3 | 互补用例全放行；可答放行率 ≥60%（补语料后）；误放行 ≤5%；两套阈值各有报告 |
| S4 | 首 token p95 ≤1.2 s；引用可定位 100%；编造率 ≤2%；门禁不过时 LLM 调用 0 次 |
| S5 | 四类路径全通；记忆 p95 ≤120 ms；删除 2 s 生效；重发幂等 |
| S6 | SLO 表逐项达标；压测报告含显存；回滚 ≤30 s；故障演练无 5xx |

---

## 8. 待拍板决策（需要你确认）

| # | 决策 | 选项 | 建议 | 影响 |
|---|---|---|---|---|
| D1 | 意图口径 | K12 六类（有 4080 训练集）/ 招生五类 | **先 K12 六类**，招生场景作为第二套业务配置 | 训练数据、标签、路由、评测切片 |
| D2 | 是否引入 LangGraph | 自研显式状态机 / LangGraph | **先自研**，S5 验收后再评估；引入需先改 `CLAUDE.md` | 依赖、可观测性、调试成本 |
| D3 | 线上是否启用重排 | GPU 全量 / GPU top20 / 关闭 | **有卡 top20**；无卡默认关 | 延迟与质量、验收口径 |
| D4 | 记忆范围 | L0/L1 / 四层全做 | **先 L0/L1** | 工期、隐私风险 |
| D5 | 冲突检测是否阻断 | shadow / block | **先 shadow 两周** | 误拒风险 |
| D6 | 语料补齐责任 | 机构提供 / 自行爬取 | **机构提供**并给清单 | 能否达到 Recall 目标 |

---

## 9. 风险登记册

| 风险 | 概率 | 影响 | 缓解 | 触发信号 |
|---|---|---|---|---|
| 语料覆盖不足，Recall 上不去 | 高 | 高 | S0 缺口清单 + 机构补文档；先测"库里有答案"的子集 | 可答题正确文档召回 <50% |
| 分数尺度混用导致门禁失效 | 高 | 高 | `GateInput` + 两套阈值 + 报告分 mode | 同一 query 两种 mode 结论相反 |
| 无 GPU，重排不可上线 | 中 | 高 | 关闭重排 + 融合分标定；离线评估保留 | `degrade` 里 `rerank_fallback` 飙升 |
| 意图标签口径反复 | 中 | 中 | S0 拍板；标签配置化，不写死 | 训练集与路由规则不一致 |
| 记忆污染回答 | 中 | 高 | L0/L1 先行；记忆不进引用；删除 2 s 生效 | 出现"记忆里的事实"被当证据 |
| 编排层引入后延迟回退 | 中 | 中 | 预算扣减 + 每节点超时 + 压测门禁 | p95 超出 S2 基线 20% |
| 索引切换期间不可用 | 低 | 高 | 双索引 + alias + 回滚演练 | 切换时 5xx |
| 工期被"企业级"拖长 | 高 | 中 | 每阶段有出口，先闭环后深度；L2/L3、多智能体、GraphRAG 全部后置 | 连续两阶段无可演示产出 |

---

## 10. 附录

### 10.1 文件地图（目标态）

```
base/            config / logger / labels / contracts / db        # 地基
qa_system/       api.py + res/qa_*.py                             # FAQ（已实现）
rag_system/      api.py + core/{retriever,reranker,evidence_gate,aspects,conflict,llm_gateway,citation}.py
                 document_loaders/ text_splitters/ embedding/ store/ ingest.py
understanding/   router.py（S2 规则）/ intent_service.py（S2.5）  # 待建
orchestration/   graph.py budget.py nodes.py templates.py         # 待建（S2/S5）
memory/          service.py writer.py summarizer.py store/        # 待建（S5）
scripts/         ingest / eval_retrieval / eval_rag / eval_faq / audit_corpus / loadtest
docs/            架构设计 / 分阶段技术方案（本文）/ analysis/
```

### 10.2 常用命令

```bash
# 摄入与建索引
python scripts/ingest.py --src datasets --kb-version v1

# 只看检索（不调 LLM）
python scripts/eval_retrieval.py --set eval/golden_v0.jsonl --report out/retrieval.md

# 证据门标定（分 mode）
python scripts/eval_rag.py --mode fusion --sweep
python scripts/eval_rag.py --mode rerank --sweep

# 端到端冒烟
python scripts/smoke.py --ws
python scripts/loadtest.py --qps 20 --duration 120s
```

### 10.3 术语表

| 术语 | 含义 |
|---|---|
| 父子块 | 父块 1200 字用于生成上下文，子块 300 字用于检索；子块命中后回填父块 |
| 融合分 | dense 0.7 + sparse 0.3 经 WeightedRanker 得到的分数 |
| 重排分 | cross-encoder 对 (query, 片段) 打分后 sigmoid 归一 |
| qcov | 问句实词在证据中的覆盖率 |
| aspect coverage | 问句拆出的语义方面被证据覆盖的比例 |
| 放行率 / 误放行率 | 可答题通过证据门的比例 / 不可答题被放行的比例 |
| shadow 模式 | 新判据只记录不生效，用真实流量验证误伤 |

---

> 本文与 `CLAUDE.md` 冲突时以 `CLAUDE.md` 为准；本文提出的架构变更（废弃 margin、引入 aspect coverage、自研状态机替代 LangGraph）需先改 `CLAUDE.md` 再改代码。
