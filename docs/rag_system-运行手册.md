# rag_system（PR3）运行手册：切分 · 入库 · 检索

知识库链路：**文件 → 加载 → 父/子块切分 → BGE-M3 稠密+稀疏 → Milvus → 混合检索 → 重排 → 证据门**。
与 qa_system 的分工：FAQ 答"标准问法"，rag_system 答"要读材料才能答的问题"，两者由编排层按意图分流。

```
datasets/01_textbook  ...md
datasets/02_lesson_plans ...docx
datasets/08_subtitles ...srt
        │  rag_system/ingest.py（先按 source 删旧块，再写）
        ▼
  Milvus edurag.edurag_chunk（dense 1024 + sparse，HNSW + 稀疏倒排）
        │  rag_system/core/retriever.py（混合召回 → 父块去重 → 重排）
        ▼
  evidence_gate（分数 / 领先幅度 / 证据数三条件）── 通过 → 进 prompt；不通过 → 拒答转人工
```

## 1. 依赖

```powershell
cd F:\heima\RAG\RAGpro_智能教育系统
.\.venv\Scripts\python.exe -m pip install -r requirements-rag.txt
# torch 按显卡驱动单独装（cu121 与 CUDA 12.x 驱动配套）。CPU 版可先跑通链路，但线上不可用。
.\.venv\Scripts\python.exe -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

模型权重放本地（默认指向旧项目目录，可用 `RAG_EMBED_MODEL_PATH` / `RAG_RERANKER_PATH` 指到别处）：

| 模型 | 体积 | 用途 |
|---|---|---|
| `BAAI/bge-m3` | 4.4 GB | 一次前向出 dense(1024) + sparse 词权重 |
| `BAAI/bge-reranker-v2-m3` | 2.2 GB | 交叉编码重排，GPU 上 40 条 ≈ 150 ms |

## 2. 起 Milvus

```powershell
.\scripts\docker_up.ps1 -Full        # etcd + minio + milvus(19531) + attu(3001)
# 只想起数据面（MySQL/Redis）：.\scripts\docker_up.ps1
```

端口刻意避开旧项目：MySQL 3307、Redis 6380、Milvus 19531、MinIO 9002/9003、Attu 3001。

## 3. 入库

```powershell
# 全量（RAG_DOCS_DIR 配置的目录）
.\.venv\Scripts\python.exe -m rag_system.ingest --defaults

# 指定目录 / 追加 PDF / 先看切分效果不入库
.\.venv\Scripts\python.exe -m rag_system.ingest --paths datasets/01_textbook
.\.venv\Scripts\python.exe -m rag_system.ingest --paths datasets/01_textbook --pdf datasets/09_textbooks/人教版九年级数学.pdf
.\.venv\Scripts\python.exe -m rag_system.ingest --paths datasets/01_textbook --dry-run

# 全量重灌（集合删了重建，模型换代后必须走这条）
.\.venv\Scripts\python.exe -m rag_system.ingest --defaults --recreate
```

- **幂等**：每个文件入库前按 `source` 删旧块，重复跑不会出现两份答案。
- **单文件失败不中断**：失败进 `data/ingest_manifest.json` 的 `failures`，退出码为 1。
- **PDF**：`loader_pdf_mineru.py` 走 MinerU 云解析，结果缓存在 `data/mineru/`，同一份 PDF 只花一次额度（`MINERU_TOKEN`）。

## 4. 检索冒烟

```powershell
# 只验证管线（不加载模型，秒级）
.\.venv\Scripts\python.exe scripts\smoke_rag.py --backend hash --paths datasets/01_textbook --query "顶点式怎么写"

# 验证真实质量（需要 torch + 权重）
.\.venv\Scripts\python.exe scripts\smoke_rag.py --paths datasets/01_textbook --query "顶点式怎么写"
```

冒烟脚本会打印：命中父块数、top3 的分数/标题/小节/文件名、证据门结论。
**拒答不是失败**——`below_threshold` / `ambiguous` / `single_weak_evidence` 都是设计行为。

## 5. 参数（`.env`，基线见 CLAUDE.md §6.3）

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_PARENT_SIZE` / `RAG_CHILD_SIZE` | 1200 / 300 | 父块进 prompt，子块只被检索 |
| `RAG_TOP_K` | 40 | 每路 ANN 召回数 |
| `RAG_FINAL_N` | 6 | 进 prompt 的证据条数 |
| `RAG_RERANK_ENABLED` | true | CPU 上建议 false（否则单查数秒） |
| `RAG_RERANK_CANDIDATES` | 40 | 只重排前 N 条 |
| `RAG_EVIDENCE_THRESHOLD` | 0.35 | 低于此分不生成 |
| `RAG_EVIDENCE_MARGIN` | 0.03 | top1-top2 最小间隔，防"同题族"误答 |
| `RAG_EMBED_BACKEND` | auto | `hash` 仅用于管线验证，检索质量无意义 |

## 6. 已知边界

1. **重排必须上 GPU**：CPU 上 40 条候选 2–4 s，会打穿 1.2 s 首 token 预算；没有 GPU 就 `RAG_RERANK_ENABLED=false`，按融合分排序。
2. **证据门阈值未按真实语料标定**：0.35/0.03 是起点，语料扩充后应用黄金集重标（同 FAQ 阈值流程）。
3. **MinerU 走云端**：断网环境需改本地 `magic-pdf` 部署，当前只做了云 API + 缓存。
4. **未做索引版本化**：现在 `--recreate` 是全量重建，重建期间服务不可用；blue/green + alias 切换在 PR3 第二阶段。
