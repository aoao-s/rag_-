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
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# torch 单独装（体积 200MB+，按驱动选）：
#   GPU：.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#   CPU：.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

**这台机器的实测版本组合（已在 CPU 上跑通端到端，不要随手升级）**：

| 包 | 版本 | 为什么卡这个版本 |
|---|---|---|
| torch | 2.5.1 (CPU wheel) | 本机无可用 CUDA 运行时；cu121 版换卡后直接替换即可 |
| transformers | 4.44.2 | FlagEmbedding 1.3.2 的适配上限，4.5x 会因 `sentencepiece` 导入链报错 |
| sentencepiece | **0.2.0** | 0.2.2 的 cp312 wheel 在 Windows 上 import 即 0xC0000005 崩溃 |
| datasets | **2.21.0** | 与 `huggingface-hub 0.36.2` 配套；5.x 要求 hub≥1.31，与 transformers 4.44 冲突 |
| huggingface-hub | 0.36.2 | transformers 4.44 要求 `<1.0` |
| peft | 0.12.0 | FlagEmbedding 的重排器在 import 期就会 `from peft import PeftModel` |
| pymilvus | 2.4.9 | 与服务端 Milvus 2.4.0 对齐；依赖 `setuptools<81`（`pkg_resources` 被移除会直接崩） |
| marshmallow | 3.22.0 | pymilvus 2.4.9 未声明该可选依赖，缺了会 `ImportError` |

两个**必踩的坑**：

1. 仓库根有 `datasets/` 目录（项目方语料），Python 会把它当 namespace 包，`import datasets` 拿到的是目录而不是 HF 库 → 装了 `datasets` 包后由 site-packages 的正规包胜出，别删这个目录即可。
2. 下载 torch 走公网很慢时用 `scripts/fetch_torch.ps1`（分段 curl + 断点重试 + 拼装校验），`scripts/fetch_wheel.py` 是备用下载器。

### 1.1 换用 conda 环境 `ragpro`

项目当前（2026-10-20 起）跑在 `F:\miniconda3\envs\ragpro`（Python 3.12.15）上，里面已经有
torch 2.14.1+cpu、transformers 5.19、FlagEmbedding、pymilvus 3.0.2、python-docx 1.2、python-pptx 1.0.2、
PyMuPDF 1.28.2、rapidocr-onnxruntime、DBUtils。缺的只有 3 个（已装）：

```powershell
F:\miniconda3\envs\ragpro\python.exe -m pip install structlog prometheus-client pytest
```

跑单测、入库、起服务都用这个解释器：

```powershell
$env:PYTHONPATH='.'
F:\miniconda3\envs\ragpro\python.exe -m pytest
F:\miniconda3\envs\ragpro\python.exe -m rag_system.ingest --defaults
F:\miniconda3\envs\ragpro\python.exe main.py
```

`ragpro` 里的版本比上表（`requirements-rag.txt` 那套）更新：torch **2.14.1+cpu**、transformers **5.19.0**、
pymilvus **3.0.2**、sentencepiece 0.2.2、FlagEmbedding 1.3.x。这套组合已实测可用（101 条单测、622 块入库、
检索接口 200），**不要**再按上表降版本；上表只适用于新建干净 venv，按它装会遇到 sentencepiece 0.2.2 崩溃、
pymilvus 缺 `pkg_resources`、hub/datasets 版本互斥等旧坑。


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

### 3.1 支持哪些文件、用哪个解析器

| 后缀 | 加载器 | 说明 |
|---|---|---|
| `.md` `.markdown` `.txt` | `loader_text.py` | 多编码兜底（utf-8-sig→utf-8→gbk→gb18030），解不出编码直接报错 |
| `.srt` | `loader_text.py::SubtitleLoader` | 去时间轴与序号，只留台词；`doc_type=subtitle` |
| `.docx` | `loader_office.py::DocxLoader` | 按文档流取块（段落/表格交错顺序不丢）；标题样式 → Markdown 标题；表格 → Markdown 表格 |
| `.pptx` | `loader_office.py::PptxLoader` | 逐页 + 形状按 `(top, left)` 排阅读顺序；表格同样转 Markdown |
| `.jpg` `.jpeg` `.png` `.bmp` `.webp` | `loader_image.py` | 走 OCR；OCR 不可用或识别为空 → 报错（不静默空文档） |
| `.pdf` | `loader_pdf_mineru.py` / `loader_pdf_local.py` | 见下面的 `RAG_PDF_PARSER` |

PDF 解析器选择（`.env`）：

```ini
RAG_PDF_PARSER=auto     # auto | mineru | local
RAG_OCR_IMAGES=false    # Office/图片里的内嵌图是否 OCR
```

| 取值 | 行为 | 什么时候用 |
|---|---|---|
| `auto`（默认） | 有 `MINERU_TOKEN` 就先云解析；云侧任何异常自动退本地 PyMuPDF，并打 `pdf_parser_fallback`（`meta` 里留 `parser_fallback` + `mineru_error`）；没有 token 直接用本地 | 日常 |
| `mineru` | 只用云解析，没有 token 或失败就报错 | 要保证解析质量（公式/表格/多栏） |
| `local` | 只用 PyMuPDF：逐页抽文本 + 大图/扫描页 OCR，每页打 `<!-- page: n -->` | 断网、批量、或不想花 MinerU 额度 |

两者的取舍：MinerU 出 Markdown（标题、表格、公式结构都在），质量上限高但要联网；本地路径零成本但双栏/扫描件容易串行、公式会变乱码。


## 4. 检索冒烟

```powershell
# 只验证管线（不加载模型，秒级）
.\.venv\Scripts\python.exe scripts\smoke_rag.py --backend hash --paths datasets/01_textbook --query "顶点式怎么写"

# 验证真实质量（需要 torch + 权重）
.\.venv\Scripts\python.exe scripts\smoke_rag.py --paths datasets/01_textbook --query "顶点式怎么写"
```

冒烟脚本会打印：命中父块数、top3 的分数/标题/小节/文件名、证据门结论。
**拒答不是失败**——`below_threshold` / `ambiguous` / `single_weak_evidence` 都是设计行为。

## 4.1 走服务调检索（`/v1/rag/*`）

```powershell
# 启动时会把 Milvus 句柄与向量化模型一起装好；rag 起不来时 /v1/rag/* 返回 503 + reason，FAQ 不受影响
python main.py
curl "http://127.0.0.1:8000/readyz"
# {"status":"ready","faq_docs":2000,"rag":true,"rag_docs":17,"rag_reason":"ok"}

curl "http://127.0.0.1:8000/v1/rag/search?q=勾股定理的证明方法&top_k=8"
curl "http://127.0.0.1:8000/v1/rag/search?q=二次函数&subject=数学&doc_type=textbook,faq&rerank=false"
curl "http://127.0.0.1:8000/v1/rag/stats"
```

返回体关键字段：`evidence`（证据门：`passed/reason/top_score/margin/n_docs`）、`citations`（引用卡片：`file/title/heading/page`）、`evidences`（父块正文与子块命中数）、`degraded`（如 `rerank_failed` 说明按融合分排序）。
`rerank=false` 或环境里没有 FlagEmbedding 时不会报错，只在 `degraded` 里标明——这就是"降级不报错"的验收点。

## 5. 参数（`.env`，基线见 CLAUDE.md §6.3）

| 变量 | 默认 | 说明 |
|---|---|---|
| `RAG_PARENT_SIZE` / `RAG_CHILD_SIZE` | 1200 / 300 | 父块进 prompt，子块只被检索 |
| `RAG_TOP_K` | 40 | 每路 ANN 召回数 |
| `RAG_FINAL_N` | 6 | 进 prompt 的证据条数 |
| `RAG_RERANK_ENABLED` | `auto` | `auto` = 只在有 CUDA 时启用；CPU 上 40 条候选实测 **115 s**，必须走融合分 |
| `RAG_RERANK_CANDIDATES` | 40 | 只重排前 N 条 |
| `RAG_EVIDENCE_THRESHOLD` | 0.74 | 低于此分不生成（黄金集标定） |
| `RAG_EVIDENCE_MARGIN` | 0.01 | top1 与**另一篇文档**最好证据的最小间隔（同文档多块算互相印证） |
| `RAG_EVIDENCE_QCOV` | 0.20 | 问句实词在 top1 证据里的最低覆盖率，挡"同主题不同知识点" |
| `RAG_EMBED_BACKEND` | auto | `hash` 仅用于管线验证，检索质量无意义 |

## 6. 已知边界

1. **重排必须上 GPU**：CPU 上 40 条候选实测 **115 s**（bge-reranker-v2-m3），远超 1.2 s 首 token 预算；`RAG_RERANK_ENABLED=auto` 会在无 CUDA 时自动跳过并按融合分排序。
2. **证据门已标定，但语料是瓶颈**：`datasets/07_eval` 1000 条里 763 条可判定，其中只有 **20%** 能在当前语料里召回到正确文档（黄金集覆盖 7–9 年级，库里只有 1 本九年级数学 + 6 份短教材）。取舍点 0.74/0.01/0.20 是"误放行 ≤5%"口径下可答放行 34%；要提升覆盖率得先补语料，不是调阈值。复跑：`python scripts/eval_rag.py analyze` / `calibrate`（报告 `docs/analysis/rag_threshold_report.md`）。
3. **MinerU 走云端**：断网环境需改本地 `magic-pdf` 部署，当前只做了云 API + 缓存。
4. **未做索引版本化**：现在 `--recreate` 是全量重建，重建期间服务不可用；blue/green + alias 切换在 PR3 第二阶段。

