# qa_system（PR2）跑通手册

FAQ 精确检索模块：启动把 MySQL 里 `published` 的语料装进内存（分学科 BM25 + 精确哈希 + 别名表），
请求路径零 IO、零模型，未命中时返回 top3 候选给 RAG 做 query 扩展。

```
datasets/04_faq/edu_faq.csv
        │  scripts/import_faq.py --publish
        ▼
   MySQL faq / faq_alias
        │  启动装载 + 发布后重建（原子换快照）
        ▼
   FAQService（内存索引） ──► /v1/faq/search（p95 ≤15 ms）
```

## 1. 准备

```powershell
cd F:\heima\RAG\RAGpro_智能教育系统
F:\miniconda3\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env      # 按需改 MYSQL_PASSWORD
docker compose -f docker/docker-compose.yml up -d
```

## 2. 建表与导数据

```powershell
.\.venv\Scripts\python.exe scripts\init_db.py                       # 建库建表（旧库会自动补列）
.\.venv\Scripts\python.exe scripts\init_db.py --prune-legacy-subjects # 一次性：下线非 K12 学科的历史行
.\.venv\Scripts\python.exe scripts\import_faq.py --publish          # CSV → faq（发布才进检索语料）
```

## 3. 启动

```powershell
.\.venv\Scripts\python.exe main.py                     # 读 .env 的 HTTP_HOST/HTTP_PORT
.\.venv\Scripts\python.exe -m uvicorn main:app --port 8124 --reload
```

### 3.1 没有 MySQL 也能跑（无库模式）

`.env` 里设 `FAQ_CORPUS_CSV=datasets/04_faq/edu_faq.csv`，启动时直接从 CSV 装载语料，
日志会打 `faq_corpus_from_csv`（`level=warning`，提醒这不是生产配置）。
写接口（publish/deprecate）改的是内存仓库，重启即失效；MySQL 才是生产真相源。

启动顺序固定：配置自检 → 装载语料 → 预热（jieba 词典 + 三条查询）→ `/readyz` 置 200。
装载期间 `/healthz` 仍是 200，`/readyz` 返回 503，负载均衡不会把冷启动请求打到用户脸上。

## 4. 验证

```powershell
curl "http://127.0.0.1:8124/readyz"
curl "http://127.0.0.1:8124/v1/faq/search?q=勾股定理的内容是什么"
curl "http://127.0.0.1:8124/v1/faq/stats"
curl "http://127.0.0.1:8124/metrics"
.\.venv\Scripts\python.exe -m pytest tests -q
```

## 5. 接口

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/v1/faq/search?q=&subject=` | 查询；返回 `FAQHit`（含 `candidates` 与 `latency_ms`） |
| GET | `/v1/faq/stats` | 语料版本 + 命中率/延迟统计 |
| GET | `/v1/faq/{id}` | 单条排查 |
| POST | `/v1/faq/{id}/publish` | 发布并重建（需 `X-Admin-Token`） |
| POST | `/v1/faq/{id}/deprecate` | 下线并重建（需 `X-Admin-Token`） |
| POST | `/v1/faq/rebuild` | 手动重建 |
| GET | `/healthz` `/readyz` `/metrics` | 存活 / 就绪 / Prometheus |

## 6. 已知待办

- `FAQ_THRESHOLD=0.25` / `FAQ_MARGIN=0.10` 已用 480 条合成同义问法 + 80 条题库负样本标定
  （`docs/analysis/faq_threshold_report.md`，precision 0.950 / recall 0.637）。
  合成样本只能当起点：上线两周后要用真实点击与转人工记录重标。
- 同题族（勾股定理「内容 / 证明 / 应用」）靠 margin 兜住；若线上仍出现"答成兄弟题"，优先补别名，不要再降阈值。
- 别名表目前只做「整串等价」（`faq_alias`），子串替换会产生"学费学费多少"这类脏串，已明确不做。
- 语料 >5 万条时把全量重建换成文档级增量（现在 160 条约 50 ms，不值得提前优化）。

## 7. 阈值标定（可复跑，不需要数据库）

```bash
# 1) 生成同义问法（deepseek-flash，160 条 FAQ × 3 变体，断点续跑）
python scripts/eval_faq.py gen --concurrency 6

# 2) 二维扫描（threshold × margin）并写报告
python scripts/eval_faq.py calibrate
```

产物：

- `docs/analysis/faq_paraphrases.jsonl`：480 条同义问法（含 faq_id 标签）
- `docs/analysis/faq_threshold_report.md`：分数分布 + 一维/二维扫描 + 推荐点

当前推荐点：`threshold=0.25`、`margin=0.10`（precision 0.950、recall 0.637）。
分数尺度由 `FAQ_NORM_K`（默认 12）决定，改 K、改分词、改同义词表都要重跑标定。

