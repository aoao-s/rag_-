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

- `FAQ_THRESHOLD=0.62` 尚未用黄金集标定：当前近义问法（"勾股定理的证明方法"）得 0.35 会落到 RAG，
  标定时用 `datasets/07_eval` 的 157 条跑 P/R 曲线（PR2 第二步，脚本未写）。
- 别名表目前只做「整串等价」（`faq_alias`），子串替换会产生"学费学费多少"这类脏串，已明确不做。
- 语料 >5 万条时把全量重建换成文档级增量（现在 160 条约 50 ms，不值得提前优化）。
