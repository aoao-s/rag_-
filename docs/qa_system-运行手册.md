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
Copy-Item .env.example .env            # 按需改 MYSQL_PASSWORD / 端口
.\scripts\docker_up.ps1                # 起 MySQL(3307) + Redis(6380)，等健康检查
```

> 离线机器先跑 `.\docker\offline\load.ps1` 导入镜像，再跑上面最后一条。
> 端口/密码只改 `.env`，compose 和应用读同一份，详见 `docs/docker-离线部署.md`。

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

### 3.2 命令行测试台（不启服务也能查）

`qa_system/main.py` 是给调参与排障用的 CLI：不经过 HTTP，直接把 `FAQService` 拉起来问。

```powershell
.\.venv\Scripts\python.exe -m qa_system.main                      # 交互式，读 MySQL
.\.venv\Scripts\python.exe -m qa_system.main --csv                # 不连库，读 datasets CSV
.\.venv\Scripts\python.exe -m qa_system.main -q "勾股定理咋证明啊"  # 单条查询后退出
.\.venv\Scripts\python.exe -m qa_system.main --csv --rank -q "什么是光合作用"
```

交互命令：`/stats`（语料与命中率）、`/rebuild`（重建索引）、`/rank <问题>`（只看分数与分差）、
`/subject <学科>`（限定学科检索，`off` 取消）。

`/rank` 是调阈值的主要工具：`/v1/faq/search` 只回最终结论，看不出"差多少分"，
而 `FAQ_THRESHOLD` / `FAQ_MARGIN` 的判断完全取决于 top1 分数与 top1−top2 分差。

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

## 6. 日志与缓存（2026-10-20 新增）

### 6.1 日志长什么样

控制台与文件是**同一行格式**，文件更详细（`LOG_FILE_LEVEL=DEBUG`，控制台 `LOG_LEVEL=INFO`）：

```
2026-10-20 19:21:35.097 [INFO] [15624:MainThread] EduRAG.qa_system.res.qa_search (qa_search.py:537): 检索命中 [来源: MySQL 索引] 问题: [2次函数的顶点式] | faq_id: 148 | 学科: 数学 | 得分: 1.0 | 耗时: 6.21ms
2026-10-20 19:21:35.150 [INFO] [15624:MainThread] EduRAG.qa_system.res.qa_search (qa_search.py:537): 检索命中 [来源: Redis 缓存] 问题: [2次函数的顶点式] | faq_id: 148 | 学科: 数学 | 得分: 1.0 | 耗时: 2.93ms
2026-10-20 19:21:30.991 [INFO] [15624:MainThread] EduRAG.qa_system.res.qa_search (qa_search.py:550): 检索未命中 [来源: MySQL 索引] 问题: [顶点式怎么写] | 原因: ambiguous | 最高得分: 0.4165 (阈值: 0.25) | 分差: 0.028 | 候选: [150, 148, 1516] | 耗时: 7.1ms
2026-10-20 19:21:30.991 [INFO] [15624:MainThread] EduRAG.qa_system.res.qa_search (qa_search.py:551): FAQ 未命中，转交 RAG 检索 问题: [顶点式怎么写] | 候选问题: ['一般式的顶点坐标', '二次函数的顶点式', 'y=2x²-8x+1 顶点']
2026-10-20 19:21:44.133 [INFO] [15624:MainThread] EduRAG.qa_system.res.qa_cache (qa_cache.py:197): Redis 缓存续命 [Key: edurag:dev:faq:v1:*:d80bf0e909b254dacebd], 剩余 59s → 3600s
```

- 文件名 `logs/app.log`（`RotatingFileHandler`，50 MB 一份、保留 10 份，UTF-8），stdout 同时一份。
- 业务日志与 uvicorn 访问日志走同一套 handler，所以"有请求没日志"不会再出现。
- 文案由 `base/log_messages.py` 统一维护：改口径只改一处，不要在业务代码里拼字符串。

| 事件（event） | 渲染后的样子 | 关键字段 |
| --- | --- | --- |
| `faq_search_hit` | 检索命中 `[来源: Redis 缓存 / MySQL 索引]` … | `faq_id`、`subject`、`score`、`latency_ms` |
| `faq_search_miss` | 检索未命中 … `最高得分 (阈值)`、`分差`、`候选` | `reason`(below_threshold/ambiguous)、`margin` |
| `faq_handoff_rag` | FAQ 未命中，转交 RAG 检索 | `candidates` |
| `faq_cache_write` / `faq_cache_slide` / `faq_cache_stale` | 回填 / 续命 / 缓存指向已下线条目 | `ttl`、`remaining_s` |
| `faq_db_query` | MySQL 命中 [操作/行数/耗时] | `op`、`rows`、`cost_ms` |
| `faq_cache_error` | 缓存读写异常（fail-open，不影响回答） | `op`、`error` |
| `http_request` | HTTP GET … -> 200 [x ms] | `status`、`latency_ms` |

排查一条查询的完整链路：

```powershell
Select-String -Path logs\app.log -Pattern '检索命中|检索未命中' | Select-Object -Last 5
# 只统计命中来源分布
Select-String -Path logs\app.log -Pattern '\[来源: (.+?)\]' | ForEach-Object { $_.Matches[0].Groups[1].Value } |
  Group-Object | Sort-Object Count -Descending
# 想看原始问法：.env 里 LOG_QUERY_TEXT=true（dev 默认已开；生产必须 false）
```

### 6.2 Redis 二级缓存

```powershell
# 看有哪些热数据
docker exec edurag-redis redis-cli -a 1234 --no-auth-warning --scan --pattern 'edurag:*'
# 单条内容与剩余寿命
docker exec edurag-redis redis-cli -a 1234 --no-auth-warning get  "edurag:dev:faq:v1:*:<hash>"
docker exec edurag-redis redis-cli -a 1234 --no-auth-warning ttl  "edurag:dev:faq:v1:*:<hash>"
# 想清空热数据（安全：缓存内容可由 MySQL 重算）
docker exec edurag-redis redis-cli -a 1234 --no-auth-warning flushdb
```

规则：

- key 带语料版本 `v{n}`；发布/下线后版本 +1，旧 key 自然失效，不需要遍历删除。
- 正缓存 1 h、绝对寿命 6 h；命中时剩余 ≤50% 才 `EXPIRE` 续命（避免每次命中都写 Redis）。
- 负缓存 30 s，**带 top3 候选**（交棒 RAG 要用）。
- Redis 挂了不影响回答：直接走内存索引，指标 `edurag_faq_cache_errors_total` 会涨。

## 7. 已知待办

- `FAQ_THRESHOLD=0.25` / `FAQ_MARGIN=0.10` 已用 480 条合成同义问法 + 80 条题库负样本标定
  （`docs/analysis/faq_threshold_report.md`，precision 0.950 / recall 0.637）。
  合成样本只能当起点：上线两周后要用真实点击与转人工记录重标。
- 同题族（勾股定理「内容 / 证明 / 应用」）靠 margin 兜住；若线上仍出现"答成兄弟题"，优先补别名，不要再降阈值。
- 别名表目前只做「整串等价」（`faq_alias`），子串替换会产生"学费学费多少"这类脏串，已明确不做。
- 语料 >5 万条时把全量重建换成文档级增量（现在 160 条约 50 ms，不值得提前优化）。

## 8. 阈值标定（可复跑，不需要数据库）

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

