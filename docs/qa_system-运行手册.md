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

输出里 `来源` 一栏就是判定依据，三行连看就知道数据是怎么流动的：

```
[i] 缓存：Redis 127.0.0.1:6380/0

问题：自然数是什么东西
结论：命中   来源：MySQL 索引（缓存未命中，已回源）   分数 0.4247   耗时 10.03 ms
原因：相似命中（分数与领先幅度均达标）
原问：自然数是什么
学科：数学   faq_id=1
------------------------------------------------------------
答案：用来表示物体个数的数，如 0、1、2、3、4……，自然数从 0 开始，最小的自然数是 0。
------------------------------------------------------------

问题：自然数是什么东西
结论：命中   来源：Redis 缓存（缓存命中，未回源）   分数 0.4247   耗时 2.36 ms
原因：相似命中（分数与领先幅度均达标）（结果直接取自缓存）
```

`来源` 的三个取值：`Redis 缓存` / `MySQL 索引`（含 CSV 无库模式，此时显示 CSV）/ `RAG 检索`（未命中转交）；
括号里是缓存状态（命中、未命中已回源、未启用、异常降级）。CLI 默认接同一套 Redis，
所以「连问同一句两次」是验证回填与续命最快的方式。

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
| GET | `/v1/faq/search?q=&subject=&selected_faq_id=` | 查询；返回 `FAQHit`（含 `tier` / `subject_state` / `degraded` / `candidates` / `latency_ms`）。`subject` 不在词表内（含别名）→ **400 `subject_invalid`**，响应体列出合法学科；不传 = 全库。`selected_faq_id` 是 mid 档追问被采纳后的**用户确认通道**（返回该条答案，逐字，`reason=user_confirmed`），id 无效只记 `degraded` 不报错 |
| GET | `/v1/faq/stats` | 语料版本 + 命中率/延迟统计；`stats.tiers` 为 high/mid/low 三档计数（`routes` 是检索范围计数，两者别混读） |
| GET | `/v1/faq/{id}` | 单条排查 |
| POST | `/v1/faq/{id}/publish` | 发布并重建（需 `X-Admin-Token`） |
| POST | `/v1/faq/{id}/deprecate` | 下线并重建（需 `X-Admin-Token`） |
| POST | `/v1/faq/rebuild` | 手动重建 |
| GET | `/healthz` `/readyz` `/metrics` | 存活 / 就绪 / Prometheus |

**行为变更（2026-10-27，三档路由 §7.1.9）**：`hit` 现在等价于 `tier == "high"`。以前 `hit=true` 的命中里，
margin 落在 `[FAQ_MARGIN, FAQ_HIGH_MARGIN)` 的那部分会变成 `tier="mid"` / `hit=false`、`answer` 为空
（只给 `matched_question` 与候选，交给 RAG + 追问）。**回滚**：把 `FAQ_HIGH_MARGIN` 设成与 `FAQ_MARGIN` 相同，
即精确恢复单档行为（不需要改代码）。

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

- `FAQ_THRESHOLD=0.25` / `FAQ_MARGIN=0.10` 是在 **160 条**语料上用 480 条合成同义问法 + 80 条题库负样本标定的
  （`docs/analysis/faq_threshold_report.md`，precision 0.950 / recall 0.637）。
  **该结论已失效**：语料扩到 2000 条后 480 条标签全部对不上（CLAUDE.md §7.1.7），
  重标定完成前只能当保守兜底；同时要按 `FAQ_MARGIN_MODE` 比较 relative/normalized/raw 三种分差。
- 同题族（勾股定理「内容 / 证明 / 应用」）靠 margin 兜住；若线上仍出现"答成兄弟题"，优先补别名，不要再降阈值。
- 别名表目前只做「整串等价」（`faq_alias`），子串替换会产生"学费学费多少"这类脏串，已明确不做。
- 语料 >5 万条时把全量重建换成文档级增量（现在 160 条约 50 ms，不值得提前优化）。
- **旧库历史数据**：2026-10-26 之前导入的 MySQL 里，97 行"道德与法治"的 `subject` 是空串
  （旧 `SUBJECTS` 只有"政治"，导入时被归一成 None）。补了 `SUBJECT_ALIASES` 之后需要
  重跑 `python scripts/import_faq.py --csv datasets/04_faq/edu_faq.csv --publish`
  （或 `UPDATE faq SET subject_name='道德与法治' WHERE subject_name=''`）才能按学科检索到它们；
  内存/CSV 模式不受影响。

### 6.3 分数是怎么算出来的

检索分不是"模型打分"，是 **BM25 原始分 → 有理函数压缩到 (0,1)**，四步：

1. **归一化 + 分词**：`qa_normalize.py` 去标点/全半角/lower、中文数字转阿拉伯、口语替换（咋→怎么），
   `jieba` 精确模式 + 学科词入词典。如 `绝对值什么意思` → `['绝对值','什么','意思']`。
2. **分学科建索引**：每个 `subject` 一份 `BM25Index`（当前 2000 条语料合并重复问法后：数学 331 条、物理 …），
   IDF 各自独立——否则小科目（地理、生物）的分数会被大科目稀释，一个阈值无法跨学科通用。
3. **BM25 打分**（`qa_corpus.py:53`，`k1=1.5`、`b=0.75`、IDF 取 `log(1+(N-df+0.5)/(df+0.5))` 保证非负）：

   ```
   score = Σ_t idf(t) · (tf·(k1+1)) / (tf + k1·(1-b + b·|d|/avgdl)) · q_weight(t)
   ```

4. **归一化**（`qa_matcher.py:34`）：`s = raw / (raw + 12)`，`FAQ_NORM_K=12`。
   用有理函数而非 min-max，是为了不依赖"先知道语料里的最高分"——语料增删后尺度不漂。

实算一条（数学库 `N=331`、`avgdl=5.08`，查 `绝对值什么意思`，命中 FAQ `什么是绝对值？`，该文档 `|d|=2`）：

| 词 | df | idf | tf | 贡献 |
| --- | --- | --- | --- | --- |
| 绝对值 | 1 | 5.3997 | 1 | 7.4273 |
| 什么 | 229 | 0.3692 | 1 | 0.5079 |
| 意思 | 0 | 0 | 0 | 0（不在语料，不计） |

`raw = 7.9352 → 7.9352/(7.9352+12) = 0.3980`，就是控制台那个分数。
注意 `什么` 出现在 229/331 条问题里（df 高、idf 只有 0.37），全靠 `绝对值` 这个学科词撑分——
这也是为什么"问句框架词"不能进索引区分度：它们在所有问题上都出现。

> 本表由 `scripts/eval_faq.py baseline` 同版本语料实测；语料或分词改动后必须重算，别沿用旧数字
> （2026-10-26 语料改写后，旧版手册里基于"自然数"的例子已整条失效）。

**判定不是只看这个分**（`decide()` 三条 + 路由一条，见 CLAUDE.md §7.1.2 / §7.1.8）：

- `top1 ≥ FAQ_THRESHOLD(0.25)`：绝对分够；
- 候选数 ≥ `FAQ_MIN_CANDIDATES(2)`：只有 1 条候选时"领先第二名"没有意义，直接判 `single_candidate`；
- 分差 ≥ `FAQ_MARGIN(0.10)`：领先第二名够多。上例 top2 是 `0.0423`（`什么是圆周角？圆周角定理是什么？`），
  margin `0.3557` → 命中；query 与 top2 之间没有任何共用实词时，top2 就会落到这种地板分。
  反例（题族）：`角的平分线的性质` 与 `角的平分线` 两条问题的 margin 只有 `0.039`
  （见 `docs/analysis/faq_diagnosis.md` 的 D2 表），任何第三种问法都会被判 `ambiguous` 而交棒 RAG——
  这正是 §7.1.7「语料重复合并」与第四步三档路由要处理的区间。
  分差所在空间由 `FAQ_MARGIN_MODE` 决定：`relative`（默认目标，`(raw1−raw2)/raw1`）/
  `normalized`（现网默认，`score1−score2`）/ `raw`（仅同学科索引内可解释）；
- 学科状态：只有 `subject_source=explicit` 允许直接答。`inferred/context` 看得到该学科索引但
  必须交 RAG 或追问；`invalid` 对外 400、内部全库兜底并带 `degraded`；`empty` 返回
  `subject_empty` 且**不**兜底。返回值里的 `subject_state`（filtered/multi/full/empty）
  就是本次实际检索范围。

**两个 1.0 不是 BM25 分**：精确哈希命中（归一化后整串相等）与别名命中直接给 `score=1.0`，
接口上看到 1.0 就说明"逐字/等价命中"，不是"最相似"。

#### `FAQ_NORM_K=12` 是怎么定的

`s = raw/(raw+K)` 里 K 是**半分点**：`raw = K` 时 `s = 0.5`。它有三个性质：

1. **单调变换**：K 不影响排序，只影响"阈值/分差"这两个人类可读数字的刻度。
   所以 K 不是模型参数，是**刻度因子**。
2. **阈值可换算回原始分**：`raw = K·t/(1−t)`。K=12 时

   | 归一化阈值 | 等价原始分 | 含义 |
   | --- | --- | --- |
   | 0.25（当前） | **4.0** | 专有词命中 + 少量通用词 |
   | 0.50 | 12.0 | 接近整句重合 |
   | 0.62（旧默认值） | **19.58** | 几乎要求逐字相同，等于把 FAQ 废掉 |

   本语料里一条"学科专有词 + 通用词"的好匹配 raw ≈ 6–9（0.33–0.43），
   纯通用词堆出来的伪匹配 raw ≈ 3–4（0.20–0.25）。K=12 让分界落在 4.0，
   即阈值 0.25 —— 数字看着合理，是因为**先按分数分布选了 K，再让阈值落在中间**，
   而不是反过来。
3. **影响分差刻度**：`margin = K(r₁−r₂) / ((r₁+K)(r₂+K))`。K 越大，同样的原始分差被压得越小，
   所以 `FAQ_MARGIN=0.10` 是**与 K=12 绑定**的值，换 K 必须重跑标定（`scripts/eval_faq.py calibrate`）。

诚实地说：K 不是从数据推导出的唯一解，它是让阈值/分差落在可读区间的尺度因子。
旧值 0.62 恰恰是"没换算过 raw 就拍脑袋定的阈值"——它等价于要求 raw ≥ 19.6，
于是 480 条同义问法召回只有 0.069。

## 8. 阈值标定（可复跑，不需要数据库）

```bash
# 0) 语料体检：条数 / 指纹 / 归一化指纹 / 原句自检，三项任一不达标 exit=2
python scripts/eval_faq.py baseline

# 1) 重建样本（正样本口语变体 + 负样本陷阱），旧样本自动归档到 docs/analysis/archive/
python scripts/eval_faq.py gen
python scripts/eval_faq.py negatives

# 2) 二维扫描（threshold × margin，三套 margin 空间各扫一遍）并写报告
python scripts/eval_faq.py calibrate

# 3) 诊断：召回丢在哪一层（阈值/候选数/margin），题族值不值得标题族
python scripts/eval_faq.py diagnose
```

产物：

- `docs/analysis/faq_corpus_baseline.{md,json}`：语料指纹、归一化指纹、原句自检结果
- `docs/analysis/faq_paraphrases.jsonl` / `faq_negatives.jsonl`：正/负样本（带 `faq_id`、类别、来源）
- `docs/analysis/faq_threshold_report.md`：分数分布 + dev 选点 + test 报数 + 邻域切片 + 可达性
- `docs/analysis/faq_diagnosis.md`：D1（召回丢在哪一层）+ D2（题族代理）+ 机制预演

**现行线上参数**：`normalized / 0.25 / 0.10`（`FAQ_MARGIN_MODE` / `FAQ_THRESHOLD` / `FAQ_MARGIN`）。
分数尺度由 `FAQ_NORM_K`（默认 12）决定，改 K、改分词、改同义词表、改语料都要重跑标定。

三轮标定的诚实结论（详见报告与 CLAUDE.md §7.1.7）：

| 轮次 | 改了什么 | test 召回 / 精度 | 过闸 |
| --- | --- | --- | --- |
| 第一轮 | 关键词式短句语料，负样本 840 | 0.078 / 0.808（全网格精度上界 0.897） | 否（精度不可达） |
| 第二轮 | 项目方把 FAQ 问题改写成完整问句；`unrelated` 归零 | 0.382 / 0.824 | 否 |
| 第三轮 | **装载期合并重复问法**（2000 → 1374 进索引）+ 补回 `unrelated` 300 条 | **0.735 / 0.859** | 否（召回过线，精度未到 0.95） |

第三轮是唯一一次**召回过线**（目标 0.637）的改动，靠的是语料装载而不是调参：
同学科 + 答案逐字相同 + 去问句框架词后 Jaccard ≥0.8 的问法并成一条，只留最小 `faq_id` 进 BM25 索引，
其余登记为精确命中别名（原问法仍可直接答）。开关 `FAQ_DEDUPE_NEAR_DUP`（默认 `true`，置 `false` 仅用于对照实验）。
`/v1/faq/stats` 的 `docs` 是载入条数、`indexed` 是实际进索引条数、`merged_duplicates` 是被合并条数——
三者对不上时先看这里。答案不同但问法几乎相同的对**不合并**，日志里打 `faq_near_dup_answer_conflict` 交教研。


