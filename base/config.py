"""集中配置：所有可调参数从环境变量 / .env 读取，业务代码禁止硬编码阈值。

启动自检（见 CLAUDE.md §2.5）：关键项缺失时给出可执行的修复提示，而不是等第一个请求 401。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from base.labels import INTENT_LABELS, SUBJECTS

# 仓库根目录（base/ 的父目录）。.env 与 datasets 都按它定位，
# 不能用相对路径——否则换个工作目录启动（PyCharm 直接 run、定时任务）就读不到配置，
# 表现为 "Access denied ... (using password: NO)"，极难排查。
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    # ；alias 不改变来源。来源的优先级固定。
    # alias 决定“在给定来源里找哪个键名”，不涉及优先级。
    # ---- 应用 ----
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    # 文件比控制台更详细：控制台看结论（INFO），文件留现场（DEBUG：缓存 key/SQL/续命）
    log_file_level: str = Field(default="DEBUG", alias="LOG_FILE_LEVEL")
    http_host: str = Field(default="127.0.0.1", alias="HTTP_HOST")
    http_port: int = Field(default=8000, alias="HTTP_PORT")

    # ---- 日志（落盘 + 轮转，见 CLAUDE.md §5.2）----
    log_dir: str = Field(default="logs", alias="LOG_DIR")
    log_file: str = Field(default="app.log", alias="LOG_FILE")
    log_max_bytes: int = Field(default=50 * 1024 * 1024, alias="LOG_MAX_BYTES")
    log_backup_count: int = Field(default=10, alias="LOG_BACKUP_COUNT")
    # 是否在日志里打原始 query。生产建议 false：query 可能含学生隐私（姓名/手机号）
    log_query_text: bool = Field(default=False, alias="LOG_QUERY_TEXT")

    # ---- MySQL ----
    mysql_host: str = Field(default="127.0.0.1", alias="MYSQL_HOST")
    mysql_port: int = Field(default=3306, alias="MYSQL_PORT")
    mysql_user: str = Field(default="root", alias="MYSQL_USER")
    mysql_password: str = Field(default="", alias="MYSQL_PASSWORD")
    mysql_database: str = Field(default="itcast", alias="MYSQL_DATABASE")
    mysql_pool_max: int = Field(default=16, alias="MYSQL_POOL_MAX")

    # ---- Redis ----
    redis_host: str = Field(default="127.0.0.1", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="1234", alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="REDIS_DB")

    # ---- qa_system 参数（CLAUDE.md §7.1.2）----
    faq_threshold: float = Field(default=0.25, alias="FAQ_THRESHOLD")
    # margin 的数值含义随 margin_mode 变：relative=(raw1-raw2)/raw1，normalized=score1-score2，raw=raw1-raw2。
    # 重标定完成前保持 normalized（现网基线）；切 relative 必须重扫并写报告（§7.1.7 第三步）
    faq_margin: float = Field(default=0.10, alias="FAQ_MARGIN")
    faq_margin_mode: str = Field(default="normalized", alias="FAQ_MARGIN_MODE")
    # 三档路由的高置信下限（CLAUDE.md §7.1.9）：与 FAQ_MARGIN **同一 margin_mode 空间**解释。
    # 过 ①②③ 且 margin ≥ 它才是 high（唯一允许直接作答的一档）；[FAQ_MARGIN, 它) 是 mid，
    # 只交 RAG + 追问，不返回答案。设成与 FAQ_MARGIN 相等即精确回退为单档——灰度和回滚都靠它，不改代码。
    faq_high_margin: float = Field(default=0.22, alias="FAQ_HIGH_MARGIN")
    # 候选数下限：只有 1 条候选时"领先第二名"不成立，直接不算命中（§7.1.2 判据 ②）
    faq_min_candidates: int = Field(default=2, alias="FAQ_MIN_CANDIDATES")
    # 多学科场景：主学科 best 与另一学科 best 的归一化分差下限（跨学科只比归一化分）
    faq_cross_subject_delta: float = Field(default=0.05, alias="FAQ_CROSS_SUBJECT_DELTA")
    faq_top_k: int = Field(default=20, alias="FAQ_TOP_K")
    faq_norm_k: float = Field(default=12.0, alias="FAQ_NORM_K")
    faq_candidate_n: int = Field(default=3, alias="FAQ_CANDIDATE_N")
    faq_alias_enabled: bool = Field(default=True, alias="FAQ_ALIAS_ENABLED")
    # 语料重复合并（§7.1.7 语料重复合并契约）：同学科 + 答案逐字相同 + 去问句框架词后 Jaccard ≥0.8 的 FAQ
    # 并成一条，只留最小 faq_id 进 BM25 索引，其余登记为精确命中别名。
    # "什么是X" 与 "X是什么" 的 BM25 分数一分不差，任何同义问法都会撞出 margin 0、两条都答不了。
    # 置 false 只用于对照实验，生产保持 true。
    faq_dedupe_near_dup: bool = Field(default=True, alias="FAQ_DEDUPE_NEAR_DUP")
    faq_admin_token: str = Field(default="", alias="FAQ_ADMIN_TOKEN")
    # 本地无数据库时从 CSV 装载语料（相对仓库根目录）；生产留空，走 MySQL 真相源
    faq_corpus_csv: str = Field(default="", alias="FAQ_CORPUS_CSV")
    # -----model意图识别模型路径
    MODELS_DIR:Path=Field(default="",alias="MODELS_DIR")
    # ---- 启动行为 ----
    # dev 下允许数据库不可用时以"空 FAQ 语料"启动，便于纯本地跑通链路；生产必须关闭
    allow_degraded_start: bool = Field(default=True, alias="ALLOW_DEGRADED_START")

    # ---- 文档与切分（CLAUDE.md §7.2 知识库）----
    # 待入库的原始资料目录：相对仓库根目录，逗号分隔可写多个。
    # 2026-10-29 起语料换成按 `阶段/学科/年级/` 组织的一体化目录（教材/教案/试卷/题库/复习卷
    # 都在这一个根下，`doc_type` 由文件名推断），旧的 `01_textbook` 等平铺目录不再默认入库。
    rag_docs_dir: str = Field(
        default="datasets/初高中教材教案试卷-按年级",
        alias="RAG_DOCS_DIR",
    )
    # 父块：语义完整、随答案一起进 prompt；子块：只用来被检索命中
    rag_parent_size: int = Field(default=1200, alias="RAG_PARENT_SIZE")
    rag_parent_overlap: int = Field(default=100, alias="RAG_PARENT_OVERLAP")
    rag_child_size: int = Field(default=300, alias="RAG_CHILD_SIZE")
    rag_child_overlap: int = Field(default=20, alias="RAG_CHILD_OVERLAP")
    # 单篇文档最多产出多少子块，防止一份畸形文件把库撑爆。
    # 2026-10-29 由 2000 上调：一份学科题库（`题库.json`，渲染后 78 万字符）按 300 字子块切
    # 就有 2600 块，2000 会把最后 1/4 的题目**静默截断**（只留一条 `ingest_truncated` 日志）。
    # 这是个"防炸"阈值，不是质量旋钮，定低了就是在删语料。
    rag_max_chunks_per_doc: int = Field(default=8000, alias="RAG_MAX_CHUNKS_PER_DOC")
    # PDF 解析器：auto=有 MINERU_TOKEN 走云解析、失败自动退本地；mineru=只用云；local=只用 PyMuPDF
    rag_pdf_parser: str = Field(default="auto", alias="RAG_PDF_PARSER")
    # Office/图片里的内嵌图是否 OCR：慢且贵，默认关闭（教材 PDF 走 MinerU 自带版面解析）
    rag_ocr_images: bool = Field(default=False, alias="RAG_OCR_IMAGES")

    # ---- 近重复文档检测（CLAUDE.md §7.2.6「未做」项的落地）----
    # 同一本教材的两次下载会让同一段证据以"两篇文档"的身份互为对手，
    # 证据门的跨文档 margin 直接判 ambiguous。`datasets/` 只读，只能在摄入侧拒绝。
    rag_dedup_enabled: bool = Field(default=True, alias="RAG_DEDUP_ENABLED")
    # 64 位 simhash 的汉明距离上限：3 能拦住"换页眉/换标点"的同一份文档，
    # 又不会把同一学科的两本不同教材判成重复（实测不同教材距离普遍 >12）
    rag_dedup_distance: int = Field(default=3, alias="RAG_DEDUP_DISTANCE")
    # 取正文前多少个字符算指纹：两次下载的差异常在页眉水印上，取开头即可覆盖
    rag_dedup_prefix_chars: int = Field(default=8000, alias="RAG_DEDUP_PREFIX_CHARS")

    # ---- 在线检索与重排 ----
    # 关闭后 /v1/rag/* 直接 503，编排层只走 FAQ（本地没有 Milvus 时用）
    rag_enabled: bool = Field(default=True, alias="RAG_ENABLED")
    # 每路 ANN 召回条数：融合后按父块去重，实际留存的父块远少于这个数
    rag_top_k: int = Field(default=40, alias="RAG_TOP_K")
    # 最终进 prompt 的父块数（证据条数）
    rag_final_n: int = Field(default=6, alias="RAG_FINAL_N")
    # 重排开关：auto=只在有 CUDA 时启用（CPU 上重排 40 条要上百秒，见 2026-10-20 实测）
    rag_rerank_enabled: str = Field(default="auto", alias="RAG_RERANK_ENABLED")
    # 只对前 N 条候选重排：CPU 上重排 80 条要数秒，GPU 上 200ms 内可跑完 40 条
    rag_rerank_candidates: int = Field(default=40, alias="RAG_RERANK_CANDIDATES")
    # 送入重排的父块截断字符数，约等于 512 token
    rag_rerank_chars: int = Field(default=2000, alias="RAG_RERANK_CHARS")
    rag_rerank_max_length: int = Field(default=512, alias="RAG_RERANK_MAX_LENGTH")
    # 权重放在**本仓库**的 `rag_system/models/`（2026-10-29 从上级旧仓库 `../Ragpro` 迁出）。
    # 指向别的仓库时，交付机上没有那份目录，`build_embedder` 只会 warn 一句然后**静默退成
    # HashEmbedder**——检索质量断崖式下降而服务照常 200，是最难发现的一类事故。
    rag_reranker_path: str = Field(
        default="rag_system/models/BAAI/bge-reranker-v2-m3", alias="RAG_RERANKER_PATH"
    )
    # 重排模型用 fp16 需要 GPU；CPU 上 fp16 会直接报错
    rag_use_fp16: bool = Field(default=True, alias="RAG_USE_FP16")
    # 证据门（CLAUDE.md §7.2.1 第⑥步）：低于门槛不生成，宁可转人工
    # 默认值来自 2026-10-20 黄金集标定（datasets/07_eval 763 条可判定样本），
    # 取舍点为「误放行 ≤5%」：0.74 / 0.01 / 0.20 → 可答题放行 34%，不可答题误放 3%。
    # 想多答就放松到 0.72，想更稳就收紧到 0.76（见 docs/analysis/rag_threshold_report.md）。
    rag_evidence_threshold: float = Field(default=0.74, alias="RAG_EVIDENCE_THRESHOLD")
    # 跨文档分差（同文档多块不算歧义）；0 = 关闭歧义判定
    rag_evidence_margin: float = Field(default=0.01, alias="RAG_EVIDENCE_MARGIN")
    rag_evidence_min_docs: int = Field(default=1, alias="RAG_EVIDENCE_MIN_DOCS")
    # 问句实词在 top1 证据里的最低覆盖率：挡"同主题但没答这个问题"的块
    rag_evidence_qcov: float = Field(default=0.20, alias="RAG_EVIDENCE_QCOV")

    # ---- 检索边界（§7.2.7-A）----
    # 安全三项一律以 Milvus 标量过滤下推，**不允许**在检索后用 Python 过滤（§5.6 第 5 条），
    # 也**不允许**被"放宽业务过滤"的修复动作清掉（`RetrievalFilters.relaxed()` 只清
    # subjects / doc_types）。改了 `KB_VERSION` 等于换一代索引，缓存 key 随之自然失效。
    kb_version: str = Field(default="v1", alias="KB_VERSION")
    tenant_id: str = Field(default="default", alias="TENANT_ID")

    # ---- 生成层：LLM 网关（CLAUDE.md §7.8）----
    # 走 OpenAI 兼容协议，换供应商只改 base_url / model，不改代码。
    # 为空时**不阻断启动**：/v1/chat 走兜底话术，FAQ 与 RAG 照常；生产用 ALLOW_DEGRADED_START=false 兜住。
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="https://api.deepseek.com", alias="LLM_BASE_URL")
    llm_qa_model: str = Field(default="deepseek-chat", alias="LLM_QA_MODEL")
    # 首 token / 连接失败时换这个模型（空 = 不换）。换模型算降级，不算重试（§5.3）
    llm_fallback_model: str = Field(default="", alias="LLM_FALLBACK_MODEL")
    # 路由/改写模型留给 PR5，先占位，避免以后再改一遍环境变量
    llm_router_model: str = Field(default="deepseek-chat", alias="LLM_ROUTER_MODEL")
    llm_timeout_s: int = Field(default=60, alias="LLM_TIMEOUT_S")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    # 网关级有限重试：**只覆盖传输层故障**（超时 / 连接中断 / HTTP 408 / 429）。
    # 这与 §5.3「生成节点不重试」不冲突，两者管的是不同层级：
    #   - 节点级不重试 = 不把整条检索 + 生成链再跑一遍（那会翻倍吃预算、翻倍计费）；
    #   - 网关级有限重试 = 一个字都还没生成前的**同一次请求**重发，最多 `LLM_MAX_RETRIES` 次。
    # 4xx（含 401/403）与 5xx 不在这里重试：前者重试也不会好，后者直接换模型更划算（§7.8）。
    llm_max_retries: int = Field(default=1, alias="LLM_MAX_RETRIES")

    # ---- 生成层：本地小模型（闲聊档专用，§7.7）----
    # 闲聊既不需要知识库证据，也不需要大模型的推理能力：一句"你好"付一次云端往返与计费不划算。
    # 这里单独给一条本地端点（Ollama 的 OpenAI 兼容层），**与主网关完全独立**——
    # 端点 / 模型 / key / 超时各自一份。不能靠 `model=` 参数复用一个网关：网关的
    # `base_url`、key、`enabled`、超时全部绑在 `Settings` 上，换模型仍然打到云端 base_url、
    # 发云端 key，而没有 cloud key 的本地端点会被 `enabled` 判成"永久不可用"。
    llm_local_base_url: str = Field(
        default="http://127.0.0.1:11434/v1", alias="LLM_LOCAL_BASE_URL"
    )
    llm_local_model: str = Field(default="qwen3.5:0.8b", alias="LLM_LOCAL_MODEL")
    # 本地 Ollama 不校验 key；留空时不发 Authorization 头（空 Bearer 反而可能被某些代理拒掉）
    llm_local_api_key: str = Field(default="", alias="LLM_LOCAL_API_KEY")
    # 比云端短：本地小模型要么很快，要么就是卡住了，等 60 s 没有意义
    llm_local_timeout_s: int = Field(default=30, alias="LLM_LOCAL_TIMEOUT_S")
    # qwen3.5:0.8b 是**思考型**模型：默认把 token 全花在 reasoning 上、`content` 恒为空
    # （本机实测 max_tokens=1024 时 `finish_reason=length` / reasoning 4302 字 / content=''），
    # 直接接进来每一轮闲聊都会被 `_extract_text` 判成"LLM 返回空答案"而走兜底。
    # 实测只有 `reasoning_effort: "none"` 被 `/v1/chat/completions` 认（`think:false` 与
    # `chat_template_kwargs.enable_thinking` 都被静默忽略）。置空 = 不发这个字段。
    llm_local_reasoning_effort: str = Field(
        default="none", alias="LLM_LOCAL_REASONING_EFFORT"
    )

    # ---- 编排层（/v1/chat）----
    # 单请求总预算：生成拿剩余部分，超了截断/兜底（§6.1）
    total_budget_ms: int = Field(default=8000, alias="TOTAL_BUDGET_MS")
    # 句级引用支持度下限：句子实词被所引证据覆盖的比例低于它就算"无支撑句"（§7.8）
    citation_min_support: float = Field(default=0.30, alias="CITATION_MIN_SUPPORT")
    # 兜底话术里的联系方式；留空就只说"转人工"，不要编造电话号码
    chat_support_contact: str = Field(default="", alias="CHAT_SUPPORT_CONTACT")
    # 意图档位开关（§5.3 每个节点都能单独关掉）。关掉等于 §7.7 A/B 里的"全量 DIRECT 基线"：
    # 不按 doc_type 收窄、不跳过检索，用于灰度与回滚，不改代码。
    feature_intent: bool = Field(default=True, alias="FEATURE_INTENT")
    # 意图识别改用大模型（§7.7）：与查询改写**合并成一次调用**，位置在 FAQ 未命中之后。
    # 关掉回落到 `understanding/intent.py` 的规则分类器——规则版是 §7.8.4 A/B 的基线，
    # 两条路径的 F1 不可互换，切换时要重新量。
    feature_intent_llm: bool = Field(default=True, alias="FEATURE_INTENT_LLM")
    # BERT 精调分类器做**初始**意图（任务 2.2，CLAUDE.md §7.7）：
    # 置信度够就直接定档，一次大模型调用都不发；不够才把意图交给大模型辅助判断。
    # 关掉（或检查点缺失 / 标签映射对不上）时整条链路退回"大模型 → 规则"，
    # 与加这个开关之前**逐字节一致**——这是它作为回滚开关的含义。
    feature_intent_bert: bool = Field(default=True, alias="FEATURE_INTENT_BERT")
    # BERT 推理的硬上限：CPU 上单条 forward 是毫秒级，超过 200 ms 说明线程池被占满
    # （torch 在跑别的活），这时宁可放弃这条结论让大模型接手，也别拖住首 token
    intent_bert_timeout_ms: int = Field(default=200, alias="INTENT_BERT_TIMEOUT_MS")

    # 大模型查询改写。`false` 时只保留确定性归一化（NFKC/空白/标点），不花任何一次调用
    feature_query_rewrite: bool = Field(default=True, alias="FEATURE_QUERY_REWRITE")
    # 年级/学段下推成 Milvus 标量过滤（§7.2.7-E）。年级误抽（"八年级的题高一还会考吗"）
    # 会把高一文档挡在外面，一键回滚用这个开关
    feature_grade_filter: bool = Field(default=True, alias="FEATURE_GRADE_FILTER")
    # HyDE（阶段 4.3）：短且抽象的问句先让模型写一段"假设性答案"，再拿它去检索。
    # **2026-11-01 默认打开**（用户决定：HyDE 交给 DeepSeek 做）。如实记录两件事：
    # ① 它会多付**一次生成调用**，正好压在首 token 预算里（§6.1 的 1.2 s SLO 尚未重测）；
    # ② "打开后召回变好"仍然**没有任何评估数据**——§5.4 要求的黄金集对照报告没跑过，
    #    所以这是一个按需求打开的实验性能力，不是已验证的优化。回滚开关就是这一行。
    # 无论开关如何，HyDE 文本都不会进证据集合（`rag_pipeline._drop_hyde_echo`），
    # 它只影响"拿哪句话去检索"
    feature_hyde: bool = Field(default=True, alias="FEATURE_HYDE")
    # HyDE 的独立预算：它自己就是一次生成调用，必须单独上闸。
    # 不能拿 `LLM_TIMEOUT_S`（60 s）当它的上限——那等于没有上限，一次慢调用会把
    # 整个 `TOTAL_BUDGET_MS`（8 s）吃穿，而 HyDE 只是"试试看能不能查得更准"
    hyde_budget_ms: int = Field(default=1500, alias="HYDE_BUDGET_MS")
    # 复合问句拆分走**本地小模型**（任务 2.4，2026-11-01）。本地端点见 `llm_local_*`：
    # 不计费、不出网，所以它可以默认打开；关掉 = 纯规则拆分，与接模型之前逐字节一致，
    # 这既是回滚开关，也是"模型拆分到底有没有用"的 A/B 分组
    feature_compound_llm: bool = Field(default=True, alias="FEATURE_COMPOUND_LLM")
    # 拆分的预算上限。它比理解层更短：这一步只是把逗号分段重新组织成问句，
    # 0.8B 的本地模型在室温下几十毫秒就该出结果；到点就退回规则，不让它拖住首 token
    compound_timeout_ms: int = Field(default=400, alias="COMPOUND_TIMEOUT_MS")
    # 槽位的"模型补空"（任务 2.3）：只有章节名 / 知识点 / 任务参数这几项走大模型，
    # 学科 / 年级 / 教材版本 / 学期**永远是规则**（§7.1.8 的安全不对称：模型抽错是
    # "正确答案被挡在索引外"，规则抽不到只是"全库多查一轮"）。
    # 关掉等于只保留规则抽到的槽位——这是槽位链路的回滚开关
    feature_slot_llm: bool = Field(default=True, alias="FEATURE_SLOT_LLM")
    # 必要槽位缺失时先追问一句再干活（教案制作缺课题、出题缺学科…见
    # `understanding/slots.py::clarification()`）。**默认关闭**：任务 2.3 的要求是
    # "只有影响任务执行的必要信息才触发澄清"，而"多拦一句话"的口子一旦默认开，
    # 误拦一次用户就走了；先按关闭发布，用真实流量判断该不该开
    feature_slot_clarify: bool = Field(default=False, alias="FEATURE_SLOT_CLARIFY")
    # 检索策略交给大模型选（`direct` / `sub_query` / `hyde`，2026-11-01）。
    # **只换"选哪种检索方式"这一件事**：`no_retrieval` 与 `clarify` 两条决策仍由规则判，
    # 因为它们编码的是结构事实（这个意图根本不检索 / 指代解不开），不是偏好——
    # 让模型去判"要不要检索"等于把"知识库里有没有"变成一次采样。
    # 关了就是**纯规则规划**，与加这个开关之前逐字节一致：这是唯一的回滚手段，
    # 所以模型超时 / 不给 / 给了跑不了的建议，都得能退回规则那条路径
    feature_planner_llm: bool = Field(default=True, alias="FEATURE_PLANNER_LLM")
    # 归一化 + 一次大模型理解（改写 + 意图）的预算上限。它和生成抢的是同一个
    # `TOTAL_BUDGET_MS`，所以必须小于总预算（`self_check()` 会拦）
    understanding_timeout_ms: int = Field(default=800, alias="UNDERSTANDING_TIMEOUT_MS")
    # 中置信 FAQ 的 top3（问题 + 答案）作为**辅助上下文**进生成 prompt（§7.1.9 边界 2，
    # 2026-10-31 修订）。它**不是证据**：不编号、不进【资料】、不进 citations，
    # `FAQHit.answer` 在中置信档仍是 None。关掉 = 回到"中置信原文一个字都不进生成器"，
    # 这是本改动的回滚开关（§5.3），也是一次干净的 A/B 分组
    feature_faq_candidates: bool = Field(default=True, alias="FEATURE_FAQ_CANDIDATES")
    # 业务权限（阶段 3）：角色 × 意图的判定在 `orchestration/admission.py::permissions_for()`，
    # 结论（`PermissionDecision`）随 `RequestContext` 一路带到编排层，**在 FAQ 之前**生效。
    # 置 `false` = 空拒绝集，等于整条门禁不存在——这是紧急回滚开关，与 `FEATURE_INTENT`
    # 是同一个角色。**默认开**：权限属于"不做会出事"的那一类，不能靠别人记得打开
    feature_permission: bool = Field(default=True, alias="FEATURE_PERMISSION")

    # ---- 接入层（阶段 1，docs/EduRAG-Pro-阶段1-请求上下文与入口设计.md §3.5）----
    # 业务硬限：`admit()` 按它拒 400 `query_too_long`。**唯一出处**——
    # `rag_system/core/query_processor` 不再自带常量、也不再静默截断（同一次请求里
    # 出现两种长度的问句，FAQ 精确哈希与 qcov 分母会各指向一份，排障时对不上）
    chat_max_query_chars: int = Field(default=400, alias="CHAT_MAX_QUERY_CHARS")
    # 请求体字节上限。它是**流量层**的粗筛（不需要解析 JSON 就能拒），
    # 与 `chat_max_query_chars` 分工不同：后者是业务口径（中文字符数），
    # 前者挡的是"把 200 MB 的 body 读进内存再校验"——那一步做完就已经被打了。
    # 只按 Content-Length 判，宁可漏拦也不留半截 body 在内存里。
    max_request_bytes: int = Field(default=65536, alias="MAX_REQUEST_BYTES")
    # JWT（HS256）验签密钥。**空 = 不启用 JWT**：此时只有白名单环境才认 `X-User-Id`
    # 一类的开发期请求头，生产环境一律匿名。空值**不是**"随便签一个"的邀请，
    # 生产漏配的表现是"所有请求都成了 ANONYMOUS"，而不是"所有人都能冒充"
    jwt_secret: str = Field(default="", alias="JWT_SECRET")
    # 签发时用的过期时长。**认证服务自己不用它**（改用下面 `AUTH_*_TTL_S`，
    # 单位是秒、access 与 refresh 分开），保留是因为 tools/网关注入的令牌仍按它算。
    jwt_expire_min: int = Field(default=120, alias="JWT_EXPIRE_MIN")
    # 验签的时钟漂移容差。`exp` / `nbf` / `iat` 三个时间断言都按它放宽：
    # 多机部署里各节点时钟差几十秒是常态，而"过了 3 秒就 401"会表现为
    # 随机一批用户间歇性掉登录，最难复现的那种故障。放宽的代价是过期令牌
    # 多活 60 秒，远小于它带来的稳定性收益。
    jwt_leeway_s: int = Field(default=60, alias="JWT_LEEWAY_S")
    # ---- 认证接口（`auth/`，用户 2026-10-31 定案）----
    # access token 寿命。**600 秒替换掉接口文档示例里的 3600**：登出只作废 refresh
    # （JWT 无状态，access 不落库，见 `auth/store.py` 模块注释），
    # 所以"登出后多久真正失效"完全由这个数字决定。调大 = 延长"已登出但仍可用"的窗口，
    # 调小 = 前端刷新更频繁。10 分钟是两者之间用户拍板的口径。
    auth_access_ttl_s: int = Field(default=600, alias="AUTH_ACCESS_TTL_S")
    # refresh 寿命：默认 7 天；勾选「记住我」时 30 天
    auth_refresh_ttl_s: int = Field(default=604800, alias="AUTH_REFRESH_TTL_S")
    auth_refresh_ttl_remember_s: int = Field(default=2592000, alias="AUTH_REFRESH_TTL_REMEMBER_S")

    # ---- 会话与记忆（CLAUDE.md §7.9，对标 §7.3 的裁剪版）----
    # 会话 L0 窗口双上限：40 轮或 28 000 token，先到先压（§3.7 第 2 条）
    mem_window_turns: int = Field(default=40, alias="MEM_WINDOW_TURNS")
    mem_window_tokens: int = Field(default=28000, alias="MEM_WINDOW_TOKENS")
    # 每次压缩吃掉最旧的 10 轮，滚动进摘要
    mem_compress_batch_turns: int = Field(default=10, alias="MEM_COMPRESS_BATCH_TURNS")
    mem_summary_max_tokens: int = Field(default=400, alias="MEM_SUMMARY_MAX_TOKENS")
    # 原子记忆：召回条数、置信度门槛、合并阈值、单用户活跃上限
    mem_recall_top_k: int = Field(default=5, alias="MEM_RECALL_TOP_K")
    mem_min_confidence: float = Field(default=0.6, alias="MEM_MIN_CONFIDENCE")
    mem_dedupe_sim: float = Field(default=0.92, alias="MEM_DEDUPE_SIM")
    mem_atom_limit_per_user: int = Field(default=500, alias="MEM_ATOM_LIMIT_PER_USER")
    # 记忆段注入 prompt 的 token 预算：画像 300 / 摘要 400 / 原子 300（§3.7 第 3 条）
    mem_profile_max_tokens: int = Field(default=300, alias="MEM_PROFILE_MAX_TOKENS")
    mem_atom_max_tokens: int = Field(default=300, alias="MEM_ATOM_MAX_TOKENS")
    # 记忆召回超时（§3.7 第 8 条）：到点就返回空记忆 + degraded，主链路继续跑检索
    mem_recall_budget_ms: int = Field(default=120, alias="MEM_RECALL_BUDGET_MS")
    # 会话数据在 Redis 里的留存时间；Redis 不可用时退化为进程内（重启即丢）
    mem_session_ttl_s: int = Field(default=86400, alias="MEM_SESSION_TTL_S")
    # 物理清理窗口（§3.7 第 5 条：软删 30 天后物理清理）。`scripts/purge_memory.py` 用它，
    # 且**只由它一个旋钮**决定三张表（memory_atom / chat_message / session_summary）的窗口：
    # 拆成三个变量后，"到底哪一个是留存策略"就没人说得清了。要分开必须连文档一起改。
    mem_purge_after_days: int = Field(default=30, alias="MEM_PURGE_AFTER_DAYS")
    # 单条 DELETE 的行数上限。分批删是为了不长时间持锁：一次删十万行会把写入全卡住
    mem_purge_batch: int = Field(default=2000, alias="MEM_PURGE_BATCH")
    # 总开关：关掉等于"无会话、无记忆"的老行为，用于灰度与排障
    feature_memory: bool = Field(default=True, alias="FEATURE_MEMORY")
    # L2 原子记忆的向量召回（§3.7 第 8 条）。关掉 / RAG 未就绪时退回字面 bigram 排序，
    # 这是**功能降级不是故障**：召回顺序变差，回答照常。
    feature_memory_vector: bool = Field(default=True, alias="FEATURE_MEMORY_VECTOR")

    # ---- 向量库 Milvus（CLAUDE.md §8.1）----
    milvus_uri: str = Field(default="http://127.0.0.1:19541", alias="MILVUS_URI")
    milvus_database: str = Field(default="edurag", alias="MILVUS_DATABASE")
    milvus_collection: str = Field(default="edurag_chunk", alias="MILVUS_COLLECTION")
    # 原子记忆的向量集合（§3.7 第 8 条）。**与知识库集合分开**：一个是可重建的索引，
    # 一个是用户数据（重建知识库时不该把记忆一起 drop 掉），且两边的过滤条件完全不同。
    milvus_memory_collection: str = Field(
        default="edurag_memory", alias="MILVUS_MEMORY_COLLECTION"
    )
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")
    # BGE-M3 输出 1024 维；改模型必须同步改这里，否则 upsert 直接报维度不匹配
    milvus_dim: int = Field(default=1024, alias="MILVUS_DIM")

    # ---- 向量化 ----
    # auto：有 FlagEmbedding+模型就用真模型，否则用确定性哈希向量（只给测试/离线跑通用）
    rag_embed_backend: str = Field(default="auto", alias="RAG_EMBED_BACKEND")
    rag_embed_model_path: str = Field(
        default="rag_system/models/BAAI/bge-m3", alias="RAG_EMBED_MODEL_PATH"
    )
    # cuda / cpu / auto
    rag_embed_device: str = Field(default="auto", alias="RAG_EMBED_DEVICE")
    rag_embed_fp16: bool = Field(default=True, alias="RAG_EMBED_FP16")
    rag_embed_batch_size: int = Field(default=8, alias="RAG_EMBED_BATCH_SIZE")
    rag_embed_max_length: int = Field(default=512, alias="RAG_EMBED_MAX_LENGTH")

    # ---- MinerU 云解析（PDF 专用，CLAUDE.md §7.2.1）----
    mineru_token: str = Field(default="", alias="MINERU_TOKEN")
    mineru_base: str = Field(default="https://mineru.net/api/v4", alias="MINERU_BASE")
    mineru_model: str = Field(default="vlm", alias="MINERU_MODEL")
    mineru_poll_interval: int = Field(default=5, alias="MINERU_POLL_INTERVAL")
    mineru_timeout_s: int = Field(default=900, alias="MINERU_TIMEOUT_S")
    # 上传超时单独一根：与轮询超时共用一个数会让 60 MB 的教材在慢上行链路上必然写超时
    # （2026-10-29 实测 `WriteTimeout: The write operation timed out` → 退本地 PyMuPDF）
    mineru_upload_timeout_s: int = Field(default=900, alias="MINERU_UPLOAD_TIMEOUT_S")
    # 解析结果缓存目录：同一份 PDF 只花一次额度（相对仓库根目录）
    mineru_cache_dir: str = Field(default="data/mineru", alias="MINERU_CACHE_DIR")

    # ---- FAQ 二级缓存（Redis，见 CLAUDE.md §7.1.2）----
    faq_cache_enabled: bool = Field(default=True, alias="FAQ_CACHE_ENABLED")
    # 命中后写入 Redis 的热数据存活时间（滑动续命），单位秒
    faq_cache_ttl: int = Field(default=3600, alias="FAQ_CACHE_TTL")
    # 单条热数据的绝对寿命上限：滑动续命不能无限延长，否则改了答案也换不掉
    faq_cache_max_ttl: int = Field(default=6 * 3600, alias="FAQ_CACHE_MAX_TTL")
    # 剩余寿命低于该比例时才续命，避免每次命中都写一次 Redis
    faq_cache_slide_ratio: float = Field(default=0.5, alias="FAQ_CACHE_SLIDE_RATIO")
    # 未命中的短缓存：挡住热门脏查询的重复打分，版本变更即失效
    faq_cache_negative_ttl: int = Field(default=30, alias="FAQ_CACHE_NEGATIVE_TTL")

    @property
    def log_path(self) -> Path:
        return BASE_DIR / self.log_dir / self.log_file

    @property
    def rag_docs_paths(self) -> list[Path]:
        """RAG_DOCS_DIR 允许逗号分隔多个目录，统一解析成绝对路径。"""
        items = [p.strip() for p in self.rag_docs_dir.split(",") if p.strip()]
        return [(BASE_DIR / p).resolve() for p in items]

    @property
    def embed_model_path(self) -> Path:
        p = Path(self.rag_embed_model_path).expanduser()
        return p if p.is_absolute() else (BASE_DIR / p).resolve()

    @property
    def reranker_model_path(self) -> Path:
        """与 `embed_model_path` 同一口径：相对路径一律按仓库根解析。

        不能直接 `Path(settings.rag_reranker_path)`——PyCharm 直接 run、计划任务、
        `python scripts/xxx.py` 三种启动方式的工作目录各不相同，相对路径会各指一处。
        """
        p = Path(self.rag_reranker_path).expanduser()
        return p if p.is_absolute() else (BASE_DIR / p).resolve()

    @property
    def mineru_cache_path(self) -> Path:
        return (BASE_DIR / self.mineru_cache_dir).resolve()

    @property
    def rerank_enabled(self) -> bool:
        """把 `RAG_RERANK_ENABLED=auto|true|false` 解析成布尔。

        默认 auto：**只有 CUDA 可用时才开重排**。实测 CPU 上 40 条候选要 115 s
        （bge-reranker-v2-m3），比整条链路其它步骤加起来还慢两个数量级，
        对在线请求不可接受；关掉后按融合分排序，质量降一点但 1–4 s 内返回。
        """
        value = str(self.rag_rerank_enabled).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return _cuda_available()

    @property
    def intent_labels(self) -> tuple[str, ...]:
        return INTENT_LABELS

    @property
    def llm_enabled(self) -> bool:
        """有 key 才允许调生成；没有 key 时 chat 走兜底，FAQ / RAG 不受影响。"""
        return bool(self.llm_api_key.strip())

    @property
    def llm_endpoint(self) -> str:
        """OpenAI 兼容的补全地址。base_url 允许带或不带结尾斜杠。"""
        return self.llm_base_url.rstrip("/") + "/chat/completions"

    @property
    def llm_local_endpoint(self) -> str:
        """本地端点的补全地址（Ollama 的 OpenAI 兼容层 `/v1`）。"""
        return self.llm_local_base_url.rstrip("/") + "/chat/completions"

    @property
    def llm_local_enabled(self) -> bool:
        """本地端点是否配置齐全。

        **刻意不看 `LLM_API_KEY`**：本地 Ollama 不需要 key，把云端 key 当开关会让
        "没配云 key 的开发机"连闲聊都走不了本地——而这恰恰是最需要它的场景。
        """
        return bool(self.llm_local_base_url.strip() and self.llm_local_model.strip())

    @property
    def subjects(self) -> tuple[str, ...]:
        return SUBJECTS

    def self_check(self) -> list[str]:
        """返回阻断启动的问题列表；空列表表示可以启动。"""
        problems: list[str] = []
        if not self.mysql_password:
            problems.append("MYSQL_PASSWORD 为空：请在 .env 里填入 MySQL 密码（compose 默认 123456）")
        if self.faq_threshold <= 0 or self.faq_threshold >= 1:
            problems.append(f"FAQ_THRESHOLD={self.faq_threshold} 必须在 (0,1) 之间")
        if self.faq_margin < 0:
            problems.append("FAQ_MARGIN 不能为负")
        if self.faq_margin_mode not in ("relative", "normalized", "raw"):
            problems.append(f"FAQ_MARGIN_MODE={self.faq_margin_mode} 必须是 relative/normalized/raw")
        # 三档是"中置信不许直接答"，高置信线低于命中线会让 mid 区间为空、配置自相矛盾
        if self.faq_high_margin < self.faq_margin:
            problems.append(
                f"FAQ_HIGH_MARGIN={self.faq_high_margin} 不能小于 FAQ_MARGIN={self.faq_margin}"
                "（两者同一 margin_mode 空间；相等即回退为单档）"
            )
        if self.faq_min_candidates < 1:
            problems.append("FAQ_MIN_CANDIDATES 必须 ≥1")
        if self.faq_top_k <= 0:
            problems.append("FAQ_TOP_K 必须为正")
        if self.llm_timeout_s <= 0:
            problems.append("LLM_TIMEOUT_S 必须为正")
        if self.llm_local_timeout_s <= 0:
            problems.append("LLM_LOCAL_TIMEOUT_S 必须为正")
        if self.llm_max_tokens <= 0:
            problems.append("LLM_MAX_TOKENS 必须为正")
        if not 0 <= self.llm_temperature <= 2:
            problems.append(f"LLM_TEMPERATURE={self.llm_temperature} 必须在 [0,2] 之间")
        # 上界是"别把一次超时放大成一串超时"：3 次以上重试一定吃掉整个生成预算
        if not 0 <= self.llm_max_retries <= 3:
            problems.append(f"LLM_MAX_RETRIES={self.llm_max_retries} 必须在 [0,3] 之间")
        if self.total_budget_ms <= 0:
            problems.append("TOTAL_BUDGET_MS 必须为正")
        if not 0 <= self.citation_min_support <= 1:
            problems.append("CITATION_MIN_SUPPORT 必须在 [0,1] 之间")
        # 理解层预算必须**严格小于**总预算：相等时生成只剩 0 ms，等于每条请求都截断
        if not 0 < self.understanding_timeout_ms < self.total_budget_ms:
            problems.append(
                f"UNDERSTANDING_TIMEOUT_MS={self.understanding_timeout_ms} 必须在 (0, {self.total_budget_ms}) 之间"
            )
        # HyDE 与拆句各自又是**两笔**独立开销，与理解层一起压在首 token 预算上。
        # 这里只保证单项没写反（0 或负数等于"每次都超时"，比总预算还大等于没有闸门）；
        # 三者之和的提醒放在 `soft_warnings()`，因为叠加超支是性能问题，不是配置错误
        if not 0 < self.hyde_budget_ms < self.total_budget_ms:
            problems.append(
                f"HYDE_BUDGET_MS={self.hyde_budget_ms} 必须在 (0, {self.total_budget_ms}) 之间"
            )
        if not 0 < self.compound_timeout_ms < self.total_budget_ms:
            problems.append(
                f"COMPOUND_TIMEOUT_MS={self.compound_timeout_ms} 必须在 (0, {self.total_budget_ms}) 之间"
            )
        # 下界 100：比短问句还短的业务上限等于"几乎全拒"，那不是配置是事故；
        # 上界 4000：再长的问句 embedding 与 BM25 都不划算，且传输层已另有 64 KB 硬上限
        if not 100 <= self.chat_max_query_chars <= 4000:
            problems.append(
                f"CHAT_MAX_QUERY_CHARS={self.chat_max_query_chars} 必须在 [100, 4000] 之间"
            )
        # 下界 1024：比任何合法请求体都小的话，正常问句会被 413 拦在门外；
        # 上界 1 MiB：再往上不是问答请求，是上传
        if not 1024 <= self.max_request_bytes <= 1024 * 1024:
            problems.append(f"MAX_REQUEST_BYTES={self.max_request_bytes} 必须在 [1024, 1048576] 之间")
        # 容差超过一分钟就不再是"漂移补偿"：过期令牌能多活的时间与配置名对不上
        if not 0 <= self.jwt_leeway_s <= 300:
            problems.append(f"JWT_LEEWAY_S={self.jwt_leeway_s} 必须在 [0, 300] 之间")
        # access 必须短于 refresh：反过来的话"用 refresh 换新 access"这件事
        # 每次换出来的有效期比原令牌还长，轮换就变成无限续命。上限 24 小时是
        # 防止"登出后 access 仍可用一整天"——那与 `auth/store.py` 的取舍前提相反
        if not 0 < self.auth_access_ttl_s <= 86400:
            problems.append(f"AUTH_ACCESS_TTL_S={self.auth_access_ttl_s} 必须在 (0, 86400] 之间")
        if self.auth_refresh_ttl_s < self.auth_access_ttl_s:
            problems.append(
                f"AUTH_REFRESH_TTL_S={self.auth_refresh_ttl_s} 不能小于 AUTH_ACCESS_TTL_S={self.auth_access_ttl_s}"
            )
        # 「记住我」只许放宽，不许收紧：比默认还短等于勾了反而更快掉线
        if self.auth_refresh_ttl_remember_s < self.auth_refresh_ttl_s:
            problems.append(
                f"AUTH_REFRESH_TTL_REMEMBER_S={self.auth_refresh_ttl_remember_s} 不能小于 AUTH_REFRESH_TTL_S={self.auth_refresh_ttl_s}"
            )
        return problems

    def soft_warnings(self) -> list[str]:
        """**不阻断启动**的配置提醒（与 `self_check()` 分开）。

        `self_check()` 的返回值在 `ALLOW_DEGRADED_START=false` 下会让进程退出，
        所以"开发机没配 key"这种可以降级跑的情况不能放进去——放进去就等于
        一台没有 key 的机器连 FAQ 都起不来（§2.5 已针对 LLM 做过同样的修正）。
        """
        warnings: list[str] = []
        if (self.feature_query_rewrite or self.feature_intent_llm) and not self.llm_api_key.strip():
            warnings.append(
                "FEATURE_QUERY_REWRITE/FEATURE_INTENT_LLM 已开启但 LLM_API_KEY 为空："
                "改写与 LLM 意图均降级为「只用归一化 + 规则分类」"
            )
        if self.feature_query_rewrite and self.understanding_timeout_ms >= self.total_budget_ms // 2:
            warnings.append(
                f"UNDERSTANDING_TIMEOUT_MS={self.understanding_timeout_ms} 已占超过总预算的一半，"
                "首 token 延迟会被改写挤压（§6.1）"
            )
        # 理解 + 拆句 + HyDE 三段都在首 token 之前，各自都有闸门，但**加起来**仍可能
        # 吃掉半个总预算。这不是配置错误（每一项单看都合法），是必须被看见的性能账：
        # 全线超时时用户等的是 1.2 s 首 token 变成 4 s
        pre_answer_ms = (
            self.understanding_timeout_ms + self.compound_timeout_ms
            + (self.hyde_budget_ms if self.feature_hyde else 0)
        )
        if pre_answer_ms >= self.total_budget_ms // 2:
            warnings.append(
                f"理解/拆句/HyDE 预算合计 {pre_answer_ms} ms 已超过总预算的一半"
                f"（TOTAL_BUDGET_MS={self.total_budget_ms}）：首 token SLO 可能不达标（§6.1）"
            )
        if self.feature_hyde and not self.llm_api_key.strip():
            warnings.append(
                "FEATURE_HYDE 已开启但 LLM_API_KEY 为空：HyDE 生成不出来，"
                "planner 会如实降级为 DIRECT + hyde_unavailable"
            )
        # 生产没配 JWT_SECRET 不阻断（匿名问答是合法降级），但必须大声说出来：
        # 此时 `X-User-Id` 一律不认，所有请求都是 ANONYMOUS——记不住会话、也不挂记忆
        if not self.jwt_secret.strip() and not is_trusted_env(self):
            warnings.append(
                f"APP_ENV={self.app_env} 下 JWT_SECRET 为空：所有请求将按 ANONYMOUS 处理"
                "（不认任何请求头身份，会话与记忆都不会落库）"
            )
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# ---- 环境可信度（唯一出处）-------------------------------------------------
# 只有这几个环境才信任客户端自报的身份 / 租户头：都是"本地或测试"。
# **staging 不在白名单里**——预发环境连的常常是生产数据，把它算成"安全"等于让一个
# 改请求头的人读到别人的知识库或别人的记忆，而这正是预发最容易被忽略的地方。
#
# 为什么放在 `base/` 而不是各业务包里：`orchestration/api.py`（租户/ACL 头）与
# `memory/api.py`（记忆身份头）都要用同一条规则，而 `memory/` 不许 import
# `orchestration/`（§7.0 硬边界 1）。同一份白名单写两遍，下一次改它必然只改一处，
# 于是两处行为在某个环境上悄悄分叉——安全开关分叉比没有开关更糟。
TRUSTED_ENVS = frozenset({"dev", "development", "local", "test", "testing"})


def is_trusted_env(settings: object) -> bool:
    """该环境是否信任客户端自报的身份头。**白名单**：没见过的 `APP_ENV` 一律不可信。

    用白名单而不是 `env != "prod"`：将来多出一个没见过的取值时，白名单让它落到
    "不信任"（安全的那一侧），黑名单让它落到"信任"（危险的那一侧）。
    """
    env = str(getattr(settings, "app_env", "dev") or "dev").strip().lower()
    return env in TRUSTED_ENVS


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    """是否有可用 CUDA。torch 没装或 CUDA 运行时缺失都按 False 处理（不抛错）。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - 环境相关
        return False



