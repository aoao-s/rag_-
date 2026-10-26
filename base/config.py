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
    # 待入库的原始资料目录：相对仓库根目录，逗号分隔可写多个
    rag_docs_dir: str = Field(
        default="datasets/01_textbook,datasets/02_lesson_plans,datasets/08_subtitles",
        alias="RAG_DOCS_DIR",
    )
    # 父块：语义完整、随答案一起进 prompt；子块：只用来被检索命中
    rag_parent_size: int = Field(default=1200, alias="RAG_PARENT_SIZE")
    rag_parent_overlap: int = Field(default=100, alias="RAG_PARENT_OVERLAP")
    rag_child_size: int = Field(default=300, alias="RAG_CHILD_SIZE")
    rag_child_overlap: int = Field(default=20, alias="RAG_CHILD_OVERLAP")
    # 单篇文档最多产出多少子块，防止一份几百页的 PDF 把库撑爆
    rag_max_chunks_per_doc: int = Field(default=2000, alias="RAG_MAX_CHUNKS_PER_DOC")
    # PDF 解析器：auto=有 MINERU_TOKEN 走云解析、失败自动退本地；mineru=只用云；local=只用 PyMuPDF
    rag_pdf_parser: str = Field(default="auto", alias="RAG_PDF_PARSER")
    # Office/图片里的内嵌图是否 OCR：慢且贵，默认关闭（教材 PDF 走 MinerU 自带版面解析）
    rag_ocr_images: bool = Field(default=False, alias="RAG_OCR_IMAGES")

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
    rag_reranker_path: str = Field(
        default="../Ragpro/rag_system/models/BAAI/bge-reranker-v2-m3", alias="RAG_RERANKER_PATH"
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

    # ---- 向量库 Milvus（CLAUDE.md §8.1）----
    milvus_uri: str = Field(default="http://127.0.0.1:19531", alias="MILVUS_URI")
    milvus_database: str = Field(default="edurag", alias="MILVUS_DATABASE")
    milvus_collection: str = Field(default="edurag_chunk", alias="MILVUS_COLLECTION")
    milvus_token: str = Field(default="", alias="MILVUS_TOKEN")
    # BGE-M3 输出 1024 维；改模型必须同步改这里，否则 upsert 直接报维度不匹配
    milvus_dim: int = Field(default=1024, alias="MILVUS_DIM")

    # ---- 向量化 ----
    # auto：有 FlagEmbedding+模型就用真模型，否则用确定性哈希向量（只给测试/离线跑通用）
    rag_embed_backend: str = Field(default="auto", alias="RAG_EMBED_BACKEND")
    rag_embed_model_path: str = Field(
        default="../Ragpro/rag_system/models/BAAI/bge-m3", alias="RAG_EMBED_MODEL_PATH"
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
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    """是否有可用 CUDA。torch 没装或 CUDA 运行时缺失都按 False 处理（不抛错）。"""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - 环境相关
        return False



