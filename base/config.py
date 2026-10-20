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
    faq_margin: float = Field(default=0.10, alias="FAQ_MARGIN")
    faq_top_k: int = Field(default=20, alias="FAQ_TOP_K")
    faq_norm_k: float = Field(default=12.0, alias="FAQ_NORM_K")
    faq_candidate_n: int = Field(default=3, alias="FAQ_CANDIDATE_N")
    faq_alias_enabled: bool = Field(default=True, alias="FAQ_ALIAS_ENABLED")
    faq_admin_token: str = Field(default="", alias="FAQ_ADMIN_TOKEN")
    # 本地无数据库时从 CSV 装载语料（相对仓库根目录）；生产留空，走 MySQL 真相源
    faq_corpus_csv: str = Field(default="", alias="FAQ_CORPUS_CSV")

    # ---- 启动行为 ----
    # dev 下允许数据库不可用时以"空 FAQ 语料"启动，便于纯本地跑通链路；生产必须关闭
    allow_degraded_start: bool = Field(default=True, alias="ALLOW_DEGRADED_START")

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
        if self.faq_top_k <= 0:
            problems.append("FAQ_TOP_K 必须为正")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


