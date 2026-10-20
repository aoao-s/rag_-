"""集中配置：所有可调参数从环境变量 / .env 读取，业务代码禁止硬编码阈值。

启动自检（见 CLAUDE.md §2.5）：关键项缺失时给出可执行的修复提示，而不是等第一个请求 401。
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from base.labels import INTENT_LABELS, SUBJECTS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- 应用 ----
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    http_host: str = Field(default="127.0.0.1", alias="HTTP_HOST")
    http_port: int = Field(default=8000, alias="HTTP_PORT")

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
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
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


