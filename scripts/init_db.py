"""初始化 MySQL 表结构（幂等）。

用法：
    python scripts/init_db.py            # 建库建表，保留已有数据
    python scripts/init_db.py --drop     # 先删表再建（仅开发环境）

对应 CLAUDE.md §8：qa_system 只需要 faq / faq_alias / faq_feedback 三张表。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymysql  # noqa: E402

from base.config import get_settings  # noqa: E402
from base.logger import get_logger  # noqa: E402

log = get_logger("scripts.init_db")

DDL = [
    """
    CREATE TABLE IF NOT EXISTS faq (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        subject_name VARCHAR(32) NOT NULL DEFAULT '',
        question VARCHAR(1000) NOT NULL,
        answer TEXT NOT NULL,
        status ENUM('draft','pending','published','deprecated') NOT NULL DEFAULT 'draft',
        content_hash CHAR(32) NOT NULL DEFAULT '',
        source VARCHAR(64) NOT NULL DEFAULT 'import',
        hit_count INT NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        updated_by VARCHAR(64) NOT NULL DEFAULT '',
        UNIQUE KEY uk_question_hash (content_hash),
        KEY idx_status_subject (status, subject_name)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='FAQ 主表（qa_pair 的稳定问题表）'
    """,
    """
    CREATE TABLE IF NOT EXISTS faq_alias (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        faq_id BIGINT NOT NULL,
        alias VARCHAR(255) NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uk_alias (alias),
        KEY idx_faq (faq_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='别名：同一问题的不同问法'
    """,
    """
    CREATE TABLE IF NOT EXISTS faq_feedback (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        request_id VARCHAR(64) NOT NULL,
        faq_id BIGINT NULL,
        verdict ENUM('up','down') NOT NULL,
        tags JSON NULL,
        comment VARCHAR(1000) NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_request (request_id),
        KEY idx_faq_created (faq_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='回答反馈（错误归因与差评聚类来源）'
    """,
]


# 旧项目已有 faq(id, subject_name, question, answer)：CREATE IF NOT EXISTS 不会补齐新列，
# 因此启动时按 information_schema 做一次幂等列迁移。
FAQ_COLUMNS = [
    ("status", "ENUM('draft','pending','published','deprecated') NOT NULL DEFAULT 'published' AFTER answer"),
    ("content_hash", "CHAR(32) NOT NULL DEFAULT '' AFTER status"),
    ("source", "VARCHAR(64) NOT NULL DEFAULT 'legacy' AFTER content_hash"),
    ("hit_count", "INT NOT NULL DEFAULT 0 AFTER source"),
    ("created_at", "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP AFTER hit_count"),
    ("updated_at", "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP AFTER created_at"),
    ("updated_by", "VARCHAR(64) NOT NULL DEFAULT '' AFTER updated_at"),
]


def _table_exists(cur, database: str, table: str) -> bool:
    cur.execute(
        "SELECT 1 AS ok FROM information_schema.tables WHERE table_schema = %s AND table_name = %s",
        (database, table),
    )
    return cur.fetchone() is not None


def _columns(cur, database: str, table: str) -> set[str]:
    cur.execute(
        "SELECT COLUMN_NAME AS col FROM information_schema.columns WHERE table_schema = %s AND table_name = %s",
        (database, table),
    )
    return {str(row["col"]).lower() for row in cur.fetchall()}


def _indexes(cur, database: str, table: str) -> set[str]:
    cur.execute(
        "SELECT DISTINCT INDEX_NAME AS idx FROM information_schema.statistics WHERE table_schema = %s AND table_name = %s",
        (database, table),
    )
    return {str(row["idx"]).lower() for row in cur.fetchall()}


def migrate_faq(cur, database: str) -> None:
    if not _table_exists(cur, database, "faq"):
        return

    existing = _columns(cur, database, "faq")
    for name, ddl in FAQ_COLUMNS:
        if name not in existing:
            cur.execute(f"ALTER TABLE faq ADD COLUMN {name} {ddl}")
            log.warning("column_added", table="faq", column=name)

    # 旧数据补 hash/status：hash 用于 import 幂等去重
    cur.execute("UPDATE faq SET content_hash = LOWER(MD5(TRIM(question))) WHERE content_hash = ''")
    cur.execute("UPDATE faq SET status = 'published' WHERE status IS NULL OR status = ''")

    indexes = _indexes(cur, database, "faq")
    if {"uk_question_hash", "idx_question_hash"} & indexes:
        return
    cur.execute("SELECT LEFT(content_hash, 32) AS h, COUNT(*) AS c FROM faq GROUP BY h HAVING c > 1 LIMIT 1")
    if cur.fetchone():
        cur.execute("CREATE INDEX idx_question_hash ON faq (content_hash)")
        log.warning("faq_hash_duplicated", detail="存在重复问题，content_hash 只建普通索引，导入按问题文本去重")
    else:
        cur.execute("ALTER TABLE faq ADD UNIQUE KEY uk_question_hash (content_hash)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop", action="store_true", help="先删表再建（危险）")
    parser.add_argument(
        "--prune-legacy-subjects",
        action="store_true",
        help="把学科不在 K12 清单内的历史行置为 deprecated（旧项目的 Python/JAVA 学科 FAQ）",
    )
    args = parser.parse_args()

    settings = get_settings()
    problems = settings.self_check()
    if problems:
        for problem in problems:
            log.error("config_problem", detail=problem)
        return 2

    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci"
            )
            cur.execute(f"USE `{settings.mysql_database}`")
            if args.drop:
                for table in ("faq_feedback", "faq_alias", "faq"):
                    cur.execute(f"DROP TABLE IF EXISTS `{table}`")
                    log.warning("table_dropped", table=table)
            for ddl in DDL:
                cur.execute(ddl)
            migrate_faq(cur, settings.mysql_database)
            if args.prune_legacy_subjects:
                from base.labels import SUBJECTS

                placeholders = ", ".join(["%s"] * len(SUBJECTS))
                cur.execute(
                    f"UPDATE faq SET status = 'deprecated' "
                    f"WHERE subject_name NOT IN ({placeholders}) AND status = 'published'",
                    SUBJECTS,
                )
                log.warning("legacy_subjects_deprecated", rows=cur.rowcount)
            cur.execute("SHOW TABLES")
            tables = sorted(row[f"Tables_in_{settings.mysql_database}"] for row in cur.fetchall())
    finally:
        conn.close()

    log.info("db_ready", database=settings.mysql_database, tables=tables)
    print(f"[ok] database={settings.mysql_database} tables={', '.join(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


