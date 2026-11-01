"""初始化 MySQL 表结构（幂等）。

用法：
    python scripts/init_db.py            # 建库建表，保留已有数据
    python scripts/init_db.py --drop     # 先删表再建（仅开发环境）

对应 CLAUDE.md §6.4：qa_system 三张表（faq / faq_alias / faq_feedback），
记忆六张表（memory_atom / memory_audit / memory_grant / user_profile / chat_message / session_summary，
见 §7.3.3 / §7.9）。`memory_grant` 是 M4 的顾问只读授权表，方向只有 owner→grantee；
`user_profile` 是 L3 画像（只有人工写过的才落库，derived 版现算）。
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
    # ---- 会话与记忆（CLAUDE.md §7.3.3 / §7.9）----
    # 时间列一律 DATETIME、Python 侧 epoch 秒：转换只在 memory/store.py 做（见该文件头注释）
    """
    CREATE TABLE IF NOT EXISTS memory_atom (
        id VARCHAR(40) PRIMARY KEY,
        tenant_id VARCHAR(32) NOT NULL DEFAULT 'default',
        user_id VARCHAR(32) NOT NULL,
        fact_type ENUM('preference','identity','constraint','decision','progress') NOT NULL,
        fact_key VARCHAR(64) NOT NULL,
        content VARCHAR(500) NOT NULL,
        confidence DECIMAL(3,2) NOT NULL DEFAULT 0.60,
        mention_count INT NOT NULL DEFAULT 1,
        source_sid VARCHAR(40) NOT NULL DEFAULT '',
        source_turn INT NOT NULL DEFAULT 0,
        source_span VARCHAR(200) NOT NULL DEFAULT '',
        valid_from DATETIME NOT NULL,
        valid_to DATETIME NULL,
        superseded_by VARCHAR(40) NULL,
        chain_root_id VARCHAR(40) NULL,
        deleted_at DATETIME NULL,
        subject VARCHAR(32) NOT NULL DEFAULT '',
        grade VARCHAR(16) NOT NULL DEFAULT '',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_user_alive (tenant_id, user_id, valid_to, deleted_at),
        KEY idx_chain (chain_root_id),
        KEY idx_user_key (tenant_id, user_id, fact_key),
        KEY idx_user_subject_grade (tenant_id, user_id, subject, grade)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='原子记忆（L2）；valid_to/deleted_at 都为空才算活着'
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_audit (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        atom_id VARCHAR(40) NOT NULL,
        action VARCHAR(32) NOT NULL,
        tenant_id VARCHAR(32) NOT NULL DEFAULT 'default',
        user_id VARCHAR(32) NOT NULL,
        operator VARCHAR(64) NOT NULL DEFAULT '',
        detail VARCHAR(500) NOT NULL DEFAULT '',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_atom (atom_id),
        KEY idx_user_created (user_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='记忆读写审计（顾问只读授权后必须能追溯）'
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_message (
        id BIGINT PRIMARY KEY AUTO_INCREMENT,
        session_id VARCHAR(40) NOT NULL,
        tenant_id VARCHAR(32) NOT NULL DEFAULT 'default',
        user_id VARCHAR(32) NOT NULL,
        role ENUM('user','assistant') NOT NULL,
        content TEXT NOT NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_session (session_id, id),
        KEY idx_user_created (tenant_id, user_id, created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话逐字历史（L0 真相源；Redis 只是热副本）'
    """,
    """
    CREATE TABLE IF NOT EXISTS session_summary (
        session_id VARCHAR(40) PRIMARY KEY,
        tenant_id VARCHAR(32) NOT NULL DEFAULT 'default',
        user_id VARCHAR(32) NOT NULL,
        summary TEXT NOT NULL,
        compressed_turns INT NOT NULL DEFAULT 0,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        KEY idx_user_updated (tenant_id, user_id, updated_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='会话摘要（L1，每 session 一份）'
    """,
    # 顾问只读授权（§3.7 第 7 条 / 实施手册 M4）。**方向是 owner 授予 grantee**：
    # 学员把自己的记忆开放给顾问，反过来没有任何入口——"管理员默认不看原文"
    # 这条约束靠"表里没有 admin 角色"来保证，而不是靠调用方记得别查。
    # `UNIQUE (tenant_id, owner_id, grantee_id)` 让"撤销后再授权"是一次覆盖而不是堆行：
    # 堆行会让 `list_by_owner` 里同一个人出现多次，前端还要自己去重。
    """
    CREATE TABLE IF NOT EXISTS memory_grant (
        id VARCHAR(40) PRIMARY KEY,
        tenant_id VARCHAR(32) NOT NULL DEFAULT 'default',
        owner_id VARCHAR(32) NOT NULL,
        grantee_id VARCHAR(32) NOT NULL,
        scope ENUM('read') NOT NULL DEFAULT 'read',
        status ENUM('active','revoked') NOT NULL DEFAULT 'active',
        granted_by VARCHAR(64) NOT NULL DEFAULT '',
        granted_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        expires_at DATETIME NULL,
        revoked_at DATETIME NULL,
        UNIQUE KEY uk_owner_grantee (tenant_id, owner_id, grantee_id),
        KEY idx_grantee (tenant_id, grantee_id, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='记忆只读授权（owner→grantee，只有 read）'
    """,
    # L3 用户画像（§3.7 第 1 条 / §7.3.1）。一人一行，**主键是 (tenant_id, user_id)**：
    # 画像本来就是"每个用户一份"，用自增 id 只会让 upsert 变成"先查再决定插还是改"。
    # `source` 只区分调用方有没有手工写过（`manual`）：没写过的画像由身份类原子记忆
    # 现算（`derived`），**不落库**——否则每次原子记忆变化都要同步一张派生表，
    # 而它的内容随时可以重算。清除画像 = 删行，下次读又回到 derived。
    """
    CREATE TABLE IF NOT EXISTS user_profile (
        tenant_id VARCHAR(32) NOT NULL DEFAULT 'default',
        user_id VARCHAR(32) NOT NULL,
        content TEXT NOT NULL,
        source ENUM('manual','derived') NOT NULL DEFAULT 'manual',
        updated_by VARCHAR(64) NOT NULL DEFAULT '',
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        PRIMARY KEY (tenant_id, user_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户画像（L3；只有人工写过的才落库）'
    """,
    # ---- 认证（auth/store.py）--------------------------------------------
    # `role` 在这里是**唯一真相源**：登录接口只认这一列，绝不从请求体或旧 token 取。
    # `password_hash` 存 Argon2id 全串（`$argon2id$v=19$m=...$salt$hash`），
    # 算法与参数都编码在串里，所以不需要额外的列记"这条是用什么算的"，
    # 换参靠 `passwords.needs_rehash()` 在用户下次登录时静默升级。
    # `UNIQUE (tenant_id, username)` 是**大小写不敏感**的（utf8mb4_0900_ai_ci）：
    # 这正是服务层把用户名统一转小写的原因（`AuthService._normalize_username`），
    # 否则 `Student001` 能注册、`student001` 登录时查不到。
    """
    CREATE TABLE IF NOT EXISTS sys_user (
        user_id VARCHAR(32) PRIMARY KEY,
        tenant_id VARCHAR(32) NOT NULL DEFAULT 'default',
        username VARCHAR(64) NOT NULL,
        nickname VARCHAR(64) NOT NULL DEFAULT '',
        password_hash VARCHAR(255) NOT NULL,
        role ENUM('STUDENT','TEACHER','ADMIN') NOT NULL DEFAULT 'STUDENT',
        status ENUM('active','disabled') NOT NULL DEFAULT 'active',
        acl_tags VARCHAR(512) NOT NULL DEFAULT '',
        avatar VARCHAR(512) NOT NULL DEFAULT '',
        last_login_at DATETIME NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        UNIQUE KEY uk_tenant_username (tenant_id, username),
        KEY idx_tenant_role (tenant_id, role, status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='账号与角色（角色只认这里，请求体里的 role 一律不看）'
    """,
    # 只记 refresh（access 不落库，见 `auth/store.py` 模块注释）。
    # `jti` 是主键：轮换与登出都是按它 UPDATE，不需要再建二级索引。
    """
    CREATE TABLE IF NOT EXISTS auth_token (
        jti VARCHAR(40) PRIMARY KEY,
        user_id VARCHAR(32) NOT NULL,
        tenant_id VARCHAR(32) NOT NULL DEFAULT 'default',
        kind ENUM('refresh','access') NOT NULL DEFAULT 'refresh',
        expires_at DATETIME NOT NULL,
        revoked_at DATETIME NULL,
        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        KEY idx_user_kind (tenant_id, user_id, kind, revoked_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='已签发的 refresh 令牌（jti + 吊销时间；access 不入库）'
    """,
]

# `--drop` 的删除顺序：先删子表/引用表，再删主表
DROP_ORDER = (
    "faq_feedback",
    "faq_alias",
    "faq",
    "memory_audit",
    "memory_grant",
    "memory_atom",
    "chat_message",
    "session_summary",
    "user_profile",
    "auth_token",
    "sys_user",
)


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


# 记忆的结构化坐标（§7.9 的 M0）。取值域是 `base/labels` 的 SUBJECTS / GRADES，
# 空串 = 未归类——**不是 NULL**：记忆页按学科筛选时要能"筛出未归类的"，
# NULL 会让 `= ''` 这类查询静默少一半行。
MEMORY_COLUMNS = [
    ("subject", "VARCHAR(32) NOT NULL DEFAULT '' AFTER deleted_at"),
    ("grade", "VARCHAR(16) NOT NULL DEFAULT '' AFTER subject"),
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


def migrate_memory(cur, database: str) -> None:
    """`memory_atom` 的幂等列迁移：老库已经在跑（记忆先于坐标上线），
    `CREATE TABLE IF NOT EXISTS` 不会给它补列，而 `_COLS` 里已经带上了 subject/grade —
    **不补的话每一次记忆读写都会报 Unknown column**，不是慢，是整块功能 500。
    """
    if not _table_exists(cur, database, "memory_atom"):
        return

    existing = _columns(cur, database, "memory_atom")
    for name, ddl in MEMORY_COLUMNS:
        if name not in existing:
            cur.execute(f"ALTER TABLE memory_atom ADD COLUMN {name} {ddl}")
            log.warning("column_added", table="memory_atom", column=name)

    # 索引单独补：列加上了不等于能按学科/年级筛得动（记忆页的筛选走这条）
    if "idx_user_subject_grade" not in _indexes(cur, database, "memory_atom"):
        cur.execute(
            "ALTER TABLE memory_atom ADD KEY idx_user_subject_grade (tenant_id, user_id, subject, grade)"
        )
        log.warning("index_added", table="memory_atom", index="idx_user_subject_grade")


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
                for table in DROP_ORDER:
                    cur.execute(f"DROP TABLE IF EXISTS `{table}`")
                    log.warning("table_dropped", table=table)
            for ddl in DDL:
                cur.execute(ddl)
            migrate_faq(cur, settings.mysql_database)
            migrate_memory(cur, settings.mysql_database)
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
            # 列名大小写随库名/系统而定（`Tables_in_rag_abc` vs `Tables_in_RAG_abc`），
            # 不能按库名拼 key，直接取每行第一个值。
            tables = sorted(next(iter(row.values())) for row in cur.fetchall())
    finally:
        conn.close()

    log.info("db_ready", database=settings.mysql_database, tables=tables)
    print(f"[ok] database={settings.mysql_database} tables={', '.join(tables)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


