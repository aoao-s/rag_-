"""把 edu_faq.csv 导入 faq 表（幂等，靠 content_hash 唯一键去重）。

用法：
    python scripts/import_faq.py --dry-run          # 只看统计，不写库
    python scripts/import_faq.py --publish          # 导入后直接置为 published
    python scripts/import_faq.py --csv <路径>

默认导入后状态是 draft：FAQ 必须经过发布动作才进检索语料，避免脏数据直接上线。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pymysql  # noqa: E402

from base.config import get_settings  # noqa: E402
from base.labels import normalize_subject  # noqa: E402
from base.logger import get_logger  # noqa: E402
from qa_system.res.qa_db import question_hash  # noqa: E402

log = get_logger("scripts.import_faq")

DEFAULT_CSV = ROOT / "datasets" / "04_faq" / "edu_faq.csv"


def read_rows(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for line_no, row in enumerate(reader, start=2):
            subject = (row.get("学科名称") or "").strip()
            question = (row.get("问题") or "").strip()
            answer = (row.get("答案") or "").strip()
            if not question or not answer:
                log.warning("skip_incomplete_row", line=line_no, question=question[:20])
                continue
            rows.append((normalize_subject(subject) or "", question, answer))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true", help="导入后立即发布")
    parser.add_argument("--operator", default="import")
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        print(f"[error] csv 不存在：{path}")
        return 2

    rows = read_rows(path)
    uniq = {(subject, question): answer for subject, question, answer in rows}
    log.info("csv_parsed", path=str(path), rows=len(rows), unique=len(uniq))
    print(f"[ok] 读取 {len(rows)} 行，去重后 {len(uniq)} 条")

    if args.dry_run:
        return 0

    settings = get_settings()
    status_value = "published" if args.publish else "draft"
    conn = pymysql.connect(
        host=settings.mysql_host,
        port=settings.mysql_port,
        user=settings.mysql_user,
        password=settings.mysql_password,
        database=settings.mysql_database,
        charset="utf8mb4",
        autocommit=False,
        cursorclass=pymysql.cursors.DictCursor,
    )
    inserted = updated = 0
    try:
        with conn.cursor() as cur:
            for (subject, question), answer in uniq.items():
                # 用 content_hash 显式查重而不是 ON DUPLICATE KEY：老库的 faq 表可能没有唯一键
                cur.execute("SELECT id FROM faq WHERE content_hash = %s LIMIT 1", (question_hash(question),))
                row = cur.fetchone()
                if row:
                    cur.execute(
                        "UPDATE faq SET subject_name = %s, answer = %s, status = %s, updated_by = %s WHERE id = %s",
                        (subject, answer, status_value, args.operator, row["id"]),
                    )
                    updated += 1
                else:
                    cur.execute(
                        "INSERT INTO faq (subject_name, question, answer, status, content_hash, source, updated_by) "
                        "VALUES (%s, %s, %s, %s, %s, 'csv', %s)",
                        (subject, question, answer, status_value, question_hash(question), args.operator),
                    )
                    inserted += 1
        conn.commit()
    except Exception as exc:
        conn.rollback()
        log.error("import_failed", error=str(exc))
        raise
    finally:
        conn.close()

    log.info("import_done", inserted=inserted, updated=updated, status=status_value)
    print(f"[ok] 新增 {inserted} 条，更新 {updated} 条，状态={status_value}")
    if not args.publish:
        print("提示：draft 不进检索语料，确认无误后执行 python scripts/import_faq.py --publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

