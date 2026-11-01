"""把 edu_faq.csv 导入 faq 表（幂等，靠 content_hash 唯一键去重）。

用法：
    python scripts/import_faq.py --dry-run          # 只看统计，不写库
    python scripts/import_faq.py --publish          # 导入后直接置为 published
    python scripts/import_faq.py --csv <路径>
    python scripts/import_faq.py --publish --deprecate-missing   # 全量替换语料

默认导入后状态是 draft：FAQ 必须经过发布动作才进检索语料，避免脏数据直接上线。

**为什么需要 `--deprecate-missing`**：语料被整代改写时（2026-10-26 项目方把 788 条问题从关键词式短句
改成完整问句），新旧两代的 `content_hash` 完全不同，导入只会**新增**，老的一代仍然 `published`——
服务随后装载的是 4000 条、其中一半是已作废的问法，`unlabeled` 也回来了。全量替换语料必须显式声明
"CSV 里没有的旧行请下线"，所以放在开关里而不是默认行为：局部导入（只导数学）时默认下线其余学科
会让整个 FAQ 库瞬间只剩数学。
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


def deprecate_missing(conn, keep_hashes: set[str], operator: str) -> int:
    """把库里已发布、但本次 CSV 里没有的行置为 deprecated。

    按 `content_hash` 比对而不是按 `question` 文本：同一句话在不同归一化规则下文本可能不同，
    而 `content_hash` 就是服务装载时判重用的那一列（`question_hash()`）。
    分批 UPDATE，避免一次 IN 上千个参数撞 mysql 的包大小限制。
    """
    with conn.cursor() as cur:
        cur.execute("SELECT id, content_hash FROM faq WHERE status = 'published'")
        stale = [row["id"] for row in cur.fetchall() if (row.get("content_hash") or "") not in keep_hashes]
        for i in range(0, len(stale), 500):
            batch = stale[i : i + 500]
            placeholders = ",".join(["%s"] * len(batch))
            cur.execute(
                f"UPDATE faq SET status = 'deprecated', updated_by = %s WHERE id IN ({placeholders})",
                (operator, *batch),
            )
    return len(stale)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--publish", action="store_true", help="导入后立即发布")
    parser.add_argument(
        "--deprecate-missing",
        action="store_true",
        help="把库里已发布但 CSV 里没有的行下线（全量替换语料时用；局部导入不要开）",
    )
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
    inserted = updated = deprecated = 0
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
            if args.deprecate_missing:
                deprecated = deprecate_missing(conn, {question_hash(q) for _, q in uniq}, args.operator)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        log.error("import_failed", error=str(exc))
        raise
    finally:
        conn.close()

    log.info("import_done", inserted=inserted, updated=updated, status=status_value)
    if args.deprecate_missing:
        # 只有开了开关才可能下线数据：默认叙事里不该出现"已下线 N 行"
        log.info("faq_deprecated_missing", rows=deprecated)
    print(f"[ok] 新增 {inserted} 条，更新 {updated} 条，状态={status_value}")
    if args.deprecate_missing:
        print(f"[ok] CSV 里没有的旧行已下线 {deprecated} 条")
    if not args.publish:
        print("提示：draft 不进检索语料，确认无误后执行 python scripts/import_faq.py --publish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

