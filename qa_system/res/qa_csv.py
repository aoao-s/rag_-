"""CSV 语料装载：无数据库时也能把 FAQ 跑起来（本地演示 / 单测 / 阈值标定共用）。

为什么单独一个模块：`scripts/import_faq.py`（写库）与 `scripts/eval_faq.py`（标定）
都要读同一份 CSV，如果在两处各写一遍，字段名一改就两边不一致。

CSV 契约（`datasets/04_faq/edu_faq.csv`）：表头 `学科名称,问题,答案`，UTF-8 with BOM。
"""

from __future__ import annotations

import csv
from pathlib import Path

from base.labels import normalize_subject
from qa_system.res.qa_db import FAQRecord

CSV_SUBJECT = "学科名称"
CSV_QUESTION = "问题"
CSV_ANSWER = "答案"


def load_csv_records(path: Path | str, *, limit: int | None = None) -> list[FAQRecord]:
    """读 CSV → FAQRecord 列表。faq_id 用 CSV 行序（从 1 开始），保证离线可复现。

    空问题行直接跳过：语料里出现空问题，归一化后是空串，进了索引只会污染精确哈希表。
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"FAQ CSV 不存在：{file}")

    records: list[FAQRecord] = []
    with file.open(encoding="utf-8-sig", newline="") as fh:
        for idx, row in enumerate(csv.DictReader(fh), start=1):
            question = (row.get(CSV_QUESTION) or "").strip()
            if not question:
                continue
            records.append(
                FAQRecord(
                    faq_id=idx,
                    subject=normalize_subject((row.get(CSV_SUBJECT) or "").strip()),
                    question=question,
                    answer=(row.get(CSV_ANSWER) or "").strip(),
                )
            )
            if limit is not None and len(records) >= limit:
                break
    return records
