#!/usr/bin/env python
"""FAQ 阈值标定：同义问法生成 + 三组样本 + 阈值扫描。

为什么要这个脚本：`FAQ_THRESHOLD=0.62` 最初是拍的。0.62 太高会把同义问法推给 RAG（贵且慢），
太低会把不相干的问题答成 FAQ（错误答案比"不知道"更糟）。所以要用**带标签的样本**量出来，
而不是凭感觉调。

样本三组：
1. 同义问法（正样本）：对每条 FAQ 用 deepseek-chat 生成 3 个"学员真实会怎么问"的变体，
   标签 = 该 FAQ 的 id。生成一次落盘复用，不重复花钱。
2. 题库干扰（负样本）：datasets/03_question_bank 抽样，这些是考试题，不该命中 FAQ。
3. 原句自检：160 条原问题逐字查，应当 100% 命中（精确哈希路径），用来确认脚本没接错。

用法：
    # 1) 生成同义问法（只跑一次，已生成过的会跳过）
    .\\.venv\\Scripts\\python.exe scripts\\eval_faq.py gen --api-key sk-xxx
    # 2) 扫阈值，输出报告
    .\\.venv\\Scripts\\python.exe scripts\\eval_faq.py calibrate

输出：docs/analysis/faq_threshold_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base.config import get_settings  # noqa: E402
from base.db import MySQL  # noqa: E402
from qa_system.res.qa_csv import load_csv_records  # noqa: E402
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository, MySQLFAQRepository  # noqa: E402
from qa_system.res.qa_matcher import MatchPolicy  # noqa: E402
from qa_system.res.qa_search import FAQService  # noqa: E402

PARAPHRASE_FILE = ROOT / "docs" / "analysis" / "faq_paraphrases.jsonl"
REPORT_FILE = ROOT / "docs" / "analysis" / "faq_threshold_report.md"
QUESTION_BANK = ROOT / "datasets" / "03_question_bank" / "question_bank.jsonl"

PROMPT = (
    "你是 K12 在线辅导机构的客服质检员。下面是一条标准 FAQ 问题，"
    "请写出 3 个学员在聊天框里真实可能的问法：口语化、可省略主语、可用不同句式，"
    "但**语义必须与原问题完全一致**（问的必须是同一件事，不能扩展也不能缩小范围）。\n"
    "要求：\n"
    "1) 不要照抄原句，不要加“请问老师”以外的多余背景；\n"
    "2) 每个问题不超过 25 个字；\n"
    "3) 不要编号，不要解释。\n"
    '只输出 JSON：{"paraphrases": ["问法1", "问法2", "问法3"]}\n\n'
    "标准问题：{question}"
)


def _api_key(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    for name in ("DEEPSEEK_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith(("LLM_API_KEY=", "DEEPSEEK_API_KEY=")) and "=" in line:
                value = line.split("=", 1)[1].strip().strip('"')
                if value:
                    return value
    raise SystemExit("缺少 API key：用 --api-key 传入，或设置 DEEPSEEK_API_KEY / LLM_API_KEY")


async def _one_call(client, sem, api_key: str, record: FAQRecord, retries: int = 3) -> list[str]:
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": PROMPT.replace("{question}", record.question)}],
        "temperature": 1.0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
    }
    async with sem:
        for attempt in range(retries):
            try:
                resp = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    timeout=60.0,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"http {resp.status_code}")
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                data = json.loads(content)
                items = [str(x).strip() for x in data.get("paraphrases", [])]
                return [x for x in items if x and x != record.question][:3]
            except Exception as exc:  # noqa: BLE001 - 重试后仍失败只跳过该条
                if attempt == retries - 1:
                    print(f"  ! faq {record.faq_id} 生成失败：{exc}", flush=True)
                    return []
                await asyncio.sleep(1.5 * (attempt + 1))
    return []


async def gen_paraphrases(api_key: str, limit: int | None = None, concurrency: int = 6) -> int:
    import httpx

    records, source = await _load_records()
    print(f"语料来源：{source}")
    if limit:
        records = records[:limit]

    PARAPHRASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if PARAPHRASE_FILE.exists():
        for line in PARAPHRASE_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(int(json.loads(line)["faq_id"]))
    todo = [r for r in records if r.faq_id not in done]
    print(f"总 {len(records)} 条，已有 {len(done)} 条，本次生成 {len(todo)} 条")

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(trust_env=False) as client:
        tasks = [_one_call(client, sem, api_key, r) for r in todo]
        for record, variants in zip(todo, await asyncio.gather(*tasks)):
            if not variants:
                continue
            with PARAPHRASE_FILE.open("a", encoding="utf-8") as fh:
                for text in variants:
                    fh.write(
                        json.dumps(
                            {
                                "faq_id": record.faq_id,
                                "subject": record.subject,
                                "original": record.question,
                                "paraphrase": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    total = sum(1 for line in PARAPHRASE_FILE.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"完成，累计 {total} 条 → {PARAPHRASE_FILE.relative_to(ROOT)}")
    return total


async def _load_records() -> tuple[list[FAQRecord], str]:
    """装载语料：优先 datasets/04_faq/edu_faq.csv（无需数据库，任何人都能复现），
    CSV 缺失时才回退到 MySQL。返回 (records, 来源说明)。
    """
    csv_path = ROOT / "datasets" / "04_faq" / "edu_faq.csv"
    if csv_path.exists():
        records = load_csv_records(csv_path)
        return records, f"datasets/04_faq/edu_faq.csv（{len(records)} 条，id=CSV 行序）"

    from base.config import get_settings
    from base.db import MySQL

    mysql = MySQL(get_settings())
    repository = MySQLFAQRepository(mysql)
    return await repository.fetch_published(), "MySQL itcast.faq（published）"


async def _scores(service: FAQService, queries: list[str]) -> list[tuple[str, int | None, int | None, float, float]]:
    """返回 (query, top_id, hit_id, score, margin)。用 threshold=0 的策略取"永远给 top1"。"""
    out = []
    for query in queries:
        hit = await service.search(query)
        out.append(
            (
                query,
                hit.faq_id,
                hit.faq_id if hit.hit else None,
                hit.score,
                getattr(hit, "margin", 0.0) or 0.0,
            )
        )
    return out


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


async def calibrate() -> None:
    if not PARAPHRASE_FILE.exists():
        raise SystemExit(f"没有同义问法样本：先跑 `gen`，期望文件 {PARAPHRASE_FILE}")

    records, source = await _load_records()
    repository = InMemoryFAQRepository(records)
    # threshold=0 / margin=0：对所有查询都返回 top1，便于离线扫参数
    service = FAQService(repository, policy=MatchPolicy(threshold=0.0, margin=0.0))
    await service.load()

    by_id = {r.faq_id: r for r in records}

    rows = [
        json.loads(line)
        for line in PARAPHRASE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r["faq_id"] in by_id]
    positives = [(r["paraphrase"], r["faq_id"]) for r in rows]

    negatives: list[str] = []
    if QUESTION_BANK.exists():
        bank = [
            json.loads(line)["question"]
            for line in QUESTION_BANK.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        random.Random(42).shuffle(bank)
        negatives = bank[:300]

    print(f"正样本 {len(positives)} 条 / 负样本 {len(negatives)} 条 / FAQ {len(records)} 条")

    pos_scores: list[float] = []
    neg_scores: list[float] = []
    pos_rows: list[tuple[int, int | None, float, float]] = []  # (expected, predicted, top1_score, margin)
    for text, expected in positives:
        ranked = await service.rank(text, top_k=2)
        if not ranked:
            pos_rows.append((expected, None, 0.0, 0.0))
            continue
        top_id, top_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        pos_scores.append(top_score)
        pos_rows.append((expected, top_id, top_score, top_score - second))

    neg_rows: list[tuple[float, float]] = []
    for text in negatives:
        ranked = await service.rank(text, top_k=2)
        if not ranked:
            neg_scores.append(0.0)
            neg_rows.append((0.0, 0.0))
            continue
        top_score = ranked[0][1]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        neg_scores.append(top_score)
        neg_rows.append((top_score, top_score - second))

    # 原句自检
    self_miss = 0
    for record in records:
        hit = await service.search(record.question)
        if not hit.hit or hit.faq_id != record.faq_id:
            self_miss += 1

    # 二维扫描：命中 = score >= t 且 margin >= m。
    # 单看绝对分，会把"同题族"（勾股定理内容 / 勾股定理证明）判成一题，margin 才是分水岭。
    def evaluate(t: float, m: float) -> tuple[float, float, int]:
        tp = fp = 0
        for expected, predicted, top_score, margin in pos_rows:
            if top_score < t or margin < m:
                continue
            if predicted == expected:
                tp += 1
            else:
                fp += 1  # 命中但答错，比不答更糟
        for top_score, margin in neg_rows:
            if top_score >= t and margin >= m:
                fp += 1
        fn = len(pos_rows) - tp
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        return recall, precision, fp

    best = None
    table = []
    t = 0.20
    while t <= 0.9001:
        recall, precision, fp = evaluate(round(t, 3), 0.05)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        table.append((round(t, 2), recall, precision, f1, int(fp)))
        if precision >= 0.97 and (best is None or recall > best[1]):
            best = (round(t, 2), recall, precision, f1, int(fp))
        t += 0.02

    # 二维：阈值 × margin。margin 越大越保守，用来"宁可不答，也不答成同题族的另一条"
    margins = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    grid: list[tuple[float, float, float, float, int]] = []
    for m in margins:
        for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
            recall, precision, fp = evaluate(t, m)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            grid.append((t, m, recall, precision, int(fp)))

    # 选点：precision ≥ 0.95 下召回最高；再平手取 F1
    cands = [g for g in grid if g[3] >= 0.95]
    pick = max(cands, key=lambda g: (g[2], g[3])) if cands else None

    lines = [
        "# FAQ 阈值标定报告",
        "",
        f"- 语料：{len(records)} 条 published FAQ（{source}）",
        f"- 正样本：{len(positives)} 条同义问法（deepseek-flash 生成，标签=对应 faq_id）",
        f"- 负样本：{len(negatives)} 条题库问题（`datasets/03_question_bank`，期望不命中）",
        f"- 原句自检：{len(records) - self_miss}/{len(records)} 命中自身（应为 100%）",
        "",
        "## 分数分布",
        "",
        "| 集合 | p50 | p90 | p95 | max |",
        "|---|---|---|---|---|",
        f"| 同义问法（正） | {_pct(pos_scores, .5):.3f} | {_pct(pos_scores, .9):.3f} | {_pct(pos_scores, .95):.3f} | {max(pos_scores or [0]):.3f} |",
        f"| 题库问题（负） | {_pct(neg_scores, .5):.3f} | {_pct(neg_scores, .9):.3f} | {_pct(neg_scores, .95):.3f} | {max(neg_scores or [0]):.3f} |",
        "",
        "## 一维扫描（margin 固定 0.05）",
        "",
        "| 阈值 | 召回 | 精度 | F1 | 误命中数 |",
        "|---|---|---|---|---|",
    ]
    for row in table:
        t_, recall, precision, f1, fp = row
        mark = " ←" if best and t_ == best[0] else ""
        lines.append(f"| {t_:.2f} | {recall:.3f} | {precision:.3f} | {f1:.3f} | {fp} |{mark}")
    lines.append("")

    lines += [
        "## 二维扫描（阈值 × margin）",
        "",
        "| 阈值 \\ margin | " + " | ".join(f"{m:.2f}" for m in margins) + " |",
        "|---" * (len(margins) + 1) + "|",
    ]
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
        cells = []
        for m in margins:
            recall, precision, _fp = evaluate(t, m)
            cells.append(f"{recall:.2f}/{precision:.2f}")
        lines.append(f"| {t:.2f} | " + " | ".join(cells) + " |")
    lines += ["", "单元格为「召回/精度」。", ""]

    if best:
        lines += [
            "## 建议",
            "",
            f"- 只看绝对分：**{best[0]:.2f}**（召回 {best[1]:.3f}、精度 {best[2]:.3f}、误命中 {best[4]} 条）。",
        ]
    if pick:
        t_, m_, recall, precision, fp = pick
        lines += [
            f"- 加 margin 约束：**threshold={t_:.2f} + margin={m_:.2f}** "
            f"→ 召回 {recall:.3f}、精度 {precision:.3f}、误命中 {fp} 条。",
            "",
            "选点原则：FAQ 答错（把 A 问题的答案给 B）比不答更伤信任，不答还能交棒 RAG。"
            "因此先卡精度 ≥0.95，再最大化召回；margin 是在同题族（内容 / 证明 / 应用）之间保命的那个旋钮。",
        ]
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-14:]))
    print(f"\n报告 → {REPORT_FILE.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FAQ 阈值标定")
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("gen", help="生成同义问法样本")
    gen.add_argument("--api-key", default=None)
    gen.add_argument("--limit", type=int, default=None, help="只生成前 N 条，便于先试")
    gen.add_argument("--concurrency", type=int, default=6)
    sub.add_parser("calibrate", help="扫阈值并输出报告")
    args = parser.parse_args()

    if args.cmd == "gen":
        asyncio.run(gen_paraphrases(_api_key(args.api_key), args.limit, args.concurrency))
    else:
        asyncio.run(calibrate())


if __name__ == "__main__":
    main()




