"""RAG 证据门标定：用黄金集扫 `RAG_EVIDENCE_THRESHOLD × RAG_EVIDENCE_MARGIN`。

为什么要单独标定：证据门决定"答不答"，它的两个参数（最低分、跨文档领先幅度）
在融合分上和在重排分上完全不是一个量纲。实测（2026-10-20）融合分在不同文档间只差
0.01–0.02，沿用 `margin=0.03` 会把 0.78 分的正确答案也拒掉。

相关性判定（不依赖人工标注每一条检索结果）：
    黄金集每条问题自带 2 段 `contexts`（事实依据）。把证据文本与这两段做**词覆盖**：
    `覆盖 = |contexts 词 ∩ 证据词| / |contexts 词|`，≥ `--rel`（默认 0.5）即认为
    这条证据足以支撑答案。闲聊题没有 contexts，不参与标定（它们本该被意图层拦掉）。

两个用法：

    python scripts/eval_rag.py collect --limit 200       # 跑检索，落 runs.jsonl
    python scripts/eval_rag.py calibrate                 # 扫参数，出报告
    python scripts/eval_rag.py all                       # 两步一起

退出码：0 正常；1 无可用样本 / 无满足精度约束的参数点。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from base.config import get_settings  # noqa: E402
from base.logger import configure, get_logger  # noqa: E402
from rag_system.core.retriever import Retriever  # noqa: E402

log = get_logger(__name__)

EVAL_FILE = BASE_DIR / "datasets" / "07_eval" / "ragas_eval.jsonl"
RUN_FILE = BASE_DIR / "docs" / "analysis" / "rag_eval_runs.jsonl"
REPORT_FILE = BASE_DIR / "docs" / "analysis" / "rag_threshold_report.md"

# 只保留有信息量的词：过滤虚词/标点/单字，否则"的、是、了"会把覆盖度虚高
_STOPWORDS = set("的 了 是 在 和 与 或 也 都 就 而 及 等 这 那 有 我 你 他 它 们 个 中 上 下 为 以 对 从 到 把 被 让 一个 什么 怎么 如何 哪些 多少 请 问".split())
_TIME_PREFIX = re.compile(r"^[^：:]{0,12}[：:]")


def _tokens(text: str) -> set[str]:
    """jieba 分词后留下有信息量的词（长度≥2 或纯数字）。"""
    import jieba

    out: set[str] = set()
    for tok in jieba.lcut(text or ""):
        tok = tok.strip().lower()
        if not tok or tok in _STOPWORDS:
            continue
        if len(tok) >= 2 or tok.isdigit():
            out.add(tok)
    return out


def _context_tokens(contexts: list[str]) -> set[str]:
    """黄金事实词集合。去掉「数学七年级上册：」这类来源前缀，避免把书名当答案内容。"""
    toks: set[str] = set()
    for c in contexts or []:
        toks |= _tokens(_TIME_PREFIX.sub("", c))
    return toks


def _coverage(evidence_text: str, gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(gold & _tokens(evidence_text)) / len(gold)


def _doc_key(parent_id: str) -> str:
    """与 evidence_gate 一致：`{doc_id}:p{n}` 取前缀；没有冒号就整串当文档。"""
    return parent_id.split(":", 1)[0] if ":" in parent_id else parent_id


def _cross_doc_margin(evidences) -> tuple[float, int]:
    if not evidences:
        return 0.0, 0
    top = evidences[0]
    top_doc = _doc_key(top.parent_id)
    others = [e for e in evidences if _doc_key(e.parent_id) != top_doc]
    docs = {_doc_key(e.parent_id) for e in evidences}
    best_other = others[0].score if others else 0.0
    return round(top.score - best_other, 6), len(docs)


async def _collect(limit: int | None, top_k: int, final_n: int, rel: float) -> int:
    rows = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [r for r in rows if r.get("contexts")]
    if limit:
        samples = samples[:limit]
    if not samples:
        print("!! 黄金集里没有带 contexts 的样本")
        return 1

    settings = get_settings()
    retriever = Retriever(settings)
    retriever.store.count()  # 顺便确认集合在
    print(f"== 待标定 {len(samples)} 条（黄金集共 {len(rows)} 条，闲聊/无 contexts 的不参与）")
    RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RUN_FILE.exists():
        RUN_FILE.unlink()

    t0 = time.perf_counter()
    done = 0
    with RUN_FILE.open("w", encoding="utf-8") as fh:
        for row in samples:
            question = row["question"]
            gold = _context_tokens(row["contexts"])
            q_tokens = _tokens(question)
            res = await retriever.search(question, top_k=top_k, final_n=final_n, rerank=False)
            margin, n_docs = _cross_doc_margin(res.evidences)
            evs = [
                {
                    "pid": e.parent_id,
                    "source": Path(e.source).name if e.source else "",
                    "title": e.title,
                    "heading": e.heading,
                    "score": round(float(e.score), 6),
                    "cov": round(_coverage(e.text, gold), 4),
                    # 问题词覆盖率：问题里有几成词真的出现在这条证据里。
                    # 等价于一次极便宜的"词面重排"：余弦分对"同主题但没答到点"的证据不敏感，
                    # 词覆盖能把"提到了问题实词"和"只是话题相近"分开。
                    "qcov": round(len(q_tokens & _tokens(e.text)) / len(q_tokens), 4) if q_tokens else 0.0,
                }
                for e in res.evidences
            ]
            fh.write(
                json.dumps(
                    {
                        "q": question,
                        "intent": row.get("intent", ""),
                        "top_score": evs[0]["score"] if evs else 0.0,
                        "margin": margin,
                        "n_docs": n_docs,
                        "evidences": evs,
                        "cost_ms": res.cost_ms,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fh.flush()
            done += 1
            if done % 25 == 0:
                rate = (time.perf_counter() - t0) / done
                print(f"   {done}/{len(samples)} 平均 {rate:.2f}s/条 预计剩余 {rate * (len(samples) - done) / 60:.1f} 分钟", flush=True)
    print(f"== 检索完成 {done} 条，用时 {(time.perf_counter() - t0) / 60:.1f} 分钟 → {RUN_FILE}")
    return 0


def _load_runs() -> list[dict]:
    if not RUN_FILE.exists():
        return []
    return [json.loads(line) for line in RUN_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sweep(runs: list[dict], rel: float, thresholds: list[float], margins: list[float], qcovs: list[float]) -> list[dict]:
    """在「可答 / 不可答」两个子集上分别评估。

    语料只有几份教材，黄金集却覆盖 7–9 年级全部知识点，所以不能用全体召回率评价门槛：
    那会把"语料里根本没有"算到门槛头上。正确的问法是两个：

    * 可答子集（检索已经找到了支撑答案的证据）：门槛**应该放行**——放行率越高越好；
    * 不可答子集（语料里找不到任何支撑）：门槛**应该拒答**——放行就是误答。
    """
    answerable = [r for r in runs if r["evidences"] and max(e["cov"] for e in r["evidences"][:10]) >= rel]
    unanswerable = [r for r in runs if r not in answerable]
    rows: list[dict] = []
    for t in thresholds:
        for m in margins:
            for qc in qcovs:
                def passed(r: dict, t=t, m=m, qc=qc) -> bool:
                    if not r["evidences"]:
                        return False
                    head = r["evidences"][0]
                    return r["top_score"] >= t and r["margin"] >= m and head["qcov"] >= qc

                ans_ok = sum(1 for r in answerable if passed(r))
                unans_ok = sum(1 for r in unanswerable if passed(r))
                answered = [r for r in runs if passed(r)]
                strict = sum(1 for r in answered if r["evidences"] and r["evidences"][0]["cov"] >= rel)
                rows.append(
                    {
                        "t": t,
                        "m": m,
                        "qc": qc,
                        "pass": round(ans_ok / len(answerable), 4) if answerable else 0.0,
                        "wrong": round(unans_ok / len(unanswerable), 4) if unanswerable else 0.0,
                        "prec1": round(strict / len(answered), 4) if answered else 0.0,
                        "n": len(answered),
                    }
                )
    return rows


def _recall_at(runs: list[dict], rel: float, k: int) -> float:
    hit = sum(1 for r in runs if any(e["cov"] >= rel for e in r["evidences"][:k]))
    return round(hit / max(len(runs), 1), 4)


def _calibrate(rel: float, target_wrong: float) -> int:
    runs = _load_runs()
    if not runs:
        print("!! 没有 runs.jsonl，先跑 collect")
        return 1

    thresholds = [round(0.20 + 0.02 * i, 2) for i in range(31)]  # 0.20 → 0.80
    margins = [round(0.005 * i, 3) for i in range(0, 25)]  # 0 → 0.12
    qcovs = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    rows = _sweep(runs, rel, thresholds, margins, qcovs)

    answerable = [r for r in runs if r["evidences"] and max(e["cov"] for e in r["evidences"][:10]) >= rel]
    unanswerable = [r for r in runs if r not in answerable]

    # 选点规则：先把"该拒的拒掉"（不可答误放行 ≤ target_wrong），再让"该答的答上"
    feasible = [r for r in rows if r["wrong"] <= target_wrong]
    feasible.sort(key=lambda r: (-r["pass"], r["wrong"], -r["qc"], -r["m"], -r["t"]))
    best = feasible[0] if feasible else None

    current = next((r for r in rows if abs(r["t"] - 0.35) < 1e-9 and abs(r["m"] - 0.03) < 1e-9 and r["qc"] == 0.0), None)

    r_at = {k: _recall_at(runs, rel, k) for k in (1, 3, 6, 10)}
    latencies = sorted(r["cost_ms"] for r in runs)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95) - 1] if len(latencies) > 1 else latencies[0]
    avg_cov_top1 = sum(r["evidences"][0]["cov"] for r in runs if r["evidences"]) / max(
        sum(1 for r in runs if r["evidences"]), 1
    )

    lines: list[str] = []
    lines.append("# RAG 证据门标定报告\n")
    lines.append(f"- 样本：{len(runs)} 条（黄金集 `datasets/07_eval` 中带 contexts 的题），"
                 f"其中可答 {len(answerable)} / 不可答 {len(unanswerable)}")
    lines.append(f"- 相关性判据：某条证据对黄金事实的词覆盖 ≥ **{rel}**（`contexts` 是事实依据，不是原文）")
    lines.append(f"- 检索：融合分（无重排），top_k=40 / final_n=10，CPU 上 p50 {p50} ms / p95 {p95} ms")
    lines.append(f"- Recall@1 / @3 / @6 / @10（全样本）：**{r_at[1]} / {r_at[3]} / {r_at[6]} / {r_at[10]}**")
    lines.append(f"- top1 证据对黄金事实的平均覆盖：{avg_cov_top1:.3f}")
    lines.append("")
    lines.append("> 召回率低主要是**语料不齐**：当前语料只有 6 份短教材 + 6 份教案 + 1 本九年级数学 PDF，")
    lines.append("> 黄金集却覆盖 7–9 年级全部知识点（有理数、数轴这些七年级内容语料里没有）。")
    lines.append("> 所以门槛只在「可答 / 不可答」两个子集上评，不能拿全体召回率评价门槛。\n")

    lines.append("## 结论\n")
    if best:
        lines.append(
            f"建议 **RAG_EVIDENCE_THRESHOLD={best['t']}**、**RAG_EVIDENCE_MARGIN={best['m']}**、"
            f"**RAG_EVIDENCE_QCOV={best['qc']}**："
            f"不可答题的误放行率 **{best['wrong']:.1%}**（≤{target_wrong:.0%}），"
            f"同时可答题放行 **{best['pass']:.1%}**；被放行的答案里 top1 相关率 {best['prec1']:.1%}。\n"
        )
        print(
            f"建议 threshold={best['t']} margin={best['m']} qcov={best['qc']} "
            f"可答放行={best['pass']:.1%} 不可答误放={best['wrong']:.1%}"
        )
    else:
        lines.append(f"没有任何参数组合能把误放行压到 ≤{target_wrong:.0%}，说明**分数本身区分不出来**：先改检索（重排/切分/语料），别调门槛。\n")
        print("!! 没有可行参数点")

    if current:
        lines.append(
            f"当前线上 `0.35 / 0.03`：可答放行 {current['pass']:.1%}、不可答误放行 {current['wrong']:.1%}、"
            f"top1 相关率 {current['prec1']:.1%}（放行 {current['n']} 条）——这就是「多数问题被拒答」的量化来源。\n"
        )

    lines.append(f"## 扫描明细（qcov={best['qc'] if best else 0.0}，单元格为「可答放行率 / 不可答误放行率」）\n")
    show_margins = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    show_thresholds = [0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
    table_qc = best["qc"] if best else 0.0
    header = "| 阈值 \\ margin | " + " | ".join(f"{m:.2f}" for m in show_margins) + " |"
    lines.append(header)
    lines.append("|---" * (len(show_margins) + 1) + "|")
    for t in show_thresholds:
        cells = []
        for m in show_margins:
            row = next(
                (r for r in rows if abs(r["t"] - t) < 1e-9 and abs(r["m"] - m) < 1e-9 and r["qc"] == table_qc),
                None,
            )
            cells.append(f"{row['pass']:.0%} / {row['wrong']:.0%}" if row else "-")
        lines.append(f"| {t:.2f} | " + " | ".join(cells) + " |")

    lines.append("\n## qcov 的边际收益（固定 threshold=0.0、margin=0.0，只看词覆盖门槛）\n")
    lines.append("| qcov | 可答放行率 | 不可答误放行率 | 说明 |")
    lines.append("|---|---|---|---|")
    for qc in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        row = next((r for r in rows if r["t"] == 0.2 and r["m"] == 0.0 and r["qc"] == qc), None)
        if row:
            note = "不问词面（当前线上行为）" if qc == 0.0 else ""
            lines.append(f"| {qc:.1f} | {row['pass']:.0%} | {row['wrong']:.0%} | {note} |")

    lines.append("\n## 复跑方式\n")
    lines.append("```powershell")
    lines.append("$env:PYTHONPATH='.'")
    lines.append("F:\\miniconda3\\envs\\ragpro\\python.exe scripts\\eval_rag.py collect")
    lines.append("F:\\miniconda3\\envs\\ragpro\\python.exe scripts\\eval_rag.py calibrate")
    lines.append("```")
    lines.append("\n改了切分尺寸、embedding 模型、语料之后必须重跑——分数尺度会变。")

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"== 报告 → {REPORT_FILE}")
    return 0 if best else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["collect", "calibrate", "all"])
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条（先小样本试水）")
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--final-n", type=int, default=10)
    parser.add_argument("--rel", type=float, default=0.5, help="证据覆盖黄金事实多少算相关")
    parser.add_argument("--target-wrong", type=float, default=0.05, help="不可答题允许的误放行率上限")
    args = parser.parse_args()

    settings = get_settings()
    configure(settings.log_level, log_path=settings.log_path, file_level=settings.log_file_level)

    if args.action in {"collect", "all"}:
        code = asyncio.run(_collect(args.limit, args.top_k, args.final_n, args.rel))
        if code:
            return code
    if args.action in {"calibrate", "all"}:
        return _calibrate(args.rel, args.target_wrong)
    return 0


if __name__ == "__main__":
    sys.exit(main())


