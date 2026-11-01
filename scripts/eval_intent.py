"""理解层评估（CLAUDE.md §7.7）：意图 F1 与"档位是否改善下游决策"的 A/B。

两段验收，顺序不能反（§7.7）：

1. `f1`   —— 前置门槛：`datasets/06_intent` 上的 F1 ≥0.90、`unknown` ≤10%。
   它只是门槛，过了也不代表档位有用。
2. `ab`   —— **真正的验收**：黄金集上"档位路由 vs 全量 DIRECT 基线"按召回切片对比。
   判据是 Recall@10 降幅 ≤1pp、错误过滤率不升；伤召回就回退全量 DIRECT。

两个子命令都可以单独跑：

    python scripts/eval_intent.py f1                    # 规则版 + 大模型版并排（默认 both）
    python scripts/eval_intent.py f1 --engine rule      # 不调大模型，只量规则兜底
    python scripts/eval_intent.py ab --limit 200      # 小样本试水
    python scripts/eval_intent.py ab                  # 全量 763

`f1` 默认要跑大模型（任务 2.1 问的就是「线上 LLM 意图分类的 F1」）。没配 `LLM_API_KEY`
时它会**直接退出（码 3）而不是跳过**——跳过会留下一份没有该项数字、退出码却是 0 的报告。

它同时回答两个不同的问题，**别把两者混成一个 F1**：
① 模型能答多准 —— 用宽松的测量预算（`_MEASURE_BUDGET_MS`），见报告第 1 节；
② 线上 800 ms 的预算里还剩多少 —— 按同一批调用的耗时分布逐条重放，见报告第 5 节。
默认按线上的 800 ms 跑会把"来不及答"记成"答错"，`llm` 引擎整个退化成规则分类器。

`ab` 需要 Milvus + 向量化模型（与线上同一套接线，复用 `main.Container`），
相关性判据 import 自 `eval_rag.py`，与检索/证据门报告同一份实现——各写一份数字迟早对不上。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "scripts"))

from base.config import get_settings  # noqa: E402
from base.labels import INTENT_LABELS, normalize_intent_label  # noqa: E402
from base.logger import configure  # noqa: E402
from eval_rag import _context_tokens, _coverage  # noqa: E402
from understanding import llm_understanding  # noqa: E402
from understanding.intent import IntentClassifier  # noqa: E402
from understanding.profiles import BASELINE, resolve_profile  # noqa: E402

INTENT_DIR = BASE_DIR / "datasets" / "06_intent"
EVAL_FILE = BASE_DIR / "datasets" / "07_eval" / "ragas_eval.jsonl"
REPORT_FILE = BASE_DIR / "docs" / "analysis" / "intent_report.md"

# 判定"这条证据能不能支撑答案"的词覆盖门槛。0.5 是 eval_rag 的默认值（主口径），
# 0.2 是放宽口径：黄金集只有 46 题能在当前语料里找到 ≥0.5 的证据，
# 46 个样本量太小，1pp 的判据会被单题噪声主导，所以并列报一个更宽的切片。
REL_MAIN = 0.5
REL_LOOSE = 0.2


# ---- 1) 意图 F1 ------------------------------------------------------------
# 两个"引擎"共用同一套判定口径与同一批 test 样本，只换分类器：
#
#   rule —— `understanding/intent.py` 的规则分类器（纯 CPU、零调用成本、结果可复现）；
#   llm  —— `understanding/llm_understanding.analyze()` 的大模型路径，**并在标签不合格时
#           退回规则分类器**——这正是线上的行为（`_valid_intent()` 返回 None 就走规则）。
#
# 只测大模型裸输出会把"模型偶尔吐词表外标签"算成全错，而线上那一条其实被规则兜住了；
# 反过来只测规则版则是任务 2.1 要回答的那个问题（线上 LLM 的 F1 到底多少）的回避。
# 所以两个都报，并额外给出"退回了多少条、为什么退"，让读者自己判断。
Predictions = list[tuple[str, str]]          # [(预测标签, 引擎/降级痕迹)]

# 大模型推理的并发度。串行 500 条按 1–2 s/次要十几分钟；并发能压到几分钟。**不要调大**：
# 实测这个账号在 8 路时开始返回 429（`Your current concurrency is 8, which exceeds your
# concurrency limit`），而 `analyze()` 会把 429 统一吞成 `understanding_failed`，
# 那批样本就被记成"模型没答上来"，看着像模型不行，其实是评估脚本自己撞的墙（见 `_invalid()`）。
_LLM_CONCURRENCY = 4

# 测量用的单条预算。**刻意不等于线上的 `UNDERSTANDING_TIMEOUT_MS`（800 ms）**：
# 实测这个网关一次 `analyze()` 要 ~1.2 s，按 800 ms 跑会几乎整批超时，
# `llm` 引擎退化成规则分类器（28 条样本上 71.4% 退回，macro F1 0.94 全是规则给的）——
# 那个数字看着像"模型也不错"，实际一个字都不是模型给的。
# 所以分类质量用宽松预算量，线上预算的真实影响单独在报告第 5 节按耗时分布报告。
_MEASURE_BUDGET_MS = 20000

# 传输层失败的重发次数（不含首次）与退避步长。
# **为什么评估脚本要重发**：500 条 × 并发 4 是评估环境自己制造的突发流量，
# 网关回 429 而 `analyze()` 把它统一吞成 `understanding_failed`——那批样本会被记成
# "模型没答上来"，看着像模型不行，其实是脚本撞的墙。实测同一批样本、同样代码：
# 全量跑 236/500（47.2%）落到这个码，抽 24 条跑则 0 条。
# **只对 `understanding_failed` 重发**：超时 / 坏 JSON / 词表外 / 置信不足都是
# "模型给了但不行"，重发只会拿到同样的结果，还会把评估拖长几倍。
_LLM_RETRIES = 3
_RETRY_BACKOFF_S = 1.5


def _f1(rows: list[dict], preds: Predictions) -> dict:
    """`preds` 与 `rows` 一一对应；由调用方先算好（大模型版是异步的）。"""
    labels = list(INTENT_LABELS)
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    unknown = 0
    wrong: Counter[tuple[str, str]] = Counter()

    for row, (pred_label, method) in zip(rows, preds):
        gold = normalize_intent_label(row["label"])
        methods[method] += 1
        if pred_label == "unknown":
            unknown += 1
        if pred_label == gold:
            tp[gold] += 1
            continue
        wrong[(gold, pred_label)] += 1
        if pred_label in labels:
            fp[pred_label] += 1
        fn[gold] += 1

    per_class = {}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        r = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_class[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f, 4),
            "support": tp[label] + fn[label],
        }
    return {
        "n": len(rows),
        "macro_f1": round(sum(v["f1"] for v in per_class.values()) / len(labels), 4),
        "accuracy": round(sum(tp.values()) / len(rows), 4) if rows else 0.0,
        "unknown_rate": round(unknown / len(rows), 4) if rows else 0.0,
        "per_class": per_class,
        # 引擎/降级分布：大模型版用它回答"有多少条其实不是模型判的"
        "methods": dict(methods.most_common()),
        "confusions": [
            {"gold": g, "pred": p, "n": n} for (g, p), n in wrong.most_common(12)
        ],
    }


def _load(split: str) -> list[dict]:
    path = INTENT_DIR / f"intent_{split}.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _rule_predictions(rows: list[dict], clf: IntentClassifier) -> Predictions:
    return [(r.label, r.method) for r in (clf.classify(row["query"]) for row in rows)]


async def _llm_predictions(
    rows: list[dict],
    clf: IntentClassifier,
    llm: object,
    *,
    budget_ms: int,
    model: str | None,
) -> tuple[Predictions, list[int], int]:
    """逐条走线上路径：先问大模型，标签不可用就退回规则分类器。

    返回 `(预测, 每条耗时 ms, 重发次数)`。耗时**不管成功失败都记**：线上预算调得动不动，
    靠的正是"一次调用要多久"这条分布，只统计成功的那批会把超时整批藏掉。
    **退避睡眠不计入耗时**——那个时长只由本脚本的并发与网关限流决定，线上不存在；
    把它算进去会让第 5 节的预算换算凭空变差。

    **只关改写、不关意图**（`want_rewrite=False`）：这次评估的对象是分类质量，
    改写那一半的结果不进任何指标，开着它只是让每次调用多花几十个 token。

    并发用信号量卡在 `_LLM_CONCURRENCY`：`asyncio.gather` 一次性铺 500 个请求
    会把网关打到 429，而 `analyze()` 把 429 统一吞成 `understanding_failed`——
    那批样本就会被记成"模型没答上来"，看着像模型不行，其实是评估脚本自己撞的墙。
    信号量只挡住突发，挡不住持续限流，所以 `understanding_failed` 另有 `_LLM_RETRIES` 次重发。
    """
    sem = asyncio.Semaphore(_LLM_CONCURRENCY)
    retries = 0

    async def one(row: dict) -> tuple[str, str, int]:
        nonlocal retries
        async with sem:
            cost = 0
            for attempt in range(_LLM_RETRIES + 1):
                t0 = time.perf_counter()
                out = await llm_understanding.analyze(
                    row["query"],
                    llm,  # type: ignore[arg-type]  # 协议在 understanding 侧定义，脚本不做严格绑定
                    budget_ms=budget_ms,
                    model=model,
                    want_rewrite=False,
                    want_intent=True,
                )
                cost = int((time.perf_counter() - t0) * 1000)
                if out.intent is not None:
                    return out.intent.label, "model", cost
                # 走到这里说明模型这一条没给出可用标签（超时 / 坏 JSON / 词表外 / 置信不足），
                # 而线上此时 `_classify()` 正是退回规则分类器。**照做**，否则量出来的
                # 是一个线上不存在的引擎。方法串带上原因，报告里才回答得了"到底多少条是模型判的"。
                reason = out.degraded[0] if out.degraded else "no_intent"
                if reason != "understanding_failed" or attempt == _LLM_RETRIES:
                    break
                retries += 1
                await asyncio.sleep(_RETRY_BACKOFF_S * (attempt + 1))
            return clf.classify(row["query"]).label, f"rule:{reason}", cost

    done = await asyncio.gather(*(one(row) for row in rows))
    return (
        [(label, method) for label, method, _ in done],
        [c for _, _, c in done],
        retries,
    )


def _pct(values: list[int], q: float) -> int:
    """分位数（`q` ∈ [0,1]）。样本少时按最近秩取值，不做插值——
    报告里要的是"最慢的那几条有多慢"，插值出来的小数只会让人以为精度很高。"""
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


# 退回规则的比例超过这个数，本次大模型测量就不能当"模型 F1"引用：
# 分不清"模型判错"与"模型根本没被判"——后者是评估环境的问题，不是分类质量问题。
_MAX_FALLBACK_SHARE = 0.05


def _fallback_share(result: dict) -> float:
    """`llm` 引擎里有多少条其实不是模型判的（超时 / 429 / 坏 JSON / 词表外）。"""
    total = sum(result["methods"].values()) or 1
    return round(1 - result["methods"].get("model", 0) / total, 4)


def _gate(result: dict) -> tuple[bool, str]:
    """§7.7 第一段的门槛：macro F1 ≥0.90、unknown ≤10%。

    返回 (是否通过, 一句话结论)。两个数分开报而不是合成一个布尔，是因为
    "F1 够但 unknown 超标"与"unknown 正常但 F1 不够"要改的地方完全不同：
    前者是分类器不敢下判断，后者是关键词覆盖不够。
    """
    f1_ok = result["macro_f1"] >= 0.90
    unk_ok = result["unknown_rate"] <= 0.10
    if f1_ok and unk_ok:
        return True, "通过"
    why = []
    if not f1_ok:
        why.append(f"macro F1 {result['macro_f1']} < 0.90")
    if not unk_ok:
        why.append(f"unknown {result['unknown_rate']} > 0.10")
    return False, "未通过（" + "；".join(why) + "）"


def _engine_table(results: dict[str, dict]) -> list[str]:
    """引擎对照表：同一批 test 样本、同一套判定，只换分类器。"""
    lines = [
        "| 引擎 | macro F1 | accuracy | unknown 比例 | 样本 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    notes = {
        "rule": "规则分类器裸跑（关键词加权打分）",
        "llm": "**线上路径**：大模型判，标签不可用时退回规则",
    }
    for name, r in results.items():
        lines.append(
            f"| `{name}` | **{r['macro_f1']}** | {r['accuracy']} | {r['unknown_rate']} "
            f"| {r['n']} | {notes.get(name, '')} |"
        )
    return lines


def _per_class_table(results: dict[str, dict]) -> list[str]:
    """按类对照：把两个引擎并排放在同一行，差异一眼可见。"""
    names = list(results)
    head = "| 意图 | support |" + "".join(f" {n} P | {n} R | {n} F1 |" for n in names)
    sep = "|---|---|" + "---|---|---|" * len(names)
    lines = [head, sep]
    first = results[names[0]]["per_class"]
    for label in first:
        support = first[label]["support"]
        cells = "".join(
            f" {results[n]['per_class'][label]['precision']}"
            f" | {results[n]['per_class'][label]['recall']}"
            f" | {results[n]['per_class'][label]['f1']} |"
            for n in names
        )
        lines.append(f"| {label} | {support} |{cells}")
    return lines


def _methods_table(results: dict[str, dict]) -> list[str]:
    lines = ["| 引擎 | 判定来源分布 |", "|---|---|"]
    for name, r in results.items():
        dist = "、".join(f"`{k}` {v}" for k, v in r["methods"].items())
        lines.append(f"| `{name}` | {dist} |")
    return lines


def _confusion_table(results: dict[str, dict]) -> list[str]:
    lines = []
    for name, r in results.items():
        lines += [f"### `{name}`", "", "| 金标 | 预测 | 条数 |", "|---|---|---|"]
        if not r["confusions"]:
            lines.append("| — | — | 0 |")
        for c in r["confusions"]:
            lines.append(f"| {c['gold']} | {c['pred']} | {c['n']} |")
        lines.append("")
    return lines


def _online_budget_section(
    rows: list[dict],
    clf: IntentClassifier,
    preds: Predictions,
    latencies: list[int],
    *,
    online_budget: int,
    measure_budget: int,
) -> list[str]:
    """把测量预算下的耗时分布换算成线上预算下的实际效果。

    做法是**逐条**比较：这条调用若慢于 `online_budget`，线上它就会超时、退回规则分类器，
    于是把它的预测换成规则给的标签，再整体算一遍 F1。这不是估公式，是在同一批样本上
    重放线上的判定路径——唯一的假设是「单条耗时与预算无关」，这一点成立（预算是闸门，
    不是加速器）；另外线上并发通常高于 `_LLM_CONCURRENCY`，实测只会更慢，所以这个数偏乐观。
    """
    rule_preds = _rule_predictions(rows, clf)
    online_preds: Predictions = [
        pred if lat <= online_budget else (rule_preds[i][0], "rule:online_budget")
        for i, (pred, lat) in enumerate(zip(preds, latencies, strict=True))
    ]
    online = _f1(rows, online_preds)
    fit = sum(1 for lat in latencies if lat <= online_budget)
    total = len(latencies) or 1
    share = fit / total

    lines = [
        "## 5. 线上预算的真实影响（`UNDERSTANDING_TIMEOUT_MS`）",
        "",
        f"测量预算 {measure_budget} ms 是为了量出「模型能答多准」；线上给它的只有 "
        f"**{online_budget} ms**。两者不是一回事，这里按同一批调用的耗时分布换算：",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 单次 `analyze()` 耗时 p50 / p90 / p95 / max（ms） "
        f"| {_pct(latencies, 0.50)} / {_pct(latencies, 0.90)} / {_pct(latencies, 0.95)} "
        f"/ {_pct(latencies, 1.0)} |",
        f"| 能在 {online_budget} ms 内完成的调用 | {fit}/{total}（**{share:.1%}**） |",
        f"| `llm` 引擎按线上预算重放的 macro F1 | **{online['macro_f1']}** "
        f"（accuracy {online['accuracy']}、unknown {online['unknown_rate']}） |",
        "",
    ]
    if share < 0.5:
        lines += [
            f"> ⚠ **线上预算下多数调用来不及返回**（{1 - share:.1%} 超时）。也就是说在实际配置里，"
            f"「大模型意图分类」≈ 规则分类器，第 1 节的 `llm` 分数**不代表线上行为**。",
            "> 处置二选一，都要有据：① 调大 `UNDERSTANDING_TIMEOUT_MS`（这笔时间从首 token 预算里扣，"
            "要先量端到端 p95 再动）；② 换更快的路由模型 / 缩短 system prompt 把单次耗时压到预算内。",
            "> 在两者落地之前，不要对外宣称「已用大模型做意图分类」。",
            "",
        ]
    else:
        lines += [
            f"> 线上预算下 {share:.1%} 的调用能拿到模型结论，其余退回规则分类器。",
            "",
        ]
    return lines


async def run_f1(engine: str, limit: int | None = None, budget_ms: int | None = None) -> int:
    """跑 `intent_test.jsonl`（必跑）与 `intent_train.jsonl`（交叉检查）。

    `engine` 决定用哪套分类器：`rule` / `llm` / `both`（默认）。
    **默认 `both`** 是因为任务 2.1 问的就是「线上 LLM 意图分类的 F1 到底多少」——
    只跑规则版等于绕开这个问题。
    """
    settings = get_settings()
    # 预算默认**不取** `UNDERSTANDING_TIMEOUT_MS`：那 800 ms 是线上为了守住首 token
    # 预算给自己上的闸门，拿它来衡量分类质量会把"没来得及答"记成"答错"。
    # 这里要回答的是两个不同的问题，分开量：
    #   ① 模型能答多准 → 用宽松预算（`_MEASURE_BUDGET_MS`），见第 1 节；
    #   ② 线上 800 ms 里还能剩下多少 → 用同一批调用的**耗时分布**反推，见第 5 节。
    budget = budget_ms or _MEASURE_BUDGET_MS
    online_budget = settings.understanding_timeout_ms
    clf = IntentClassifier()

    test_rows = _load("test")
    if not test_rows:
        print(f"没有测试集：{INTENT_DIR / 'intent_test.jsonl'}")
        return 1
    if limit and limit < len(test_rows):
        # 必须**等距**抽样，不能取前 N 条：数据文件是按类聚在一起排的，
        # 取前 60 条正好全是第一类——量出来 macro F1 = 1/7，看着像分类器废了，
        # 实际是抽样把其余六类整个切掉了。
        step = len(test_rows) / limit
        test_rows = [test_rows[int(i * step)] for i in range(limit)]

    wanted = ["rule", "llm"] if engine == "both" else [engine]
    llm = None
    if "llm" in wanted:
        from rag_system.core.llm_gateway import LLMGateway

        llm = LLMGateway(settings)
        if not llm.enabled:
            # **不静默跳过**：这一项是任务 2.1 的核心交付，跳过等于没做，
            # 而"没做"看起来和"通过"一模一样（报告照样生成、退出码照样 0）。
            print(
                "== 缺少大模型配置：LLM_API_KEY / LLM_BASE_URL 未设置，无法评估线上 LLM 意图分类。\n"
                "== 补齐后重跑，或先跑 `--engine rule` 看规则兜底的水平（那**不是**线上路径）。"
            )
            return 3

    test_results: dict[str, dict] = {}
    preds: Predictions = []
    llm_latencies: list[int] = []
    llm_retries = 0
    if "rule" in wanted:
        test_results["rule"] = _f1(test_rows, _rule_predictions(test_rows, clf))
    if "llm" in wanted:
        print(f"== 大模型判定 {len(test_rows)} 条（并发 {_LLM_CONCURRENCY}，预算 {budget} ms/条）…")
        t0 = time.perf_counter()
        preds, llm_latencies, llm_retries = await _llm_predictions(
            test_rows, clf, llm, budget_ms=budget, model=settings.llm_router_model or None
        )
        test_results["llm"] = _f1(test_rows, preds)
        print(f"== 用时 {time.perf_counter() - t0:.1f}s")

    # train 只跑规则版：它是关键词表的调参面（F1 天然偏高），用途是看"test 掉多少"；
    # 2100 条大模型调用换这一个对照数不值得，而且 train 不参与任何验收。
    train_rows = _load("train")
    train_rule = _f1(train_rows, _rule_predictions(train_rows, clf)) if train_rows else None

    gated = "llm" if "llm" in test_results else "rule"
    ok, verdict = _gate(test_results[gated])

    # 测量有效性：退回规则的比例过高时，`llm` 那一行的数字主要是规则给的，
    # 拿它当"线上 LLM 的 F1"就是拿评估环境的故障冒充分类质量（§7.7「不许伪造」的同一口径）。
    invalid = ""
    if "llm" in test_results:
        share = _fallback_share(test_results["llm"])
        if share > _MAX_FALLBACK_SHARE:
            invalid = (
                f"> ⚠ **本次大模型测量无效**：{share:.1%} 的样本退回了规则分类器"
                f"（门槛 {_MAX_FALLBACK_SHARE:.0%}）。上面 `llm` 那一行的数字主要来自规则兜底，"
                f"**不能当模型 F1 引用**。\n"
                f"> 排查：429 限流（调小 `_LLM_CONCURRENCY`）、`--budget-ms` 太短、网关不可达。\n"
            )

    lines = [
        "# 意图分类评估报告（§7.7 第一段：前置门槛）",
        "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 测试集：`datasets/06_intent/intent_test.jsonl`（{len(test_rows)} 条）"
        f"、训练集 {len(train_rows)} 条",
        # 只有真跑了大模型才写这一行：规则版的报告上顶着一行模型名，
        # 读的人会以为那份数字来自模型
        *(
            [
                f"- 大模型：`{settings.llm_router_model}`，测量预算 {budget} ms/条"
                f"（线上 {online_budget} ms，换算见第 5 节）",
                # 重发次数是**测量环境的自白**：它不影响任何 F1，但读者有权知道
                # 这批数字是在"脚本撞过几次 429 并重发成功"之后拿到的
                f"- 传输层重发：{llm_retries} 次（`understanding_failed` 专用，最多 {_LLM_RETRIES} 次/条）"
            ]
            if "llm" in test_results
            else ["- 大模型：**本次未运行**（`--engine rule`）"]
        ),
        "",
        "## 0. 口径（先读这一段，否则下面的数字会被读错）",
        "",
        "1. **调参只用 train**（关键词表与 `MIN_SCORE` / `MIN_MARGIN`），test **只在最后跑一次**；",
        "   所以 train 的 F1 是「被调过的那一侧」，看到它高于 test 是正常的，不算泄漏。",
        "2. **`llm` 引擎 = 线上路径**：大模型判，标签不可用（超时 / 坏 JSON / 词表外 / 置信 <0.6）时",
        "   退回规则分类器——这正是 `orchestration/chat.py::_classify()` 的行为。",
        "   只测模型裸输出会把「偶尔吐个词表外标签」算成全错，而线上那一条其实被规则兜住了。",
        "3. `methods` 列显示判定来源：`model` 是大模型判的，`rule:<原因>` 是退回来的。",
        "   **`llm` 引擎的高分若主要来自 `rule:*`，说明大模型没起作用**——这一项必须看。",
        "4. F1 只是门槛。真正上线的依据是 `ab` 子命令的档位 A/B（§7.7：F1 高但下游不改善，不予上线）。",
        "",
        "## 1. 主口径",
        "",
        *_engine_table(test_results),
        "",
        f"**门槛（按 `{gated}` 判）：{verdict}**",
        "",
        *([invalid, ""] if invalid else []),
        "## 2. 分类别",
        "",
        *_per_class_table(test_results),
        "",
        "## 3. 判定来源",
        "",
        *_methods_table(test_results),
        "",
        "## 4. 主要混淆",
        "",
        *_confusion_table(test_results),
    ]

    if llm_latencies:
        lines += _online_budget_section(
            test_rows,
            clf,
            preds,
            llm_latencies,
            online_budget=online_budget,
            measure_budget=budget,
        )

    if train_rule is not None:
        lines += [
            "## 6. 交叉检查（`intent_train.jsonl`，规则版）",
            "",
            f"macro F1 **{train_rule['macro_f1']}**、accuracy {train_rule['accuracy']}、"
            f"unknown {train_rule['unknown_rate']}（n={train_rule['n']}）",
            "",
            "同一套关键词表在调参面上本来就会更高；两片差距过大（>10pp）才是过拟合的信号。",
            "大模型版不跑 train：2100 次调用换一个不参与验收的对照数，不划算。",
            "",
        ]

    lines += [
        "## 7. 局限",
        "",
        "- 这份数据集是按 7 类均衡构造的（每类约 300 / 71 条），**真实流量不是均衡分布**：",
        "  线上大概率以「学习讲解」和「闲聊」为主，macro F1 会把长尾类的错误放大，",
        "  读的时候同时看 `support` 与混淆表，别只看总分。",
        "- 大模型路径每次调用都会重跑，**同一批样本两次的结果可能不同**（温度 >0 时）。",
        "  报告里没有置信区间：要判断「两次差 1pp 算不算差异」，得重复跑几轮再比。",
        "- 拒答（`超出范围`）的漏判代价与别类不同：把教学问题误判成越界会**直接拒答用户**，",
        "  所以 §7.7 在 `resolve_profile()` 里留了「带教学场景词就改走检索」的安全阀；",
        "  混淆表里 `超出范围 → 学习讲解` 这类方向值得单独盯。",
        "",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"== 报告 → {REPORT_FILE}")
    print(f"== 门槛（{gated}）：{verdict}")
    if invalid:
        print("== 本次大模型测量无效（退回规则比例过高），见报告第 1 节")
        return 4
    return 0 if ok else 2


# ---- 2) 档位 A/B -----------------------------------------------------------
def _recall(pairs: list[tuple[list[float], list[float]]], rel: float, k: int = 10) -> dict:
    """`pairs` 每项是 (baseline 的覆盖度序列, treatment 的覆盖度序列)。"""
    base_hit = sum(1 for b, _ in pairs if any(c >= rel for c in b[:k]))
    treat_hit = sum(1 for b, t in pairs if any(c >= rel for c in t[:k]))
    # 错误过滤率：基线上找得到、档位路由后找不到 —— 这是唯一"会让用户彻底拿不到答案"的伤害
    dropped = sum(
        1
        for b, t in pairs
        if any(c >= rel for c in b[:k]) and not any(c >= rel for c in t[:k])
    )
    base_any = base_hit or 1
    return {
        "n": len(pairs),
        "baseline_recall": round(base_hit / len(pairs), 4) if pairs else 0.0,
        "treat_recall": round(treat_hit / len(pairs), 4) if pairs else 0.0,
        "delta_pp": round((treat_hit - base_hit) / len(pairs) * 100, 2) if pairs else 0.0,
        "error_filter_rate": round(dropped / base_any, 4),
        "dropped": dropped,
        "baseline_hits": base_hit,
    }


async def _run_ab(limit: int | None, stride: int) -> int:
    from main import Container  # 延迟导入：`f1` 子命令不该被 Milvus 依赖拖住

    settings = get_settings()
    container = Container(settings)
    await container.load_rag()
    if not container.rag.get("ready"):
        print(f"RAG 未就绪，无法做 A/B：{container.rag.get('reason')}")
        return 1
    retriever = container.rag["retriever"]

    rows = [
        json.loads(line)
        for line in EVAL_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples = [r for r in rows if r.get("contexts")]
    if stride > 1:
        samples = samples[::stride]
    if limit:
        samples = samples[:limit]

    clf = IntentClassifier()
    pairs_main: list[tuple[list[float], list[float]]] = []
    pairs_loose: list[tuple[list[float], list[float]]] = []
    per_profile: dict[str, Counter[str]] = {}
    records: list[dict] = []
    started = time.perf_counter()
    filtered = 0

    for i, row in enumerate(samples, start=1):
        question = row["question"]
        gold = _context_tokens(row["contexts"])

        intent = clf.classify(question)
        # 与线上同一个函数选档位：安全阀开不开、降到哪个档，都由它决定
        profile, degraded = resolve_profile(question, intent.label)
        doc_types = list(profile.doc_types) or None

        base = await retriever.search(question, final_n=settings.rag_final_n)
        base_cov = [_coverage(e.text, gold) for e in base.evidences]

        if not profile.retrieval:
            # `no_retrieval` 档在线上真的不检索 —— 处理组必须如实为空，
            # 不能拿基线结果冒充（那会把"跳过检索"的代价藏起来，A/B 就变成自证清白）
            treat_cov = []
        elif doc_types is None:
            # 档位不限 doc_type ⇒ 处理组就是基线本身，不必再查一次
            treat_cov = base_cov
        else:
            filtered += 1
            treat = await retriever.search(
                question, final_n=settings.rag_final_n, doc_types=doc_types
            )
            treat_cov = [_coverage(e.text, gold) for e in treat.evidences]

        pairs_main.append((base_cov, treat_cov))
        # 放宽口径：用同一批覆盖度、换更低的相关性门槛，样本量从 46 提到上百
        pairs_loose.append((base_cov, treat_cov))

        stat = per_profile.setdefault(profile.name, Counter())
        stat["n"] += 1
        # `no_retrieval` 档线上不检索，耗时记 0 才是它真实的样子
        stat["retrieval_ms"] += base.cost_ms if profile.retrieval else 0
        stat["empty"] += 1 if not base.evidences else 0
        stat["skipped"] += 0 if profile.retrieval else 1
        records.append(
            {
                "q": question,
                "intent": intent.label,
                "profile": profile.name,
                "retrieval": profile.retrieval,
                "degraded": list(degraded),
                "doc_types": doc_types or [],
                "base_top": round(base_cov[0], 3) if base_cov else 0.0,
                "treat_top": round(treat_cov[0], 3) if treat_cov else 0.0,
            }
        )
        if i % 20 == 0:
            rate = (time.perf_counter() - started) / i
            print(
                f"   {i}/{len(samples)}  {rate:.2f}s/题  剩余约 {rate * (len(samples) - i) / 60:.1f} 分钟",
                flush=True,
            )

    main_ab = _recall(pairs_main, REL_MAIN)
    loose_ab = _recall(pairs_loose, REL_LOOSE)

    lines = [
        "# 意图档位 A/B 报告（§7.7 第二段：真正的验收）",
        "",
        f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本：{len(samples)} 条黄金集题（带 contexts）；其中 {filtered} 条被档位收窄了 `doc_type`",
        f"- 基线：`{BASELINE.name}`（不限 doc_type、必须检索）= `FEATURE_INTENT=false` 的行为",
        "- 处理组：按意图档位收窄 `doc_type` 后重查（前缀过滤下推到 Milvus，不是查后过滤）；"
        "`no_retrieval` 档**按线上行为真的不检索**，证据记空——跳过检索的代价要算进去，不能拿基线结果冒充",
        "- 档位判定调 `understanding/profiles.py::resolve_profile`，与编排层是同一份代码",
        "- 相关性判据：证据对黄金事实的词覆盖 ≥ 门槛（与 `eval_rag.py` 同一份实现）",
        "",
        f"## 1. 主口径（覆盖度 ≥{REL_MAIN}）",
        "",
        "| 指标 | 值 | 判据 |",
        "|---|---|---|",
        f"| 基线 Recall@10 | {main_ab['baseline_recall']}（{main_ab['baseline_hits']}/{main_ab['n']}） | — |",
        f"| 档位 Recall@10 | {main_ab['treat_recall']} | — |",
        f"| 降幅 | {main_ab['delta_pp']} pp | **≤1 pp** |",
        f"| 错误过滤率 | {main_ab['error_filter_rate']}（{main_ab['dropped']} 条） | **不升**（基线 0） |",
        "",
        f"## 2. 放宽口径（覆盖度 ≥{REL_LOOSE}，样本量更大，用来判断 1pp 是噪声还是真伤害）",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 基线 Recall@10 | {loose_ab['baseline_recall']}（{loose_ab['baseline_hits']}/{loose_ab['n']}） |",
        f"| 档位 Recall@10 | {loose_ab['treat_recall']} |",
        f"| 降幅 | {loose_ab['delta_pp']} pp |",
        f"| 错误过滤率 | {loose_ab['error_filter_rate']}（{loose_ab['dropped']} 条） |",
        "",
        "## 3. 按档位分布",
        "",
        "| 档位 | 题数 | 平均检索耗时 | 空召回 | 跳过检索 |",
        "|---|---|---|---|---|",
    ]
    for name, stat in sorted(per_profile.items(), key=lambda kv: -kv[1]["n"]):
        lines.append(
            f"| {name} | {stat['n']} | {stat['retrieval_ms'] / stat['n']:.0f} ms "
            f"| {stat['empty']} | {stat['skipped']} |"
        )

    verdict = "**通过**" if main_ab["delta_pp"] >= -1.0 and main_ab["dropped"] == 0 else "**未通过**"
    skipped_total = sum(stat["skipped"] for stat in per_profile.values())
    lines += [
        "",
        "## 4. 结论",
        "",
        f"- 判据（Recall@10 降幅 ≤1pp 且无错误过滤）：{verdict}",
    ]
    if main_ab["dropped"]:
        lines.append(
            f"- 有 {main_ab['dropped']} 题在档位路由后丢了正确答案，逐条看 `docs/analysis/intent_ab_details.jsonl`。"
            "线上已有 `doc_type_relaxed` 兜底（收窄后 0 命中就放开重检），"
            "但它只能救「完全空召回」，救不了「还有别的证据、正确答案却被滤掉」的情形。"
        )
    lines += [
        f"- 本批 {len(samples)} 题里真正跳过检索的有 **{skipped_total}** 题"
        f"（{skipped_total / len(samples) * 100:.1f}%）。黄金集全是学业问题，"
        "这个数接近 0 是好事——说明疑问形态安全阀把「短问句被当成寒暄」挡在了检索之外；"
        "闲聊档的延迟收益要在真实闲聊流量上量，不在黄金集上量。",
        "- 档位判定与编排层共用 `understanding/profiles.py::resolve_profile`，"
        "所以这里的数字就是线上行为，不是脚本里的另一份实现。",
        "",
    ]
    REPORT_FILE.with_name("intent_ab_report.md").write_text("\n".join(lines), encoding="utf-8")
    REPORT_FILE.with_name("intent_ab_details.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"== 报告 → {REPORT_FILE.with_name('intent_ab_report.md')}")
    return 0 if verdict.startswith("**通过") else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="理解层评估：意图 F1 与档位 A/B")
    parser.add_argument("mode", choices=["f1", "ab"], help="f1=前置门槛；ab=档位 A/B")
    parser.add_argument(
        "--engine",
        choices=["rule", "llm", "both"],
        default="both",
        help="f1 用哪套分类器：rule=规则版；llm=大模型版（标签不可用退回规则）；both=并排（默认）",
    )
    parser.add_argument(
        "--budget-ms",
        type=int,
        default=None,
        help=(
            f"单条理解调用的时限，默认 {_MEASURE_BUDGET_MS}（测量用，不等于线上）；"
            "线上预算下的实际效果由脚本按耗时分布换算，见报告第 5 节"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="等距抽 N 题（试水；文件按类聚排，不能取前 N）")
    parser.add_argument("--stride", type=int, default=1, help="固定步长抽样，可复跑")
    args = parser.parse_args()

    configure("WARNING")
    if args.mode == "f1":
        return asyncio.run(run_f1(args.engine, args.limit, args.budget_ms))
    return asyncio.run(_run_ab(args.limit, args.stride))


if __name__ == "__main__":
    raise SystemExit(main())
