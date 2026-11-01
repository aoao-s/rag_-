"""端到端评估（CLAUDE.md §7.8 第 4 项）：把黄金集逐题喂给 `/v1/chat` 的编排内核。

为什么要单起一个脚本，而不是扩 `eval_rag.py`：
`eval_rag.py` 只回答"检索该不该放行"，它不经过 FAQ、不经过生成、不经过引用对齐。
本脚本跑的是**用户真正拿到的那条链路**（`orchestration.chat.ChatPipeline`），
四组指标缺一不可——答案正确性、引用正确率、拒答表现、延迟与召回（§7.8 第 4 项）。

LLM 三种模式（`--llm`）：

* `live`：真实网关。只有这一档，"答案正确性"才有意义。
* `stub`：抽取式替身——取最贴题的一句证据原文并附 `[n]`。它**不代表答案质量**，
  只验证"证据门 → 提示词装配 → 引用对齐 → 出口"这条管线能否跑通、引用能不能对上。
  没有 key 时自动落到这一档，报告头会写明。
* `off`：不生成。证据门通过就记成"本可作答"，用来单独观察检索与门的边界。

相关性判据与 `eval_rag.py` **完全同一份代码**（直接 import），否则两份报告的数字不可比：
证据文本对黄金事实的词覆盖 ≥ `--rel`（默认 0.5）即认为它能支撑答案；
一题的 top-10 里存在这样的证据 ⇒ 可答，否则不可答。

注意"不可答"的语义：黄金事实在**当前语料**里找不到，不等于"系统不该回答"。
FAQ 高置信命中是另一条知识路径（人工审核过的标准问答），单列一档，不算编造。

用法（PowerShell）：

    $env:PYTHONPATH='.'
    F:\\miniconda3\\envs\\ragpro\\python.exe scripts\\eval_chat.py --limit 60   # 小样本试水
    F:\\miniconda3\\envs\\ragpro\\python.exe scripts\\eval_chat.py              # 全量

产物：`docs/analysis/chat_eval_details.jsonl`（逐题）+ `docs/analysis/chat_eval_report.md`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
# 相关性判据只允许有一份实现：两个脚本各写一份，数字迟早对不上
sys.path.insert(0, str(BASE_DIR / "scripts"))

from base.config import get_settings  # noqa: E402
from base.logger import configure  # noqa: E402
from eval_rag import _context_tokens, _coverage, _tokens  # noqa: E402
from main import Container  # noqa: E402 - 复用线上同一套接线，避免"评估跑的是另一套代码"
from rag_system.core.llm_gateway import LLMResult  # noqa: E402

EVAL_FILE = BASE_DIR / "datasets" / "07_eval" / "ragas_eval.jsonl"
DETAIL_FILE = BASE_DIR / "docs" / "analysis" / "chat_eval_details.jsonl"
REPORT_FILE = BASE_DIR / "docs" / "analysis" / "chat_eval_report.md"

_BLOCK_HEAD = re.compile(r"^\[(\d+)\]")
_SENTENCE = re.compile(r"(?<=[。！？!?])")


class StubLLM:
    """抽取式替身：只为验证管线，不代表答案质量（报告头会写明）。

    从装配好的 prompt 里挑一条证据，取其中与问句实词重合最多的一句原文，附上它自己的编号。
    这样 `citation.align()` 拿到的是**真实证据里的字句**，句级支持度必然达标——
    它能验证"引用对得上"，但回答得好不好要 `--llm live`。
    """

    def __init__(self) -> None:
        self.enabled = True
        self.calls = 0

    async def complete(
        self, messages: list[dict], *, budget_ms: int, model: str | None = None
    ) -> LLMResult:
        self.calls += 1
        text = self._extract(messages)
        if not text:
            # 与真实模型"答不出"同构：返回空串，交给 align() 判成无引用 → 兜底
            text = "知识库暂无该内容。"
        return LLMResult(text=text, model="stub", prompt_tokens=0, completion_tokens=0)

    @staticmethod
    def _extract(messages: list[dict]) -> str:
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "【问题】" not in user:
            return ""
        body, question = user.split("【问题】", 1)[0], user.rsplit("【问题】", 1)[1]
        q_terms = _tokens(question)
        best_n, best_sentence, best_score = 0, "", 0.0
        for block in body.split("\n\n"):
            head = _BLOCK_HEAD.match(block)
            if head is None:
                continue
            n = int(head.group(1))
            for sentence in _SENTENCE.split(block[head.end() :]):
                sentence = sentence.strip().replace("\n", "")
                if len(sentence) < 8:
                    continue
                score = len(q_terms & _tokens(sentence))
                if score > best_score:
                    best_n, best_sentence, best_score = n, sentence, score
        if best_score < 2:  # 一句都贴不上题，就当模型答不出，别硬编
            return ""
        return f"{best_sentence}[{best_n}]"


class DisabledLLM:
    """`--llm off`：证据门通过也只记"本可作答"，不进生成。"""

    enabled = False

    async def complete(self, *args: Any, **kwargs: Any) -> LLMResult:  # pragma: no cover
        raise AssertionError("--llm off 不应调用生成")


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return round(ordered[idx], 1)


def _load_samples(limit: int | None, stride: int) -> list[dict]:
    rows = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = [r for r in rows if r.get("contexts")]
    if stride > 1:
        # 固定步长抽样：不随机、可复跑，且能覆盖到题库各段而不是只取前 N 条
        samples = samples[::stride]
    return samples[:limit] if limit else samples


def _cited_relevance(citations: list[dict], evidences: list[Any], gold: set[str], rel: float) -> tuple[int, int]:
    """(能定位到本轮证据的条数, 其中真的相关的条数)。"""
    by_parent = {e.parent_id: e for e in evidences}
    located = relevant = 0
    for c in citations:
        ev = by_parent.get(c.get("parent_id"))
        if ev is None:
            continue
        located += 1
        if _coverage(ev.text, gold) >= rel:
            relevant += 1
    return located, relevant


async def _evaluate(samples: list[dict], container: Container, rel: float, llm_mode: str) -> list[dict]:
    retriever = container.rag["retriever"]
    pipeline = container.chat
    assert pipeline is not None
    records: list[dict] = []
    started = time.perf_counter()
    for i, row in enumerate(samples, start=1):
        question = row["question"]
        gold = _context_tokens(row["contexts"])

        result = await retriever.search(question, final_n=container.settings.rag_final_n)
        covs = [_coverage(e.text, gold) for e in result.evidences]
        answerable = any(c >= rel for c in covs[:10])

        state = await pipeline.ask(question)
        evidence_list = state.retrieval.evidences if state.retrieval else []
        located, relevant = _cited_relevance(state.citations, evidence_list, gold, rel)

        records.append(
            {
                "q": question,
                "intent": row.get("intent", ""),
                "answerable": answerable,
                "top_cov": round(covs[0], 3) if covs else 0.0,
                "cover10": [round(c, 3) for c in covs[:10]],
                "first_rank": next((j for j, c in enumerate(covs[:10], start=1) if c >= rel), 0),
                "retrieval_ms": result.cost_ms,
                "source": state.source,
                "tier": state.faq.tier if state.faq else "-",
                "gate": state.gate.reason if state.gate else "-",
                "citations": len(state.citations),
                "cite_located": located,
                "cite_relevant": relevant,
                # 答案对黄金事实的词覆盖：自动化的"答得沾不沾边"代理指标，
                # 不是正确性判定（那要人工或裁判模型），报告里按代理指标写
                "answer_cov": round(_coverage(state.answer, gold), 3) if state.source == "rag" else 0.0,
                "degraded": list(state.degraded),
                "latency": state.latency_ms,
                "llm": llm_mode,
            }
        )
        if i % 20 == 0:
            rate = (time.perf_counter() - started) / i
            print(f"   {i}/{len(samples)}  {rate:.2f}s/题  剩余约 {rate * (len(samples) - i) / 60:.1f} 分钟", flush=True)
    return records


def _summarise(records: list[dict], rel: float, llm_mode: str, meta: dict) -> str:
    n = len(records)
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    def recall_at(k: int, subset: list[dict]) -> float:
        return round(sum(1 for r in subset if r["first_rank"] and r["first_rank"] <= k) / max(len(subset), 1), 4)

    mrr = round(sum(1 / r["first_rank"] for r in records if r["first_rank"]) / max(n, 1), 4)

    import math

    ndcg = 0.0
    for r in records:
        rels = [1 if c >= rel else 0 for c in r["cover10"]]
        dcg = sum(v / math.log2(i + 1) for i, v in enumerate(rels, start=1))
        idcg = sum(v / math.log2(i + 1) for i, v in enumerate(sorted(rels, reverse=True), start=1))
        ndcg += dcg / idcg if idcg else 0.0
    ndcg = round(ndcg / max(n, 1), 4)

    sources = Counter(r["source"] for r in records)
    tiers = Counter(r["tier"] for r in records)
    gate_reasons = Counter(r["gate"] for r in records if r["gate"] != "-")

    # 出口 × 可答性：左边"该答的答上了没有"，右边"该拒的拒掉了没有"。
    # `refusal` 必须列在这里：漏了它，`cross[r["source"]]` 会为拒答行静默建一个新计数桶，
    # 表格里看不到，两列相加也就不再等于 n（§7.7 `OUT_OF_SCOPE` 的出口）。
    cross: dict[str, Counter] = {
        s: Counter() for s in ("faq", "rag", "clarify", "fallback", "refusal")
    }
    for r in records:
        cross[r["source"]]["answerable" if r["answerable"] else "unanswerable"] += 1

    answered_ans = sum(1 for r in answerable if r["source"] in ("faq", "rag"))
    # **越界拒答不计入"拒掉了"**：黄金集里全是教学问句，这里出现 `refusal` 只可能是
    # 意图误判（把该答的题判成 `超出范围`）。把它算成"拒得对"等于让一个分类错误
    # 伪装成成绩——它与 `fallback/clarify`（"知识库没这条"）是两种语义，必须分开看。
    refused_unans = sum(1 for r in unanswerable if r["source"] in ("fallback", "clarify"))
    refusal_rows = [r for r in records if r["source"] == "refusal"]
    refusal_ans = sum(1 for r in refusal_rows if r["answerable"])
    faq_on_unans = sum(1 for r in unanswerable if r["source"] == "faq")
    rag_on_unans = sum(1 for r in unanswerable if r["source"] == "rag")

    rag_rows = [r for r in records if r["source"] == "rag"]
    cite_located = sum(r["cite_located"] for r in rag_rows)
    cite_total = sum(r["citations"] for r in rag_rows)
    cite_relevant = sum(r["cite_relevant"] for r in rag_rows)
    cite_all_rel = sum(1 for r in rag_rows if r["citations"] and r["cite_relevant"] == r["citations"])
    answer_cov = round(sum(r["answer_cov"] for r in rag_rows) / max(len(rag_rows), 1), 3)

    lat = {
        key: (_pct([r["latency"].get(key, 0) for r in records], 50), _pct([r["latency"].get(key, 0) for r in records], 95))
        for key in ("faq", "retrieve", "generate", "total")
    }

    lines: list[str] = []
    lines.append("# 端到端问答评估报告\n")
    lines.append(f"- 生成：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 样本：{n} 条（黄金集 `datasets/07_eval` 中带 contexts 的题；可答 {len(answerable)} / 不可答 {len(unanswerable)}）")
    lines.append(f"- 相关性判据：证据对黄金事实的词覆盖 ≥ **{rel}**（与 `eval_rag.py` 同一实现）")
    lines.append(f"- 语料：FAQ {meta['faq_docs']} 条（索引 {meta['faq_indexed']} 条）/ 向量库 {meta['rag_docs']} 个子块"
                 f"（embedder={meta['embedder']}，rerank={'on' if meta['rerank'] else 'off'}）")
    lines.append(f"- 生成模型：**{llm_mode}**" + ("　← 替身，只验证管线，**不代表答案质量**" if llm_mode == "stub" else ""))
    lines.append("")
    if llm_mode != "live":
        lines.append("> ⚠ 本轮没有真实 LLM（`LLM_API_KEY` 未配置或显式 `--llm stub`）：")
        lines.append("> 「引用可定位 / 引用相关 / 拒答」这些**结构性**指标有效，")
        lines.append("> 「答案正确性」只能等拿到 key 后跑 `--llm live` 再评。\n")

    lines.append("## 1. 检索质量（与生成无关，全样本）\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| Recall@1 / @3 / @6 / @10 | {recall_at(1, records)} / {recall_at(3, records)} / {recall_at(6, records)} / {recall_at(10, records)} |")
    lines.append(f"| MRR@10 | {mrr} |")
    lines.append(f"| nDCG@10 | {ndcg} |")
    lines.append(f"| 可答子集 Recall@10 | {recall_at(10, answerable)}（n={len(answerable)}） |")
    lines.append("")
    lines.append("> 全体召回天然偏低：黄金集覆盖 7–9 年级全部知识点，语料只有几份教材与教案。")
    lines.append("> 因此门槛只看子集（下面两节），召回数字用于**纵向对比**（改切分 / 换模型 / 加语料前后）。\n")

    lines.append("## 2. 出口分布\n")
    lines.append("| source | 全部 | 可答题 | 不可答题 |")
    lines.append("|---|---|---|---|")
    for s in ("faq", "rag", "clarify", "fallback", "refusal"):
        lines.append(f"| {s} | {sources.get(s, 0)} | {cross[s]['answerable']} | {cross[s]['unanswerable']} |")
    lines.append("")
    lines.append(f"- FAQ 档位分布：{dict(tiers)}")
    lines.append(f"- 证据门拒绝原因：{dict(gate_reasons) or '（无）'}")
    if refusal_rows:
        lines.append(
            f"- 越界拒答：{len(refusal_rows)} 条，其中**可答** {refusal_ans} 条"
            "（可答的那些是意图误判，不算成绩——拒答不计入下一节的「不可答题被拒」）"
        )
    else:
        # 黄金集目前没有「超出范围」样本（`scripts/export_data_inventory.py` 记的缺口），
        # 所以 0 只能说明"没测到"，不能说这条路径验证过。
        lines.append("- 越界拒答：0 条。黄金集无越界样本，**不代表**拒答路径已验证（单测见 `tests/test_chat_refusal.py`）")
    lines.append("")

    lines.append("## 3. 拒答表现\n")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| 可答题被答上 | {answered_ans}/{len(answerable)} = {answered_ans / max(len(answerable), 1):.1%} | 越低说明越保守 |")
    lines.append(f"| 不可答题被拒 | {refused_unans}/{len(unanswerable)} = {refused_unans / max(len(unanswerable), 1):.1%} | **越高越好** |")
    lines.append(f"| 不可答题走了 RAG 生成 | {rag_on_unans} | 需要人看：可能是语料其实能答 |")
    lines.append(f"| 不可答题 FAQ 高置信直答 | {faq_on_unans} | 另一条知识路径，不算编造 |")
    lines.append(f"| 越界拒答（`refusal`） | {len(refusal_rows)}（可答 {refusal_ans}） | 可答的那几条是**误拒**，需人看 |")
    lines.append("")
    lines.append("> 「不可答」= 黄金事实在当前语料里找不到，**不等于**系统不该回答。")
    lines.append("> 因此 FAQ 高置信命中单独统计，不并入「误答」。\n")

    lines.append("## 4. 引用质量（只统计 `source=rag` 的答案）\n")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|---|---|---|")
    lines.append(f"| 有引用的答案 | {len(rag_rows)} 条 | 其余都走了兜底 |")
    lines.append(f"| 引用可定位率 | {cite_located}/{cite_total} = {cite_located / max(cite_total, 1):.1%} | `parent_id` 必须能映射回本轮证据，期望 100% |")
    lines.append(f"| 引用相关率 | {cite_relevant}/{cite_total} = {cite_relevant / max(cite_total, 1):.1%} | 被引证据对黄金事实的覆盖 ≥ {rel} |")
    lines.append(f"| 题级「引用全相关」 | {cite_all_rel}/{len(rag_rows)} = {cite_all_rel / max(len(rag_rows), 1):.1%} | 有一条引错就不算 |")
    lines.append(f"| 答案覆盖黄金事实（代理） | {answer_cov} | 词面覆盖，非正确性判定 |")
    lines.append("")

    lines.append("## 5. 延迟（毫秒，含 FAQ + 检索 + 生成）\n")
    lines.append("| 阶段 | p50 | p95 |")
    lines.append("|---|---|---|")
    for key, label in (("faq", "FAQ"), ("retrieve", "检索"), ("generate", "生成"), ("total", "端到端")):
        lines.append(f"| {label} | {lat[key][0]} | {lat[key][1]} |")
    lines.append("")

    lines.append("## 6. 结论\n")
    if llm_mode != "live":
        lines.append("- 本轮**不能**给「答案正确性」结论：替身模型的回答是从证据里抄的句子，")
        lines.append("  它衡量的只是「证据选得对不对」和「引用对得上不上」。")
    lines.append(f"- 引用可定位率 {cite_located / max(cite_total, 1):.1%}：对齐环节按预期工作，没有出现无法回溯的来源。")
    if cite_total and cite_relevant / cite_total < 0.6:
        lines.append(f"- 引用相关率只有 {cite_relevant / cite_total:.1%}：**证据选错了**才是当前瓶颈，")
        lines.append("  不是生成环节。先补语料 / 改切分 / 上重排，别去调提示词。")
    lines.append(f"- 不可答题误生成 {rag_on_unans} 条：逐条看 `chat_eval_details.jsonl` 里 `source=rag` 且 `answerable=false` 的记录。")
    if refusal_ans:
        lines.append(f"- **{refusal_ans} 条可答题被越界拒答**：这是意图误判，不是「更保守」。")
        lines.append("  先看问句里有没有教学场景词（`understanding/profiles.py::looks_educational`），")
        lines.append("  再看是不是规则分类器的 `超出范围` 关键词过宽——别去改拒答话术。")
    lines.append("")
    lines.append("## 复跑\n")
    lines.append("```powershell")
    lines.append("$env:PYTHONPATH='.'")
    lines.append("F:\\miniconda3\\envs\\ragpro\\python.exe scripts\\eval_chat.py --limit 60")
    lines.append("F:\\miniconda3\\envs\\ragpro\\python.exe scripts\\eval_chat.py --llm live   # 有 key 时")
    lines.append("```")
    return "\n".join(lines) + "\n"


async def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    # 控制台只看进度（检索每题的 rag_search 日志会刷屏），细节仍进 logs/app.log
    configure("WARNING", log_path=settings.log_path)
    if args.rerank:
        settings.rag_rerank_enabled = args.rerank

    samples = _load_samples(args.limit, args.stride)
    if not samples:
        print("!! 黄金集里没有带 contexts 的样本")
        return 1

    container = Container(settings)
    container.build()
    await container.load_faq()
    await container.load_rag()
    container.load_llm()
    if args.llm == "live" and not container.llm.enabled:
        print("!! --llm live 需要 LLM_API_KEY，当前为空；改用 --llm stub 或先配置 .env")
        container.close()
        return 2
    mode = args.llm
    if mode == "auto":
        mode = "live" if container.llm.enabled else "stub"
    container.llm = {"live": container.llm, "stub": StubLLM(), "off": DisabledLLM()}[mode]
    container.build_chat()

    if not (container.rag and container.rag.get("ready")):
        print(f"!! 检索链未就绪：{(container.rag or {}).get('reason')}")
        container.close()
        return 3

    faq_state = container.faq_service.state() if container.faq_service else {}
    meta = {
        "faq_docs": faq_state.get("docs", 0),
        "faq_indexed": faq_state.get("indexed", 0),
        "rag_docs": (container.rag or {}).get("docs", 0),
        "embedder": (container.rag or {}).get("embedder_name", "unknown"),
        "rerank": settings.rerank_enabled,
    }
    print(f"== 样本 {len(samples)} 条，llm={mode}，rel={args.rel}，rerank={meta['rerank']}")
    print(f"== FAQ {meta['faq_docs']} / 索引 {meta['faq_indexed']}，向量库 {meta['rag_docs']} 子块")

    try:
        records = await _evaluate(samples, container, args.rel, mode)
    finally:
        container.close()

    DETAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DETAIL_FILE.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    REPORT_FILE.write_text(_summarise(records, args.rel, mode, meta), encoding="utf-8")

    sources = Counter(r["source"] for r in records)
    print(f"== 出口分布 {dict(sources)}")
    print(f"== 明细 → {DETAIL_FILE}")
    print(f"== 报告 → {REPORT_FILE}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端问答评估（黄金集 → ChatPipeline）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--stride", type=int, default=1, help="固定步长抽样（>1 时跨段取样，可复跑）")
    parser.add_argument("--rel", type=float, default=0.5, help="证据覆盖黄金事实多少算相关")
    parser.add_argument("--llm", choices=["auto", "live", "stub", "off"], default="auto",
                        help="auto=有 key 用真模型、否则抽取式替身")
    parser.add_argument("--rerank", choices=["auto", "true", "false"], default=None,
                        help="覆盖 RAG_RERANK_ENABLED（CPU 上开重排会慢两个数量级）")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
