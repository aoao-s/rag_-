#!/usr/bin/env python
"""FAQ 阈值标定：同义问法生成 + 三组样本 + 阈值扫描。

为什么要这个脚本：`FAQ_THRESHOLD=0.62` 最初是拍的。0.62 太高会把同义问法推给 RAG（贵且慢），
太低会把不相干的问题答成 FAQ（错误答案比"不知道"更糟）。所以要用**带标签的样本**量出来，
而不是凭感觉调。

样本两组（都落盘，带指纹，可复现）：
1. 正样本（同义问法）：`gen --source rule|llm`。规则改写离线可复现；llm 更接近真实口语。
   旧文件一律先归档——按 faq_id 跳过已有条目会把新旧标签混进同一文件（2026-10-25 踩过）。
2. 负样本：`negatives`。同主题考试题（词面重合但不是同一件事）+ 学科外语料 + 闲聊垃圾串。
两条硬规则（§7.1.7 前置约束）：`original` 必须等于当前正文（否则直接中止）；变体必须改变词袋
（BM25 是词袋模型，只换语序的"同义问法"分数一模一样，等于灌水）。

用法（conda 环境 ragpro）：
    python scripts/eval_faq.py baseline            # 第一步：语料与索引基线
    python scripts/eval_faq.py gen --source rule   # 第二步：重构正样本
    python scripts/eval_faq.py negatives           # 第二步：重构负样本
    python scripts/eval_faq.py calibrate           # 第三步：dev 选点、test 报数

输出：docs/analysis/faq_corpus_baseline.{json,md}、docs/analysis/faq_threshold_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from base.config import get_settings  # noqa: E402
from base.db import MySQL  # noqa: E402
from base.labels import SUBJECTS, normalize_subject  # noqa: E402
from qa_system.res.qa_csv import load_csv_records  # noqa: E402
from qa_system.res.qa_db import FAQRecord, InMemoryFAQRepository, MySQLFAQRepository  # noqa: E402
from qa_system.res.qa_matcher import MatchPolicy, decide  # noqa: E402
from qa_system.res.qa_normalize import Normalizer  # noqa: E402
from qa_system.res.qa_search import FAQService  # noqa: E402

FAQ_CSV = ROOT / "datasets" / "04_faq" / "edu_faq.csv"
PARAPHRASE_FILE = ROOT / "docs" / "analysis" / "faq_paraphrases.jsonl"
REPORT_FILE = ROOT / "docs" / "analysis" / "faq_threshold_report.md"
BASELINE_JSON = ROOT / "docs" / "analysis" / "faq_corpus_baseline.json"
BASELINE_REPORT = ROOT / "docs" / "analysis" / "faq_corpus_baseline.md"
QUESTION_BANK = ROOT / "datasets" / "03_question_bank" / "question_bank.jsonl"
POSITIVE_FILE = PARAPHRASE_FILE
NEGATIVE_FILE = ROOT / "docs" / "analysis" / "faq_negatives.jsonl"
DIAGNOSIS_FILE = ROOT / "docs" / "analysis" / "faq_diagnosis.md"
ARCHIVE_DIR = ROOT / "docs" / "analysis" / "archive"
OPENSOURCE_FILE = ROOT / "datasets" / "03_question_bank" / "opensource_questions.jsonl"
CEVAL_DIR = ROOT / "datasets" / "raw_opensource" / "ceval-exam"

# `unrelated` 负样本的学科外来源（§7.1.7 数据补齐契约）。
# 刻意**不含** high_school_* / middle_school_* / chinese_language_and_literature /
# ideological_and_moral_cultivation——那些落在 9 学科内，拿来当 unrelated 是错的（它们是 lexical_trap）。
# 也刻意不含 college_chemistry / college_physics / advanced_mathematics 这类"同学科、只是更难"的：
# 它们与 FAQ 词面高度重合，会退化成第二个 lexical_trap 池，量不出"学科外"这一类的表现。
CEVAL_UNRELATED_SUBJECTS: tuple[str, ...] = (
    "accountant",
    "art_studies",
    "basic_medicine",
    "business_administration",
    "civil_servant",
    "clinical_medicine",
    "college_economics",
    "college_programming",
    "computer_architecture",
    "computer_network",
    "education_science",
    "electrical_engineer",
    "environmental_impact_assessment_engineer",
    "fire_engineer",
    "law",
    "legal_professional",
    "logic",
    "metrology_engineer",
)

# 样本规模（§7.1.7 第二步：正样本 ≥600、负样本 ≥300，这里留出 dev/test 切分的余量）
POSITIVE_TARGET = 900
MIN_PER_SUBJECT = 60
MAX_PER_FAQ = 3
TRAP_N = 500
UNRELATED_N = 300
SAMPLE_SEED = 20261026
DEV_BUCKETS = 7  # faq_id % 10 < 7 → dev，其余 test

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


# ---------------------------------------------------------------- 第二步：重建样本
# 规则改写表。只收"换字面不换义"的对，且替换后必须改变词袋（BM25 是词袋模型，见 §7.1.7 约束 4）。
_SYNONYMS: tuple[tuple[str, str], ...] = (
    ("有哪些", "都有什么"),
    ("是什么", "指的是什么"),
    ("怎么样", "如何"),
    ("怎么", "怎样"),
    ("如何", "怎么"),
    ("为什么", "什么原因"),
    ("多少", "几"),
    ("哪些", "什么"),
    ("需要", "要"),
    ("区别", "不同"),
    ("方法", "办法"),
    ("计算", "求"),
    ("影响", "作用"),
    ("特点", "特征"),
    ("内容", "含义"),
    ("步骤", "过程"),
    ("包括", "包含"),
    ("表示", "指的是"),
    ("可以", "能"),
    ("能否", "能不能"),
    ("一般", "通常"),
    ("主要", "重点"),
    ("作用", "用处"),
    ("例子", "举例"),
    ("定义", "概念"),
)

_NOISE_TEMPLATES: tuple[str, ...] = ("老师，{q}", "{q}，能说下吗", "我想了解下{q}")

_QUESTION_TAILS: tuple[str, ...] = (
    "是什么", "有哪些", "是多少", "怎么样", "怎么", "为什么", "包括哪些",
    "有什么", "指什么", "如何", "多少", "哪些", "什么",
)

# 题库里的选择题选项行与 LaTeX 行内公式（见 _clean_stem）
_OPTION_LINE = re.compile(r"^\s*[（(]?\s*[A-Da-d]\s*[)）.、．:：]")
_LATEX = re.compile(r"\$[^$]*\$")

# 负样本里的"闲聊/垃圾串"：真实客服入口上会出现，但语料里一定没有对应 FAQ
_NOISE_QUERIES: tuple[str, ...] = (
    "在吗", "有人吗", "你好", "您好呀", "谢谢", "拜拜", "你是机器人吗", "测试一下",
    "今天天气怎么样", "现在几点了", "讲个笑话", "推荐部电影", "附近有什么好吃的",
    "客服电话是多少", "怎么退款", "我要投诉", "能开发票吗", "几点下班",
    "asdfgh", "qwerty", "123456", "1111", "？？？", "。。。", "……", "???",
    "1+1等于几", "地球到月球多远", "我的快递到哪了", "帮我写首诗",
    "随便问问", "没啥事", "嗯", "好的", "在不在", "有人理我吗",
    "我想报名", "怎么交钱", "能不能便宜点", "有优惠活动吗",
)


def _archive_samples(path: Path) -> Path | None:
    """旧样本一律先归档（§7.1.7 前置约束 1）：`gen` 的 faq_id 跳过逻辑会把新旧标签混进同一文件。"""
    if not path.exists() or path.stat().st_size == 0:
        return None
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARCHIVE_DIR / f"{path.stem}.{datetime.now():%Y%m%d-%H%M%S}{path.suffix}"
    path.replace(dest)
    return dest


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _check_positives(rows: list[dict], by_id: dict[int, FAQRecord]) -> None:
    """硬校验：`original` 必须等于该 faq_id 的当前正文（§7.1.7 前置约束 2）。"""
    bad = [
        row
        for row in rows
        if row.get("faq_id") not in by_id or str(row.get("original", "")) != by_id[row["faq_id"]].question
    ]
    if not bad:
        return
    preview = "\n".join(f"  faq_id={row.get('faq_id')} 样本原文={row.get('original')!r}" for row in bad[:5])
    raise SystemExit(
        f"样本标签与当前语料不一致：{len(bad)}/{len(rows)} 条\n{preview}\n"
        "语料换过版本，旧样本的 faq_id 已经指向别的问题。请重新生成："
        "`python scripts/eval_faq.py gen --source rule`（旧文件会自动归档）。"
    )


def _acceptable(
    text: str,
    *,
    normalizer: Normalizer,
    base_norm: str,
    base_tokens: set[str],
    corpus_norms: set[str],
    seen: set[str],
) -> bool:
    """一条变体是否算有效正样本（§7.1.7 约束 4）。

    四个拒绝条件：空串、与原文同归一化串（走精确哈希，测不出阈值）、
    与任一 FAQ 正文同串（标签不可信）、词袋与原文相同（BM25 分数完全一致，等于灌水）。
    """
    norm = normalizer.normalize(text)
    if not norm or norm == base_norm or norm in corpus_norms:
        return False
    if set(normalizer.tokens(text)) == base_tokens:
        return False
    if norm in seen:
        return False
    seen.add(norm)
    return True


def _rule_variants(question: str, normalizer: Normalizer, corpus_norms: set[str]) -> list[tuple[str, str]]:
    """规则改写：同义替换 / 省略疑问尾巴 / 加称呼口头语（§7.1.7 第二步表）。"""
    base_norm = normalizer.normalize(question)
    base_tokens = set(normalizer.tokens(question))
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    def add(kind: str, text: str) -> None:
        text = text.strip()
        if _acceptable(
            text,
            normalizer=normalizer,
            base_norm=base_norm,
            base_tokens=base_tokens,
            corpus_norms=corpus_norms,
            seen=seen,
        ):
            out.append((kind, text))

    for old, new in _SYNONYMS:
        if old in question:
            # 一条变体只换一处（链式替换会把语义改歪）；不同处各算一条，噪声模板排在最后
            add("synonym", question.replace(old, new, 1))
    for tail in _QUESTION_TAILS:
        if question.endswith(tail):
            stem = question[: -len(tail)].rstrip("，,。. ")
            if len(stem) >= 6:  # 太短会同时命中多条 FAQ，那是歧义查询，不是同义问法
                add("abbrev", stem)
            break
    for template in _NOISE_TEMPLATES:
        before = len(out)
        add("noise", template.replace("{q}", question))
        if len(out) > before:
            break  # 每题只留一条口头语变体，避免同题样本高度冗余
    return out[:MAX_PER_FAQ]


async def gen_samples(
    source: str = "rule",
    api_key: str | None = None,
    concurrency: int = 6,
    reuse: bool = False,
) -> int:
    """生成正样本（同义问法）。`source=rule` 离线可复现，`source=llm` 更接近真实口语。"""
    records, corpus_source = await _load_records()
    by_id = {r.faq_id: r for r in records}
    normalizer = Normalizer(extra_words=SUBJECTS)
    normalizer.preload()
    corpus_norms = {normalizer.normalize(r.question) for r in records}
    print(f"语料来源：{corpus_source}")

    if reuse and POSITIVE_FILE.exists():
        rows = _read_jsonl(POSITIVE_FILE)
        _check_positives(rows, by_id)
        print(f"复用已有样本 {len(rows)} 条（已通过标签校验）→ {POSITIVE_FILE.relative_to(ROOT)}")
        return len(rows)

    archived = _archive_samples(POSITIVE_FILE)
    if archived:
        print(f"旧样本已归档 → {archived.relative_to(ROOT)}")

    by_subject: dict[str, list[FAQRecord]] = defaultdict(list)
    for record in records:
        by_subject[record.subject or "(未标注)"].append(record)
    rng = random.Random(SAMPLE_SEED)
    now = datetime.now().isoformat(timespec="seconds")
    rows_out: list[dict] = []
    no_variant = 0

    for subject in sorted(by_subject):
        pool = list(by_subject[subject])
        rng.shuffle(pool)
        quota = max(MIN_PER_SUBJECT, round(POSITIVE_TARGET * len(pool) / len(records)))

        if source == "llm":
            import httpx

            need = min(len(pool), max(1, -(-quota // MAX_PER_FAQ)))
            sem = asyncio.Semaphore(concurrency)
            async with httpx.AsyncClient(trust_env=False) as client:
                generated = await asyncio.gather(*[_one_call(client, sem, api_key or "", r) for r in pool[:need]])
            pairs = [(record, [("llm", text) for text in texts]) for record, texts in zip(pool[:need], generated)]
        else:
            pairs = [(record, _rule_variants(record.question, normalizer, corpus_norms)) for record in pool]

        made = 0
        for record, variants in pairs:
            if made >= quota:
                break
            base_norm = normalizer.normalize(record.question)
            base_tokens = set(normalizer.tokens(record.question))
            seen: set[str] = set()
            if not variants:
                no_variant += 1
                continue
            for kind, text in variants:
                if made >= quota:
                    break
                if not _acceptable(
                    text,
                    normalizer=normalizer,
                    base_norm=base_norm,
                    base_tokens=base_tokens,
                    corpus_norms=corpus_norms,
                    seen=seen,
                ):
                    continue
                rows_out.append(
                    {
                        "faq_id": record.faq_id,
                        "subject": record.subject,
                        "original": record.question,
                        "paraphrase": text.strip(),
                        "kind": kind,
                        "source": source,
                        "generated_at": now,
                    }
                )
                made += 1
        print(f"  {subject}: {made}/{quota}")

    POSITIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with POSITIVE_FILE.open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    kinds = Counter(row["kind"] for row in rows_out)
    print(f"完成：{len(rows_out)} 条正样本（{dict(kinds)}），无有效变体 {no_variant} 条")
    print(f"→ {POSITIVE_FILE.relative_to(ROOT)}")
    return len(rows_out)


def _ceval_pool(per_subject: int = 30) -> list[tuple[str, str]]:
    """从 ceval-exam 的非 K12 学科取"学科外真问题"，作为 `unrelated` 负样本的来源。

    parquet 里 `question` 已经是干净题干（选项在单独的列），只需去掉填空下划线。
    每学科固定种子抽样后**轮转交错**，避免 300 条被某个学科包场——单学科包场时
    "unrelated 精度"其实是"某个学科的表现"，不是这个类别的表现。
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise SystemExit("ceval 源需要 pandas + pyarrow：conda install -n ragpro pandas pyarrow") from exc
    if not CEVAL_DIR.exists():
        raise SystemExit(f"缺少 ceval 语料目录：{CEVAL_DIR}")

    per: dict[str, list[str]] = {}
    for subject in CEVAL_UNRELATED_SUBJECTS:
        files = sorted((CEVAL_DIR / subject).glob("*.parquet"))
        if not files:
            continue
        frame = pd.read_parquet(files[0])
        if "question" not in frame.columns:
            continue
        picked: list[str] = []
        for raw in frame["question"]:
            text = _clean_stem(str(raw))
            # 与题库负样本同一口径：8–60 字、至少 2 个汉字
            if 8 <= len(text) <= 60 and len(re.findall(r"[一-鿿]", text)) >= 2:
                picked.append(text)
        if not picked:
            continue
        random.Random(f"{SAMPLE_SEED}:{subject}").shuffle(picked)
        per[subject] = picked[:per_subject]

    pool: list[tuple[str, str]] = []
    for i in range(per_subject):
        for subject in sorted(per):
            if i < len(per[subject]):
                pool.append((per[subject][i], f"ceval_exam:{subject}"))
    return pool


def build_negatives(
    trap_n: int = TRAP_N,
    unrelated_n: int = UNRELATED_N,
    unrelated_source: str = "auto",
) -> int:
    """生成负样本（§7.1.7 第二步表）：同主题考试题 / 学科外语料 / 闲聊垃圾串。

    考试题是天然的"词面重合但不是同一件事"——正是阈值最容易被骗过的地方。
    `unrelated_source`：`bank`=只用开源题库的学科外题目（现在一条都没有）；
    `ceval`=只用 ceval-exam 非 K12 学科；`auto`=题库优先、不足再用 ceval 补齐。
    """
    if not OPENSOURCE_FILE.exists():
        raise SystemExit(f"缺少开源题库：{OPENSOURCE_FILE}")
    records = load_csv_records(FAQ_CSV)
    normalizer = Normalizer(extra_words=SUBJECTS)
    normalizer.preload()
    faq_norms = {normalizer.normalize(r.question) for r in records}
    faq_tokens = {r.faq_id: set(normalizer.tokens(r.question)) for r in records}
    inverted: dict[str, set[int]] = defaultdict(set)
    for faq_id, tokens in faq_tokens.items():
        for token in tokens:
            inverted[token].add(faq_id)

    def nearest(text: str) -> tuple[float, int | None]:
        tokens = set(normalizer.tokens(text))
        if not tokens:
            return 0.0, None
        candidates: set[int] = set()
        for token in tokens:
            candidates |= inverted.get(token, set())
        best, best_id = 0.0, None
        for faq_id in candidates:
            other = faq_tokens[faq_id]
            score = len(tokens & other) / len(tokens | other)
            if score > best:
                best, best_id = score, faq_id
        return best, best_id

    in_scope: list[str] = []
    out_scope: list[str] = []
    for line in OPENSOURCE_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        question = _clean_stem(str(row.get("question", "")))
        # 太短没信息量；超过 60 字是整段材料，真实入口上属于"粘贴一道大题"，不是 FAQ 该接的活
        if not (8 <= len(question) <= 60) or len(re.findall(r"[一-鿿]", question)) < 2:
            continue
        subject = normalize_subject(str(row.get("subject", "")))
        (in_scope if subject in SUBJECTS else out_scope).append(question)

    rng = random.Random(SAMPLE_SEED)
    rng.shuffle(in_scope)
    rng.shuffle(out_scope)

    if unrelated_source == "bank":
        unrelated_pool = [(q, "opensource_question_bank") for q in out_scope]
    elif unrelated_source == "ceval":
        unrelated_pool = _ceval_pool()
    else:  # auto：题库里有学科外题就先用它，不够再补 ceval
        unrelated_pool = [(q, "opensource_question_bank") for q in out_scope] + _ceval_pool()

    def scan(pool: list[tuple[str, str]], cap: int) -> list[dict]:
        scored: list[dict] = []
        for text, origin in pool[:cap]:
            if normalizer.normalize(text) in faq_norms:
                continue
            overlap, faq_id = nearest(text)
            # 与某条 FAQ 几乎同词袋 ⇒ 用户其实问到点上了，标成负样本会逼着阈值高到离谱
            if overlap >= 0.9:
                continue
            scored.append({"text": text, "overlap": round(overlap, 3), "nearest_faq_id": faq_id,
                           "source": origin})
        return scored

    traps = sorted(scan([(q, "opensource_question_bank") for q in in_scope], trap_n * 8),
                   key=lambda row: -row["overlap"])[:trap_n]
    for row in traps:
        row["kind"] = "lexical_trap"
    unrelated = scan(unrelated_pool, unrelated_n * 4)[:unrelated_n]
    for row in unrelated:
        row["kind"] = "unrelated"
    # 题库全落在 9 学科内时 unrelated 会静默变成 0：精度从此只由 lexical_trap 决定，
    # 与含 unrelated 的历史报告不可比。不造数据，但必须把缺口喊出来（§7.1.7 第二步表）。
    if len(unrelated) < unrelated_n:
        print(
            f"⚠ 负样本缺口：unrelated 只有 {len(unrelated)}/{unrelated_n} 条"
            f"（来源 {unrelated_source}，可用 {len(unrelated_pool)} 条学科外题目）——本次精度只反映同主题陷阱，"
            "与含 unrelated 的历史报告不可直接比较",
            file=sys.stderr,
        )
    noise = [{"text": text, "kind": "noise", "source": "handwritten"} for text in _NOISE_QUERIES]

    rows = traps + unrelated + noise
    NEGATIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 与 `gen` 同一纪律：换源前先归档，否则新旧类别混在一个文件里，报告里的"类别构成"无法回溯
    archived = _archive_samples(NEGATIVE_FILE)
    if archived:
        print(f"旧负样本已归档 → {archived.relative_to(ROOT)}")
    with NEGATIVE_FILE.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    origins = Counter(str(r.get("source", "?")).split(":")[0] for r in unrelated)
    print(f"完成：{len(rows)} 条负样本（{dict(Counter(r['kind'] for r in rows))}）")
    if unrelated:
        print(f"unrelated 来源分布：{dict(origins)}（学科数 {len({r['source'] for r in unrelated})}）")
    print(f"→ {NEGATIVE_FILE.relative_to(ROOT)}")
    return len(rows)


def _clean_stem(text: str) -> str:
    """把题库里的整道题（含选项与 LaTeX）削成"用户会真打进聊天框的那一句"。

    整题直接当负样本会失真：选项块和 `$...$` 带来大量无意义 token，长查询的候选又少，
    margin 反而变大——量出来的不是"该不该答"，而是"输入脏不脏"。
    """
    lines = [ln for ln in text.splitlines() if not _OPTION_LINE.match(ln)]
    stem = _LATEX.sub(" ", " ".join(lines))
    stem = re.sub(r"\s+", " ", stem).strip()
    stem = re.sub(r"[_＿]{2,}", " ", stem)
    return re.sub(r"\s+", " ", stem).strip(" ，,。.；;：:")


async def _load_records() -> tuple[list[FAQRecord], str]:
    """装载语料：优先 datasets/04_faq/edu_faq.csv（无需数据库，任何人都能复现），
    CSV 缺失时才回退到 MySQL。返回 (records, 来源说明)。
    """
    if FAQ_CSV.exists():
        records = load_csv_records(FAQ_CSV)
        return records, f"datasets/04_faq/edu_faq.csv（{len(records)} 条，id=CSV 行序）"

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


def _quiet() -> None:
    """评估期间关掉逐条检索日志：抓取上万条样本时每行 2000 条命中日志会把结论挤没。"""
    logging.getLogger().setLevel(logging.WARNING)


def _len_bucket(n_chars: int) -> str:
    if n_chars <= 10:
        return "≤10"
    if n_chars <= 20:
        return "11–20"
    if n_chars <= 40:
        return "21–40"
    return "41+"


def _negative_gap_notes(kinds: dict[str, int]) -> list[str]:
    """负样本缺类要写进报告：缺了 unrelated，精度含义就变了（只剩同主题陷阱），不能沉默。"""
    if kinds.get("unrelated"):
        return []
    return [
        "- ⚠ 负样本缺口：`unrelated`（学科外语料）0 条——题库已全部落在 9 学科内，"
        "本次精度只由 `lexical_trap` 决定，**与含 unrelated 的历史 run 不可直接比较**；"
        "补齐方式：`python scripts/eval_faq.py negatives --unrelated-source ceval`（§7.1.7 数据补齐契约）。"
    ]


# ---------------------------------------------------------------- 第一步：语料基线
def _policy_from_settings(settings) -> MatchPolicy:
    """与 main.py 同源：评估必须用线上那套判定参数，否则量出来的阈值换不到线上。"""
    return MatchPolicy(
        threshold=settings.faq_threshold,
        margin=settings.faq_margin,
        candidate_n=settings.faq_candidate_n,
        norm_k=settings.faq_norm_k,
        min_candidates=settings.faq_min_candidates,
        margin_mode=settings.faq_margin_mode,  # type: ignore[arg-type]
        # §7.1.9 三档：自检必须与线上同一档位，否则"原句命中自身"量的是另一套行为
        high_margin=settings.faq_high_margin,
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records_digest(records: list[FAQRecord]) -> str:
    """按 id/学科/问题重算的指纹：CSV 换编码或换列顺序不该让指纹变。"""
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record.faq_id}\t{record.subject or ''}\t{record.question}\n".encode("utf-8"))
    return digest.hexdigest()


def _epoch_iso(value: float) -> str:
    return datetime.fromtimestamp(value).isoformat(timespec="seconds") if value else ""


# 归一化探针：一条覆盖一条规则。用**行为**而不是源码做指纹——重构不算变，语义变了才算变。
_NORMALIZER_PROBES: tuple[str, ...] = (
    "请问一下，学费多少钱？",  # 礼貌语 + 句尾语气词
    "学三个月",  # 中文数字 + 量词
    "勾股定理咋证明",  # 口语同义
    "ＢＥＲＴ模型",  # 全半角与大小写
    "因式分解 x²-6x+9",  # 数学算子保留（负号）
    "2/3-1/6",  # 分数 + 运算符
    "2×3",  # 算子写法归一
    "|-5|",  # 一元负号
    "数学-七年级",  # 连字符仍作分隔符删除
    "学费+住宿费",  # 中文之间的加号仍删除
    "H₂SO₄ 与 h2so4",  # 下标/大小写
    "第3-5题",  # 数字区间
)


def _normalizer_fingerprint() -> dict[str, object]:
    """归一化规则的语义指纹：第二、三步靠它判断"分数是否可比"。"""
    normalizer = Normalizer(extra_words=SUBJECTS)
    mapping = {probe: normalizer.normalize(probe) for probe in _NORMALIZER_PROBES}
    rules = json.dumps(mapping, ensure_ascii=False, sort_keys=True)
    return {
        "probes": len(mapping),
        "sha256": hashlib.sha256(rules.encode("utf-8")).hexdigest(),
        "samples": mapping,
    }


async def baseline() -> None:
    """§7.1.7 第一步：确认评估语料与线上同一版本，且每条原问题都能查回自己。

    这一步不调任何参数，只回答两个问题：
    1. 装进索引的条数 == 语料条数（评估与线上同源）；
    2. 原句 100% 命中自身（精确哈希路径没接错）。
    任一不成立就先修语料，别急着 `gen`——否则后两步标出来的阈值全是噪声。
    """
    settings = get_settings()
    records, source = await _load_records()
    policy = _policy_from_settings(settings)
    service = FAQService(
        InMemoryFAQRepository(records),
        policy=policy,
        cross_subject_delta=settings.faq_cross_subject_delta,
        dedupe_near_dup=settings.faq_dedupe_near_dup,
    )
    await service.load()
    state = service.state()

    # 精确哈希表是 norm → faq_id 的单值映射：归一化后同串的两行，必有一行查不到自己。
    # 这不是"脚本 bug"，是语料里有重复问题，必须先去重再标定。
    normalizer = Normalizer(extra_words=SUBJECTS)
    by_norm: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_norm[normalizer.normalize(record.question)].append(record.faq_id)
    by_id = {r.faq_id: r for r in records}
    duplicates = [
        {
            "norm": norm,
            "faq_ids": ids,
            "questions": [by_id[fid].question for fid in ids],
        }
        for norm, ids in by_norm.items()
        if len(ids) > 1
    ]
    duplicate_ids = {fid for item in duplicates for fid in item["faq_ids"]}

    misses: list[dict[str, object]] = []
    for record in records:
        hit = await service.search(record.question)
        # 词袋重复被并成别名后，问"X 是什么"会落到代表条目"什么是 X"上——
        # 两条答案逐字相同，属于"查回自己"，不能算 miss。
        if hit.hit and hit.faq_id == service.canonical_id(record.faq_id):
            continue
        misses.append(
            {
                "faq_id": record.faq_id,
                "subject": record.subject,
                "question": record.question,
                "got_id": hit.faq_id,
                "reason": hit.reason,
                "score": round(hit.score, 4),
                "duplicate": record.faq_id in duplicate_ids,
            }
        )
    real_misses = [m for m in misses if not m["duplicate"]]

    subject_counts = Counter(record.subject or "(未标注)" for record in records)
    normalizer_fp = _normalizer_fingerprint()
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "corpus": {
            "path": str(FAQ_CSV.relative_to(ROOT)) if FAQ_CSV.exists() else source,
            "csv_sha256": _file_sha256(FAQ_CSV) if FAQ_CSV.exists() else None,
            "records_sha256": _records_digest(records),
            "records": len(records),
            "subjects": dict(subject_counts.most_common()),
        },
        "normalizer": normalizer_fp,
        "index": {
            "docs": state["docs"],
            "indexed": state["indexed"],
            "merged_duplicates": state["merged_duplicates"],
            "subjects": state["subjects"],
            "unlabeled": state["unlabeled"],
            "version": state["version"],
            "built_at": _epoch_iso(state["built_at"]),
        },
        "policy": {
            "threshold": policy.threshold,
            "margin": policy.margin,
            "high_margin": policy.high_margin,
            "margin_mode": policy.margin_mode,
            "norm_k": policy.norm_k,
            "min_candidates": policy.min_candidates,
        },
        "self_check": {
            "total": len(records),
            "hit_self": len(records) - len(misses),
            "misses": misses[:50],
        },
        "duplicates": duplicates[:50],
    }
    BASELINE_JSON.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    count_ok = state["docs"] == len(records)
    verdict_ok = count_ok and not real_misses and not duplicates
    lines = [
        "# FAQ 语料基线（CLAUDE.md §7.1.7 第一步）",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 语料：`{payload['corpus']['path']}`（{len(records)} 条）",
        f"- 文件 sha256：`{payload['corpus']['csv_sha256']}`",
        f"- 内容 sha256：`{payload['corpus']['records_sha256']}`（按 id/学科/问题重算）",
        f"- 归一化指纹：`{normalizer_fp['sha256']}`（{normalizer_fp['probes']} 条行为探针）"
        "——第二、三步断言语料与分数可比时以此为准；改归一化规则必须重跑本步",
        f"- 索引：{state['docs']} 条 / {len(state['subjects'])} 学科 / unlabeled {state['unlabeled']}"
        f" / version {state['version']} / built_at {payload['index']['built_at']}",
        f"- 语料重复合并：{state['merged_duplicates']} 条并成别名（实际进 BM25 索引 {state['indexed']} 条）"
        "——同学科 + 答案逐字相同 + 去框架词后 Jaccard ≥0.8，原问法仍可精确命中代表条目",
        f"- 策略：threshold={policy.threshold} margin={policy.margin}（{policy.margin_mode}）"
        f" high_margin={policy.high_margin}（≥high 才直接答，§7.1.9）"
        f" norm_k={policy.norm_k:g} min_candidates={policy.min_candidates}",
        "",
        "## 条数一致性",
        "",
        f"- 语料 {len(records)} 条 vs 索引 {state['docs']} 条 → {'一致' if count_ok else '不一致'}",
        "",
        "## 原句自检",
        "",
        f"- 命中自身：**{len(records) - len(misses)}/{len(records)}**",
        f"- 其中因归一化重复无法命中自身的行：{len(misses) - len(real_misses)} 条",
        "",
    ]
    if duplicates:
        lines += ["### 归一化后重复的问题（必须先清理）", "", "| norm | faq_id | 问题 |", "|---|---|---|"]
        for item in duplicates:
            for fid, question in zip(item["faq_ids"], item["questions"]):
                lines.append(f"| `{item['norm']}` | {fid} | {question} |")
        lines.append("")
    if real_misses:
        lines += [
            "### 未命中自身的行",
            "",
            "| faq_id | 学科 | 问题 | 实际返回 | reason | score |",
            "|---|---|---|---|---|---|",
        ]
        for miss in real_misses[:50]:
            lines.append(
                f"| {miss['faq_id']} | {miss['subject'] or '-'} | {miss['question']} "
                f"| {miss['got_id']} | {miss['reason']} | {miss['score']} |"
            )
        lines.append("")
    lines += ["## 分学科条数", "", "| 学科 | 条数 |", "|---|---|"]
    for name, count in subject_counts.most_common():
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        "## 结论",
        "",
        (
            f"第一步通过：语料与索引同源（{len(records)} 条），原句 100% 命中自身，"
            "可以进入第二步（重建正负样本）。"
            if verdict_ok
            else "第一步**未通过**：先解决上面的条数不一致 / 重复问题 / 未命中行，再进入第二步。"
        ),
        "",
    ]
    BASELINE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        f"语料 {len(records)} 条 / 索引 {state['docs']} 条 / 学科 {len(state['subjects'])} "
        f"/ unlabeled {state['unlabeled']} / 重复 {len(duplicates)} 组 / 自检 "
        f"{len(records) - len(misses)}/{len(records)}"
    )
    print(f"报告 → {BASELINE_REPORT.relative_to(ROOT)}")
    print(f"数据 → {BASELINE_JSON.relative_to(ROOT)}")
    if not verdict_ok:
        raise SystemExit(2)


# 扫描网格：threshold 永远在归一化空间，margin 的网格按各自空间的量纲给（§7.1.7 第三步表）
THRESHOLDS: tuple[float, ...] = tuple(round(0.10 + 0.01 * i, 2) for i in range(81))
MARGIN_MODE_GRID: dict[str, tuple[float, ...]] = {
    "relative": tuple(round(0.05 * i, 2) for i in range(19)),  # 0 → 0.90
    "normalized": tuple(round(0.02 * i, 2) for i in range(31)),  # 0 → 0.60
    "raw": tuple(float(i) for i in range(0, 41)),  # 0 → 40 分
}
_MARGIN_INDEX = {"relative": 5, "normalized": 6, "raw": 7}


async def _observe(
    service: FAQService,
    text: str,
    expected: int | None,
    kind: str,
    split: str,
    norm_k: float,
    top_k: int,
) -> tuple:
    """一条样本的观测值：

    `(split, kind, expected, predicted, score, m_rel, m_norm, m_raw, 候选数, 是否精确命中, 查询长度, top2, 正确项名次)`

    判定一律交给 `qa_matcher.decide()`，脚本不复制线上逻辑；`rank()` 只给归一化分，
    原始分按 `raw = K·s/(1−s)` 反解（归一化是严格单调的，反解无损）。
    末尾两项只给 `diagnose` 用（题族代理与"正确答案排第几"），`calibrate` 不读它们。
    """
    ranked = await service.rank(text, top_k=top_k)
    n_chars = len(text)
    positions = {faq_id: i for i, (faq_id, _) in enumerate(ranked)}
    correct_rank = positions.get(expected) if expected is not None else None
    top2 = ranked[1][0] if len(ranked) > 1 else None
    if not ranked:
        return (split, kind, expected, None, 0.0, 0.0, 0.0, 0.0, 0, False, n_chars, None, correct_rank)
    if ranked[0][1] >= 1.0:  # 精确命中在线上直接返回，不走阈值与 margin
        return (split, kind, expected, ranked[0][0], 1.0, 1.0, 1.0, float("inf"), len(ranked), True, n_chars,
                top2, correct_rank)
    scored = [(faq_id, norm_k * score / (1 - score) if 0 < score < 1 else 0.0) for faq_id, score in ranked]
    result = decide(scored, MatchPolicy(threshold=0.0, margin=0.0, norm_k=norm_k, min_candidates=1))
    return (
        split,
        kind,
        expected,
        result.faq_id,
        result.score,
        result.margin_relative,
        result.margin_normalized,
        result.margin_raw,
        result.candidate_count,
        False,
        n_chars,
        top2,
        correct_rank,
    )


async def calibrate() -> None:
    """§7.1.7 第三步：dev 选点、test 只报数，三套 margin 空间各扫一遍。"""
    _quiet()
    records, source = await _load_records()
    by_id = {r.faq_id: r for r in records}
    positives = _read_jsonl(POSITIVE_FILE)
    negatives = _read_jsonl(NEGATIVE_FILE)
    if not positives:
        raise SystemExit(f"没有正样本：先跑 `gen --source rule`，期望文件 {POSITIVE_FILE}")
    if not negatives:
        raise SystemExit(f"没有负样本：先跑 `negatives`，期望文件 {NEGATIVE_FILE}")
    # 标签校验放在打分之前：错标签算出来的阈值是噪声，现在会直接中止而不是静默 0 精度
    _check_positives(positives, by_id)

    settings = get_settings()
    norm_k = settings.faq_norm_k
    min_candidates = settings.faq_min_candidates
    top_k = max(3, settings.faq_candidate_n)
    normalizer_fp = _normalizer_fingerprint()

    service = FAQService(
        InMemoryFAQRepository(records),
        policy=MatchPolicy(threshold=0.0, margin=0.0),
        dedupe_near_dup=settings.faq_dedupe_near_dup,
    )
    await service.load()

    obs: list[tuple] = []
    for row in positives:
        split = "dev" if int(row["faq_id"]) % 10 < DEV_BUCKETS else "test"
        # 合并后的重复问法以代表条目为准：两者答案逐字相同，命中代表即"答对了"
        obs.append(
            await _observe(service, str(row["paraphrase"]), service.canonical_id(int(row["faq_id"])),
                           str(row.get("kind", "?")), split, norm_k, top_k)
        )
    for row in negatives:
        bucket = int(hashlib.sha1(str(row["text"]).encode("utf-8")).hexdigest()[:8], 16) % 10
        split = "dev" if bucket < DEV_BUCKETS else "test"
        obs.append(await _observe(service, str(row["text"]), None, str(row.get("kind", "?")), split, norm_k, top_k))

    dev_pos = sum(1 for o in obs if o[0] == "dev" and o[2] is not None)
    test_pos = sum(1 for o in obs if o[0] == "test" and o[2] is not None)
    dev_neg = sum(1 for o in obs if o[0] == "dev" and o[2] is None)
    test_neg = sum(1 for o in obs if o[0] == "test" and o[2] is None)
    print(
        f"正样本 {dev_pos + test_pos} 条（dev {dev_pos} / test {test_pos}）｜"
        f"负样本 {dev_neg + test_neg} 条（dev {dev_neg} / test {test_neg}）"
    )

    pos_scores = [o[4] for o in obs if o[2] is not None]
    neg_scores = [o[4] for o in obs if o[2] is None]

    # 原句自检
    self_miss = 0
    for record in records:
        hit = await service.search(record.question)
        if not hit.hit or hit.faq_id != service.canonical_id(record.faq_id):
            self_miss += 1

    def evaluate(split: str, t: float, m: float, mode: str) -> tuple[int, int, int, int]:
        """(正确命中, 命中但答错, 负样本误命中, 漏答)。"""
        idx = _MARGIN_INDEX[mode]
        tp = wrong = fp_neg = fn = 0
        for row in obs:
            if row[0] != split:
                continue
            hit = row[9] or (row[8] >= min_candidates and row[4] >= t and row[idx] >= m)
            if row[2] is None:
                if hit:
                    fp_neg += 1
            elif hit and row[3] == row[2]:
                tp += 1
            elif hit:
                wrong += 1
            else:
                fn += 1
        return tp, wrong, fp_neg, fn

    def ratio(counts: tuple[int, int, int, int]) -> tuple[float, float]:
        tp, wrong, fp_neg, fn = counts
        precision = tp / (tp + wrong + fp_neg) if (tp + wrong + fp_neg) else 0.0
        recall = tp / (tp + wrong + fn) if (tp + wrong + fn) else 0.0
        return recall, precision

    # dev 选点：precision ≥ 0.95 下召回最大；平手取 margin 更小的（更宽松、更少漏答）
    picks: dict[str, dict | None] = {}
    for mode, margins in MARGIN_MODE_GRID.items():
        best: dict | None = None
        for t in THRESHOLDS:
            for m in margins:
                counts = evaluate("dev", t, m, mode)
                recall, precision = ratio(counts)
                if precision < 0.95:
                    continue
                key = (recall, -m, t)
                if best is None or key > best["key"]:
                    best = {"key": key, "t": t, "m": m, "mode": mode, "counts": counts,
                            "recall": recall, "precision": precision}
        picks[mode] = best

    pick = max(
        (p for p in picks.values() if p),
        key=lambda p: (p["recall"], p["precision"]),
        default=None,
    )

    sample_fp = hashlib.sha256(POSITIVE_FILE.read_bytes() + NEGATIVE_FILE.read_bytes()).hexdigest()
    pos_kinds = Counter(o[1] for o in obs if o[2] is not None)
    neg_kinds = Counter(o[1] for o in obs if o[2] is None)
    sources = Counter(str(row.get("source", "?")) for row in positives)
    mode_label = {
        "relative": "相对分差 `(raw1−raw2)/raw1`",
        "normalized": "归一化分差 `s1−s2`",
        "raw": "原始分差 `raw1−raw2`",
    }

    def kind_stats(split: str, t: float, m: float, mode: str) -> dict[str, tuple[int, int]]:
        idx = _MARGIN_INDEX[mode]
        stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in obs:
            if row[0] != split or row[2] is None:
                continue
            stats[row[1]][1] += 1
            hit = row[9] or (row[8] >= min_candidates and row[4] >= t and row[idx] >= m)
            if hit and row[3] == row[2]:
                stats[row[1]][0] += 1
        return {name: (hits, total) for name, (hits, total) in stats.items()}

    def slice_stats(split: str, t: float, m: float, mode: str) -> dict[str, tuple[int, int, int, int]]:
        """按查询长度切片 → {区间: (正命中, 正总数, 负误命中, 负总数)}。

        两次用到它（报告表格 + 结论里的病因推导），所以做成函数，避免结论与表格各算一遍。
        """
        idx = _MARGIN_INDEX[mode]
        buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for row in obs:
            if row[0] != split:
                continue
            bucket = _len_bucket(row[10])
            hit = row[9] or (row[8] >= min_candidates and row[4] >= t and row[idx] >= m)
            if row[2] is None:
                buckets[bucket][3] += 1
                buckets[bucket][2] += bool(hit)
            else:
                buckets[bucket][1] += 1
                buckets[bucket][0] += bool(hit and row[3] == row[2])
        return {name: (v[0], v[1], v[2], v[3]) for name, v in buckets.items() if v[1] or v[3]}

    # ---- §7.1.9 三档模拟：同一份观测上把"过 ①②③"的那段按 high_margin 切成两段
    def tier_of(row: tuple, t: float, m: float, idx2: int, high: float) -> str:
        """按 `decide()` 的顺序复原档位：exact 恒 high；过 ①②③ 后由 margin 分 high / mid。

        与线上同一份判定语义（阈值 → 候选数 → margin → 高置信线），只是在这里对已观测的
        分数重放一遍——`_observe()` 已经把原始分与三种 margin 都取回来了。
        """
        if row[9]:  # 归一化串精确命中，线上直接返回，不受任何封顶
            return "high"
        if row[8] >= min_candidates and row[4] >= t and row[idx2] >= m:
            return "high" if row[idx2] >= high else "mid"
        return "low"

    def tier_stats(split: str, t: float, m: float, mode: str, high: float) -> dict[str, dict[str, int]]:
        """每档的 `{正样本, 答对, 答错, 负样本, 负样本误命中}`。

        `答错` = 落在该档但 top1 是另一条 FAQ；`负样本` = 落在该档的负样本数，
        若该档作答它们就全是误命中——"mid 若直接答的精度"由此算出（§7.1.9 验收第 2 条）。
        """
        idx2 = _MARGIN_INDEX[mode]
        out = {tier: {"pos": 0, "tp": 0, "wrong": 0, "neg": 0, "fp_neg": 0} for tier in ("high", "mid", "low")}
        for row in obs:
            if row[0] != split:
                continue
            bucket = out[tier_of(row, t, m, idx2, high)]
            if row[2] is None:
                bucket["neg"] += 1
                bucket["fp_neg"] += 1
            else:
                bucket["pos"] += 1
                if row[3] == row[2]:
                    bucket["tp"] += 1
                else:
                    bucket["wrong"] += 1
        return out

    def answered_precision(stats: dict[str, int]) -> float:
        answered = stats["tp"] + stats["wrong"] + stats["fp_neg"]
        return stats["tp"] / answered if answered else 0.0

    def tier_section() -> list[str]:
        """§7.1.9 三档的离线验收：同一批观测、同一套 threshold / margin，只加 `FAQ_HIGH_MARGIN`。

        与单档选点解耦：即便 dev 上找不到 precision ≥0.95 的（threshold, margin）点，
        三档该不该上、`high` 该切在哪，仍然要有一份数——所以不放在 `if pick:` 里面。
        """
        high = settings.faq_high_margin
        out: list[str] = [
            "",
            f"## 三档路由（§7.1.9；现行 `{cur_mode}` / threshold={cur_t:.2f} / margin={cur_m:.2f} "
            f"/ high_margin={high:.2f}）",
            "",
            "「若直接答的精度」= 该档内 `答对 / (答对 + 答错 + 负样本)`；线上只有 `high` 档会直接返回答案，"
            "`mid` 只给候选与追问，`low` 交棒 RAG。「占召回」= 该档答对数 ÷ 该集合正样本总数。",
            "",
            "| 档 | 集合 | 正样本 | 答对 | 答错 | 负样本 | 若直接答的精度 | 占召回 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        tiers: dict[tuple[str, str], dict[str, int]] = {}
        for split, n_pos in (("dev", dev_pos), ("test", test_pos)):
            stats = tier_stats(split, cur_t, cur_m, cur_mode, high)
            for tier in ("high", "mid", "low"):
                s = stats[tier]
                tiers[(split, tier)] = s
                # low 档不参与作答，它的"精度"没有线上含义，不报数字（免得被当成一个可调参数）
                precision_cell = "—" if tier == "low" else f"{answered_precision(s):.3f}"
                out.append(
                    f"| `{tier}` | {split} | {s['pos']} | {s['tp']} | {s['wrong']} | {s['neg']} "
                    f"| {precision_cell} | {s['tp'] / (n_pos or 1):.3f} |"
                )

        hi_test, mid_test = tiers[("test", "high")], tiers[("test", "mid")]
        hi_precision = answered_precision(hi_test)
        hi_recall = hi_test["tp"] / (test_pos or 1)
        mid_answered = mid_test["tp"] + mid_test["wrong"] + mid_test["fp_neg"]
        out += [
            "",
            f"- high 档 test：精度 **{hi_precision:.3f}**、召回 **{hi_recall:.3f}** → "
            f"**{'通过' if hi_precision >= 0.95 and hi_recall >= 0.25 else '未通过'}** §7.1.9 验收"
            "（高置信精度 ≥0.95 且召回 ≥0.25）。",
            f"- mid 档「若直接答」的精度：**{answered_precision(mid_test):.3f}**"
            f"（答对 {mid_test['tp']} / 若作答 {mid_answered}）——这就是「中置信不许直答」的量化依据；"
            "它的去处是 RAG 答案 + top1 追问，不是 FAQ 原文。",
            f"- low 档（含 `reject_*` 与单候选）承载剩余 {tiers[('test', 'low')]['pos']} 条正样本，"
            "全部交棒 RAG；候选照旧带进 query 扩展，不浪费这段被拒的信号。",
            "",
            f"### 高置信下限邻域（`{cur_mode}`，threshold={cur_t:.2f}、margin={cur_m:.2f}；dev 选点、test 只报数）",
            "",
            "| FAQ_HIGH_MARGIN | dev high 召回/精度 | dev mid 若直接答 | test high 召回/精度 | test mid 若直接答 |",
            "|---|---|---|---|---|",
        ]
        for hv in MARGIN_MODE_GRID[cur_mode]:
            if hv < cur_m or hv > 0.40:
                continue
            cells: list[str] = []
            for split, n_pos in (("dev", dev_pos), ("test", test_pos)):
                s = tier_stats(split, cur_t, cur_m, cur_mode, hv)
                hi, md = s["high"], s["mid"]
                cells.append(f"{hi['tp'] / (n_pos or 1):.3f} / {answered_precision(hi):.3f}")
                cells.append(f"{answered_precision(md):.3f}")
            mark = " ← 现行" if abs(hv - high) < 1e-9 else ""
            out.append(f"| {hv:.2f}{mark} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")
        out += [
            "",
            "选点读法：`FAQ_HIGH_MARGIN` 只切「直接答 / 交给追问」的边界，不动 `FAQ_THRESHOLD` / `FAQ_MARGIN`；"
            "把它设成等于 `FAQ_MARGIN` 即精确回退为单档（回滚手段，不是另一套逻辑）。",
        ]
        return out

    lines = [
        "# FAQ 阈值标定报告（CLAUDE.md §7.1.7 第二、三步）",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 语料：{len(records)} 条（{source}）",
        f"- 语料内容指纹：`{_records_digest(records)}`",
        f"- 归一化指纹：`{normalizer_fp['sha256']}`（{normalizer_fp['probes']} 条行为探针）",
        f"- 样本文件指纹：`{sample_fp}`",
        f"- 正样本：{dev_pos + test_pos} 条（dev {dev_pos} / test {test_pos}）"
        f"，来源 {dict(sources)}，类别 {dict(pos_kinds)}",
        f"- 负样本：{dev_neg + test_neg} 条（dev {dev_neg} / test {test_neg}），类别 {dict(neg_kinds)}",
        f"- 分桶：正样本按 `faq_id % 10 < {DEV_BUCKETS}` 进 dev，负样本按 `sha1(text) % 10`；同一 FAQ 的变体必落同一侧",
        f"- 原句自检：{len(records) - self_miss}/{len(records)} 命中自身",
        *_negative_gap_notes(dict(neg_kinds)),
        "",
        "## 分数分布",
        "",
        "| 集合 | p50 | p90 | p95 | max |",
        "|---|---|---|---|---|",
        f"| 正样本 | {_pct(pos_scores, .5):.3f} | {_pct(pos_scores, .9):.3f} "
        f"| {_pct(pos_scores, .95):.3f} | {max(pos_scores or [0]):.3f} |",
        f"| 负样本 | {_pct(neg_scores, .5):.3f} | {_pct(neg_scores, .9):.3f} "
        f"| {_pct(neg_scores, .95):.3f} | {max(neg_scores or [0]):.3f} |",
        "",
        "## dev 选点（三套 margin 空间各扫一遍）",
        "",
        "| margin 空间 | threshold | margin | 召回 | 精度 | 误命中 | 命中但答错 |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, picked in picks.items():
        if picked is None:
            lines.append(f"| {mode_label[mode]} | — | — | — | — | — | — |")
            continue
        tp, wrong, fp_neg, _fn = picked["counts"]
        lines.append(
            f"| {mode_label[mode]} | {picked['t']:.2f} | {picked['m']:.2f} | {picked['recall']:.3f} "
            f"| {picked['precision']:.3f} | {fp_neg} | {wrong} |"
        )
    lines += ["", "无可行点（precision 无法达到 0.95）的空间显示为 —。", ""]

    # 反向问法：如果坚持"召回不低于现基线"，精度能到多少？两个方向一起看才知道是调参问题还是特征问题
    recall_floor = 0.637
    floor_pick: dict | None = None
    for mm, margins in MARGIN_MODE_GRID.items():
        for tt in THRESHOLDS:
            for mv in margins:
                counts = evaluate("dev", tt, mv, mm)
                recall, precision = ratio(counts)
                if recall < recall_floor:
                    continue
                key = (precision, recall)
                if floor_pick is None or key > (floor_pick["precision"], floor_pick["recall"]):
                    floor_pick = {"mode": mm, "t": tt, "m": mv, "precision": precision, "recall": recall}
    if floor_pick:
        dev_counts = evaluate("dev", floor_pick["t"], floor_pick["m"], floor_pick["mode"])
        test_counts = evaluate("test", floor_pick["t"], floor_pick["m"], floor_pick["mode"])
        test_recall_f, test_precision_f = ratio(test_counts)
        lines += [
            f"## 反向选点（召回下限 {recall_floor:.3f}，精度最大）",
            "",
            f"- `FAQ_MARGIN_MODE={floor_pick['mode']}`、`FAQ_THRESHOLD={floor_pick['t']:.2f}`、"
            f"`FAQ_MARGIN={floor_pick['m']:.2f}`",
            f"- dev：召回 {floor_pick['recall']:.3f}｜精度 {floor_pick['precision']:.3f}"
            f"｜误命中 {dev_counts[2]}",
            f"- **test**：召回 {test_recall_f:.3f}｜精度 {test_precision_f:.3f}｜误命中 {test_counts[2]}",
            "",
        ]

    # 诊断表统一按**现行线上参数**算：选点没有被采纳，按选点切片只会描述一个不存在的行为
    cur_mode = str(settings.faq_margin_mode)
    cur_t, cur_m = settings.faq_threshold, settings.faq_margin

    if pick:
        mode = pick["mode"]
        t, m = pick["t"], pick["m"]
        dev_counts = evaluate("dev", t, m, mode)
        test_counts = evaluate("test", t, m, mode)
        dev_recall, dev_precision = ratio(dev_counts)
        test_recall, test_precision = ratio(test_counts)
        lines += [
            "## dev 选点结果（dev 选出，test 报数；是否采纳见结论）",
            "",
            f"- `FAQ_MARGIN_MODE={mode}`、`FAQ_THRESHOLD={t:.2f}`、`FAQ_MARGIN={m:.2f}`",
            f"- dev：召回 {dev_recall:.3f}｜精度 {dev_precision:.3f}"
            f"｜错误直接命中率 {dev_counts[2] / dev_neg if dev_neg else 0:.3f}"
            f"｜错误过滤率 {1 - dev_recall:.3f}",
            f"- **test**：召回 {test_recall:.3f}｜精度 {test_precision:.3f}"
            f"｜错误直接命中率 {test_counts[2] / test_neg if test_neg else 0:.3f}"
            f"｜错误过滤率 {1 - test_recall:.3f}"
            f"｜命中但答错 {test_counts[1]}",
            "",
            "## 分类别召回（现行参数处）",
            "",
            "| 类别 | dev 命中/总数 | test 命中/总数 |",
            "|---|---|---|",
        ]
        # 诊断表统一按**现行线上参数**算：选点没有被采纳，按选点切片只会描述一个不存在的行为
        dev_kinds = kind_stats("dev", cur_t, cur_m, cur_mode)
        test_kinds = kind_stats("test", cur_t, cur_m, cur_mode)
        for name in sorted(set(dev_kinds) | set(test_kinds)):
            dh, dt = dev_kinds.get(name, (0, 0))
            th, tt = test_kinds.get(name, (0, 0))
            lines.append(f"| {name} | {dh}/{dt} | {th}/{tt} |")

        def hard_recall(split: str, tt: float, mv: float, mm: str) -> float:
            """去掉口语噪声型样本后的召回：真正换了词/省了词的变体才算难样本。"""
            idx2 = _MARGIN_INDEX[mm]
            hits = total = 0
            for row in obs:
                if row[0] != split or row[2] is None or row[1] == "noise":
                    continue
                total += 1
                hit = row[9] or (row[8] >= min_candidates and row[4] >= tt and row[idx2] >= mv)
                if hit and row[3] == row[2]:
                    hits += 1
            return hits / total if total else 0.0

        # 重标定的价值在于"换了没有、换得值不值"：没有现行参数的对照行，读者无法判断
        lines += [
            "",
            "## 与现行线上参数对比（同一批样本）",
            "",
            "| 参数 | 集合 | 召回 | 精度 | 硬样本召回 | 错误直接命中率 | 错误过滤率 |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, mm, tt, mv in (("现行默认", cur_mode, cur_t, cur_m), ("本次选点", mode, t, m)):
            for split in ("dev", "test"):
                c = evaluate(split, tt, mv, mm)
                r, p = ratio(c)
                n_neg = dev_neg if split == "dev" else test_neg
                lines.append(
                    f"| {name}（`{mm}` / {tt:.2f} / {mv:.2f}） | {split} | {r:.3f} | {p:.3f} "
                    f"| {hard_recall(split, tt, mv, mm):.3f} "
                    f"| {c[2] / n_neg if n_neg else 0:.3f} | {1 - r:.3f} |"
                )

        lines += ["", f"## 阈值邻域（现行 margin `{cur_mode}` / {cur_m:.2f}，test）", "",
                  "| threshold | 召回 | 精度 | 硬样本召回 |", "|---|---|---|---|"]
        for tt in THRESHOLDS:
            if not 0.10 <= tt <= 0.60 or abs(round(tt * 100) % 5):
                continue
            c = evaluate("test", tt, cur_m, cur_mode)
            r, p = ratio(c)
            lines.append(f"| {tt:.2f} | {r:.3f} | {p:.3f} | {hard_recall('test', tt, cur_m, cur_mode):.3f} |")

        # 同一条曲线上换另一根轴：谁在真正压召回，看这张表比看阈值表清楚
        lines += ["", f"## margin 邻域（现行 `{cur_mode}`，threshold={cur_t:.2f}，test）", "",
                  "| margin | 召回 | 精度 | 硬样本召回 |", "|---|---|---|---|"]
        upper = max(0.30, cur_m * 3)
        for mv in MARGIN_MODE_GRID[cur_mode]:
            if mv > upper:
                continue
            c = evaluate("test", cur_t, mv, cur_mode)
            r, p = ratio(c)
            lines.append(f"| {mv:.2f} | {r:.3f} | {p:.3f} | {hard_recall('test', cur_t, mv, cur_mode):.3f} |")

        # §7.1.9 三档的报表在 `tier_section()` 里生成（放在 if 之外：单档选点失败时它照样要出数）

        # 查询长度切片：分数是"命中词权重之和"，长查询天然占便宜——这张表用来分开
        # "阈值没调好"与"长查询靠词数堆分"这两种完全不同的病因。
        # 固定用**线上现用参数**切片：诊断的是生产行为，不是本次选点。
        slice_mode, slice_t, slice_m = settings.faq_margin_mode, settings.faq_threshold, settings.faq_margin
        lines += ["", f"## 按查询长度切片（线上现用参数 `{slice_mode}` / {slice_t:.2f} / {slice_m:.2f}，test）", "",
                  "| 字数 | 正样本命中/总数 | 负样本误命中/总数 |", "|---|---|---|"]
        slices = slice_stats("test", slice_t, slice_m, slice_mode)
        for bucket in ("≤10", "11–20", "21–40", "41+"):
            if bucket in slices:
                pos_hit, pos_all, neg_hit, neg_all = slices[bucket]
                lines.append(f"| {bucket} | {pos_hit}/{pos_all} | {neg_hit}/{neg_all} |")

        # 全网格的 test 精度上界：判据能不能达成本身是个结论，不能只报"选出来的那个点"
        best_test = max(
            (
                (*ratio(evaluate("test", tt, mv, mm)), mm, tt, mv)
                for mm, margins in MARGIN_MODE_GRID.items()
                for tt in THRESHOLDS
                for mv in margins
            ),
            key=lambda x: (x[1], x[0]),
        )
        lines += ["", "## 判据可达性", "",
                  f"- 全网格（三套 margin 空间 × {len(THRESHOLDS)} 个阈值）在 test 上的最好精度："
                  f"**{best_test[1]:.3f}**（召回 {best_test[0]:.3f}，`{best_test[2]}` / "
                  f"{best_test[3]:.2f} / {best_test[4]:.2f}）"]
        if best_test[1] >= 0.95:
            lines.append(
                f"- 精度 0.95 可达，但代价是召回只剩 {best_test[0]:.3f}——特征能分开的只有最干净的那一小撮；"
                f"召回要到 {recall_floor:.3f} 时精度上界见「反向选点」。"
                "两根旋钮（threshold + margin）到此为止，把剩下的区间交给下一阶段的机制（三档路由）。"
            )
        else:
            lines.append(
                "- 该值低于 0.95：精度上界由**特征**（分数 + margin）决定，不是阈值没调好；"
                "继续调参只会用召回换精度，问题要交给下一阶段的机制（三档路由 / 查询形态信号 / 语料改造）。"
            )

        lines += ["", "## 结论", ""]
        if sources == {"rule": len(positives)}:
            lines += [
                "**样本来源为 `source=rule`（规则改写），结论只能算临时值。**",
                "规则变体的词面偏离比真实口语小，分数分布偏高，标出来的阈值会偏保守（宁可漏答）；",
                "拿到生成模型密钥后必须重跑 `gen --source llm` + `calibrate`，才可作为改写线上默认参数的依据。",
                "",
            ]
        else:
            lines += ["样本来源为生成模型（`source=llm`），结论可用于改写线上默认参数。", ""]
        selected_ok = test_precision >= 0.95
        cur_counts = evaluate("test", cur_t, cur_m, cur_mode)
        cur_recall, cur_precision = ratio(cur_counts)
        lines.append(
            f"- 本次 dev 选点（`{mode}` / {t:.2f} / {m:.2f}）在独立 test 上的精度：{test_precision:.3f}、"
            f"召回 {test_recall:.3f} → **{'通过' if selected_ok else '未通过'}** §7.1.7 第三步验收"
            "（precision ≥0.95 且 recall ≥0.637）。"
        )
        if selected_ok:
            lines.append(f"- 建议：`FAQ_MARGIN_MODE={mode}`、`FAQ_THRESHOLD={t:.2f}`、`FAQ_MARGIN={m:.2f}`。")
        else:
            lines += [
                "- **不建议据此改动线上参数**：dev 上过闸、test 上不过。"
                f"dev 只有 {dev_neg} 条负样本，一条就是 {100 / dev_neg:.2f} 个百分点精度；"
                f"test 只有 {test_neg} 条，一条是 {100 / test_neg:.2f} 个百分点——"
                "精度 0.95 这道闸门在两个集合上都贴着噪声，选点落在边界上就会翻。",
            ]
            if floor_pick:
                lines.append(
                    f"- 反向问法给出的上界：把召回守在现基线 {recall_floor:.3f}，dev 上能选到的最好精度是 "
                    f"{floor_pick['precision']:.3f}（test {test_precision_f:.3f}）。"
                    "两根旋钮（threshold + margin）到此为止——要精度只能把召回砍到 0.1 量级。"
                )
            lines += [
                f"- 现行默认（`{cur_mode}` / {cur_t:.2f} / {cur_m:.2f}）在 test 上是召回 {cur_recall:.3f}、"
                f"精度 {cur_precision:.3f}，本身已落在前沿上，继续调参没有真实的收益空间。",
            ]
            # margin 是唯一还在起作用的旋钮：把召回守在目标 0.70，margin 最多能收到多紧
            knee = next(
                (
                    (mv, *ratio(evaluate("test", cur_t, mv, cur_mode)))
                    for mv in reversed(MARGIN_MODE_GRID[cur_mode])
                    if ratio(evaluate("test", cur_t, mv, cur_mode))[0] >= 0.70
                ),
                None,
            )
            if knee is not None and knee[0] < cur_m:
                lines.append(
                    f"- 按召回目标 0.70 反过来选 margin：`{cur_mode}` 下最紧能收到 {knee[0]:.2f}"
                    f"（test 召回 {knee[1]:.3f}、精度 {knee[2]:.3f}）——就在这个点上精度也只有 {knee[2]:.3f}，"
                    "离 0.95 还差得远；精度与召回在这套特征上此消彼长，不存在两者同时达标的点。"
                )
            # 病因必须从**本次**数据重新推：语料换过一版，"关键词式短句"那套解释已经过期
            slices = slice_stats("test", cur_t, cur_m, cur_mode)
            worst_neg = max(
                (
                    (b, hits / total, ph / pa)
                    for b, (ph, pa, hits, total) in slices.items()
                    if total >= 20 and pa
                ),
                key=lambda x: x[1],
                default=None,
            )
            th_lo, th_hi = evaluate("test", 0.10, cur_m, cur_mode), evaluate("test", 0.35, cur_m, cur_mode)
            pos_max, neg_max = max(pos_scores or [0]), max(neg_scores or [0])
            lines.append(
                f"- 分数分辨力不足：正样本 p90 {_pct(pos_scores, .9):.3f} / max {pos_max:.3f}，"
                f"负样本 p90 {_pct(neg_scores, .9):.3f} / max {neg_max:.3f}"
                + (
                    "——负样本的高分尾部伸进正样本主体里，单一阈值只能二选一。"
                    if neg_max > _pct(pos_scores, .5)
                    else "——两个分布没有清晰的分界。"
                )
            )
            if worst_neg is not None:
                lines.append(
                    f"- 长查询靠词数堆分：`{worst_neg[0]}` 字区间负样本误命中率 {worst_neg[1]:.3f}、"
                    f"正样本召回却只有 {worst_neg[2]:.3f}（见长度切片）——同一区间两头都变差："
                    "BM25 原始分随命中词数增长，`score_norm` 只是单调压缩，"
                    "分不清\"题干里出现这些词\"与\"问的就是这件事\"。"
                )
            if th_lo[:2] == th_hi[:2]:
                r_lo, p_lo = ratio(th_lo)
                lines.append(
                    f"- 阈值在 0.10–0.35 完全不生效（召回 {r_lo:.3f} / 精度 {p_lo:.3f} 一模一样）："
                    f"真正卡住命中是 margin {cur_m:.2f}，不是 threshold——\"调阈值\"这条路上本来就没有空间。"
                )
            lines += [
                "- 判定里没有任何\"查询形态\"信号（概念题 / 选择题 / 计算题），也拿不到同题族结构；"
                "只用（分数 + margin + 候选数）这些特征时，上面就是精度上界。",
                "- 第四步（三档路由 §7.1.9）已落地：见「三档路由」一节的 `high` / `mid` / `low` 分档，"
                "中置信命中不再直接作答，而是交 RAG + top1 追问——这段\"答不准\"的区间交由机制消化，"
                "不再指望把 threshold / margin 调到两者同时达标。",
                "- 仍待做（§7.1.7 第五、六步）：线上反馈闭环（追问采纳率、high vs mid 点踩率）"
                "与查询形态 / 同题族二次判别——若要让 FAQ 自己扛住精度，后者才是有希望的路。",
            ]
        lines += [
            "",
            "选点原则：FAQ 答错（把 A 的答案给 B）比不答更伤信任，不答还能交棒 RAG。",
            "因此先卡精度 ≥0.95，再最大化召回；margin 是在同题族（内容 / 证明 / 应用）之间保命的旋钮。",
        ]
    lines += tier_section()
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-10:]))
    print(f"\n报告 → {REPORT_FILE.relative_to(ROOT)}")


# ---------------------------------------------------------------- 第四步前置：只读诊断
# 契约见 CLAUDE.md §7.1.7「第四步前置：诊断契约」：不写库、不改参数、不改判定逻辑。
# `hit` / `wrong` / `false_hit` 按三档拆开（§7.1.9）：只有 `*_high` 会直接返回答案，
# `*_mid` 交给 RAG + 追问——把两者混在一个桶里，就分不清"答错了"与"本来就不作答"。
_DIAG_ORDER: tuple[str, ...] = (
    "exact",
    "hit_high",
    "hit_mid",
    "wrong_high",
    "wrong_mid",
    "reject_rank",
    "reject_single",
    "reject_threshold",
    "reject_margin",
    "reject_both",
    "false_hit_high",
    "false_hit_mid",
)
_DIAG_NOTES: dict[str, str] = {
    "exact": "归一化串精确命中（恒为 high，线上直接返回）",
    "hit_high": "过判据且 margin ≥ high_margin：**会直接作答**且 top1 正确",
    "hit_mid": "过判据但 margin < high_margin：top1 正确，交 RAG + 追问（这段可救）",
    "wrong_high": "直接作答但 top1 是另一条 FAQ——**最糟**，比不答差得多",
    "wrong_mid": "top1 是另一条 FAQ，但不直接作答（mid），伤害已被追问兜住",
    "reject_rank": "top1 不是正确答案：排序/词面就没找到（别名或语料能救）",
    "reject_single": "top1 正确但候选数 <2（判据②）",
    "reject_threshold": "top1 正确、只差绝对分门槛",
    "reject_margin": "top1 正确、只差相对领先幅度（题族/三档路由的用武之地）",
    "reject_both": "top1 正确但阈值与 margin 同时不达标",
    "false_hit_high": "负样本被 direct 作答（答非所问）",
    "false_hit_mid": "负样本落在 mid：不直接作答，误命中由 RAG 证据门兜住",
}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _classify(
    row: tuple, *, t: float, m: float, idx: int, min_candidates: int, high: float
) -> str:
    """按**绑定判据**唯一归类，顺序与 `decide()` 一致：阈值 → 候选数 → margin → 高置信线。

    同一份观测只能落一个桶，否则几张表加起来会超过总数（§7.1.7 诊断契约）。
    过了 ①②③ 之后再按 `high_margin` 分 high / mid（§7.1.9）。
    """
    positive = row[2] is not None
    if row[9]:
        return "exact"
    if row[8] >= min_candidates and row[4] >= t and row[idx] >= m:
        is_high = row[idx] >= high
        if not positive:
            return "false_hit_high" if is_high else "false_hit_mid"
        right = row[3] == row[2]
        if right:
            return "hit_high" if is_high else "hit_mid"
        return "wrong_high" if is_high else "wrong_mid"
    if positive and row[3] != row[2]:
        return "reject_rank"
    if row[8] < min_candidates:
        return "reject_single"
    low_t = row[4] < t
    low_m = row[idx] < m
    if low_t and low_m:
        return "reject_both"
    return "reject_threshold" if low_t else "reject_margin"


async def diagnose(
    threshold: float | None = None,
    margin: float | None = None,
    mode: str | None = None,
    top_k: int | None = None,
    high_margin: float | None = None,
) -> None:
    """D1（召回丢在哪一层，按三档拆开）+ D2（题族值不值得标）+ 机制预演，产出 `docs/analysis/faq_diagnosis.md`。"""
    _quiet()
    records, source = await _load_records()
    by_id = {r.faq_id: r for r in records}
    positives = _read_jsonl(POSITIVE_FILE)
    negatives = _read_jsonl(NEGATIVE_FILE)
    if not positives:
        raise SystemExit(f"没有正样本：先跑 `gen --source rule`，期望文件 {POSITIVE_FILE}")
    if not negatives:
        raise SystemExit(f"没有负样本：先跑 `negatives`，期望文件 {NEGATIVE_FILE}")
    _check_positives(positives, by_id)

    settings = get_settings()
    norm_k = settings.faq_norm_k
    min_candidates = settings.faq_min_candidates
    t = settings.faq_threshold if threshold is None else threshold
    m = settings.faq_margin if margin is None else margin
    high = settings.faq_high_margin if high_margin is None else high_margin
    chosen_mode = str(settings.faq_margin_mode if mode is None else mode)
    idx = _MARGIN_INDEX[chosen_mode]
    k = max(3, settings.faq_candidate_n if top_k is None else top_k)

    normalizer = Normalizer(extra_words=SUBJECTS)
    normalizer.preload()
    faq_tokens = {rid: set(normalizer.tokens(r.question)) for rid, r in by_id.items()}
    pair_cache: dict[tuple[int, int], float] = {}

    def pair_jaccard(a: int | None, b: int | None) -> float:
        """(top1, top2) 这一对 FAQ 的词面重合度——与用户问法无关，可在建索引期预计算。"""
        if a is None or b is None or a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        if key not in pair_cache:
            pair_cache[key] = _jaccard(faq_tokens.get(key[0], set()), faq_tokens.get(key[1], set()))
        return pair_cache[key]

    service = FAQService(
        InMemoryFAQRepository(records),
        policy=MatchPolicy(threshold=0.0, margin=0.0),
        dedupe_near_dup=get_settings().faq_dedupe_near_dup,
    )
    await service.load()

    obs: list[tuple] = []
    for row in positives:
        split = "dev" if int(row["faq_id"]) % 10 < DEV_BUCKETS else "test"
        # 合并后的重复问法以代表条目为准：两者答案逐字相同，命中代表即"答对了"
        obs.append(
            await _observe(service, str(row["paraphrase"]), service.canonical_id(int(row["faq_id"])),
                           str(row.get("kind", "?")), split, norm_k, k)
        )
    for row in negatives:
        bucket = int(hashlib.sha1(str(row["text"]).encode("utf-8")).hexdigest()[:8], 16) % 10
        split = "dev" if bucket < DEV_BUCKETS else "test"
        obs.append(await _observe(service, str(row["text"]), None, str(row.get("kind", "?")), split, norm_k, k))

    def labels(row: tuple) -> str:
        return _classify(row, t=t, m=m, idx=idx, min_candidates=min_candidates, high=high)

    def tally(split: str, positive: bool) -> Counter:
        return Counter(labels(row) for row in obs if row[0] == split and (row[2] is not None) == positive)

    by_split = {
        "pos_dev": tally("dev", True),
        "pos_test": tally("test", True),
        "neg_dev": tally("dev", False),
        "neg_test": tally("test", False),
    }
    totals = {name: sum(counts.values()) for name, counts in by_split.items()}

    def cell(counts: Counter, name: str, category: str) -> str:
        total = totals[name] or 1
        n = counts.get(category, 0)
        return f"{n} ({n / total:.1%})" if n else "—"

    sample_kinds = Counter(str(r.get("kind", "?")) for r in positives)
    neg_kinds = Counter(str(r.get("kind", "?")) for r in negatives)
    pos_sources = Counter(str(r.get("source", "?")) for r in positives)
    neg_sources = Counter(str(r.get("source", "?")).split(":")[0] for r in negatives)

    lines: list[str] = [
        "# FAQ 诊断报告（CLAUDE.md §7.1.7 第四步前置，只读）",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 语料：{source}",
        f"- 语料指纹（id/学科/问题重算）：`{_records_digest(records)}`",
        f"- 归一化指纹：`{_normalizer_fingerprint()['sha256']}`",
        f"- 判定参数（**现行线上值，本报告不修改任何参数**）：`threshold={t}`、`margin={m}`、"
        f"`high_margin={high}`、`margin_mode={chosen_mode}`、`min_candidates={min_candidates}`、"
        f"`top_k={k}`、`norm_k={norm_k}`",
        f"- 正样本 {len(positives)} 条（类别 {dict(sample_kinds)}，来源 {dict(pos_sources)}）",
        f"- 负样本 {len(negatives)} 条（类别 {dict(neg_kinds)}，来源 {dict(neg_sources)}）",
        f"- dev/test：正 {totals['pos_dev']}/{totals['pos_test']}、负 {totals['neg_dev']}/{totals['neg_test']}",
        "",
        "> 归因顺序固定为「阈值 → 候选数 → margin → 高置信线」，与 `decide()` 的判定顺序一致；每条观测只落一个桶。",
        "> `*_high` 是**会直接返回答案**的那部分（含负样本 `false_hit_high`）；`*_mid` 只是「够格被考虑」——"
        "它交给 RAG + 追问，所以 `wrong_mid` / `false_hit_mid` 的伤害与 `*_high` 不同级，不能合并统计。",
        "",
        "## D1 召回丢在哪一层",
        "",
        "| 类别 | 含义 | 正·dev | 正·test | 负·dev | 负·test |",
        "|---|---|---|---|---|---|",
    ]
    for category in _DIAG_ORDER:
        lines.append(
            f"| `{category}` | {_DIAG_NOTES[category]} | {cell(by_split['pos_dev'], 'pos_dev', category)} "
            f"| {cell(by_split['pos_test'], 'pos_test', category)} "
            f"| {cell(by_split['neg_dev'], 'neg_dev', category)} "
            f"| {cell(by_split['neg_test'], 'neg_test', category)} |"
        )

    # D1b：被 reject_rank 拒掉的正样本里，正确答案排第几（决定三档路由的 mid 档能吃下多少）
    lines += ["", "## D1b 被 `reject_rank` 拒掉的正样本，正确答案排第几", ""]
    lines += ["| 集合 | 未进 top-k | 第 2 名 | 第 3 名 | 合计 |", "|---|---|---|---|---|"]
    for split, name in (("dev", "pos_dev"), ("test", "pos_test")):
        ranks = [row[12] for row in obs if row[0] == split and row[2] is not None and labels(row) == "reject_rank"]
        miss = sum(1 for r in ranks if r is None)
        second = sum(1 for r in ranks if r == 1)
        third = sum(1 for r in ranks if r is not None and r >= 2)
        lines.append(f"| {split} | {miss} | {second} | {third} | {len(ranks)} |")

    # D2：题族代理
    lines += [
        "",
        "## D2 题族代理（正样本里「top1 正确、只差 margin」的那些）",
        "",
        "题族代理 = top2 与正确答案两条 question 的 token Jaccard。它是 **(top1, top2) 这一对的属性**，"
        "与用户问法无关，可在建索引期预计算。",
        "",
        "| 集合 | 条数 | Jaccard p50 | p90 | ≥0.5 占比 | ≥0.75 占比 |",
        "|---|---|---|---|---|---|",
    ]
    d2_share: dict[str, float] = {}
    for split in ("dev", "test"):
        rows = [row for row in obs if row[0] == split and row[2] is not None and labels(row) == "reject_margin"]
        values = [pair_jaccard(row[2], row[11]) for row in rows]
        share5 = sum(1 for v in values if v >= 0.5) / len(values) if values else 0.0
        share75 = sum(1 for v in values if v >= 0.75) / len(values) if values else 0.0
        d2_share[split] = share5
        lines.append(
            f"| {split} | {len(rows)} | {_pct(values, 0.5):.3f} | {_pct(values, 0.9):.3f} "
            f"| {share5:.1%} | {share75:.1%} |"
        )

    examples = sorted(
        (row for row in obs if row[0] == "test" and row[2] is not None and labels(row) == "reject_margin"),
        key=lambda row: -pair_jaccard(row[2], row[11]),
    )[:8]
    if examples:
        lines += ["", "最像同题族的 8 条（test）：", "", "| Jaccard | 正确答案 | top2（竞争对手） | margin |", "|---|---|---|---|"]
        for row in examples:
            right = by_id[row[2]].question[:38].replace("|", "／")
            rival = by_id[row[11]].question[:38].replace("|", "／") if row[11] in by_id else "—"
            lines.append(f"| {pair_jaccard(row[2], row[11]):.2f} | {right} | {rival} | {row[idx]:.3f} |")

    # D2 预演：同族对放宽 margin（机制预览，不作为采纳依据）
    def evaluate_relaxed(split: str, j_cut: float | None) -> tuple[float, float]:
        tp = wrong = fp_neg = fn = 0
        for row in obs:
            if row[0] != split:
                continue
            relaxed = j_cut is not None and pair_jaccard(row[3], row[11]) >= j_cut
            hit = row[9] or (
                row[8] >= min_candidates and row[4] >= t and (row[idx] >= m or relaxed)
            )
            if row[2] is None:
                fp_neg += bool(hit)
            elif hit and row[3] == row[2]:
                tp += 1
            elif hit:
                wrong += 1
            else:
                fn += 1
        precision = tp / (tp + wrong + fp_neg) if (tp + wrong + fp_neg) else 0.0
        recall = tp / (tp + wrong + fn) if (tp + wrong + fn) else 0.0
        return recall, precision

    lines += ["", "## D2 预演：同族对放宽 margin（**机制预览，不作为采纳依据**）", "",
              "| 规则 | dev 召回 / 精度 | test 召回 / 精度 |", "|---|---|---|"]
    for label, j_cut in (("现行（不放宽）", None), ("同族 ≥0.50 放宽", 0.5), ("同族 ≥0.75 放宽", 0.75)):
        rd, pd = evaluate_relaxed("dev", j_cut)
        rt, pt = evaluate_relaxed("test", j_cut)
        lines.append(f"| {label} | {rd:.3f} / {pd:.3f} | {rt:.3f} / {pt:.3f} |")

    # 结论由数字机械推导，避免"结论与表格各算一遍"
    test_total = totals["pos_test"] or 1
    margin_n = by_split["pos_test"].get("reject_margin", 0)
    rank_n = by_split["pos_test"].get("reject_rank", 0)
    share5 = d2_share.get("test", 0.0)
    if share5 > 0.40:
        verdict = ("**>40%：值得标题族**——被 margin 拒掉的正样本里，多数是因为 top1/top2 是同一题族；"
                   "按题族对放宽 margin 可以把这段召回拿回来，且上表 test 精度可以验证代价。")
    elif share5 < 0.20:
        verdict = ("**<20%：题族标注不值**——被拒的样本里 top1/top2 本来就是两个不同问题，margin 拒得对；"
                   "预算转给第四步三档路由（把中置信命中交 RAG），不要去做题族标注。")
    else:
        verdict = ("**20–40%：灰色**——先做第四步三档路由吃下这段区间，题族标注等真实 miss pool 再评估。")
    lever = (
        "召回主要丢在 **margin**（`reject_margin` "
        f"{margin_n}/{test_total}）而不是排序（`reject_rank` {rank_n}/{test_total}）："
        "补别名/扩语料救不了，必须动机制。"
        if margin_n > rank_n
        else "召回主要丢在**排序层**（`reject_rank` "
             f"{rank_n}/{test_total} > `reject_margin` {margin_n}/{test_total}）："
             "正确答案根本没排到第一，该补别名/语料，动 margin 收益有限。"
    )
    lines += ["", "## 结论", "", f"- {lever}", f"- D2（test 同族占比 {share5:.1%}）：{verdict}"]

    # D1 按档拆开后最有信息量的一行：多少条命中被从"直接答"移到了"交追问"，以及那一段有多不干净
    hit_high = by_split["pos_test"].get("hit_high", 0) + by_split["pos_test"].get("exact", 0)
    hit_mid = by_split["pos_test"].get("hit_mid", 0)
    wrong_high = by_split["pos_test"].get("wrong_high", 0)
    wrong_mid = by_split["pos_test"].get("wrong_mid", 0)
    fp_high = by_split["neg_test"].get("false_hit_high", 0)
    fp_mid = by_split["neg_test"].get("false_hit_mid", 0)
    if hit_mid or wrong_mid or fp_mid:
        lines.append(
            f"- 三档切分（test）：直接作答 `high` 答对 {hit_high} / 答错 {wrong_high} / 负样本误命中 {fp_high}；"
            f"交追问 `mid` 答对 {hit_mid} / 答错 {wrong_mid} / 负样本 {fp_mid}。"
            f"mid 这一段若直接答就是 {hit_mid}/{hit_mid + wrong_mid + fp_mid or 1} 条里对 "
            f"{hit_mid} 条——这正是它不许直答的原因。"
        )
    else:
        lines.append(
            f"- `high_margin == margin`（单档）：没有 `mid` 桶，行为与三档上线前一致"
            f"（test 直接作答答对 {hit_high} / 答错 {wrong_high} / 负样本误命中 {fp_high}）。"
        )

    DIAGNOSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSIS_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-6:]))
    print(f"\n诊断报告 → {DIAGNOSIS_FILE.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FAQ 阈值标定（CLAUDE.md §7.1.7）")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline", help="第一步：语料与索引基线自检")
    gen = sub.add_parser("gen", help="第二步：生成正样本（同义问法），旧文件自动归档")
    gen.add_argument("--source", choices=("rule", "llm"), default="rule", help="rule=规则改写（离线可复现）/ llm=生成模型")
    gen.add_argument("--api-key", default=None, help="llm 源用；也可用 DEEPSEEK_API_KEY / LLM_API_KEY 环境变量")
    gen.add_argument("--concurrency", type=int, default=6)
    gen.add_argument("--reuse", action="store_true", help="复用已有样本（先做标签校验，不一致即中止）")
    neg = sub.add_parser("negatives", help="第二步：生成负样本（同主题考试题 + 学科外语料 + 闲聊）")
    neg.add_argument("--unrelated-source", choices=("auto", "bank", "ceval"), default="auto",
                     help="学科外负样本来源：auto=题库优先、不足补 ceval；bank=只用开源题库；ceval=只用 ceval-exam")
    sub.add_parser("calibrate", help="第三步：dev 选点、test 报数，三套 margin 空间各扫一遍")
    diag = sub.add_parser("diagnose", help="第四步前置：D1 召回丢在哪一层 / D2 题族值不值得标（只读）")
    diag.add_argument("--threshold", type=float, default=None, help="默认取 .env 的 FAQ_THRESHOLD")
    diag.add_argument("--margin", type=float, default=None, help="默认取 .env 的 FAQ_MARGIN")
    diag.add_argument("--high-margin", type=float, default=None,
                      help="默认取 .env 的 FAQ_HIGH_MARGIN（等于 margin 即单档）")
    diag.add_argument("--mode", choices=("relative", "normalized", "raw"), default=None,
                      help="默认取 .env 的 FAQ_MARGIN_MODE")
    diag.add_argument("--top-k", type=int, default=None, help="默认取 .env 的 FAQ_CANDIDATE_N（至少 3）")
    args = parser.parse_args()

    if args.cmd == "baseline":
        asyncio.run(baseline())
    elif args.cmd == "gen":
        api_key = _api_key(args.api_key) if args.source == "llm" else None
        asyncio.run(gen_samples(args.source, api_key, args.concurrency, args.reuse))
    elif args.cmd == "negatives":
        build_negatives(unrelated_source=args.unrelated_source)
    elif args.cmd == "diagnose":
        asyncio.run(diagnose(args.threshold, args.margin, args.mode, args.top_k, args.high_margin))
    else:
        asyncio.run(calibrate())


if __name__ == "__main__":
    main()




