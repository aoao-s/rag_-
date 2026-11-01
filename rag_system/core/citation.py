"""引用对齐（CLAUDE.md §3.4 / §7.8）：正文里的 `[n]` 必须映射回本轮真实检索到的 chunk。

三件事，按顺序做：

1. **越界剔除**：`n` 不在 `[1, len(evidences)]` 里的，一律从正文删掉并记 `out_of_range`。
   模型最常见的"编来源"不是凭空写 URL，而是把不存在的编号写得很自然（上下文只有 6 条却在说 `[9]`）。
2. **只收正文真出现的引用**：证据给了 6 条，答案只用了第 2、5 条，`citations` 就只有这两条。
   把没用到的证据全塞进引用卡片，等于给用户一份"看起来有依据"的假清单。
3. **句级支持度**：句子切分后，用 jieba 实词与所引证据做覆盖率（去停用词），
   低于 `CITATION_MIN_SUPPORT` 记为"无支撑句"。v1 不做自动改写（要二次调 LLM，延迟翻倍），
   但允许**一次受控修复**（`repair()`）：把无支撑句整句剔掉再验一遍，验过了才放行；
   剔完没内容了由编排层走兜底——宁可转人工，也不放一条没有依据的结论出去。

用词面覆盖而不是 embedding：这一步要在每次回答上跑，几毫秒与几十毫秒的差别会直接进首 token 延迟；
而"句子里的实词是否出现在被引证据里"已经能抓住绝大部分张冠李戴（§7.1.8 的 qcov 是同一个思路）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import jieba

from rag_system.core.retriever import Evidence

# 半角与全角方括号都收：模型偶尔会输出 ［1］，那也应当被识别，否则引用会漏进正文
_REF = re.compile(r"[\[［]\s*(\d{1,2})\s*[\]］]")
_SENTENCE = re.compile(r"(?<=[。！？；!?;\n])")
_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)

# 只去虚词。"是什么/怎么"这类问句框架词在证据里也常出现，去掉后支持度反而更准
_STOPWORDS = set(
    "的 了 是 在 和 与 或 也 都 就 而 及 等 这 那 有 我 你 他 它 们 个 中 上 下 为 以 对 从 到 把 被 让 "
    "一个 我们 你们 他们 这个 那个 可以 需要 使用 进行 以及 因为 所以 但是 如果 那么 其中 根据 通过"
    .split()
)


@dataclass(slots=True)
class Alignment:
    """对齐结果。`text` 是剔除越界引用后的正文，其余字段供编排层决策与观测。"""

    text: str
    citations: list[dict] = field(default_factory=list)
    used: list[int] = field(default_factory=list)
    out_of_range: list[int] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    support_rate: float = 1.0


def content_terms(text: str) -> set[str]:
    """实词集合：去停用词与纯标点。数字保留（"0.74" 是结论的一部分）。"""
    terms: set[str] = set()
    for raw in jieba.cut(text):
        token = raw.strip()
        if not token or token in _STOPWORDS or _PUNCT.match(token):
            continue
        terms.add(token)
    return terms


def support_ratio(sentence: str, evidence_text: str, *, evidence_terms: set[str] | None = None) -> float:
    """句子实词被证据覆盖的比例。句子没有实词（如"如下："）时视为 1.0，不误伤。"""
    terms = content_terms(_REF.sub("", sentence))
    if not terms:
        return 1.0
    pool = evidence_terms if evidence_terms is not None else content_terms(evidence_text)
    return len(terms & pool) / len(terms)


def align(
    text: str,
    evidences: list[Evidence],
    *,
    min_support: float = 0.30,
) -> Alignment:
    """校验并整理模型答案里的引用。"""
    used: list[int] = []
    out_of_range: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(evidences):
            if n not in used:
                used.append(n)
            return match.group(0)
        if n not in out_of_range:
            out_of_range.append(n)
        return ""

    cleaned = _REF.sub(_replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()

    evidence_terms = [content_terms(ev.text) for ev in evidences]
    unsupported: list[str] = []
    cited_sentences = 0
    supported_sentences = 0

    for sentence in _SENTENCE.split(cleaned):
        sentence = sentence.strip()
        if not sentence:
            continue
        refs = [int(n) for n in _REF.findall(sentence)]
        if not refs:
            continue  # 没标引用的话不在这里判：它可能是过渡句，交给 require_citation 的整体判断
        cited_sentences += 1
        best = max(
            (support_ratio(sentence, "", evidence_terms=evidence_terms[n - 1]) for n in refs),
            default=0.0,
        )
        if best >= min_support:
            supported_sentences += 1
        else:
            unsupported.append(sentence)

    citations = [_citation(n, evidences[n - 1]) for n in sorted(used)]
    return Alignment(
        text=cleaned,
        citations=citations,
        used=sorted(used),
        out_of_range=out_of_range,
        unsupported=unsupported,
        support_rate=(supported_sentences / cited_sentences) if cited_sentences else 0.0,
    )


def repair(
    text: str,
    evidences: list[Evidence],
    *,
    min_support: float = 0.30,
) -> tuple[Alignment, bool]:
    """**最多一次**受控修复：剔除无支撑句后重新验证（实施手册 §6.3）。

    返回 `(对齐结果, 是否真的删过句子)`。第二项是给编排层与指标用的——"没删过"与
    "删完还能答"是两种不同的结果，混成一句会让"引用修复率"变成假数。

    三条不做，都有理由：

    - **不调模型改写**：改写要二次调用 LLM，延迟与成本翻倍，而且等于让模型给自己
      补依据——它最擅长的事就是把没有的话说得像有。
    - **不改来源元数据迁就模型**：把 `snippet` 换成模型说的那句，`[n]` 是对上了，
      可它指向的已经不是原文。这比没有引用更糟。
    - **不重新分配编号**：无支撑句里的 `[3]` 删掉就是删掉，不许改挂到 `[1]` 上继续用。

    只剩事实性剔除这一种改动，所以"修完的正文"每一句都还对得上第二遍验证的结果。
    """
    first = align(text, evidences, min_support=min_support)
    if not first.unsupported:
        return first, False

    dropped = {s.strip() for s in first.unsupported}
    kept = [piece for piece in _SENTENCE.split(first.text) if piece.strip() and piece.strip() not in dropped]
    repaired = align("".join(kept), evidences, min_support=min_support)
    # 越界引用在第一遍就被删掉了，第二遍看不到它们；但上层要能报告"模型编过编号"，
    # 所以把第一遍的观测带过来。其余字段以第二遍为准——它才是真正要发出去的正文。
    repaired.out_of_range = first.out_of_range
    return repaired, True


def _citation(n: int, evidence: Evidence) -> dict:
    parent_id = evidence.parent_id or ""
    snippet = evidence.child_hits[0] if evidence.child_hits else evidence.text
    return {
        "n": n,
        "doc_id": parent_id.split(":p", 1)[0],
        "file_name": Path(evidence.source).name if evidence.source else "",
        # `page_no` 只在真有页码时才非空。新语料的 HTML 教案/试卷没有页码、
        # 教材解析出的页标记也没落到 chunk 上，实测 3256/3256 行 `page=0`——
        # 所以**定位靠 `locator`**（有页码用页码，没有退化为「标题 · 章节」），
        # 由 `Evidence.locator` 统一给出，诊断接口与这里必须是同一句话。
        "page_no": evidence.page or None,
        "section": evidence.heading or None,
        "title": evidence.title or evidence.file_name,
        "locator": evidence.locator,
        "locator_kind": evidence.locator_kind,
        "parent_id": parent_id,
        "score": round(evidence.score, 4),
        "snippet": snippet.strip()[:200],
        # 学段/年级/学科跟着引用走：同一学科的 `教材.pdf` 在 7 个年级目录下各有一份，
        # 只显示文件名的话前端卡片分不清"初中九年级数学教材"与"初中七年级数学教材"
        "subject": evidence.subject or None,
        "stage": evidence.stage or None,
        "grade": evidence.grade or None,
    }
