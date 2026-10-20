"""证据门：什么时候该答"不知道"。

只看重排分是不够的——同题族（"顶点式怎么写" vs "一般式怎么化成顶点式"）分数能到 0.6
以上却答非所问。所以这里用四条并列判据，任一不满足就不生成：

1. 分数门槛：top1 分数 ≥ `RAG_EVIDENCE_THRESHOLD`；
2. **跨文档**领先幅度：top1 与"另一篇文档里的最好证据"之差 ≥ `RAG_EVIDENCE_MARGIN`；
   同一篇文档的多块证据是**互相印证**，不算歧义；
3. 覆盖度：命中的父块数 ≥ `RAG_EVIDENCE_MIN_DOCS`，只有 1 块时要求 top1 明显更高；
4. **问题词覆盖**：问句里的实词有多大比例真的出现在 top1 证据里（`RAG_EVIDENCE_QCOV`）。

第 2 条是 2026-10-20 实测后改的：原先拿 top1 - top2 比，但 top2 往往就是同一份教案的
相邻父块，分数天然接近（实测 margin 0.0006–0.02），导致 top1 已经 0.99 还判 `ambiguous`。
真正的歧义是"两篇不同资料给出不同答案"，按文档比较才对应这个语义。

第 4 条是标定之后加的（763 条黄金集，见 `docs/analysis/rag_threshold_report.md`）：
融合分在"同主题不同知识点"的语料上区分度很差（可答与不可答的 top1 中位数只差 0.04），
而问句实词是否出现在证据里是**独立信号**——它等价于一次极便宜的词面重排，不花钱不掉速。
代价是改写得很厉害的问句（同义替换多）会被误拒，所以门槛定得低（0.2 = 五个实词里中一个）。

阈值是保守的（宁可拒答，不可编造），因为拒答还能转人工，编造会直接毁信任。
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_system.core.retriever import RetrievalResult

# 与 eval_rag.py 的判据保持一致的停用词：只过滤虚词，不影响实词覆盖
_STOPWORDS = set(
    "的 了 是 在 和 与 或 也 都 就 而 及 等 这 那 有 我 你 他 它 们 个 中 上 下 为 以 对 从 到 把 被 让 "
    "一个 什么 怎么 如何 哪些 多少 请 问 怎么办 怎么样 咋办 咋样 为何 干什么 是什么".split()
)


@dataclass(slots=True)
class GateDecision:
    passed: bool
    reason: str
    top_score: float
    margin: float
    n_docs: int
    qcov: float = 0.0

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "top_score": round(self.top_score, 4),
            "margin": round(self.margin, 4),
            "n_docs": self.n_docs,
            "qcov": round(self.qcov, 3),
        }


def query_coverage(query: str, text: str) -> float:
    """问句实词有多大比例出现在证据里。0.0 = 一个都没中，1.0 = 全中。"""
    import jieba

    def tokens(s: str) -> set[str]:
        return {
            t.strip().lower()
            for t in jieba.lcut(s or "")
            if t.strip() and t.strip().lower() not in _STOPWORDS and (len(t.strip()) >= 2 or t.strip().isdigit())
        }

    q = tokens(query)
    if not q:
        # 问句里没有可判定的实词（"怎么办""谢谢"），覆盖度无从谈起——不能因为判不了就拒答
        return 1.0
    return len(q & tokens(text)) / len(q)


def evaluate(
    result: RetrievalResult,
    *,
    threshold: float = 0.35,
    margin: float = 0.03,
    min_docs: int = 1,
    single_doc_bonus: float = 0.10,
    min_qcov: float = 0.20,
) -> GateDecision:
    evidences = result.evidences
    if not evidences:
        return GateDecision(False, "no_evidence", 0.0, 0.0, 0)

    top_evidence = evidences[0]
    top = top_evidence.score
    top_doc = _doc_key(top_evidence)
    n = len(evidences)
    n_docs = len({_doc_key(e) for e in evidences})
    qcov = query_coverage(result.query, top_evidence.text)

    # 跨文档领先幅度：只在"另一篇文档"里有可比证据时才构成竞争
    rival = next((e.score for e in evidences[1:] if _doc_key(e) != top_doc), 0.0)
    gap = top - rival if rival else top

    if top < threshold:
        return GateDecision(False, "below_threshold", top, gap, n_docs, qcov)

    # 只有一块证据时，用更高的门槛兜底：孤证最容易是"看着像"的噪声
    if n < max(min_docs, 2) and top < threshold + single_doc_bonus:
        return GateDecision(False, "single_weak_evidence", top, gap, n_docs, qcov)

    if rival and gap < margin:
        return GateDecision(False, "ambiguous", top, gap, n_docs, qcov)

    if qcov < min_qcov:
        return GateDecision(False, "low_query_coverage", top, gap, n_docs, qcov)

    return GateDecision(True, "ok", top, gap, n_docs, qcov)


def _doc_key(evidence) -> str:  # noqa: ANN001 - 避免与 retriever 循环导入
    """文档身份：优先进 parent_id 的 doc 前缀（`{doc_id}:p{n}`），否则退回 source。"""
    pid = getattr(evidence, "parent_id", "") or ""
    return pid.split(":p", 1)[0] if ":p" in pid else (getattr(evidence, "source", "") or pid)
