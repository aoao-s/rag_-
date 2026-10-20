"""证据门：什么时候该答"不知道"。

只看重排分是不够的——同题族（"顶点式怎么写" vs "一般式怎么化成顶点式"）分数能到 0.6
以上却答非所问。所以这里用三条并列判据，任一不满足就不生成：

1. 分数门槛：top1 分数 ≥ `RAG_EVIDENCE_THRESHOLD`；
2. **跨文档**领先幅度：top1 与"另一篇文档里的最好证据"之差 ≥ `RAG_EVIDENCE_MARGIN`；
   同一篇文档的多块证据是**互相印证**，不算歧义；
3. 覆盖度：命中的父块数 ≥ `RAG_EVIDENCE_MIN_DOCS`，只有 1 块时要求 top1 明显更高。

第 2 条是 2026-10-20 实测后改的：原先拿 top1 - top2 比，但 top2 往往就是同一份教案的
相邻父块，分数天然接近（实测 margin 0.0006–0.02），导致 top1 已经 0.99 还判 `ambiguous`。
真正的歧义是"两篇不同资料给出不同答案"，按文档比较才对应这个语义。

阈值是保守的（宁可拒答，不可编造），因为拒答还能转人工，编造会直接毁信任。
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_system.core.retriever import RetrievalResult


@dataclass(slots=True)
class GateDecision:
    passed: bool
    reason: str
    top_score: float
    margin: float
    n_docs: int

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "top_score": round(self.top_score, 4),
            "margin": round(self.margin, 4),
            "n_docs": self.n_docs,
        }


def evaluate(
    result: RetrievalResult,
    *,
    threshold: float = 0.35,
    margin: float = 0.03,
    min_docs: int = 1,
    single_doc_bonus: float = 0.10,
) -> GateDecision:
    evidences = result.evidences
    if not evidences:
        return GateDecision(False, "no_evidence", 0.0, 0.0, 0)

    top_evidence = evidences[0]
    top = top_evidence.score
    top_doc = _doc_key(top_evidence)
    n = len(evidences)
    n_docs = len({_doc_key(e) for e in evidences})

    # 跨文档领先幅度：只在"另一篇文档"里有可比证据时才构成竞争
    rival = next((e.score for e in evidences[1:] if _doc_key(e) != top_doc), 0.0)
    gap = top - rival if rival else top

    if top < threshold:
        return GateDecision(False, "below_threshold", top, gap, n_docs)

    # 只有一块证据时，用更高的门槛兜底：孤证最容易是"看着像"的噪声
    if n < max(min_docs, 2) and top < threshold + single_doc_bonus:
        return GateDecision(False, "single_weak_evidence", top, gap, n_docs)

    if rival and gap < margin:
        return GateDecision(False, "ambiguous", top, gap, n_docs)

    return GateDecision(True, "ok", top, gap, n_docs)


def _doc_key(evidence) -> str:  # noqa: ANN001 - 避免与 retriever 循环导入
    """文档身份：优先进 parent_id 的 doc 前缀（`{doc_id}:p{n}`），否则退回 source。"""
    pid = getattr(evidence, "parent_id", "") or ""
    return pid.split(":p", 1)[0] if ":p" in pid else (getattr(evidence, "source", "") or pid)
