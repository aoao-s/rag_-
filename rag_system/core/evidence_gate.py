"""证据门：什么时候该答"不知道"。

只看重排分是不够的——同题族（"顶点式怎么写" vs "一般式怎么化成顶点式"）分数能到 0.6
以上却答非所问。所以这里用三条并列判据，任一不满足就不生成：

1. 分数门槛：top1 分数 ≥ `RAG_EVIDENCE_THRESHOLD`；
2. 领先幅度：top1 - top2 ≥ `RAG_EVIDENCE_MARGIN`（候选 ≥2 时），否则说明"好几个都差不多像"；
3. 覆盖度：命中的父块数 ≥ `RAG_EVIDENCE_MIN_DOCS`，只有 1 块时要求 top1 明显更高。

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

    top = evidences[0].score
    second = evidences[1].score if len(evidences) > 1 else 0.0
    gap = top - second
    n = len(evidences)

    if top < threshold:
        return GateDecision(False, "below_threshold", top, gap, n)

    # 只有一块证据时，用更高的门槛兜底：孤证最容易是"看着像"的噪声
    if n < max(min_docs, 2) and top < threshold + single_doc_bonus:
        return GateDecision(False, "single_weak_evidence", top, gap, n)

    if n > 1 and gap < margin:
        return GateDecision(False, "ambiguous", top, gap, n)

    return GateDecision(True, "ok", top, gap, n)
