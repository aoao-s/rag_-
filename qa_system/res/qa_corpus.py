"""分学科 BM25 索引。

两个刻意的设计：
1. **自己实现 BM25，不用 rank_bm25**：要控制 IDF 公式（用 log(1 + ...) 保证非负）与
   空语料行为，且避免小语料上 idf 为负导致分数不可解释。
2. **按 subject 分片**：旧项目全库单索引，长尾学科（地理、生物）的分数被大科目稀释，
   换学科阈值就失效。分片后每个学科独立 IDF，阈值才可跨学科复用。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class IndexedDoc:
    faq_id: int
    question: str
    subject: str | None
    tokens: tuple[str, ...]


class BM25Index:
    """不可变索引：构建完即只读，因此可安全地整体替换（双缓冲热更新）。"""

    __slots__ = ("_docs", "_tf", "_doc_len", "_avgdl", "_idf", "k1", "b")

    def __init__(self, docs: Sequence[IndexedDoc], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs = tuple(docs)
        self._tf: tuple[Counter[str], ...] = tuple(Counter(d.tokens) for d in self._docs)
        self._doc_len: tuple[int, ...] = tuple(len(d.tokens) for d in self._docs)
        total = sum(self._doc_len)
        self._avgdl = (total / len(self._docs)) if self._docs else 0.0

        n_docs = len(self._docs)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def __len__(self) -> int:
        return len(self._docs)

    @property
    def avgdl(self) -> float:
        return self._avgdl

    def search(self, tokens: Sequence[str], top_k: int = 20) -> list[tuple[IndexedDoc, float]]:
        """返回 (文档, 原始 BM25 分)，按分数降序；空查询/空语料返回空列表。"""
        if not self._docs or not tokens or self._avgdl <= 0:
            return []
        query_terms = Counter(tok for tok in tokens if tok in self._idf)
        if not query_terms:
            return []

        scores: list[tuple[IndexedDoc, float]] = []
        for idx, tf in enumerate(self._tf):
            doc_len = self._doc_len[idx] or 1
            denom_norm = self.k1 * (1.0 - self.b + self.b * doc_len / self._avgdl)
            score = 0.0
            for term, q_weight in query_terms.items():
                freq = tf.get(term)
                if not freq:
                    continue
                score += self._idf[term] * (freq * (self.k1 + 1.0)) / (freq + denom_norm) * q_weight
            if score > 0:
                scores.append((self._docs[idx], score))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]
