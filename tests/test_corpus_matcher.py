"""BM25 索引与命中判定的单测。"""

from __future__ import annotations

import pytest

from qa_system.res.qa_corpus import BM25Index, IndexedDoc
from qa_system.res.qa_matcher import MatchPolicy, decide, normalize_score


def _doc(faq_id: int, tokens: str) -> IndexedDoc:
    return IndexedDoc(faq_id=faq_id, question=f"q{faq_id}", subject="数学", tokens=tuple(tokens.split()))


@pytest.fixture()
def index() -> BM25Index:
    return BM25Index(
        [
            _doc(1, "勾股定理 证明 面积法"),
            _doc(2, "勾股定理 内容 斜边"),
            _doc(3, "一元二次方程 求根 公式"),
        ]
    )


class TestBM25:
    def test_ranked_by_relevance(self, index: BM25Index) -> None:
        hits = index.search(["面积法"], top_k=5)
        assert hits and hits[0][0].faq_id == 1

    def test_unknown_token_returns_empty(self, index: BM25Index) -> None:
        assert index.search(["量子纠缠"], top_k=5) == []

    def test_empty_query_returns_empty(self, index: BM25Index) -> None:
        assert index.search([], top_k=5) == []

    def test_empty_corpus(self) -> None:
        empty = BM25Index([])
        assert len(empty) == 0
        assert empty.search(["勾股定理"], top_k=3) == []

    def test_top_k_limits(self, index: BM25Index) -> None:
        assert len(index.search(["勾股定理"], top_k=1)) == 1

    def test_idf_non_negative(self, index: BM25Index) -> None:
        # 自己实现 IDF 的原因：语料极小时 rank_bm25 会出现负分
        hits = index.search(["勾股定理"], top_k=10)
        assert all(score > 0 for _, score in hits)


class TestMatcher:
    def test_normalize_score_monotonic(self) -> None:
        assert normalize_score(0) == 0.0
        assert normalize_score(6, 12) < normalize_score(24, 12) < 1.0

    def test_hit_when_clear_winner(self) -> None:
        result = decide([(1, 40.0), (2, 5.0)], MatchPolicy(threshold=0.62, margin=0.05))
        assert result.hit and result.faq_id == 1

    def test_below_threshold(self) -> None:
        result = decide([(1, 1.0)], MatchPolicy(threshold=0.62))
        assert not result.hit and result.reason == "below_threshold"

    def test_ambiguous_when_margin_small(self) -> None:
        result = decide([(1, 30.0), (2, 29.0)], MatchPolicy(threshold=0.5, margin=0.5))
        assert not result.hit and result.reason == "ambiguous"

    def test_empty_corpus(self) -> None:
        assert decide([]).reason == "empty_corpus"

