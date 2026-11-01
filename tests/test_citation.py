"""引用对齐单测：越界剔除、只收正文出现的引用、句级支持度。"""

from __future__ import annotations

from rag_system.core.citation import align, content_terms, repair, support_ratio
from rag_system.core.retriever import Evidence

MATH_TEXT = (
    "二次函数 y=ax²+bx+c 可以通过配方法化为顶点式 y=a(x-h)²+k，其中 h=-b/(2a)，k=(4ac-b²)/(4a)。"
)


def _evidence(text: str = MATH_TEXT, *, parent_id: str = "doc-1:p3", page: int = 12) -> Evidence:
    return Evidence(
        parent_id=parent_id,
        text=text,
        score=0.91,
        rerank_score=0.88,
        child_hits=[text[:60]],
        title="人教版-数学-九年级上册",
        source="F:/datasets/01_textbook/人教版-数学-九年级上册.pdf",
        subject="数学",
        doc_type="textbook",
        heading="22.1 二次函数",
        page=page,
    )


def test_out_of_range_reference_is_stripped() -> None:
    result = align("顶点式可以通过配方法得到[1]，详见另一份资料[9]。", [_evidence()])
    assert result.used == [1]
    assert result.out_of_range == [9]
    assert "[9]" not in result.text
    assert len(result.citations) == 1
    assert result.citations[0]["n"] == 1


def test_only_references_actually_used_are_returned() -> None:
    evidences = [_evidence(), _evidence(parent_id="doc-2:p1", page=9)]
    result = align("顶点式可以通过配方法得到[2]。", evidences)
    assert [c["n"] for c in result.citations] == [2]
    assert result.citations[0]["page_no"] == 9


def test_citation_fields_follow_contract() -> None:
    result = align("配方法可以化为顶点式[1]。", [_evidence()])
    citation = result.citations[0]
    assert citation["doc_id"] == "doc-1"
    assert citation["file_name"] == "人教版-数学-九年级上册.pdf"
    assert citation["section"] == "22.1 二次函数"
    assert citation["parent_id"] == "doc-1:p3"
    assert citation["score"] == 0.91
    assert citation["snippet"]


def test_locator_uses_page_when_available() -> None:
    result = align("配方法可以化为顶点式[1]。", [_evidence(page=12)])
    citation = result.citations[0]
    assert citation["locator_kind"] == "page"
    assert citation["locator"] == "第12页 · 22.1 二次函数"
    assert citation["title"] == "人教版-数学-九年级上册"


def test_locator_degrades_to_title_and_section_without_page() -> None:
    """新语料 `page` 恒为 0（HTML 教案/试卷没有页码），引用必须退化为「标题 · 章节」。

    这条是整个降级规则的守门用例：如果哪天有人把 `locator` 改成无条件用页码，
    前端会渲染出"第 0 页"——看起来有定位、实际点不开。
    """
    result = align("配方法可以化为顶点式[1]。", [_evidence(page=0)])
    citation = result.citations[0]
    assert citation["page_no"] is None
    assert citation["locator_kind"] == "section"
    assert citation["locator"] == "人教版-数学-九年级上册 · 22.1 二次函数"


def test_locator_degrades_to_title_without_section() -> None:
    evidence = Evidence(parent_id="doc-2:p1", text=MATH_TEXT, title="初中八年级历史教材", page=0)
    result = align("辛亥革命推翻了清朝统治[1]。", [evidence])
    assert result.citations[0]["locator_kind"] == "title"
    assert result.citations[0]["locator"] == "初中八年级历史教材"


def test_locator_degrades_to_file_name_without_title() -> None:
    evidence = Evidence(
        parent_id="doc-3:p1", text=MATH_TEXT, source="F:/datasets/x/教材.pdf", page=0
    )
    result = align("辛亥革命推翻了清朝统治[1]。", [evidence])
    assert result.citations[0]["locator_kind"] == "file"
    assert result.citations[0]["locator"] == "教材.pdf"


def test_supported_sentence_has_no_flag() -> None:
    result = align("顶点式可以通过配方法得到[1]。", [_evidence()])
    assert result.unsupported == []
    assert result.support_rate == 1.0


def test_unsupported_sentence_is_flagged() -> None:
    result = align("今天天气不错适合出去走走[1]。", [_evidence()])
    assert len(result.unsupported) == 1
    assert result.support_rate == 0.0


def test_fullwidth_brackets_are_recognised() -> None:
    result = align("顶点式可以通过配方法得到［1］。", [_evidence()])
    assert result.used == [1]
    assert "［1］" in result.text


def test_sentence_without_reference_is_not_judged() -> None:
    result = align("下面给出推导过程。顶点式可以通过配方法得到[1]。", [_evidence()])
    assert result.unsupported == []
    assert result.support_rate == 1.0


def test_no_reference_at_all_gives_empty_citations() -> None:
    result = align("顶点式可以通过配方法得到。", [_evidence()])
    assert result.citations == []
    assert result.support_rate == 0.0


def test_content_terms_drop_stopwords() -> None:
    # jieba 的默认词典会把"二次函数"切成"二次/函数"，断言按实际分词来，别把分词结果写死
    terms = content_terms("这个是可以使用的二次函数")
    assert "二次" in terms and "函数" in terms
    assert "是" not in terms and "可以" not in terms


def test_support_ratio_ignores_reference_marker() -> None:
    ratio = support_ratio("顶点式[1]", MATH_TEXT)
    assert ratio > 0.5


# ---------------------------------------------------------------------------
# 受控修复（§6.3）：**只做事实性剔除**，一次，剔完必须重新验证
# ---------------------------------------------------------------------------
def test_repair_drops_only_the_unsupported_sentence() -> None:
    text = "顶点式可以通过配方法得到[1]。今天天气不错适合出去走走[1]。"
    result, dropped = repair(text, [_evidence()])
    assert dropped is True
    assert "配方法" in result.text
    assert "今天天气不错" not in result.text
    assert result.unsupported == []          # 修完这一遍真的过了验，不是"删了就算"
    assert [c["n"] for c in result.citations] == [1]


def test_repair_reports_nothing_dropped_when_already_clean() -> None:
    """第二项返回值是给指标用的：没删过却报 repaired，引用修复率就永远虚高。"""
    text = "顶点式可以通过配方法得到[1]。"
    result, dropped = repair(text, [_evidence()])
    assert dropped is False
    assert result.text == text


def test_repair_leaves_nothing_when_every_sentence_is_unsupported() -> None:
    """全删光 → 空正文 + 空引用。编排层据此拒答（`citation_invalid`），
    而不是拿一句"我删完了"的空答案回给用户。"""
    text = "今天天气不错适合出去走走[1]。明天大概会下雨[1]。"
    result, dropped = repair(text, [_evidence()])
    assert dropped is True
    assert result.text == ""
    assert result.citations == []


def test_repair_keeps_first_pass_out_of_range_observation() -> None:
    """越界编号在第一遍就被删了，第二遍根本看不到它。上层还要报告"模型编过编号"，
    所以这条观测必须从第一遍带过来，否则 `citation_out_of_range` 永远不触发。"""
    text = "顶点式可以通过配方法得到[1]，详见别处[9]。今天天气不错[1]。"
    result, dropped = repair(text, [_evidence()])
    assert dropped is True
    assert result.out_of_range == [9]
    assert "天气" not in result.text


def test_repair_never_renumbers_the_surviving_reference() -> None:
    """删掉 `[3]` 就断了，不许把它改挂到别的证据上继续用——那样"对上了"的是
    编号，不是原文（§7.8 硬约束 3）。"""
    evidences = [_evidence(), _evidence(parent_id="doc-2:p1"), _evidence(parent_id="doc-3:p1")]
    text = "顶点式可以通过配方法得到[1]。今天天气不错[3]。"
    result, _ = repair(text, evidences)
    assert result.used == [1]
    assert all(cite["n"] == 1 for cite in result.citations)
