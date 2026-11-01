"""加载器测试：Office / 图片 / 本地 PDF / 工厂分派与降级。

原则：不依赖外部服务与真实教材——用 python-docx / python-pptx / PyMuPDF 现造最小文件。
缺依赖时 `importorskip`，不让本地环境差异变成红叉。
"""

from __future__ import annotations

import pytest

from rag_system.document_loaders.base import LoaderError
from rag_system.document_loaders.loader_office import DocxLoader, PptxLoader, _table_to_markdown
from rag_system.document_loaders.loader_pdf_local import LocalPdfLoader


def _make_docx(path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_heading("二次函数", level=1)
    document.add_paragraph("顶点式 y=a(x-h)^2+k 可以直接读出顶点坐标。")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "形式"
    table.cell(0, 1).text = "顶点"
    table.cell(1, 0).text = "顶点式"
    table.cell(1, 1).text = "(h, k)"
    document.save(str(path))
    return path


def _make_pptx(path):
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches, Pt

    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "函数图像"
    body = slide.placeholders[1]
    body.text_frame.text = "开口方向由 a 决定"
    paragraph = body.text_frame.add_paragraph()
    paragraph.text = "|a| 越大开口越窄"
    paragraph.font.size = Pt(18)
    box = slide.shapes.add_textbox(Inches(1), Inches(5), Inches(3), Inches(1))
    box.text_frame.text = "最后一条"
    presentation.save(str(path))
    return path


class TestDocxLoader:
    def test_heading_and_table(self, tmp_path):
        path = _make_docx(tmp_path / "数学-二次函数.docx")
        doc = DocxLoader().load(path)
        assert "# 二次函数" in doc.text  # add_heading(level=1) → Heading 1 → 一级标题
        assert "| 形式 | 顶点 |" in doc.text
        assert "| 顶点式 | (h, k) |" in doc.text
        assert doc.subject == "数学"
        assert doc.doc_type == "lesson_plan"
        assert doc.title == "数学-二次函数"

    def test_empty_docx_raises(self, tmp_path):
        docx = pytest.importorskip("docx")
        path = tmp_path / "空.docx"
        docx.Document().save(str(path))
        with pytest.raises(LoaderError):
            DocxLoader().load(path)


class TestPptxLoader:
    def test_title_and_shapes(self, tmp_path):
        path = _make_pptx(tmp_path / "函数图像.pptx")
        doc = PptxLoader().load(path)
        assert "## 第 1 页：函数图像" in doc.text
        assert "开口方向由 a 决定" in doc.text
        assert "最后一条" in doc.text
        # 形状按 top 排序：正文占位符在上，底部文本框在下
        assert doc.text.index("开口方向由 a 决定") < doc.text.index("最后一条")


class TestTableToMarkdown:
    def test_ragged_rows_padded(self):
        text = _table_to_markdown([["a", "b", "c"], ["1", "2"], ["", ""]])
        assert text.splitlines()[0] == "| a | b | c |"
        assert "| 1 | 2 |  |" in text  # 空行被丢掉，短行补空格

    def test_empty(self):
        assert _table_to_markdown([["", ""], []]) == ""


class TestLocalPdfLoader:
    def test_text_pdf_has_page_marker(self, tmp_path):
        pymupdf = pytest.importorskip("pymupdf")
        path = tmp_path / "数学-讲义.pdf"
        document = pymupdf.open()
        for number in (1, 2):
            page = document.new_page()
            page.insert_text((72, 72), f"第 {number} 页正文：" + "一元二次方程的求根公式推导。" * 3)
        document.save(str(path))
        document.close()

        doc = LocalPdfLoader().load(path)
        assert "<!-- page: 1 -->" in doc.text and "<!-- page: 2 -->" in doc.text
        assert doc.meta["pages"] == 2
        assert doc.meta["parser"] == "pymupdf"

    def test_empty_pdf_raises(self, tmp_path):
        pymupdf = pytest.importorskip("pymupdf")
        path = tmp_path / "空白.pdf"
        document = pymupdf.open()
        document.new_page()
        document.save(str(path))
        document.close()
        with pytest.raises(LoaderError):
            LocalPdfLoader().load(path)

    def test_split_pages(self):
        from rag_system.document_loaders.loader_pdf_local import split_pages

        pages = split_pages("<!-- page: 1 -->\n甲\n<!-- page: 2 -->\n乙")
        assert pages == [(1, "甲"), (2, "乙")]


class TestFactoryDispatch:
    def _factory(self, tmp_path, **overrides):
        from base.config import Settings

        s = Settings(MYSQL_PASSWORD="x", **overrides)
        from rag_system.document_loaders.thy_factory import DocumentLoaderFactory

        return DocumentLoaderFactory(s, None)

    def test_supported_extensions(self, tmp_path):
        factory = self._factory(tmp_path)
        for name in ("a.md", "b.srt", "c.docx", "d.pptx", "e.pdf", "f.png"):
            assert factory.is_supported(tmp_path / name)
        assert not factory.is_supported(tmp_path / "g.xlsx")

    def test_pdf_without_token_uses_local(self, tmp_path):
        from rag_system.document_loaders.loader_pdf_local import LocalPdfLoader

        factory = self._factory(tmp_path, MINERU_TOKEN="")
        assert isinstance(factory.get_loader(tmp_path / "a.pdf"), LocalPdfLoader)

    def test_pdf_parser_local_forced(self, tmp_path):
        from rag_system.document_loaders.loader_pdf_local import LocalPdfLoader

        factory = self._factory(tmp_path, MINERU_TOKEN="", RAG_PDF_PARSER="local")
        assert isinstance(factory.get_loader(tmp_path / "a.pdf"), LocalPdfLoader)

    def test_pdf_parser_mineru_requires_token(self, tmp_path):
        factory = self._factory(tmp_path, MINERU_TOKEN="", RAG_PDF_PARSER="mineru")
        with pytest.raises(LoaderError):
            factory.get_loader(tmp_path / "a.pdf")

    def test_docx_routed(self, tmp_path):
        assert isinstance(self._factory(tmp_path).get_loader(tmp_path / "a.docx"), DocxLoader)

    def test_fallback_when_cloud_fails(self, tmp_path):
        """云解析抛错时退本地，并在 meta 里留痕（不静默）。"""
        from rag_system.document_loaders.base import LoadedDoc
        from rag_system.document_loaders.thy_factory import _MinerUWithLocalFallback

        class Boom:
            extensions = (".pdf",)

            def load(self, path):
                raise RuntimeError("mineru 500")

        class LocalOk:
            extensions = (".pdf",)

            def load(self, path):
                return LoadedDoc(text="本地正文", source=str(path))

        combined = _MinerUWithLocalFallback(Boom(), LocalOk(), path_name="a.pdf")
        doc = combined.load(tmp_path / "a.pdf")
        assert doc.text == "本地正文"
        assert doc.meta["parser_fallback"] == "pymupdf"
        assert "mineru 500" in doc.meta["mineru_error"]

    def test_both_fail_reports_both(self, tmp_path):
        from rag_system.document_loaders.thy_factory import _MinerUWithLocalFallback

        class Boom:
            extensions = (".pdf",)

            def load(self, path):
                raise RuntimeError("cloud down")

        class Boom2(Boom):
            def load(self, path):
                raise ValueError("no text layer")

        combined = _MinerUWithLocalFallback(Boom(), Boom2(), path_name="a.pdf")
        with pytest.raises(LoaderError) as info:
            combined.load(tmp_path / "a.pdf")
        assert "cloud down" in str(info.value) and "no text layer" in str(info.value)


class TestImageLoader:
    def test_ocr_missing_or_text(self, tmp_path):
        """OCR 不可用时必须明确报错，而不是返回空字符串当成功。"""
        pytest.importorskip("PIL")
        from PIL import Image, ImageDraw

        from rag_system.document_loaders.loader_image import ImageLoader

        path = tmp_path / "板书.png"
        image = Image.new("RGB", (240, 80), "white")
        ImageDraw.Draw(image).text((10, 30), "y=ax2+bx+c", fill="black")
        image.save(path)

        from rag_system.document_loaders.ocr import ocr_available

        if not ocr_available():
            with pytest.raises(LoaderError):
                ImageLoader().load(path)
            return
        doc = ImageLoader().load(path)
        assert doc.doc_type == "image"


_HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>初中数学教案</title>
<style>p { color: red }</style></head>
<body>
<h1>二次函数</h1>
<p>教学目标</p>
<ul><li>理解顶点式</li><li>会用配方法</li></ul>
<table><tr><th>形式</th><th>顶点</th></tr><tr><td>顶点式</td><td>(h, k)</td></tr></table>
<script>var secret = 1;</script>
<p>顶点式 y=a(x-h)^2+k 可直接读出顶点。</p>
</body></html>
"""


class TestHtmlLoader:
    """教案 / 试卷 HTML（新语料的主体格式）。

    `meta` / `link` 是空元素：早先它们被放进"整段跳过"集合，自增不自减，
    把整个文档吞成空串——所以这里显式放一个 `<meta charset>` 进 header 作回归。
    """

    def _load(self, tmp_path, html: str = _HTML_PAGE, name: str = "教案.html"):
        from rag_system.document_loaders.loader_html import HtmlLoader

        path = tmp_path / name
        path.write_text(html, encoding="utf-8")
        return HtmlLoader().load(path)

    def test_structure_becomes_markdown(self, tmp_path):
        doc = self._load(tmp_path)
        assert "# 二次函数" in doc.text  # h1 → 一级标题
        assert "教学目标" in doc.text  # meta charset 没有吞掉正文
        assert "- 理解顶点式" in doc.text and "- 会用配方法" in doc.text  # li → 列表
        assert "| 形式 | 顶点 |" in doc.text and "| 顶点式 | (h, k) |" in doc.text

    def test_script_and_style_are_dropped(self, tmp_path):
        doc = self._load(tmp_path)
        assert "secret" not in doc.text  # MathJax/脚本留着就是脏 chunk
        assert "color: red" not in doc.text

    def test_title_and_inline_math_kept(self, tmp_path):
        doc = self._load(tmp_path)
        assert doc.title == "初中数学教案"  # <title> 供引用卡片用
        assert "y=a(x-h)^2+k" in doc.text  # 行内公式原样保留

    def test_subject_from_path(self, tmp_path):
        folder = tmp_path / "01_textbook" / "初中" / "数学"
        folder.mkdir(parents=True)
        doc = self._load(folder, name="教材.html")
        assert doc.subject == "数学"  # 学科只在目录里，文件名是通用词

    def test_empty_html_raises(self, tmp_path):
        empty = "<html><head><meta charset='utf-8'><title>空</title></head><body><script>a</script></body></html>"
        with pytest.raises(LoaderError):
            self._load(tmp_path, empty)


class TestCorpusClassification:
    """工厂按目录/文件名判 doc_type 与标题（新语料 `学科目录/教材.pdf` 结构）。"""

    def _factory(self, **overrides):
        from base.config import Settings
        from rag_system.document_loaders.thy_factory import DocumentLoaderFactory

        return DocumentLoaderFactory(Settings(MYSQL_PASSWORD="x", **overrides), None)

    def test_doc_type_by_directory(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _doc_type

        assert _doc_type(P("datasets/10_exam_papers/初中/数学/试卷.html"), "other") == "exam"
        assert _doc_type(P("datasets/01_textbook/高中/物理/教材.pdf"), "other") == "textbook"
        assert _doc_type(P("datasets/02_lesson_plans/初中/英语/教案.html"), "other") == "lesson_plan"
        assert _doc_type(P("datasets/01_textbook/初中/数学/随便.md"), "textbook") == "textbook"

    def test_doc_type_by_filename_when_no_dir(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _doc_type

        assert _doc_type(P("F:/inbox/九年级数学试卷.pdf"), "other") == "exam"
        assert _doc_type(P("F:/inbox/二次函数教案.docx"), "other") == "lesson_plan"
        assert _doc_type(P("F:/inbox/读书笔记.md"), "other") == "other"

    def test_generic_stem_gets_built_title(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _title

        # `教材.pdf` 直接当标题，前端会显示一排同名卡片，定位不到来源
        assert _title(P("datasets/01_textbook/初中/数学/教材.pdf"), "textbook", "数学", "教材") == "初中数学教材"
        assert _title(P("datasets/10_exam_papers/高中/物理/试卷.pdf"), "exam", "物理", "试卷") == "高中物理试卷"

    def test_specific_title_is_kept(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _title

        assert _title(P("datasets/02_lesson_plans/初中/数学/教案.html"), "lesson_plan", "数学", "初中数学教案") == "初中数学教案"

    def test_navigation_pages_are_skipped(self):
        from pathlib import Path as P

        factory = self._factory()
        # 索引导航页与"本科目材料缺失"的告示都不是知识正文，进库只会被当成教材片段召回
        assert not factory.is_supported(P("datasets/初高中教材教案试卷/索引.html"))
        assert not factory.is_supported(P("datasets/01_textbook/初中/道德与法治/资源说明.html"))
        assert factory.is_supported(P("datasets/02_lesson_plans/初中/数学/教案.html"))

    def test_guess_subject_canonicalizes_aliases(self):
        from rag_system.document_loaders.base import guess_subject

        assert guess_subject("F:/datasets/01_textbook/高中/生物学/教材.pdf") == "生物"
        assert guess_subject("F:/datasets/01_textbook/高中/思想政治/教材.pdf") == "道德与法治"
        assert guess_subject("F:/datasets/01_textbook/初中/道德与法治/教材.pdf") == "道德与法治"
        assert guess_subject("F:/datasets/x/未知.pdf") is None


class TestStageAndGrade:
    """新语料按 `阶段/学科/年级/` 三层组织：学段与年级必须能从路径认出来。

    认不出的后果不是"少了两个字段"——`stage`/`grade` 是 Milvus 的标量字段，
    识别失败会让整批文档落成空串，按年级过滤就永远查不到东西。
    """

    def test_stage_and_grade_from_path(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _grade, _stage

        root = P("datasets/初高中教材教案试卷-按年级")
        assert _stage(root / "高中/数学/高一/教材.pdf") == "高中"
        assert _grade(root / "高中/数学/高一/教材.pdf") == "高一"
        assert _stage(root / "初中/化学/九年级/教材.pdf") == "初中"
        assert _grade(root / "初中/化学/九年级/教材.pdf") == "九年级"

    def test_grade_aliases_are_normalized(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _grade

        # `初三` / `高中一年级` 是同一年级的另一种写法，归一后才能当过滤值用
        assert _grade(P("datasets/x/初中/数学/初三/教材.pdf")) == "九年级"
        assert _grade(P("datasets/x/高中/物理/高中一年级/教材.pdf")) == "高一"

    def test_unrecognized_stage_and_grade_are_none(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _grade, _stage

        # 认不出就是 None，不猜（与 subject 同一口径）
        assert _grade(P("datasets/01_textbook/初中/数学/教材.pdf")) is None
        assert _stage(P("datasets/杂项/资料/教材.pdf")) is None

    def test_grade_distinguishes_titles(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _title

        # 同一学科的七/九两个年级目录下都是 `教材.pdf`：标题不带年级，
        # 引用卡片上就是一排"初中数学教材"，用户分不清拿到的是哪本
        seven = _title(P("datasets/c/初中/数学/七年级/教材.pdf"), "textbook", "数学", "教材")
        nine = _title(P("datasets/c/初中/数学/九年级/教材.pdf"), "textbook", "数学", "教材")
        assert seven == "初中七年级数学教材"
        assert nine == "初中九年级数学教材"
        assert seven != nine

    def test_question_bank_and_review_paper_doc_types(self):
        from pathlib import Path as P

        from rag_system.document_loaders.thy_factory import _doc_type

        root = P("datasets/初高中教材教案试卷-按年级")
        # 认成 other 的话，`KB_STEPWISE` 的 `doc_type in [exam, …]` 过滤永远召不回它们
        assert _doc_type(root / "初中/化学/九年级/题库.json", "other") == "question_bank"
        assert _doc_type(root / "高中/数学/高三/复习卷.html", "other") == "exam"
        assert _doc_type(root / "高中/数学/高三/试卷.pdf", "other") == "exam"


class TestQuestionBankLoader:
    """`题库.json` → 一题一段 Markdown（每道题一个 `###`）。"""

    _ITEMS = [
        {
            "subject": "初中化学",
            "ques_type": "填空题",
            "ques_difficulty": "容易",
            "ques_content": "题目内容: 水的化学式是____。",
            "ques_answer": ["H2O"],
            "ques_analyze": "水由氢氧两种元素组成。",
            "ques_knowledges": ["水的组成"],
        },
        {
            "subject": "初中化学",
            "ques_type": "单选题",
            "ques_difficulty": "一般",
            "ques_content": "下列属于混合物的是（ ）。A.氧气 B.空气",
            "ques_answer": ["B"],
            "ques_analyze": "",
            "ques_knowledges": [],
        },
    ]

    def _write(self, path, payload):
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_renders_one_section_per_question(self, tmp_path):
        from rag_system.document_loaders.loader_question_bank import QuestionBankLoader

        path = self._write(tmp_path / "题库.json", self._ITEMS)
        doc = QuestionBankLoader().load(path)
        assert doc.doc_type == "question_bank"
        assert doc.meta["questions"] == 2
        assert doc.text.count("### 第 ") == 2
        assert "答案：H2O" in doc.text
        assert "解析：水由氢氧两种元素组成。" in doc.text
        assert "知识点：水的组成" in doc.text

    def test_prefix_and_empty_analysis_are_cleaned(self, tmp_path):
        from rag_system.document_loaders.loader_question_bank import QuestionBankLoader

        path = self._write(tmp_path / "题库.json", self._ITEMS)
        text = QuestionBankLoader().load(path).text
        assert "题目内容:" not in text  # 前缀会把 318 道题的题首变成同一个词
        # 空解析不渲染占位，否则"解析：（空）"会进索引
        assert text.count("解析：") == 1

    def test_subject_falls_back_to_json_field(self, tmp_path):
        from rag_system.document_loaders.loader_question_bank import QuestionBankLoader

        # 路径里没有学科目录时（临时文件）用 JSON 自带的 `初中化学`
        doc = QuestionBankLoader().load(self._write(tmp_path / "题库.json", self._ITEMS))
        assert doc.subject == "化学"

    def test_non_bank_json_raises(self, tmp_path):
        from rag_system.document_loaders.loader_question_bank import QuestionBankLoader

        with pytest.raises(LoaderError):
            QuestionBankLoader().load(self._write(tmp_path / "配置.json", {"a": 1}))
        with pytest.raises(LoaderError):
            QuestionBankLoader().load(self._write(tmp_path / "空.json", []))

    def test_past_exam_schema_is_accepted(self, tmp_path):
        """高中题库是另一套字段（高考真题）：`question / answer / analysis / year / source`。

        只认 `ques_content` 的话，26 个高中题库会整体 LoaderError——那是语料里题量最大的一块。
        """
        from rag_system.document_loaders.loader_question_bank import QuestionBankLoader

        items = [
            {
                "year": "2010",
                "category": "（新课标）",
                "question": "1. 已知集合 $A=\\{x \\mid x \\leqslant 2\\}$，则 $A \\cap B=$\nA. $(0,2)$\nB. $[0,2]$",
                "answer": ["D"],
                "analysis": "解：由交集定义可得。",
                "source": "2010-2022_Math_I_MCQs.json",
            }
        ]
        doc = QuestionBankLoader().load(self._write(tmp_path / "题库.json", items))
        assert doc.doc_type == "question_bank"
        assert "### 第 1 题" in doc.text
        assert "年份：2010" in doc.text and "分类：（新课标）" in doc.text
        assert "答案：D" in doc.text
        assert "解析：解：由交集定义可得。" in doc.text
        # 选项行必须留下：答案只有一个字母，删了选项这个字母就无所指
        assert "A. $(0,2)$" in doc.text and "B. $[0,2]$" in doc.text

    def test_partially_broken_bank_still_loads(self, tmp_path):
        """一条空题不该把整份好题挡在库外，但序号要紧凑（不能留 `第 2 题` 的空洞）。"""
        from rag_system.document_loaders.loader_question_bank import QuestionBankLoader

        items = [
            {"question": "第一题题干", "answer": ["A"]},
            {"question": "", "answer": []},
            {"question": "第三题题干", "answer": ["B"]},
        ]
        doc = QuestionBankLoader().load(self._write(tmp_path / "题库.json", items))
        assert doc.meta["questions"] == 3
        assert doc.meta["rendered"] == 2
        assert "### 第 2 题\n第三题题干" in doc.text

    def test_splitter_keeps_questions_apart(self, tmp_path):
        """一道题一个父块：否则"引用到第 2 题"会连带第 1 题一起进上下文。"""
        from rag_system.document_loaders.loader_question_bank import QuestionBankLoader
        from rag_system.text_splitters.parent_child import ParentChildSplitter

        path = self._write(tmp_path / "题库.json", self._ITEMS)
        chunks = ParentChildSplitter().split(QuestionBankLoader().load(path))
        parents: dict[str, str] = {}
        for c in chunks:
            parents[c.parent_id] = c.parent_text
        first = [p for p in parents.values() if "水的化学式" in p]
        assert len(first) == 1
        assert "混合物" not in first[0]  # 第 2 题没有并进第 1 题的父块
        assert all(c.meta["doc_type"] == "question_bank" for c in chunks)
