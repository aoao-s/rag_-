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
