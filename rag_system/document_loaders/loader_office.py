"""Office 文档加载器：.docx / .pptx。

与旧项目（`Ragpro/rag_system/document_loaders/loader_docx.py`、`loader_pptx.py`）的差别：

1. **按阅读顺序出来结构**，而不是把段落和表格拍平成一串文本：
   docx 里段落与表格是交错的块级元素，pptx 里形状按 (top, left) 排序；
   拍平之后"表格里那一行到底属于哪个小节"就丢了，切分器没法按标题归类。
2. **表格转 Markdown 表格**（旧项目只把单元格文本一条条塞进列表）：
   行列关系是重要语义，教材/教案的对照表拍平后基本不可读。
3. **内嵌图 OCR 默认关闭**（`OCR_IMAGES=false`）：OCR 慢且贵，
   多数文档正文已含图片里的信息；需要时显式开启。
4. 不用 tqdm、不用 langchain，失败直接抛 `LoaderError`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError, guess_subject
from rag_system.document_loaders.ocr import ocr_image_bytes



def _table_to_markdown(rows: list[list[str]]) -> str:
    """二维文本 → Markdown 表格；空行/空列裁掉，超过 30 列时退化为制表符分隔。"""
    grid = [[(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    grid = [row for row in grid if any(cell for cell in row)]
    if not grid:
        return ""
    width = max(len(row) for row in grid)
    grid = [row + [""] * (width - len(row)) for row in grid]
    if width > 30:  # 极端宽的表格，Markdown 反而更难读
        return "\n".join("\t".join(row) for row in grid)
    header, *body = grid
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


class DocxLoader(BaseDocumentLoader):
    """`.docx`：段落（带标题样式层级）+ 表格（转 Markdown 表格）。"""

    extensions = (".docx",)

    def __init__(self, *, ocr_images: bool = False) -> None:
        self._ocr_images = ocr_images

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path).resolve()
        try:
            import docx  # python-docx
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:  # pragma: no cover
            raise LoaderError("python-docx 未安装，无法解析 .docx") from exc

        try:
            document = docx.Document(str(p))
        except Exception as exc:
            raise LoaderError(f"打开 docx 失败：{p.name}：{type(exc).__name__}: {exc}") from exc

        parts: list[str] = []
        for block in _iter_docx_blocks(document, Paragraph, Table):
            if isinstance(block, Paragraph):
                text = block.text.strip()
                if not text:
                    continue
                style = (block.style.name or "").lower() if block.style is not None else ""
                level = _heading_level(style)
                # 标题样式 → Markdown 标题，切分器才能按小节聚合
                parts.append(f"{'#' * level} {text}" if level else text)
                if self._ocr_images:
                    parts.extend(_ocr_paragraph_images(document, block))
            else:
                rows = [[cell.text for cell in row.cells] for row in block.rows]
                table = _table_to_markdown(rows)
                if table:
                    parts.append(table)

        text = "\n\n".join(part for part in parts if part.strip())
        if not text.strip():
            raise LoaderError(f"docx 没有可提取的文本：{p.name}")
        return LoadedDoc(
            text=text,
            source=str(p),
            doc_type="lesson_plan",
            subject=guess_subject(str(p)),
            title=p.stem,
            meta={"blocks": len(parts), "ocr_images": self._ocr_images},
        )


class PptxLoader(BaseDocumentLoader):
    """`.pptx`：按幻灯片顺序，形状按 (top, left) 排阅读顺序。"""

    extensions = (".pptx",)

    def __init__(self, *, ocr_images: bool = False) -> None:
        self._ocr_images = ocr_images

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path).resolve()
        try:
            from pptx import Presentation
            from pptx.enum.shapes import MSO_SHAPE_TYPE
        except ImportError as exc:  # pragma: no cover
            raise LoaderError("python-pptx 未安装，无法解析 .pptx") from exc

        try:
            presentation = Presentation(str(p))
        except Exception as exc:
            raise LoaderError(f"打开 pptx 失败：{p.name}：{type(exc).__name__}: {exc}") from exc

        parts: list[str] = []
        for index, slide in enumerate(presentation.slides, 1):
            title = _slide_title(slide)
            parts.append(f"## 第 {index} 页" + (f"：{title}" if title else ""))
            shapes = sorted(slide.shapes, key=lambda s: ((s.top or 0), (s.left or 0)))
            for shape in shapes:
                parts.extend(self._shape_text(shape, MSO_SHAPE_TYPE))
        text = "\n\n".join(part for part in parts if part and part.strip())
        if not text.strip():
            raise LoaderError(f"pptx 没有可提取的文本：{p.name}")
        return LoadedDoc(
            text=text,
            source=str(p),
            doc_type="lesson_plan",
            subject=guess_subject(str(p)),
            title=p.stem,
            meta={"slides": len(presentation.slides), "ocr_images": self._ocr_images},
        )

    def _shape_text(self, shape, mso) -> list[str]:  # noqa: ANN001 - python-pptx 类型不导出
        out: list[str] = []
        if shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if text:
                out.append(text)
            return out
        if getattr(shape, "has_table", False):
            rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
            table = _table_to_markdown(rows)
            if table:
                out.append(table)
            return out
        if shape.shape_type == mso.PICTURE and self._ocr_images:
            try:
                text = ocr_image_bytes(shape.image.blob)
            except Exception:
                text = ""
            if text:
                out.append(text)
            return out
        if shape.shape_type == mso.GROUP:
            for child in getattr(shape, "shapes", []):
                out.extend(self._shape_text(child, mso))
        return out


def _iter_docx_blocks(document, paragraph_cls, table_cls) -> Iterator[object]:  # noqa: ANN001
    """按文档顺序产出段落与表格（docx 的 body 是 XML 流，直接遍历 paragraphs 会丢交错顺序）。"""
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P

    for child in document.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield paragraph_cls(child, document)
        elif isinstance(child, CT_Tbl):
            yield table_cls(child, document)


def _heading_level(style_name: str) -> int:
    """'Heading 1' / '标题 1' / 'heading 2' → 级别；正文返回 0。"""
    name = style_name.strip().lower()
    if not name.startswith(("heading", "标题")):
        return 0
    digits = "".join(ch for ch in name if ch.isdigit())
    return max(1, min(6, int(digits))) if digits else 1


def _slide_title(slide) -> str:  # noqa: ANN001
    try:
        if slide.shapes.title is not None and slide.shapes.title.text_frame is not None:
            return slide.shapes.title.text_frame.text.strip()
    except Exception:
        pass
    return ""


def _ocr_paragraph_images(document, paragraph) -> list[str]:  # noqa: ANN001
    """段落里的内嵌图 → OCR 文本（旧项目同款做法，但默认关闭）。"""
    out: list[str] = []
    try:
        parts = document.part.related_parts
        for blip in paragraph._element.xpath(".//a:blip/@r:embed"):
            part = parts.get(blip)
            blob = getattr(part, "blob", None)
            if blob:
                text = ocr_image_bytes(blob)
                if text:
                    out.append(text)
    except Exception:
        return out
    return out
