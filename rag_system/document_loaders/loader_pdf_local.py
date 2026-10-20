"""本地 PDF 加载器（PyMuPDF + 大图 OCR）。

它是 MinerU 的**降级路径**，不是替代：
- MinerU 出 Markdown（标题/表格/公式都在），质量上限高，但要联网、按页计费；
- 本地路径断网可用、零成本，代价是双栏/扫描件容易串行、公式变乱码。

选用规则（见 `thy_factory.py`）：`PDF_PARSER=auto` 时优先 MinerU，token 缺失或调用失败
自动退回本地，并在日志里记 `pdf_parser_fallback`——**不静默**。

和旧项目 `Ragpro/rag_system/document_loaders/loader_pdf.py` 的差别：

1. 用 PIL 做 90° 整倍旋转，不再依赖 OpenCV（少一个 60 MB 的依赖）；
2. OCR 只在"图占页面 ≥10% 且宽高 ≥64pt"时才做，避免把页眉图标也送进识别；
3. 每页文本带上页码标记 `<!-- page: n -->`，切分时能定位引用页；
4. 一页抽不出文字（真·扫描页）时整页 OCR，而不是直接丢掉这一页。
"""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError, guess_subject
from rag_system.document_loaders.ocr import ocr_array, ocr_available

_SUBJECTS = ("语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治")

IMAGE_AREA_RATIO = 0.10  # 图占页面面积低于 10% 视为装饰图
IMAGE_MIN_SIDE = 64  # PDF 点（1/72 inch）
PAGE_TEXT_MIN = 40  # 一页正文少于这么多字，判定为扫描页


def _open_pdf(path: Path):  # noqa: ANN202
    try:
        import pymupdf  # PyMuPDF >= 1.24 的新包名
    except ImportError:  # pragma: no cover - 老版本只有 fitz
        try:
            import fitz as pymupdf  # type: ignore[no-redef]
        except ImportError as exc:
            raise LoaderError("PyMuPDF 未安装，无法解析 PDF（pip install pymupdf）") from exc
    try:
        return pymupdf.open(str(path))
    except Exception as exc:
        raise LoaderError(f"打开 PDF 失败：{path.name}：{type(exc).__name__}: {exc}") from exc


def _render_page(page, pymupdf) -> bytes:  # noqa: ANN001
    """整页渲染成 PNG（扫描页 OCR 用，150 DPI 是质量/速度的平衡点）。"""
    pix = page.get_pixmap(dpi=150)
    return pix.tobytes("png")


class LocalPdfLoader(BaseDocumentLoader):
    """`.pdf` → 逐页文本 + 大图/扫描页 OCR。"""

    extensions = (".pdf",)

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path).resolve()
        doc = _open_pdf(p)
        import pymupdf

        parts: list[str] = []
        ocr_pages: list[int] = []
        try:
            for number, page in enumerate(doc, 1):
                text = (page.get_text("text") or "").strip()
                if len(text) >= PAGE_TEXT_MIN:
                    parts.append(f"<!-- page: {number} -->\n{text}")
                    extra = self._ocr_page_images(page, number, doc, pymupdf)
                    if extra:
                        parts.append(extra)
                        ocr_pages.append(number)
                    continue

                # 正文太少：大概率是扫描页，整页 OCR
                if ocr_available():
                    rendered = _render_page(page, pymupdf)
                    page_text = _ocr_png(rendered)
                    if page_text:
                        parts.append(f"<!-- page: {number} -->\n{page_text}")
                        ocr_pages.append(number)
                        continue
                if text:
                    parts.append(f"<!-- page: {number} -->\n{text}")
            total = doc.page_count
        finally:
            doc.close()

        body = "\n\n".join(parts).strip()
        if not body:
            raise LoaderError(f"PDF 没有任何可提取文本（可能需要 OCR）：{p.name}")
        return LoadedDoc(
            text=body,
            source=str(p),
            doc_type="textbook",
            subject=guess_subject(p.name, _SUBJECTS),
            title=p.stem,
            meta={"pages": total, "ocr_pages": ocr_pages, "parser": "pymupdf"},
        )

    def _ocr_page_images(self, page, number: int, doc, pymupdf) -> str:  # noqa: ANN001
        if not ocr_available():
            return ""
        out: list[str] = []
        page_area = page.rect.width * page.rect.height or 1.0
        for info in page.get_image_info(xrefs=True):
            xref = info.get("xref")
            bbox = info.get("bbox") or (0, 0, 0, 0)
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if not xref or width < IMAGE_MIN_SIDE or height < IMAGE_MIN_SIDE:
                continue
            if (width * height) / page_area < IMAGE_AREA_RATIO:
                continue
            try:
                pix = pymupdf.Pixmap(doc, xref)
                # CMYK/带 alpha 的色彩空间 OCR 模型不支持，统一转 RGB
                if pix.n - pix.alpha >= 4:
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                png = pix.tobytes("png")
            except Exception:
                continue
            text = _ocr_png(png, rotation=page.rotation)
            if text:
                out.append(f"<!-- page: {number} image -->\n{text}")
        return "\n\n".join(out)


def _ocr_png(png: bytes, *, rotation: int = 0) -> str:
    """PNG 字节 → RGB ndarray（按需 90° 整倍纠偏）→ OCR 文本。"""
    from PIL import Image

    import numpy as np

    with Image.open(BytesIO(png)) as im:
        rgb = im.convert("RGB")
        if rotation % 360:
            # PyMuPDF 的 rotation 是"页面顺时针转了多少"，OCR 前要转回去
            rgb = rgb.rotate(-rotation, expand=True)
        return ocr_array(np.array(rgb))


_PAGE_MARK = re.compile(r"<!-- page: (\d+)(?: image)? -->")


def split_pages(text: str) -> list[tuple[int, str]]:
    """把带页码标记的 PDF 文本切成 [(页码, 该页正文)]，供引用对齐页号。"""
    chunks: list[tuple[int, str]] = []
    matches = list(_PAGE_MARK.finditer(text))
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            chunks.append((int(match.group(1)), body))
    return chunks
