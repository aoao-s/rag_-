"""图片加载器：.jpg/.png/.bmp/.webp → OCR 文本。

适用场景：教材截图、板书照片、试卷拍照。这类文件的"正文"只存在于像素里。
与旧项目 `Ragpro/rag_system/document_loaders/loader_image.py` 的差别：
不再返回空字符串假装成功——OCR 不可用或识别为空都抛 `LoaderError`，
入库报告里能看到"为什么这份没进来"。
"""

from __future__ import annotations

from pathlib import Path

from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError, guess_subject
from rag_system.document_loaders.ocr import ocr_available, ocr_image_bytes



class ImageLoader(BaseDocumentLoader):
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path).resolve()
        if not ocr_available():
            raise LoaderError("rapidocr 未安装，无法解析图片（pip install rapidocr-onnxruntime）")
        try:
            data = p.read_bytes()
        except OSError as exc:
            raise LoaderError(f"读取图片失败：{p.name}：{exc}") from exc
        text = ocr_image_bytes(data)
        if not text.strip():
            raise LoaderError(f"OCR 没有识别出文字：{p.name}")
        return LoadedDoc(
            text=text,
            source=str(p),
            doc_type="image",
            subject=guess_subject(str(p)),
            title=p.stem,
            meta={"bytes": len(data)},
        )
