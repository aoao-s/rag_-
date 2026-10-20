"""OCR 引擎：进程内单例，优先 GPU 版 rapidocr_paddle，缺了退 onnxruntime。

为什么单例：模型权重几百 MB，每份文档建一次会 OOM，且冷启动 2–5 s。
为什么不在请求路径调用：OCR 只在**入库期**跑（Office 内嵌图、扫描 PDF、图片文件），
在线查询永远不碰它。
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from base.logger import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_engine: Any = None
_unavailable: str | None = None


def _build() -> Any:
    try:
        from rapidocr_paddle import RapidOCR  # 有 GPU 时更快

        log.info("ocr_engine_ready", engine="rapidocr_paddle")
        return RapidOCR()
    except Exception:
        pass
    try:
        from rapidocr_onnxruntime import RapidOCR  # CPU 兜底

        log.info("ocr_engine_ready", engine="rapidocr_onnxruntime")
        return RapidOCR()
    except Exception as exc:  # pragma: no cover - 取决于环境
        log.warning("ocr_engine_missing", error=f"{type(exc).__name__}: {exc}")
        return None


def get_ocr() -> Any:
    """返回 OCR 引擎；不可用时返回 None（调用方自己决定降级还是报错）。"""
    global _engine, _unavailable
    if _engine is not None:
        return _engine
    if _unavailable is not None:
        return None
    with _lock:
        if _engine is None and _unavailable is None:
            _engine = _build()
            if _engine is None:
                _unavailable = "rapidocr 未安装（pip install rapidocr-onnxruntime）"
    return _engine


def ocr_available() -> bool:
    return get_ocr() is not None


def ocr_array(image: np.ndarray) -> str:
    """对 RGB ndarray 做 OCR，返回按行拼接的文本（失败返回空串，不抛）。"""
    engine = get_ocr()
    if engine is None:
        return ""
    try:
        result, _ = engine(image)
    except Exception as exc:  # pragma: no cover - 引擎内部异常
        log.warning("ocr_failed", error=f"{type(exc).__name__}: {exc}")
        return ""
    if not result:
        return ""
    lines = [str(item[1]).strip() for item in result if len(item) > 1 and str(item[1]).strip()]
    return "\n".join(lines)


def ocr_image_bytes(data: bytes, *, max_side: int = 2000, min_side: int = 600) -> str:
    """图片字节 → OCR 文本。

    两个尺寸处理都是为准确率：
    - `max_side`：超大图先等比缩小，否则检测模型会跑很久（教材插图常见 4000px）；
    - `min_side`：过小的图先放大（默认只在最长边 < 600px 时触发）。放大能救回笔画，
      但放太大反而让 `^` 被识别成 `~`，所以上限 2 倍。
    """
    from io import BytesIO

    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as im:
            rgb = im.convert("RGB")
            rgb = _rescale(rgb, max_side=max_side, min_side=min_side)
            return ocr_array(np.array(rgb))
    except Exception as exc:  # pragma: no cover - 损坏图片
        log.warning("ocr_image_failed", error=f"{type(exc).__name__}: {exc}")
        return ""


def _rescale(image, *, max_side: int, min_side: int):  # noqa: ANN001, ANN202
    """按最长边把图片缩到 [min_side, max_side] 区间内（只做整数倍或等比缩放）。"""
    w, h = image.size
    longest = max(w, h)
    if longest <= 0:
        return image
    if longest > max_side:
        scale = max_side / longest
    elif longest < min_side:
        scale = min(2.0, min_side / longest)  # 最多放大 2 倍，放太大反而变糊
    else:
        return image
    from PIL import Image

    target = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(target, Image.LANCZOS)

