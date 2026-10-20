"""base：跨模块公共底座（配置 / 日志 / 标签 / 契约 / MySQL 连接池）。

只放"谁都要用、且不含业务语义"的东西；qa_system 与 rag_system 平级引用，
禁止 base 反向 import 它们。
"""

from base.config import Settings, get_settings
from base.contracts import Chunk, Evidence, FAQHit, MemoryContext
from base.db import MySQL
from base.labels import INTENT_FALLBACK, INTENT_LABELS, SUBJECTS, normalize_subject
from base.logger import bind, clear, configure, current, get_logger, new_request_id

__all__ = [
    "Chunk",
    "Evidence",
    "FAQHit",
    "INTENT_FALLBACK",
    "INTENT_LABELS",
    "MySQL",
    "MemoryContext",
    "SUBJECTS",
    "Settings",
    "bind",
    "clear",
    "configure",
    "current",
    "get_logger",
    "get_settings",
    "new_request_id",
    "normalize_subject",
]
