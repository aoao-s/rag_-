"""结构化日志与 trace 上下文。

字段固定：trace_id / request_id / node / latency_ms / extra（CLAUDE.md §5.2）。
没有 structlog 时自动退化为标准库 JSON，保证脚本环境也能跑。
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any

try:  # pragma: no cover - 依赖存在时走 structlog
    import structlog

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover
    _HAS_STRUCTLOG = False

_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("edurag_log_ctx", default={})
_configured = False


def bind(**kwargs: Any) -> None:
    """把字段绑定到当前请求上下文（自动出现在后续所有日志里）。"""
    current = dict(_ctx.get())
    current.update({k: v for k, v in kwargs.items() if v is not None})
    _ctx.set(current)


def current() -> dict[str, Any]:
    return dict(_ctx.get())


def clear() -> None:
    _ctx.set({})


def new_request_id() -> str:
    return uuid.uuid4().hex


def configure(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    logging.basicConfig(stream=sys.stdout, level=getattr(logging, level.upper(), logging.INFO))
    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(ensure_ascii=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
            cache_logger_on_first_use=True,
        )
    _configured = True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(current())
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class _StdlibAdapter:
    """把 structlog 风格调用（`log.info("event", key=value)`）适配到标准库。

    没有 structlog 的环境（精简虚拟环境、只装了运行依赖的机器）里，如果直接返回
    `logging.Logger`，`log.warning("mysql_ping_failed", error=str(exc))` 会抛
    `TypeError: Logger._log() got an unexpected keyword argument 'error'`——
    而且是在异常处理分支里二次崩溃，掩盖真正的错误（如 1045 密码错误）。
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    @staticmethod
    def _render(event: str, kwargs: dict[str, Any]) -> str:
        if not kwargs:
            return str(event)
        extras = " ".join(f"{k}={v}" for k, v in kwargs.items() if k != "exc_info")
        return f"{event} {extras}" if extras else str(event)

    def _log(self, level: int, event: str, *args: Any, **kwargs: Any) -> None:
        exc_info = kwargs.pop("exc_info", None)
        if args:  # 兼容 printf 风格
            event = event % args if args else event
        self._logger.log(level, self._render(event, kwargs), exc_info=exc_info)

    def debug(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, *args, **kwargs)

    def info(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, event, *args, **kwargs)

    def warning(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, *args, **kwargs)

    warn = warning

    def error(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, *args, **kwargs)

    def exception(self, event: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self._log(logging.ERROR, event, *args, **kwargs)

    def critical(self, event: str, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, *args, **kwargs)


def get_logger(name: str) -> Any:
    configure()
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return _StdlibAdapter(logger)
