"""结构化日志与 trace 上下文（全局唯一日志出口）。

CLAUDE.md §5.2 的落地方式：
- 一份日志两处去处：stdout（容器采集）+ `logs/app.log`（轮转，故障复盘）；
- 每条日志字段化（JSON），带 `request_id`/`node`/`event`，禁止 `print` 与字符串拼接；
- 业务日志与 uvicorn/第三方库日志走同一套 handler，否则排查启动失败时只能看到半截。

实现要点：structlog 通过 `stdlib.LoggerFactory` 落到标准库 logging，
再由 handler 上的 `ProcessorFormatter` 统一渲染 JSON。这样"加文件 handler"这件事
对两边同时生效，不存在 structlog 写 stdout、stdlib 写文件的裂缝。
"""

from __future__ import annotations

import contextvars
import json
import logging
import logging.handlers
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:  # pragma: no cover - 依赖存在时走 structlog
    import structlog

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover
    _HAS_STRUCTLOG = False

_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("edurag_log_ctx", default={})
_configured = False
_file_handler: logging.Handler | None = None


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


class _JsonFormatter(logging.Formatter):
    """没有 structlog 时的兜底 JSON 渲染（字段与 structlog 输出保持一致）。"""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
            + f".{int(record.msecs):03d}",
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload.update(current())
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _build_formatter() -> logging.Formatter:
    if not _HAS_STRUCTLOG:
        return _JsonFormatter()
    return structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(ensure_ascii=False),
        foreign_pre_chain=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ],
    )


def configure(
    level: str = "INFO",
    *,
    log_path: Path | None = None,
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 10,
) -> None:
    """初始化日志。

    可重复调用，且**允许后补文件路径**：模块导入期的 `get_logger()` 会先用默认参数初始化一次，
    lifespan 里拿到配置后再补 `log_path`——早期版本在这里直接 return，导致 `logs/app.log`
    永远不生成（现象是"日志配置了但文件是空的"）。
    """
    global _configured, _file_handler

    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(lvl)

    if not _configured:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(_build_formatter())
        root.addHandler(console)
        _configured = True

    if log_path is not None and _file_handler is None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            handler.setFormatter(_build_formatter())
            root.addHandler(handler)
            _file_handler = handler
        except Exception as exc:  # 日志目录只读时宁可没文件，也不能起不来服务
            print(f"[logger] 无法写入日志文件 {log_path}: {exc}", file=sys.stderr)

    # 第三方库降噪：jieba 的 DEBUG 会把分词词典刷满日志
    for noisy in ("jieba", "urllib3", "pymysql", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=False),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                # 交给 handler 上的 ProcessorFormatter 渲染，业务日志与第三方日志共用一套出口
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(lvl),
            cache_logger_on_first_use=True,
        )


class _StdlibAdapter:
    """把 structlog 风格调用（`log.info("event", key=value)`）适配到标准库。

    没有 structlog 的环境里，如果直接返回 `logging.Logger`，
    `log.warning("mysql_ping_failed", error=str(exc))` 会抛
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
        if args:
            event = event % args
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
    return _StdlibAdapter(logging.getLogger(name))
