"""日志：一份人可读的行格式，两处去处（控制台 + 文件）。

目标格式（与旧项目一致，便于沿用既有排查习惯）：

    2026-10-20 19:20:11.302 [INFO] [35856:MainThread] EduRAG.qa_system.res.qa_search (qa_search.py:128): 检索命中 [来源: Redis 缓存] ...

设计要点：
- **文件与控制台同一格式**，只是级别不同：控制台 `LOG_LEVEL`（默认 INFO，看结论），
  文件 `LOG_FILE_LEVEL`（默认 DEBUG，留现场：缓存 key、SQL、续命等）。
- structlog 经 `stdlib.LoggerFactory` 落到标准库 logging，handler 上的 `ProcessorFormatter`
  统一渲染；uvicorn/第三方库日志因此走同一出口，不会出现"业务有日志、访问日志没有"。
- 事件名 → 中文文案由 `base/log_messages.py` 负责，调用侧只传结构化字段。
- `configure()` 可重复调用且允许后补文件路径（导入期先建 console，lifespan 再补文件 handler）。
"""

from __future__ import annotations

import contextvars
import logging
import logging.handlers
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from base import log_messages

try:  # pragma: no cover - 依赖存在时走 structlog
    import structlog

    _HAS_STRUCTLOG = True
except Exception:  # pragma: no cover
    _HAS_STRUCTLOG = False

_ctx: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("edurag_log_ctx", default={})
_configured = False
_file_handler: logging.Handler | None = None
LOG_PREFIX = "EduRAG"

LINE_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] [%(process)d:%(threadName)s] %(name)s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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


def _render_message(event: str, fields: dict[str, Any]) -> str:
    """事件 + 字段 → 一行中文；带上 `(file:line)` 方便直接跳代码。"""
    filename = fields.pop("filename", None)
    lineno = fields.pop("lineno", None)
    text = log_messages.render(event, fields)
    if filename:
        return f"({filename}:{lineno}): {text}"
    return text


def _console_stream() -> Any:
    """控制台流：英文控制台（GBK）遇到 `m/s²` 这类字符会抛 UnicodeEncodeError。

    logging 自己会吞掉编码异常并打印 "--- Logging error ---"，结果是**那一条日志直接丢失**
    （实测踩过：预热查询带 `s²` 的问法，日志被吞、控制台一片正常）。
    这里把 stdout 的错误策略改成反斜杠转义：见到不支持的字符就打 `\\xb2`，绝不丢整行。
    """
    stream = sys.stdout
    try:
        stream.reconfigure(errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:  # 被替换过的流（pytest capsys、重定向包装）没有 reconfigure
        pass
    return stream


class _LineFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(fmt=LINE_FORMAT, datefmt=DATE_FORMAT)


def _build_formatter() -> logging.Formatter:
    """文件与控制台共用：structlog 事件先渲染成中文，再套固定行格式。"""
    if not _HAS_STRUCTLOG:
        return _LineFormatter()

    # 注意：filename/lineno 不能在这里过滤掉，它们要交给 _render_message 拼成 (file:line)
    structural = {
        "level",
        "timestamp",
        "event",
        "logger",
        "logger_name",
        "func_name",
        "_record",
        "_from_structlog",
    }

    def processor(logger: Any, name: str, event_dict: dict[str, Any]) -> str:
        event = str(event_dict.pop("event", ""))
        for key in ("level", "timestamp", "logger", "logger_name", "_record", "_from_structlog"):
            event_dict.pop(key, None)
        fields = {k: v for k, v in event_dict.items() if k not in structural}
        return _render_message(event, fields)

    return structlog.stdlib.ProcessorFormatter(
        processor=processor,
        # 行格式必须显式传给 ProcessorFormatter：它默认 fmt="%(message)s"，
        # 不传就会得到"只有中文文案、没有时间/级别/进程/来源"的裸消息。
        fmt=LINE_FORMAT,
        datefmt=DATE_FORMAT,
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
    console_level: str | None = None,
    file_level: str = "DEBUG",
) -> None:
    """初始化日志。

    可重复调用，且**允许后补文件路径**：导入期 `get_logger()` 会先用默认参数初始化一次，
    lifespan 拿到配置后再补 `log_path`——早期版本在这里直接 return，导致 `logs/app.log`
    永远不生成（现象是"日志配置了但文件是空的"）。
    """
    global _configured, _file_handler

    lvl = getattr(logging, level.upper(), logging.INFO)
    console_lvl = getattr(logging, (console_level or level).upper(), lvl)
    root = logging.getLogger()
    root.setLevel(min(lvl, getattr(logging, file_level.upper(), logging.DEBUG)))

    if not _configured:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        console = logging.StreamHandler(_console_stream())
        console.setLevel(console_lvl)
        console.setFormatter(_build_formatter())
        root.addHandler(console)
        _configured = True
    else:
        for handler in list(root.handlers):
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(console_lvl)

    if log_path is not None and _file_handler is None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
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
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=False),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                # 行号取调用点，而不是 structlog 内部帧
                structlog.processors.CallsiteParameterAdder(
                    parameters=[
                        structlog.processors.CallsiteParameter.FILENAME,
                        structlog.processors.CallsiteParameter.LINENO,
                    ]
                ),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.make_filtering_bound_logger(min(lvl, logging.DEBUG)),
            cache_logger_on_first_use=True,
        )


class _StdlibAdapter:
    """把 structlog 风格调用（`log.info("event", key=value)`）适配到标准库。

    没有 structlog 的环境里，若直接返回 `logging.Logger`，
    `log.warning("mysql_ping_failed", error=str(exc))` 会抛
    `TypeError: Logger._log() got an unexpected keyword argument 'error'`——
    而且是在异常处理分支里二次崩溃，掩盖真正的错误（如 1045 密码错误）。
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, *args: Any, **kwargs: Any) -> None:
        exc_info = kwargs.pop("exc_info", None)
        try:
            message = log_messages.render(event, dict(kwargs))
        except Exception:  # pragma: no cover - 日志绝不能反向影响业务
            message = f"{event} {kwargs}"
        self._logger.log(level, message, exc_info=exc_info)

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
    # 只在首次初始化时配：`configure()` 每次调用都会按传入级别重设 handler，
    # 而这里的默认参数是 INFO——无条件调用会把 lifespan / 脚本设好的级别悄悄改回去。
    if not _configured:
        configure()
    if _HAS_STRUCTLOG:
        # 统一前缀，日志里直接能看出这是本项目的 logger
        return structlog.get_logger(f"{LOG_PREFIX}.{name}")
    return _StdlibAdapter(logging.getLogger(f"{LOG_PREFIX}.{name}"))
