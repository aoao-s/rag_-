"""EduRAG-Pro 应用入口（对标旧项目根目录的 main.py）。

职责：
1. 建 FastAPI 应用并挂载三个模块的路由（当前只有 qa_system，rag_system / 记忆模块按 PR 逐步接入）；
2. 管理进程级依赖容器（MySQL 连接池、Redis、FAQService），请求路径只读；
3. 启动三件事按序执行：校验配置 → 装载语料 → 预热，完成才把 /readyz 置 200；
4. 暴露 /healthz、/readyz、/metrics 三个运维端点。

启动方式：
    python main.py                     # 按 .env 里 HTTP_HOST/HTTP_PORT 起服务
    uvicorn main:app --reload          # 开发热重载
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from base.config import Settings, get_settings
from base.db import MySQL
from base.logger import bind, clear, configure, get_logger, new_request_id
from qa_system.api import router as faq_router
from qa_system.res.qa_cache import create_redis
from qa_system.res.qa_csv import load_csv_records
from qa_system.res.qa_db import (
    FAQRepository,
    InMemoryFAQRepository,
    MySQLFAQRepository,
)
from qa_system.res.qa_search import FAQService

log = get_logger(__name__)

STARTED_AT = time.time()


class Container:
    """进程级依赖容器：只有这里允许创建连接池/客户端，其余模块从 app.state 取。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mysql: MySQL | None = None
        self.redis: Any = None
        self.faq_repository: FAQRepository | None = None
        self.faq_service: FAQService | None = None
        self.degraded = False

    def build(self) -> None:
        s = self.settings
        self.mysql = MySQL(s)
        try:
            self.redis = create_redis(s)
        except Exception as exc:  # Redis 只用于缓存/限流，不可用时降级而不是拒服务
            log.warning("redis_unavailable", error=str(exc))
            self.redis = None

    async def load_faq(self) -> None:
        s = self.settings

        # 无库模式：显式配置 FAQ_CORPUS_CSV 时从 CSV 装载，写接口仍落在内存仓库
        # （用于本地演示、CI 与没有 docker 的开发机；生产必须留空，MySQL 才是真相源）
        if s.faq_corpus_csv:
            from pathlib import Path

            csv_path = Path(s.faq_corpus_csv)
            if not csv_path.is_absolute():
                csv_path = Path(__file__).resolve().parent / csv_path
            records = load_csv_records(csv_path)
            repository: FAQRepository = InMemoryFAQRepository(records)
            service = FAQService(
                repository,
                policy=_policy_from(s),
                alias_enabled=s.faq_alias_enabled,
                top_k=s.faq_top_k,
                extra_words=s.subjects,
            )
            info = await service.load()
            await service.warmup()
            self.faq_repository = repository
            self.faq_service = service
            log.warning("faq_corpus_from_csv", path=str(csv_path), docs=info.get("docs"))
            log.info("faq_ready", degraded=self.degraded, **info)
            return

        assert self.mysql is not None
        repository = MySQLFAQRepository(self.mysql)
        service = FAQService(
            repository,
            policy=_policy_from(self.settings),
            alias_enabled=self.settings.faq_alias_enabled,
            top_k=self.settings.faq_top_k,
            extra_words=self.settings.subjects,
        )
        try:
            info = await service.load()
        except Exception as exc:
            if not self.settings.allow_degraded_start:
                raise
            # 明确打日志并标记 degraded：readyz 仍 200，但 /v1/faq/stats 里能看到空语料
            log.error("faq_load_failed_degraded", error=str(exc))
            self.degraded = True
            repository = InMemoryFAQRepository()
            service = FAQService(
                repository,
                policy=_policy_from(self.settings),
                alias_enabled=self.settings.faq_alias_enabled,
                top_k=self.settings.faq_top_k,
                extra_words=self.settings.subjects,
            )
            info = await service.load()
        await service.warmup()
        self.faq_repository = repository
        self.faq_service = service
        log.info("faq_ready", degraded=self.degraded, **info)

    def close(self) -> None:
        if self.redis is not None:
            try:
                self.redis.close()
            except Exception:  # noqa: S110 - 关闭失败不影响退出
                pass
        if self.mysql is not None:
            self.mysql.close()


def _policy_from(settings: Settings):
    from qa_system.res.qa_matcher import MatchPolicy

    return MatchPolicy(
        threshold=settings.faq_threshold,
        margin=settings.faq_margin,
        candidate_n=settings.faq_candidate_n,
        norm_k=settings.faq_norm_k,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure(settings.log_level)

    problems = settings.self_check()
    if problems:
        for problem in problems:
            log.error("config_problem", detail=problem)
        if not settings.allow_degraded_start:
            raise RuntimeError("配置校验失败：" + "；".join(problems))

    container = Container(settings)
    container.build()
    await container.load_faq()

    app.state.container = container
    app.state.faq_service = container.faq_service
    app.state.ready = True
    log.info("app_ready", version=settings.app_version, env=settings.app_env)
    try:
        yield
    finally:
        app.state.ready = False
        container.close()
        log.info("app_stopped")


app = FastAPI(
    title="EduRAG-Pro 智能教育知识库系统",
    version="0.1.0",
    description="qa_system / rag_system / 记忆模块 三模块之一：FAQ 精确检索（PR2）",
    lifespan=lifespan,
)
app.include_router(faq_router)


@app.middleware("http")
async def trace_middleware(request: Request, call_next: Any) -> Response:
    """给每个请求绑定 request_id，异常统一转 500 并保留 trace 便于排查。"""
    clear()
    request_id = request.headers.get("X-Request-Id") or new_request_id()
    bind(request_id=request_id, path=request.url.path)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        log.error("unhandled_error", error=str(exc), exc_info=True)
        clear()
        return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": request_id})
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        log.info("http_request", method=request.method, status=response.status_code, latency_ms=round(elapsed_ms, 2))
    response.headers["X-Request-Id"] = request_id
    clear()
    return response


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict[str, str]:
    return {"status": "ok", "uptime_s": f"{time.time() - STARTED_AT:.0f}"}


@app.get("/readyz", include_in_schema=False)
async def readyz(request: Request) -> Response:
    container: Container | None = getattr(request.app.state, "container", None)
    ready = bool(getattr(request.app.state, "ready", False)) and container is not None
    body = {
        "status": "ready" if ready else "not_ready",
        "degraded": bool(container.degraded) if container else True,
        "faq_docs": container.faq_service.state()["docs"] if container and container.faq_service else 0,
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.get("/metrics", include_in_schema=False)
async def metrics(request: Request) -> Response:
    """极简 Prometheus 文本：FAQ 命中率、延迟、语料版本。够 Grafana 出图即可。"""
    container: Container | None = getattr(request.app.state, "container", None)
    service = container.faq_service if container else None
    lines: list[str] = []
    if service is not None:
        state, stats = service.state(), service.stats()
        lines += [
            "# HELP edurag_faq_docs 已装载 FAQ 条数",
            "# TYPE edurag_faq_docs gauge",
            f"edurag_faq_docs {state['docs']}",
            "# HELP edurag_faq_version 语料版本号",
            "# TYPE edurag_faq_version gauge",
            f"edurag_faq_version {state['version']}",
            "# HELP edurag_faq_searches_total 查询总数",
            "# TYPE edurag_faq_searches_total counter",
            f"edurag_faq_searches_total {stats['searches']}",
            "# HELP edurag_faq_hit_rate 命中率",
            "# TYPE edurag_faq_hit_rate gauge",
            f"edurag_faq_hit_rate {stats['hit_rate']}",
            "# HELP edurag_faq_latency_ms_avg 平均延迟(ms)",
            "# TYPE edurag_faq_latency_ms_avg gauge",
            f"edurag_faq_latency_ms_avg {stats['avg_latency_ms']}",
        ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


def run() -> None:
    """命令行入口：按 .env 的 HTTP_HOST/HTTP_PORT 起服务（等价于旧项目的 python main.py）。"""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "main:app",
        host=settings.http_host,
        port=settings.http_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()

