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
from rag_system.api import router as rag_router
from rag_system.core.evidence_gate import evaluate as gate_evaluate
from rag_system.core.retriever import Retriever
from qa_system.res.qa_cache import (
    BaseFAQCache,
    InMemoryFAQCache,
    RedisFAQCache,
    create_redis,
)
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
        self.faq_cache: BaseFAQCache | None = None
        self.faq_repository: FAQRepository | None = None
        self.faq_service: FAQService | None = None
        self.rag: dict[str, Any] | None = None
        self.degraded = False

    def build(self) -> None:
        s = self.settings
        self.mysql = MySQL(s)
        try:
            self.redis = create_redis(s)
            self.redis.ping()  # 早失败：省得等到第一个请求才在日志里发现连不上
            log.info("redis_ready", host=s.redis_host, port=s.redis_port, db=s.redis_db)
        except Exception as exc:  # Redis 只用于缓存/限流，不可用时降级而不是拒服务
            log.warning("redis_unavailable", error=str(exc))
            self.redis = None

        if s.faq_cache_enabled and self.redis is not None:
            self.faq_cache = RedisFAQCache(
                self.redis,
                max_ttl=s.faq_cache_max_ttl,
                slide_ratio=s.faq_cache_slide_ratio,
            )
            log.info(
                "faq_cache_ready",
                ttl=s.faq_cache_ttl,
                max_ttl=s.faq_cache_max_ttl,
                negative_ttl=s.faq_cache_negative_ttl,
                slide_ratio=s.faq_cache_slide_ratio,
            )
        elif s.faq_cache_enabled:
            # Redis 不可用时退化为进程内缓存：单副本仍有收益，多副本各自为政但结果一致
            # （命中内容由语料版本决定，不依赖副本间同步）
            self.faq_cache = InMemoryFAQCache(max_ttl=s.faq_cache_max_ttl)
            log.warning("faq_cache_disabled", reason="redis_unavailable", fallback="in_memory")
        else:
            log.info("faq_cache_disabled", reason="config")

    def _make_service(self, repository: FAQRepository) -> FAQService:
        """FAQService 的唯一构造点：缓存、策略、日志开关只在这里接线。"""
        s = self.settings
        return FAQService(
            repository,
            policy=_policy_from(s),
            alias_enabled=s.faq_alias_enabled,
            top_k=s.faq_top_k,
            extra_words=s.subjects,
            cache=self.faq_cache,
            cache_ttl=s.faq_cache_ttl,
            cache_negative_ttl=s.faq_cache_negative_ttl,
            cache_prefix=f"edurag:{s.app_env}",
            log_query=s.log_query_text,
        )

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
            service = self._make_service(repository)
            info = await service.load()
            await service.warmup()
            self.faq_repository = repository
            self.faq_service = service
            log.warning("faq_corpus_from_csv", path=str(csv_path), docs=info.get("docs"))
            log.info("faq_ready", degraded=self.degraded, **info)
            return

        assert self.mysql is not None
        repository = MySQLFAQRepository(self.mysql)
        service = self._make_service(repository)
        try:
            info = await service.load()
        except Exception as exc:
            if not self.settings.allow_degraded_start:
                raise
            # 明确打日志并标记 degraded：readyz 仍 200，但 /v1/faq/stats 里能看到空语料
            log.error("faq_load_failed_degraded", error=str(exc))
            self.degraded = True
            repository = InMemoryFAQRepository()
            service = self._make_service(repository)
            info = await service.load()
        await service.warmup()
        self.faq_repository = repository
        self.faq_service = service
        log.info("faq_ready", degraded=self.degraded, **info)

    async def load_rag(self) -> None:
        """启动期装载检索链：模型 + Milvus 句柄一次建好，请求路径零加载。

        RAG 起不来不应该拖垮 FAQ（两者是独立入口），所以这里失败只标记 not ready：
        `/v1/rag/*` 返回 503 并带上原因，FAQ 照常服务。
        """
        s = self.settings
        if not s.rag_enabled:
            self.rag = {"ready": False, "reason": "disabled"}
            log.info("rag_disabled")
            return

        import asyncio

        try:
            retriever = Retriever(s)
            # 建索引句柄 + 装载向量化模型都在这里完成（GPU 上约 15–40 s，CPU 更久）
            store = await asyncio.to_thread(lambda: retriever.store)
            docs = await asyncio.to_thread(store.count)
            embedder = await asyncio.to_thread(lambda: retriever.embedder)
            self.rag = {
                "ready": True,
                "reason": "ok",
                "retriever": retriever,
                "gate": lambda result: gate_evaluate(
                    result,
                    threshold=s.rag_evidence_threshold,
                    margin=s.rag_evidence_margin,
                    min_docs=s.rag_evidence_min_docs,
                ),
                "embedder_name": getattr(embedder, "name", "unknown"),
                "rerank_enabled": s.rerank_enabled,
                "docs": docs,
            }
            log.info(
                "rag_ready",
                docs=docs,
                embedder=getattr(embedder, "name", "unknown"),
                rerank=s.rerank_enabled,
                collection=s.milvus_collection,
            )
        except Exception as exc:  # noqa: BLE001 - Milvus/模型不可用时降级为"仅 FAQ"
            self.rag = {"ready": False, "reason": str(exc)[:200]}
            log.error("rag_unavailable", error=str(exc))
            if not s.allow_degraded_start:
                raise

    def close(self) -> None:
        if self.rag and self.rag.get("ready"):
            try:
                self.rag["retriever"].store.close()
            except Exception:  # noqa: S110 - 关闭失败不影响退出
                pass
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
    # 日志必须在任何业务初始化之前配好，否则启动期的失败会掉进 stdout 之外的虚空
    configure(
        settings.log_level,
        log_path=settings.log_path,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
        file_level=settings.log_file_level,
    )
    bind(node="startup", env=settings.app_env, version=settings.app_version)

    problems = settings.self_check()
    if problems:
        for problem in problems:
            log.error("config_problem", detail=problem)
        if not settings.allow_degraded_start:
            raise RuntimeError("配置校验失败：" + "；".join(problems))

    container = Container(settings)
    container.build()
    await container.load_faq()
    await container.load_rag()

    app.state.container = container
    app.state.faq_service = container.faq_service
    app.state.rag = container.rag
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
app.include_router(rag_router)


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
        "rag": bool(container.rag and container.rag.get("ready")) if container else False,
        "rag_docs": (container.rag or {}).get("docs", 0) if container else 0,
        "rag_reason": (container.rag or {}).get("reason") if container else "startup",
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
            "# HELP edurag_faq_cache_hits_total 缓存命中数",
            "# TYPE edurag_faq_cache_hits_total counter",
            f"edurag_faq_cache_hits_total {stats['cache_hits']}",
            "# HELP edurag_faq_cache_fills_total 缓存回填数",
            "# TYPE edurag_faq_cache_fills_total counter",
            f"edurag_faq_cache_fills_total {stats['cache_fills']}",
            "# HELP edurag_faq_cache_hit_rate 缓存命中率",
            "# TYPE edurag_faq_cache_hit_rate gauge",
            f"edurag_faq_cache_hit_rate {stats['cache_hit_rate']}",
            "# HELP edurag_faq_cache_errors_total 缓存异常数（fail-open）",
            "# TYPE edurag_faq_cache_errors_total counter",
            f"edurag_faq_cache_errors_total {stats['cache_errors']}",
        ]
    rag = container.rag if container else None
    if rag is not None:
        lines += [
            "# HELP edurag_rag_ready 检索链是否就绪（0/1）",
            "# TYPE edurag_rag_ready gauge",
            f"edurag_rag_ready {1 if rag.get('ready') else 0}",
            "# HELP edurag_rag_docs 向量库子块数",
            "# TYPE edurag_rag_docs gauge",
            f"edurag_rag_docs {rag.get('docs', 0)}",
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
        # log_config=None：否则 uvicorn 会用自己的 handler 覆盖 root，
        # 启动信息与访问日志就进不了 logs/app.log（排查启动失败时最需要的那几行）
        log_config=None,
    )


if __name__ == "__main__":
    run()


