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
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from auth.api import router as auth_router
from auth.service import AuthService
from auth.store import (
    InMemoryTokenRepo,
    InMemoryUserRepo,
    MySQLTokenRepo,
    MySQLUserRepo,
    TokenRepo,
    UserRepo,
)
from base.config import Settings, get_settings
from base.db import MySQL
from base.logger import bind, clear, configure, get_logger, new_request_id
from memory.api import router as memory_router
from memory.service import MemoryService
from memory.session import SessionStore
from memory.store import (
    AtomIndex,
    AtomRepo,
    ChatLogRepo,
    GrantRepo,
    InMemoryAtomRepo,
    InMemoryGrantRepo,
    InMemoryProfileRepo,
    MilvusAtomIndex,
    MySQLAtomRepo,
    MySQLGrantRepo,
    MySQLProfileRepo,
    ProfileRepo,
)
from orchestration.api import router as chat_router
from orchestration.chat import ChatPipeline
from understanding.bert_intent import BertIntentClassifier
from understanding.compound_query import split as compound_split
from qa_system.api import router as faq_router
from rag_system.api import router as rag_router
from rag_system.core.llm_gateway import LLMGateway, LLMTarget
from rag_system.core.prompts import hyde_messages
from rag_system.core.rag_pipeline import RagPipeline
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
        # 记忆与 FAQ 用同一个真相源：FAQ 走 CSV 无库模式时，记忆也走内存，
        # 免得"一半能落库一半不能落库"这种只有排障时才发现的半截状态
        self.backend: str = "memory"
        self.memory_repo: AtomRepo | None = None
        self.chat_log: ChatLogRepo | None = None
        self.grant_repo: GrantRepo | None = None
        self.profile_repo: ProfileRepo | None = None
        self.atom_index: AtomIndex | None = None
        self.session_store: SessionStore | None = None
        self.memory: MemoryService | None = None
        self.auth: AuthService | None = None
        self.rag: dict[str, Any] | None = None
        self.llm: LLMGateway | None = None
        # 本地小模型（闲聊档，§7.7）。**与 `self.llm` 各自独立**：端点、key、
        # 超时、可用性都不共享。`None` 表示"没配/没建起来"，编排层据此回落主网关。
        self.llm_local: LLMGateway | None = None
        # BERT 初始意图分类器（任务 2.2）。`None` 表示"不可用/未启用"，
        # 编排层据此退回"大模型 → 规则"，不额外分支。
        self.bert_intent: BertIntentClassifier | None = None
        self.chat: ChatPipeline | None = None
        # 复合问句拆分（阶段 4.2）。放在组合根而不是塞进 `rag_system/api.py`：
        # 调试接口要跑"线上那一半"的计划，就得先有子问题；但检索侧 import 理解层
        # 会让两个平级模块互相依赖（§5.1）。这里注入，检索侧只调用不 import。
        self.compound_split = compound_split
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
            cross_subject_delta=s.faq_cross_subject_delta,
            dedupe_near_dup=s.faq_dedupe_near_dup,
            # 租户进缓存键（阶段 1 验收 5）。部署时一个进程只服务一个租户，所以在这里
            # 注入是安全的；多租户同进程要改成按 `RequestContext.tenant_id` 传参。
            tenant_id=s.tenant_id,
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
            self.backend = "memory"
            log.warning("faq_corpus_from_csv", path=str(csv_path), docs=info.get("docs"))
            log.info("faq_ready", degraded=self.degraded, **info)
            return

        assert self.mysql is not None
        repository = MySQLFAQRepository(self.mysql)
        service = self._make_service(repository)
        try:
            info = await service.load()
            self.backend = "mysql"
        except Exception as exc:
            if not self.settings.allow_degraded_start:
                raise
            # 明确打日志并标记 degraded：readyz 仍 200，但 /v1/faq/stats 里能看到空语料
            log.error("faq_load_failed_degraded", error=str(exc))
            self.degraded = True
            repository = InMemoryFAQRepository()
            service = self._make_service(repository)
            info = await service.load()
            self.backend = "memory"
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
                # 生产链路由 RagPipeline 持有：计划 → 多路召回 → 融合 → 证据门 → 至多一次修复。
                # 编排层只与它打交道，不认识 Retriever/证据门的细节（§7.2.7-B）。
                # `hyde_budget_ms` 与生成器里用的是**同一个**配置项：两处各写一个数，
                # 就会出现"外层 1500 ms 掐断、内层还在等 60 s"的错位（HyDE 就是这么丢过一次）
                "pipeline": RagPipeline(
                    s, retriever, hyde=self._hyde_gateway(s), hyde_budget_ms=s.hyde_budget_ms
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

    def _hyde_gateway(self, s: Settings) -> Callable[[str], Awaitable[str]] | None:
        """HyDE 的"假设文"生成器（阶段 4.3）。**开关关着时返回 None**。

        返回值只用于"拿它去检索"，永远不进证据集合：pipeline 只把它当查询文本，
        并在证据装配完后再按整块相等过滤一次（`_drop_hyde_echo`）。

        用闭包**延迟**取 `self.llm`：`load_rag()` 跑在 `load_llm()` 之前（第 441–442 行），
        构造期拿不到网关。真正要生成时再取，取不到就抛——pipeline 记 `hyde_failed`
        并如实退回 `DIRECT`，不假装跑过 HyDE。
        """
        if not s.feature_hyde:
            return None

        async def generate(query: str) -> str:
            llm = getattr(self, "llm", None)
            if llm is None or not getattr(llm, "enabled", False):
                raise RuntimeError("hyde_llm_unavailable")
            # **显式点名生成模型**（2026-11-01）：HyDE 交给 DeepSeek 做，而 `LLM_QA_MODEL`
            # 就是生成侧那一个。不写死在这里是因为将来换网关/灰度时，假的假设文会跟着
            # 真答案一起换——两处各写一个模型名，迟早出现"答案用 A、HyDE 用 B"的对照
            # 假象。留空则走网关默认模型（与响应体里的 `model` 一致）。
            #
            # 预算是 `HYDE_BUDGET_MS` 而不是 `LLM_TIMEOUT_S`：后者 60 s 等于没有上限，
            # 一次慢调用会把整个 `TOTAL_BUDGET_MS`（8 s）吃穿，而 HyDE 只是"试试能不能
            # 查得更准"。这里给的就是硬上限，`RagPipeline` 外面还会再套一层 `wait_for`
            # 同值超时——两层都在是刻意的：外层负责"别拖住检索"，内层负责"别让 httpx
            # 在超时之后还占着连接继续跑"
            result = await llm.complete(
                hyde_messages(query),
                budget_ms=s.hyde_budget_ms,
                model=s.llm_qa_model or None,
            )
            return result.text

        return generate

    def load_llm(self) -> None:
        """LLM 网关只在进程内建一个（httpx 连接池复用）。

        `LLM_API_KEY` 为空**不退出**：FAQ 与 RAG 都不依赖它，没有 key 的开发机仍然要能起服务。
        只在生产（`ALLOW_DEGRADED_START=false`）时把缺 key 当成阻断问题——见 §7.8。
        """
        s = self.settings
        self.llm = LLMGateway(s)
        # **必须建在这段 early-return 之前**：没有云端 key 的开发机恰恰是最需要
        # 本地小模型的场景（闲聊档），把它放在 return 之后等于降级启动时永远没有本地引擎。
        if s.llm_local_enabled:
            self.llm_local = LLMGateway(s, target=LLMTarget.local_from_settings(s))
            log.info(
                "llm_local_ready",
                model=s.llm_local_model,
                base_url=s.llm_local_base_url,
                timeout_s=s.llm_local_timeout_s,
                reasoning_effort=s.llm_local_reasoning_effort,
            )
        else:
            log.warning(
                "llm_local_not_configured",
                hint="LLM_LOCAL_BASE_URL / LLM_LOCAL_MODEL 为空：闲聊档回落主网关",
            )
        if not self.llm.enabled:
            log.warning(
                "llm_not_configured",
                hint="LLM_API_KEY 为空：/v1/chat 将走兜底话术，FAQ 与 RAG 不受影响",
            )
            if not s.allow_degraded_start:
                raise RuntimeError("LLM_API_KEY 未配置（ALLOW_DEGRADED_START=false 时不允许降级启动）")
            return
        log.info("llm_ready", model=s.llm_qa_model, base_url=s.llm_base_url, timeout_s=s.llm_timeout_s)

    def build_memory(self) -> None:
        """把「会话窗口 + 原子记忆 + 记忆页」接起来（§7.9）。

        仓库实现跟随 FAQ 的真相源：`backend=="mysql"` 才用 MySQL，否则用内存实现。
        会话存储的 Redis 同理——Redis 不可用时 `SessionStore` 自己退化为进程内字典，
        这里只需把 `redis=None` 传下去，不额外分支。
        """
        s = self.settings
        if not s.feature_memory:
            log.info("memory_disabled", reason="FEATURE_MEMORY=off")
            self.session_store = SessionStore(s)
            # 关掉记忆时授权表也只进进程内：重启即丢，和"这一整块本就不生效"的语义一致。
            # 不用 `grants=None`——那会让 `/v1/memory/grants` 抛 RuntimeError 变成 500，
            # 而"功能被关掉"应该是可预期地回答"没有授权"，不是服务端出错。
            self.memory = MemoryService(
                s,
                InMemoryAtomRepo(),
                self.session_store,
                grants=InMemoryGrantRepo(),
                profiles=InMemoryProfileRepo(),
            )
            return

        repo: AtomRepo
        chat_log: ChatLogRepo | None = None
        grants: GrantRepo
        profiles: ProfileRepo
        if self.backend == "mysql" and self.mysql is not None:
            repo = MySQLAtomRepo(self.mysql)
            chat_log = ChatLogRepo(self.mysql)
            grants = MySQLGrantRepo(self.mysql)
            profiles = MySQLProfileRepo(self.mysql)
        else:
            # 无库模式（CSV 语料 / MySQL 不可用降级）：记忆只活在进程内，重启即丢。
            # 这是刻意的——宁可"这次会话记住、重启就忘"，也不要拿一个连不上的库假装持久化。
            repo = InMemoryAtomRepo()
            grants = InMemoryGrantRepo()
            profiles = InMemoryProfileRepo()
            log.warning("memory_backend_in_memory", reason=self.backend)

        self.memory_repo = repo
        self.chat_log = chat_log
        self.grant_repo = grants
        self.profile_repo = profiles
        self.session_store = SessionStore(s, redis=self.redis, chat_log=chat_log)
        self.memory = MemoryService(
            s,
            repo,
            self.session_store,
            chat_log,
            grants=grants,
            profiles=profiles,
            index=self._atom_index(),
        )
        log.info(
            "memory_ready",
            backend=self.backend if chat_log is not None else "memory",
            redis=bool(self.redis),
            window_turns=s.mem_window_turns,
            window_tokens=s.mem_window_tokens,
            vector=bool(self.memory.index),
        )

    def _atom_index(self) -> AtomIndex | None:
        """L2 的向量索引（§3.7 第 8 条）。**能降级就降级，不阻断启动。**

        三个前置条件缺一不可，缺了就返回 None，召回退回字面排序：
        ① `FEATURE_MEMORY_VECTOR=on`；② RAG 就绪（借它的 Milvus 连接与 bge-m3，
           再装载一遍 2.5 GB 的模型只为给几百条记忆做召回不值得）；
        ③ 集合可用（建不出来就说明这个 Milvus 有问题，记忆不该因此起不来）。

        `ensure()` 放在这里（启动期）而不是第一次写入时：请求路径上建集合会引入
        一次几百毫秒的连接 + 建表，而它只发生一次。
        """
        s = self.settings
        if not s.feature_memory_vector:
            log.info("memory_vector_disabled", reason="FEATURE_MEMORY_VECTOR=off")
            return None
        rag = self.rag or {}
        retriever = rag.get("retriever")
        if not rag.get("ready") or retriever is None:
            log.warning("memory_vector_unavailable", reason=str(rag.get("reason") or "rag_not_ready"))
            return None
        try:
            index = MilvusAtomIndex(
                retriever.store.client, retriever.embedder, s, tenant_id=s.tenant_id
            )
            index.ensure()
        except Exception as exc:  # Milvus 抖动不该让整个服务起不来
            log.warning("memory_vector_unavailable", reason=str(exc)[:200])
            return None
        log.info(
            "memory_vector_ready",
            collection=index.collection,
            embedder=index.name,
            rows=index.count(),
        )
        return index

    def build_auth(self) -> None:
        """账号与令牌（`/api/v1/auth/*`）。仓储选择与 `build_memory()` 同一条规则。

        **`JWT_SECRET` 为空不阻断启动**：四个写接口会回 503 `auth_not_configured`，
        其它端点（FAQ / RAG / chat）照常。理由与 `LLM_API_KEY` 为空时不退出一样——
        一台没配密钥的开发机不该连问问题都起不来。但原因必须出现在启动日志里。
        """
        s = self.settings
        users: UserRepo
        tokens: TokenRepo
        if self.backend == "mysql" and self.mysql is not None:
            users = MySQLUserRepo(self.mysql)
            tokens = MySQLTokenRepo(self.mysql)
        else:
            # 无库模式：能注册能登录，但重启即失。宁可"重启就忘"，也不要拿一个
            # 连不上的库假装持久化（与记忆模块同口径）
            users = InMemoryUserRepo()
            tokens = InMemoryTokenRepo()
            log.warning("auth_backend_in_memory", reason=self.backend)

        self.auth = AuthService(settings=s, users=users, tokens=tokens)
        if not self.auth.configured:
            log.warning(
                "auth_not_configured",
                hint="JWT_SECRET 为空：/api/v1/auth 的 register/login/refresh/me 将回 503",
            )
            return
        log.info(
            "auth_ready",
            backend=self.backend if self.mysql is not None else "memory",
            access_ttl_s=s.auth_access_ttl_s,
            refresh_ttl_s=s.auth_refresh_ttl_s,
        )

    def build_intent(self) -> None:
        """BERT 意图分类器：启动期加载（§4.2：模型不进请求路径）。

        加载失败**不阻断启动**——意图只是"挑一条先验"，挂了会退到大模型与规则分类器
        （§7.7 降级表），FAQ 与检索照常。但原因必须出现在启动日志里：
        "今天意图全走大模型"要能一眼看出来，而不是等人去翻 `bert_intent_*` 事件。
        """
        s = self.settings
        if not s.feature_intent_bert:
            self.bert_intent = None
            log.info("bert_intent_disabled", reason="FEATURE_INTENT_BERT=off")
            return
        classifier = BertIntentClassifier()
        if classifier.warmup():
            self.bert_intent = classifier
        else:
            # `warmup()` 内部已经打过 `bert_intent_unavailable`（含原因）
            self.bert_intent = None

    def build_chat(self) -> None:
        assert self.faq_service is not None and self.llm is not None
        self.chat = ChatPipeline(
            self.settings,
            faq_service=self.faq_service,
            rag=self.rag,
            llm=self.llm,
            llm_local=self.llm_local,
            bert=self.bert_intent,
            memory=self.memory,
        )
        log.info(
            "chat_ready",
            rag=bool(self.rag and self.rag.get("ready")),
            llm=self.llm.enabled,
            memory=self.memory is not None,
            bert=self.bert_intent is not None,
        )

    async def aclose(self) -> None:
        if self.llm is not None:
            await self.llm.aclose()
        if self.llm_local is not None:
            await self.llm_local.aclose()
        self.close()

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
        min_candidates=settings.faq_min_candidates,
        margin_mode=settings.faq_margin_mode,  # type: ignore[arg-type]
        # 三档路由的高置信下限（§7.1.9）；等于 faq_margin 即回退为单档
        high_margin=settings.faq_high_margin,
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

    # 不阻断启动的提醒（没配 key、理解预算偏大）：与 `self_check()` 分开，
    # 否则一台没有 key 的开发机连 FAQ 都起不来（§2.5 对 LLM 的同一口径）
    for warning in settings.soft_warnings():
        log.warning("config_warning", detail=warning)

    container = Container(settings)
    container.build()
    await container.load_faq()
    await container.load_rag()
    container.load_llm()
    container.build_memory()
    container.build_auth()
    container.build_intent()
    container.build_chat()

    app.state.container = container
    app.state.faq_service = container.faq_service
    app.state.rag = container.rag
    app.state.llm = container.llm
    app.state.chat = container.chat
    app.state.memory = container.memory
    app.state.auth = container.auth
    app.state.ready = True
    log.info("app_ready", version=settings.app_version, env=settings.app_env)
    try:
        yield
    finally:
        app.state.ready = False
        await container.aclose()
        log.info("app_stopped")


app = FastAPI(
    title="EduRAG-Pro 智能教育知识库系统",
    version="0.1.0",
    description="qa_system / rag_system / 记忆模块 三模块之一：FAQ 精确检索（PR2）",
    lifespan=lifespan,
)
app.include_router(faq_router)
app.include_router(rag_router)
app.include_router(chat_router)
app.include_router(memory_router)
app.include_router(auth_router)


@app.middleware("http")
async def trace_middleware(request: Request, call_next: Any) -> Response:
    """给每个请求绑定 request_id，异常统一转 500 并保留 trace 便于排查。"""
    clear()
    request_id = request.headers.get("X-Request-Id") or new_request_id()
    request.state.request_id = request_id
    bind(request_id=request_id, path=request.url.path)
    started = time.perf_counter()
    # 单一出口：异常也在这里落成一个 response，再走同一个 `finally`。
    # 原实现在 `except` 里直接 `return`，`finally` 仍读 `response.status_code`——
    # 那个变量此刻**没绑定**，`UnboundLocalError` 会顶掉本该返回的 500 JSON，
    # 用户拿到的是 uvicorn 的纯文本错误页，`request_id` 也没了。
    response: Response
    try:
        response = await call_next(request)
    except Exception as exc:
        log.error("unhandled_error", error=str(exc), exc_info=True)
        response = JSONResponse(
            status_code=500, content={"detail": "internal error", "request_id": request_id}
        )
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        # `path` 必须显式传：模板 `HTTP {method} {path} -> {status}` 缺字段会渲染成 `-`，
        # 而排查 404 时"哪个路径 404 了"正是唯一有用的信息（实测日志里出现过 `HTTP GET - -> 404`）
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=getattr(response, "status_code", 500),
            latency_ms=round(elapsed_ms, 2),
        )
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
        "llm": bool(container.llm and container.llm.enabled) if container else False,
        "llm_model": container.llm.model if container and container.llm else None,
        "memory": bool(container.memory) if container else False,
        "memory_backend": container.backend if container else "none",
        # `auth` 是"能不能签发令牌"（密钥配了没），不是"进程起来没"——
        # 空密钥时进程照样 ready，登录接口回 503，这里如实暴露 false
        "auth": bool(container.auth and container.auth.configured) if container else False,
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
        # 三档路由计数（§7.1.9）：与 routes 分开——一个是"查了多大范围"，一个是"结论有多可信"
        for tier, note in (("high", "高置信可直接作答"), ("mid", "中置信只给候选与追问"), ("low", "低置信交棒 RAG")):
            lines += [
                f"# HELP edurag_faq_route_{tier}_total {note}",
                f"# TYPE edurag_faq_route_{tier}_total counter",
                f"edurag_faq_route_{tier}_total {stats['tiers'].get(tier, 0)}",
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
    llm = container.llm if container else None
    if llm is not None:
        lines += [
            "# HELP edurag_llm_ready 生成模型是否可用（0/1，无 key 时为 0）",
            "# TYPE edurag_llm_ready gauge",
            f"edurag_llm_ready {1 if llm.enabled else 0}",
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


