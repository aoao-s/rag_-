"""FAQService：qa_system 的唯一对外入口。

职责：
- 启动时把 published 语料装载成"分学科 BM25 索引 + 精确哈希表"，之后请求路径**零 IO、零模型**；
- 发布/下线后重建索引并**原子替换快照**，请求侧不加锁、不阻塞（双缓冲）；
- 未命中时返回 top3 候选，交给 RAG 做 query 扩展，而不是留一句"答不上来"。

性能取向：`search()` 全程内存计算，p95 目标 ≤15 ms（CLAUDE.md §7.1.5）。

关于"按学科重建"：当前语料规模（数千条）全量重建 <100 ms，且别名变更会影响所有学科的
归一化结果，局部重建反而容易出错。因此 `rebuild(subject=...)` 的 subject 仅用于日志与指标，
实际执行全量重建；语料超过 5 万条时再引入文档级增量。
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from base.contracts import FAQCandidate, FAQHit
from base.labels import SUBJECTS, normalize_subject
from base.logger import get_logger
from qa_system.res.qa_cache import BaseFAQCache, CacheEntry, SingleFlight
from qa_system.res.qa_corpus import BM25Index, IndexedDoc
from qa_system.res.qa_matcher import MatchPolicy, decide, normalize_score
from qa_system.res.qa_normalize import Normalizer
from qa_system.res.qa_db import FAQRecord, FAQRepository

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """不可变快照：search 只读一次引用，重建时整体替换，天然线程安全。"""

    version: int
    indexes: dict[str | None, BM25Index]
    exact: dict[str, int]
    records: dict[int, FAQRecord]
    normalizer: Normalizer
    built_at: float = 0.0

    @property
    def doc_count(self) -> int:
        return sum(len(idx) for idx in self.indexes.values())


@dataclass(slots=True)
class _Stats:
    searches: int = 0
    hits: int = 0
    exact_hits: int = 0
    alias_hits: int = 0
    misses: int = 0
    candidates_served: int = 0
    total_latency_ms: float = 0.0
    rebuilds: int = 0
    cache_hits: int = 0
    cache_fills: int = 0
    cache_errors: int = 0

    def as_dict(self) -> dict[str, Any]:
        avg = self.total_latency_ms / self.searches if self.searches else 0.0
        hit_rate = self.hits / self.searches if self.searches else 0.0
        exact_rate = self.exact_hits / self.searches if self.searches else 0.0
        cache_rate = self.cache_hits / self.searches if self.searches else 0.0
        return {
            "searches": self.searches,
            "hits": self.hits,
            "misses": self.misses,
            "exact_hits": self.exact_hits,
            "alias_hits": self.alias_hits,
            "hit_rate": round(hit_rate, 4),
            "exact_rate": round(exact_rate, 4),
            "avg_latency_ms": round(avg, 3),
            "candidates_served": self.candidates_served,
            "rebuilds": self.rebuilds,
            "cache_hits": self.cache_hits,
            "cache_fills": self.cache_fills,
            "cache_errors": self.cache_errors,
            "cache_hit_rate": round(cache_rate, 4),
        }


class FAQService:
    def __init__(
        self,
        repository: FAQRepository,
        *,
        policy: MatchPolicy | None = None,
        alias_enabled: bool = True,
        top_k: int = 20,
        extra_words: tuple[str, ...] = SUBJECTS,
        cache: BaseFAQCache | None = None,
        cache_ttl: int = 3600,
        cache_negative_ttl: int = 30,
        cache_prefix: str = "edurag:dev",
        log_query: bool = False,
    ) -> None:
        self._repo = repository
        self._policy = policy or MatchPolicy()
        self._alias_enabled = alias_enabled
        self._top_k = top_k
        self._extra_words = extra_words
        self._cache = cache
        self._cache_ttl = cache_ttl
        self._cache_negative_ttl = cache_negative_ttl
        self._cache_prefix = cache_prefix
        self._single_flight = SingleFlight()
        # 原始 query 可能含学生隐私（姓名/手机号），默认只记长度与指纹
        self._log_query = log_query
        self._snapshot = _Snapshot(
            version=0,
            indexes={},
            exact={},
            records={},
            normalizer=Normalizer(extra_words=extra_words),
            built_at=0.0,
        )
        self._rebuild_lock = threading.Lock()
        self._stats = _Stats()

    # ------------------------------------------------------------------ 生命周期
    async def load(self) -> dict[str, Any]:
        """启动期装载。失败必须向上抛：带着空语料接流量等于线上静默失效。"""
        started = time.perf_counter()
        records, aliases = await self._fetch_all()
        snapshot = self._build(records, aliases, version=1)
        self._snapshot = snapshot
        cost_ms = (time.perf_counter() - started) * 1000
        log.info(
            "faq_loaded",
            docs=snapshot.doc_count,
            subjects=len(snapshot.indexes),
            aliases=len(aliases),
            cost_ms=round(cost_ms, 1),
        )
        return {"version": snapshot.version, "docs": snapshot.doc_count, "cost_ms": round(cost_ms, 1)}

    async def warmup(self, samples: list[str] | None = None) -> int:
        """预热：跑一遍真实查询，让 jieba 词典与分词缓存进入热态，并重置预热期的指标。"""
        self._snapshot.normalizer.preload()
        queries = samples or ["学费多少", "课程学多久", "怎么报名"]
        for query in queries:
            await self.search(query)
        self._stats = _Stats(rebuilds=self._stats.rebuilds)
        return len(queries)

    # ------------------------------------------------------------------ 查询
    async def search(self, query: str, subject: str | None = None) -> FAQHit:
        started = time.perf_counter()
        snapshot = self._snapshot

        if snapshot.version == 0 or not snapshot.records:
            return self._finish(False, started, reason="not_loaded", source="none")

        normalizer = snapshot.normalizer
        if not normalizer.normalize(query or ""):
            return self._finish(False, started, reason="empty_query", source="none")

        raw_norm = normalizer.normalize(query or "", apply_alias=False)
        norm = normalizer.normalize(query or "")
        subject_key = normalize_subject(subject)

        # ① Redis 二级缓存：多副本共享 + 省掉一次索引计算。
        # key 带 snapshot.version，发布/下线后旧 key 自然不可达，无需遍历删除（见 qa_cache 文档）。
        key = self._cache_key(snapshot.version, norm, subject_key)
        cached, cache_state = await self._cache_lookup(key)
        if cached is not None:
            if cached.faq_id is None:
                # 负缓存：热门脏查询不再反复打分（TTL 很短，版本变更即失效）
                self._stats.cache_hits += 1
                cached_candidates = tuple(
                    FAQCandidate(
                        faq_id=fid,
                        question=snapshot.records[fid].question,
                        score=score,
                    )
                    for fid, score in cached.candidates
                    if fid in snapshot.records
                )
                self._stats.candidates_served += len(cached_candidates)
                return self._finish(
                    False,
                    started,
                    reason=f"cached_{cached.reason}",
                    score=cached.score,
                    subject=cached.subject,
                    candidates=cached_candidates,
                    source="cache",
                    cache_state="hit",
                    norm=norm,
                )
            record = snapshot.records.get(cached.faq_id)
            if record is not None:
                self._stats.cache_hits += 1
                if self._cache is not None:
                    await self._cache.slide(key, self._cache_ttl)
                return self._finish(
                    True,
                    started,
                    record=record,
                    score=cached.score,
                    reason=f"cache_{cached.reason}",
                    alias_hit=cached.reason == "alias_exact",
                    source="cache",
                    cache_state="hit",
                    norm=norm,
                )
            # 缓存指向的条目已被删除：当作未命中，走一次索引
            log.info("faq_cache_stale", faq_id=cached.faq_id, version=snapshot.version)

        # ② 精确命中（含别名归一结果）：O(1)，最常见的索引路径
        faq_id = snapshot.exact.get(norm)
        if faq_id is not None:
            via_alias = raw_norm != norm
            if via_alias:
                self._stats.alias_hits += 1
            else:
                self._stats.exact_hits += 1
            reason = "alias_exact" if via_alias else "exact"
            await self._cache_fill(key, CacheEntry(faq_id=faq_id, score=1.0, reason=reason), self._cache_ttl)
            return self._finish(
                True,
                started,
                record=snapshot.records[faq_id],
                score=1.0,
                reason=reason,
                alias_hit=via_alias,
                source="index",
                cache_state=cache_state,
                norm=norm,
            )

        # ③ 分学科 BM25；未指定学科则全学科打分后取每问最大值
        tokens = normalizer.tokens(norm)
        if not tokens:
            return self._finish(
                False, started, reason="empty_tokens", source="none", cache_state=cache_state, norm=norm
            )

        indexes = self._select_indexes(snapshot, subject_key)
        best: dict[int, float] = defaultdict(float)
        for index in indexes:
            for doc, score in index.search(tokens, top_k=self._top_k):
                if score > best[doc.faq_id]:
                    best[doc.faq_id] = score

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]
        if not ranked:
            # 一个 token 都没命中：没有候选可给，交给 RAG 全量检索
            await self._cache_fill(
                key, CacheEntry(faq_id=None, score=0.0, reason="no_match"), self._cache_negative_ttl
            )
            return self._finish(
                False, started, reason="no_match", source="index", cache_state=cache_state, norm=norm
            )

        result = decide(ranked, self._policy)
        top_subject = snapshot.records[result.faq_id].subject if result.faq_id in snapshot.records else None

        if not result.hit or result.faq_id is None:
            candidates = tuple(
                FAQCandidate(
                    faq_id=fid,
                    question=snapshot.records[fid].question,
                    score=round(raw / (raw + self._policy.norm_k), 4),
                )
                for fid, raw in ranked[: self._policy.candidate_n]
                if fid in snapshot.records
            )
            self._stats.candidates_served += len(candidates)
            await self._cache_fill(
                key,
                CacheEntry(
                    faq_id=None,
                    score=result.score,
                    reason=result.reason,
                    subject=top_subject,
                    candidates=tuple((c.faq_id, c.score) for c in candidates),
                ),
                self._cache_negative_ttl,
            )
            return self._finish(
                False,
                started,
                reason=result.reason,
                score=result.score,
                candidates=candidates,
                subject=top_subject,
                source="index",
                cache_state=cache_state,
                norm=norm,
            )

        await self._cache_fill(
            key,
            CacheEntry(faq_id=result.faq_id, score=result.score, reason=result.reason, subject=top_subject),
            self._cache_ttl,
        )
        return self._finish(
            True,
            started,
            record=snapshot.records[result.faq_id],
            score=result.score,
            reason=result.reason,
            source="index",
            cache_state=cache_state,
            norm=norm,
        )

    # ------------------------------------------------------------------ 离线评估
    async def rank(self, query: str, subject: str | None = None, top_k: int = 3) -> list[tuple[int, float]]:
        """返回 (faq_id, 归一化分) 降序，不做命中判定。

        供离线阈值标定与排障使用：`search()` 只能看到"最终结论"，看不出 top1/top2 的分差，
        而 margin 恰恰是区分"同题族"（勾股定理内容 / 勾股定理证明）的关键信号。
        """
        snapshot = self._snapshot
        if snapshot.version == 0 or not snapshot.records:
            return []

        normalizer = snapshot.normalizer
        norm = normalizer.normalize(query or "")
        if not norm:
            return []

        faq_id = snapshot.exact.get(norm)
        if faq_id is not None:
            return [(faq_id, 1.0)]

        tokens = normalizer.tokens(norm)
        if not tokens:
            return []

        best: dict[int, float] = defaultdict(float)
        for index in self._select_indexes(snapshot, normalize_subject(subject)):
            for doc, score in index.search(tokens, top_k=self._top_k):
                if score > best[doc.faq_id]:
                    best[doc.faq_id] = score

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(fid, normalize_score(raw, self._policy.norm_k)) for fid, raw in ranked]

    # ------------------------------------------------------------------ 写路径
    async def publish(self, faq_id: int, operator: str = "api") -> dict[str, Any]:
        if not await self._repo.publish(faq_id, operator):
            raise LookupError(f"faq {faq_id} 不存在或当前状态不可发布")
        return await self.rebuild()

    async def deprecate(self, faq_id: int, operator: str = "api") -> dict[str, Any]:
        if not await self._repo.deprecate(faq_id, operator):
            raise LookupError(f"faq {faq_id} 不存在")
        return await self.rebuild()

    async def rebuild(self, subject: str | None = None) -> dict[str, Any]:
        """重建索引并原子替换快照。重建期间的读请求继续用旧快照，不做任何等待。"""
        with self._rebuild_lock:
            started = time.perf_counter()
            records, aliases = await self._fetch_all()
            snapshot = self._build(records, aliases, version=self._snapshot.version + 1)
            self._snapshot = snapshot
            self._stats.rebuilds += 1
            cost_ms = (time.perf_counter() - started) * 1000
            log.info(
                "faq_rebuilt",
                version=snapshot.version,
                docs=snapshot.doc_count,
                subject=subject,
                cost_ms=round(cost_ms, 1),
                # 缓存不主动删：key 带版本号，旧版本条目在 TTL 内自然淘汰
                cache_scope="version_bump",
            )
            return {"version": snapshot.version, "docs": snapshot.doc_count, "cost_ms": round(cost_ms, 1)}

    # ------------------------------------------------------------------ 管理读
    async def get(self, faq_id: int) -> FAQRecord | None:
        """按 id 取单条（管理端排障用，不进请求热路径）。"""
        return await self._repo.fetch_by_id(faq_id)

    # ------------------------------------------------------------------ 观测
    def state(self) -> dict[str, Any]:
        snapshot = self._snapshot
        # 学科为空的行（数据集里有 97 条「认识自己」类问题没有标学科）单独计为 unlabeled，
        # 不能 str(None) 混进 subjects，否则监控面板上会出现一个叫 "None" 的学科。
        subjects = sorted(k for k in snapshot.indexes if k)
        unlabeled = sum(len(snapshot.indexes[k]) for k in snapshot.indexes if not k)
        return {
            "loaded": snapshot.version > 0,
            "version": snapshot.version,
            "docs": snapshot.doc_count,
            "subjects": subjects,
            "unlabeled": unlabeled,
            "built_at": snapshot.built_at,
        }

    def stats(self) -> dict[str, Any]:
        return self._stats.as_dict()

    # ------------------------------------------------------------------ 内部
    def _cache_key(self, version: int, norm: str, subject: str | None) -> str:
        """key = faq:v{语料版本}:{学科}:{归一化问法指纹}。

        用指纹而不是原文：key 长度稳定、不把学生问法写进 Redis 键空间。
        带版本号是刻意的——发布/下线只需让路由到新版本，旧键靠 TTL 自然淘汰。
        """
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:20]
        return f"{self._cache_prefix}:faq:v{version}:{subject or '*'}:{digest}"

    async def _cache_lookup(self, key: str) -> tuple[CacheEntry | None, str]:
        if self._cache is None:
            return None, "disabled"
        try:
            entry = await self._cache.get(key)
        except Exception as exc:
            self._stats.cache_errors += 1
            log.warning("faq_cache_error", op="get", error=str(exc))
            return None, "error"
        return (entry, "hit") if entry is not None else (None, "miss")

    async def _cache_fill(self, key: str, entry: CacheEntry, ttl: int) -> None:
        """回填热数据（MySQL/索引命中后写入 Redis）。用单飞避免并发重复写。"""
        if self._cache is None:
            return
        lock = await self._single_flight.lock(key)
        async with lock:
            try:
                if await self._cache.get(key) is None:
                    await self._cache.put(key, entry, ttl)
                    self._stats.cache_fills += 1
                    log.info(
                        "faq_cache_fill",
                        faq_id=entry.faq_id,
                        reason=entry.reason,
                        ttl=ttl,
                    )
            except Exception as exc:
                self._stats.cache_errors += 1
                log.warning("faq_cache_error", op="fill", error=str(exc))
        await self._single_flight.release(key)

    async def _fetch_all(self) -> tuple[list[FAQRecord], list[tuple[int, str]]]:
        records = await self._repo.fetch_published()
        aliases = await self._repo.fetch_aliases() if self._alias_enabled else []
        return records, aliases

    def _select_indexes(self, snapshot: _Snapshot, subject: str | None) -> list[BM25Index]:
        if subject:
            index = snapshot.indexes.get(subject)
            if index is not None:
                return [index]
            # 学科未知 → 退化为全学科检索，不能因为拼错学科就让整条链路失效
        return list(snapshot.indexes.values())

    def _build(
        self,
        records: list[FAQRecord],
        aliases: list[tuple[int, str]],
        *,
        version: int,
    ) -> _Snapshot:
        by_id: dict[int, FAQRecord] = {rec.faq_id: rec for rec in records if rec.question}

        alias_pairs: dict[str, str] = {}
        if self._alias_enabled:
            for faq_id, alias in aliases:
                rec = by_id.get(faq_id)
                if rec and alias:
                    alias_pairs[alias] = rec.question

        normalizer = Normalizer(aliases=alias_pairs, extra_words=self._extra_words)

        exact: dict[str, int] = {}
        grouped: dict[str | None, list[IndexedDoc]] = defaultdict(list)
        for rec in by_id.values():
            norm = normalizer.normalize(rec.question)
            if not norm:
                continue
            exact.setdefault(norm, rec.faq_id)
            tokens = tuple(normalizer.tokens(rec.question))
            if not tokens:
                continue
            grouped[rec.subject].append(
                IndexedDoc(faq_id=rec.faq_id, question=rec.question, subject=rec.subject, tokens=tokens)
            )

        indexes = {subject: BM25Index(docs) for subject, docs in grouped.items()}
        return _Snapshot(
            version=version,
            indexes=indexes,
            exact=exact,
            records=by_id,
            normalizer=normalizer,
            built_at=time.time(),
        )

    def _finish(
        self,
        hit: bool,
        started: float,
        *,
        record: FAQRecord | None = None,
        score: float = 0.0,
        reason: str = "",
        candidates: tuple[FAQCandidate, ...] = (),
        subject: str | None = None,
        alias_hit: bool = False,
        source: str = "index",
        cache_state: str = "",
        norm: str = "",
    ) -> FAQHit:
        latency_ms = (time.perf_counter() - started) * 1000
        self._stats.searches += 1
        self._stats.total_latency_ms += latency_ms
        if hit:
            self._stats.hits += 1
        else:
            self._stats.misses += 1

        # 每次查询一行结构化日志：命中来自缓存还是索引、命中哪条、分数多少、耗时多少。
        # 排查"为什么答错"时，这一行是唯一的现场；没有它只能靠复现。
        fields: dict[str, Any] = {
            "hit": hit,
            "reason": reason,
            "source": source,
            "cache": cache_state,
            "faq_id": record.faq_id if record else None,
            "subject": record.subject if record else subject,
            "score": round(score, 4),
            "candidates": [c.faq_id for c in candidates],
            "latency_ms": round(latency_ms, 2),
            "version": self._snapshot.version,
            "q_len": len(norm),
            "q_hash": hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12] if norm else "",
        }
        if self._log_query and norm:
            # 原始问法属于个人数据，只在显式开启时以 DEBUG 落盘
            log.debug("faq_search_query", q=norm)
        log.info("faq_search", **fields)

        if record is not None:
            return FAQHit(
                hit=hit,
                faq_id=record.faq_id,
                answer=record.answer,
                score=score,
                subject=record.subject,
                matched_question=record.question,
                alias_hit=alias_hit,
                candidates=candidates,
                reason=reason,
                latency_ms=latency_ms,
                source=source,
                cache=cache_state,
            )
        return FAQHit(
            hit=False,
            score=score,
            subject=subject,
            candidates=candidates,
            reason=reason,
            latency_ms=latency_ms,
            source=source,
            cache=cache_state,
        )

