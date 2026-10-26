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
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any

from base.contracts import FAQCandidate, FAQHit, SlotResult, SubjectState, Tier
from base.labels import SUBJECTS
from base.logger import get_logger
from qa_system.res import qa_slots
from qa_system.res.qa_cache import BaseFAQCache, CacheEntry, SingleFlight
from qa_system.res.qa_corpus import BM25Index, IndexedDoc
from qa_system.res.qa_matcher import MatchPolicy, decide, decide_multi, normalize_score
from qa_system.res.qa_normalize import Normalizer, content_tokens
from qa_system.res.qa_db import FAQRecord, FAQRepository

log = get_logger(__name__)

# 命中来源的中文说法：日志一行内必须能直接看出"这次是缓存、索引还是 RAG"
_SOURCE_CN = {"cache": "Redis 缓存", "index": "MySQL 索引", "rag": "RAG 检索", "none": "无"}

# 这些来源的学科结论**不足以硬过滤**：可以看到该学科索引，但不允许直接答
# （CLAUDE.md §7.1.8 路由表：推断/上下文置信度未验证，冲突则是规则与模型打架；
#  invalid 是"参数不在词表里"，内部编排兜底成全库后更不能当成确定答案）
_NO_DIRECT: frozenset[str] = frozenset({"inferred", "context", "conflict", "invalid"})

# 语料重复合并的相似度门槛（CLAUDE.md §7.1.7 语料重复合并契约）。
# 去问句框架词后 Jaccard ≥0.8 且**答案逐字相同**才合并：实测在该语料上同答案 903 对 / 异答案 6 对。
_NEAR_DUP_JACCARD = 0.8
# 冲突检测（同科不同答案）的候选生成：共享内容词 ≥3 才值得算精确 Jaccard。
# 这是**近似**筛查，只用于告警；合并判定不走这条路径（合并按"同答案"分组，是完备的）。
_CONFLICT_MIN_SHARED = 3
_CONFLICT_MAX_DF = 100  # 词文档频率超过它的多半是通用词，跳过以免候选爆炸


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _plan_near_duplicate_merges(
    records: dict[int, FAQRecord],
    token_map: dict[int, tuple[str, ...]],
) -> tuple[dict[int, int], list[tuple[int, int]]]:
    """把"同一事实的多种问法"并成一条，返回 (被合并 faq_id → 代表 faq_id, 答案冲突对)。

    判据（三条同时成立）：同学科 + 答案逐字相同 + 去框架词后 Jaccard ≥ `_NEAR_DUP_JACCARD`。
    逐对判定后用并查集取传递闭包，代表取组内最小 faq_id——不整组一刀切，
    因为同一答案组里也可能有问法差异过大的行（实测 367 组里 61 组如此）。

    为什么必须做：BM25 给"什么是X"与"X指什么"几乎相同的分数，`margin` 必然 ≈0，
    两条都答不了；语料里有 318 组三胞胎，等于把 1/3 的语料锁死在"谁也命中不了"。
    """
    content: dict[int, frozenset[str]] = {
        fid: content_tokens(tokens) for fid, tokens in token_map.items() if tokens
    }
    parent: dict[int, int] = {fid: fid for fid in content}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # ① 合并：按 (学科, 答案) 分组，组内两两比相似度——同答案才可能是同一事实
    by_answer: dict[tuple[str | None, str], list[int]] = defaultdict(list)
    for fid in content:
        rec = records[fid]
        if rec.answer.strip():
            by_answer[(rec.subject, rec.answer.strip())].append(fid)
    merged_pairs = 0
    for fids in by_answer.values():
        if len(fids) < 2:
            continue
        for a, b in combinations(fids, 2):
            if _jaccard(content[a], content[b]) >= _NEAR_DUP_JACCARD:
                union(a, b)
                merged_pairs += 1

    groups: dict[int, list[int]] = defaultdict(list)
    for fid in content:
        groups[find(fid)].append(fid)
    canonical = {
        fid: min(members) for members in groups.values() if len(members) > 1 for fid in members if fid != min(members)
    }

    # ② 冲突：相似度够高但答案不同——同一件事写了两份答案，交教研，不自动合并
    conflicts: list[tuple[int, int]] = []
    postings: dict[str, list[int]] = defaultdict(list)
    for fid, toks in content.items():
        for tok in toks:
            postings[tok].append(fid)
    shared: Counter[tuple[int, int]] = Counter()
    for fids in postings.values():
        if len(fids) < 2 or len(fids) > _CONFLICT_MAX_DF:
            continue
        for a, b in combinations(sorted(fids), 2):
            shared[(a, b)] += 1
    for (a, b), n in shared.items():
        if n < _CONFLICT_MIN_SHARED or records[a].subject != records[b].subject:
            continue
        if records[a].answer.strip() == records[b].answer.strip():
            continue  # 合并路径已处理
        if _jaccard(content[a], content[b]) >= _NEAR_DUP_JACCARD:
            conflicts.append((a, b))
    if merged_pairs or conflicts:
        log.info("faq_near_dup_scan", pairs_merged=merged_pairs, conflicts=len(conflicts))
    return canonical, conflicts


@dataclass(frozen=True, slots=True)
class _Scope:
    """一次查询实际要扫的索引集合 + 对外可解释的状态（§7.1.8 路由表）。"""

    indexes: tuple[tuple[str | None, BM25Index], ...]
    state: SubjectState
    degraded: tuple[str, ...]
    key_part: str  # 进缓存 key 的检索范围标识



@dataclass(frozen=True, slots=True)
class _Snapshot:
    """不可变快照：search 只读一次引用，重建时整体替换，天然线程安全。"""

    version: int
    indexes: dict[str | None, BM25Index]
    exact: dict[str, int]
    records: dict[int, FAQRecord]
    normalizer: Normalizer
    built_at: float = 0.0
    # 词袋重复合并：被合并的 faq_id → 代表 faq_id（§7.1.7 数据补齐契约）
    merges: dict[int, int] = field(default_factory=dict)

    @property
    def merged(self) -> int:
        return len(self.merges)

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
    # 检索范围分布：filtered/multi/full/empty。不看这个数就无法判断"命中率变化"
    # 是阈值调出来的还是路由改出来的（CLAUDE.md §7.1.7 第四步）
    routes: dict[str, int] = field(default_factory=dict)
    # 置信档分布：high/mid/low（§7.1.9）。与 routes 分开计数——一个是"查了多大范围"，
    # 一个是"结论有多可信"，混在一起就再也分不清命中率变化是谁造成的
    tiers: dict[str, int] = field(default_factory=dict)

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
            "routes": dict(self.routes),
            # 三档恒齐备：缺一个键，面板上就分不清"没发生"与"埋点漏了"
            "tiers": {k: self.tiers.get(k, 0) for k in ("high", "mid", "low")},
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
        cross_subject_delta: float = 0.05,
        dedupe_near_dup: bool = True,
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
        self._cross_subject_delta = cross_subject_delta
        self._dedupe_near_dup = dedupe_near_dup
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
            docs=len(snapshot.records),
            indexed=snapshot.doc_count,
            merged=snapshot.merged,
            subjects=len(snapshot.indexes),
            aliases=len(aliases),
            cost_ms=round(cost_ms, 1),
        )
        return {
            "version": snapshot.version,
            "docs": len(snapshot.records),
            "indexed": snapshot.doc_count,
            "merged_duplicates": snapshot.merged,
            "cost_ms": round(cost_ms, 1),
        }

    async def warmup(self, samples: list[str] | None = None) -> int:
        """预热：跑一遍真实查询，让 jieba 词典与分词缓存进入热态，并重置预热期的指标。"""
        self._snapshot.normalizer.preload()
        queries = samples or ["学费多少", "课程学多久", "怎么报名"]
        for query in queries:
            await self.search(query)
        self._stats = _Stats(rebuilds=self._stats.rebuilds)
        return len(queries)

    # ------------------------------------------------------------------ 查询
    async def search(
        self,
        query: str,
        subject: str | None = None,
        *,
        slots: SlotResult | None = None,
        selected_faq_id: int | None = None,
    ) -> FAQHit:
        """检索。`slots=None` 等价于 `subject_source="not_provided"`（用户真的没限定学科）。

        `subject` 参数保留给公开 API 与离线脚本；两者同时给出时以 `slots` 为准——
        编排层已经算好了模型版槽位，不该被一个字符串参数覆盖回去。

        `selected_faq_id` 是用户确认通道（§7.1.9）：mid 档追问被采纳后，客户端带着它重新提交，
        直接返回该条答案（逐字）。决定权交给用户，而不是让分类器猜。
        """
        started = time.perf_counter()
        snapshot = self._snapshot

        if snapshot.version == 0 or not snapshot.records:
            return self._finish("low", started, reason="not_loaded", source="none")

        normalizer = snapshot.normalizer
        if not normalizer.normalize(query or ""):
            return self._finish("low", started, reason="empty_query", source="none")

        raw_norm = normalizer.normalize(query or "", apply_alias=False)
        norm = normalizer.normalize(query or "")
        if slots is None:
            slots = qa_slots.from_param(subject)
        scope = self._resolve_scope(snapshot, slots)
        self._stats.routes[scope.state] = self._stats.routes.get(scope.state, 0) + 1

        # 学科合法但该学科 0 条：明确回答 subject_empty，**不**兜底成全库（§7.1.4）
        if scope.state == "empty":
            return self._finish(
                "low",
                started,
                reason="subject_empty",
                subject=slots.subject,
                subject_state=scope.state,
                degraded=scope.degraded,
                source="index",
                norm=norm,
            )

        # 用户确认通道优先级最高：用户已经点头，不再比阈值，也不写缓存
        # （一条只对本次点击成立的结论，写进共享缓存等于让别人的点错也生效）
        if selected_faq_id is not None:
            confirmed = self._confirmed_record(snapshot, selected_faq_id)
            if confirmed is None:
                # 用户点的是旧卡片（已下线/被合并掉）：忽略参数走正常链路，**不返回 4xx**
                scope = replace(scope, degraded=scope.degraded + ("faq_selected_invalid",))
                log.info("faq_selected_invalid", faq_id=selected_faq_id)
            else:
                return self._finish(
                    "high",
                    started,
                    record=confirmed,
                    score=1.0,
                    reason="user_confirmed",
                    source="index",
                    subject_state=scope.state,
                    degraded=scope.degraded,
                    norm=norm,
                )

        # ① Redis 二级缓存：多副本共享 + 省掉一次索引计算。
        # key 带 snapshot.version，发布/下线后旧 key 自然不可达，无需遍历删除（见 qa_cache 文档）。
        # key 还带检索范围（filtered/multi/full/empty），否则"限定数学"与"全库"会互相串答案。
        key = self._cache_key(snapshot.version, norm, scope.key_part)
        cached, cache_state = await self._cache_lookup(key)
        # 没有 `tier` 字段 = 旧版本写入的存量条目（滚动发布期间）。一律按 low 处理并**覆盖写**：
        # 猜成 high 等于把没验过的答案直接返回；不覆盖写则每来一次查询都要重算一遍（§7.1.9）。
        force_fill = False
        if cached is not None and cached.tier is None:
            log.info("faq_cache_no_tier", faq_id=cached.faq_id)
            cached = None
            force_fill = True

        if cached is not None:
            cached_tier: Tier = cached.tier if cached.tier is not None else "low"
            if cached_tier == "high":
                record = snapshot.records.get(cached.faq_id) if cached.faq_id is not None else None
                if record is not None:
                    # 高置信正缓存：唯一一条不跑检索、不比重排的路径
                    self._stats.cache_hits += 1
                    if self._cache is not None:
                        await self._cache.slide(key, self._cache_ttl)
                    return self._finish(
                        "high",
                        started,
                        record=record,
                        score=cached.score,
                        reason=f"cache_{cached.reason}",
                        alias_hit=cached.reason == "alias_exact",
                        source="cache",
                        cache_state="hit",
                        subject_state=scope.state,
                        degraded=scope.degraded,
                        norm=norm,
                    )
                # 缓存指向的条目已被删除：当作未命中，走一次索引（顺手把这条脏缓存覆盖掉）
                log.info("faq_cache_stale", faq_id=cached.faq_id, version=snapshot.version)
                force_fill = True
            else:
                # 中/低置信：都**不返回答案**，只回候选；mid 多一个可确认的 top1。
                # 按 entry.tier 复原而不重新判定——同一 query 不能因为 margin 抖动在两档间反复跳。
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
                record = (
                    snapshot.records.get(cached.faq_id)
                    if cached_tier == "mid" and cached.faq_id is not None
                    else None
                )
                return self._finish(
                    cached_tier,
                    started,
                    record=record,
                    reason=f"cached_{cached.reason}",
                    score=cached.score,
                    subject=cached.subject,
                    candidates=cached_candidates,
                    source="cache",
                    cache_state="hit",
                    subject_state=scope.state,
                    degraded=scope.degraded,
                    norm=norm,
                )

        # ② 精确命中（含别名归一结果）：O(1)，最常见的索引路径。
        # 命中必须落在本次检索范围内，否则"限定数学"会直接吐出物理的答案。
        faq_id = snapshot.exact.get(norm)
        if faq_id is not None and not self._exact_in_scope(snapshot, faq_id, scope):
            faq_id = None
        if faq_id is not None:
            via_alias = raw_norm != norm
            if via_alias:
                self._stats.alias_hits += 1
            else:
                self._stats.exact_hits += 1
            reason = "alias_exact" if via_alias else "exact"
            # 精确命中恒为 high（§7.1.9）：归一化串完全相同是字面事实，不受学科来源封顶
            await self._cache_fill(
                key,
                CacheEntry(faq_id=faq_id, score=1.0, reason=reason, tier="high"),
                self._cache_ttl,
                overwrite=force_fill,
            )
            return self._finish(
                "high",
                started,
                record=snapshot.records[faq_id],
                score=1.0,
                reason=reason,
                alias_hit=via_alias,
                source="index",
                cache_state=cache_state,
                subject_state=scope.state,
                degraded=scope.degraded,
                norm=norm,
            )

        # ③ 按检索范围打分：单索引直接比原始分；多学科各算各的、只在归一化空间合并
        tokens = normalizer.tokens(norm)
        if not tokens:
            return self._finish(
                "low", started, reason="empty_tokens", source="none", cache_state=cache_state, norm=norm
            )

        if scope.state == "multi":
            per_subject: dict[str, list[tuple[int, float]]] = {}
            for name, index in scope.indexes:
                ranked_i = sorted(
                    ((doc.faq_id, score) for doc, score in index.search(tokens, top_k=self._top_k)),
                    key=lambda kv: kv[1],
                    reverse=True,
                )[: self._top_k]
                if ranked_i:
                    per_subject[name or "*"] = ranked_i
            result = decide_multi(
                per_subject, self._policy, cross_delta=self._cross_subject_delta
            )
            merged: dict[int, float] = defaultdict(float)
            for ranked_i in per_subject.values():
                for fid, raw in ranked_i:
                    if raw > merged[fid]:
                        merged[fid] = raw
            ranked = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]
        else:
            best: dict[int, float] = defaultdict(float)
            for _, index in scope.indexes:
                for doc, score in index.search(tokens, top_k=self._top_k):
                    if score > best[doc.faq_id]:
                        best[doc.faq_id] = score
            ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[: self._top_k]
            result = decide(ranked, self._policy)

        if not ranked:
            # 一个 token 都没命中：没有候选可给，交给 RAG 全量检索
            await self._cache_fill(
                key,
                CacheEntry(faq_id=None, score=0.0, reason="no_match", tier="low"),
                self._cache_negative_ttl,
                overwrite=force_fill,
            )
            return self._finish(
                "low",
                started,
                reason="no_match",
                subject_state=scope.state,
                degraded=scope.degraded,
                source="index",
                cache_state=cache_state,
                norm=norm,
            )

        # 档位只降不升（§7.1.9）：基准档由 decide() 给，这里只按学科状态往下压。
        # 推断/上下文/冲突/非法四类来源即使分数够也不直接答：交 RAG 或追问（§7.1.8 路由表）
        if result.tier == "high" and slots.subject_source in _NO_DIRECT:
            result = replace(result, tier="mid", reason=f"subject_{slots.subject_source}_handoff")
        # 多学科封顶 mid：跨学科只在归一化分上比，CROSS_SUBJECT_DELTA 没经过精度标定，
        # 不能让这条路径直接作答（精确命中已在上面按 high 返回，不受此限）
        elif result.tier == "high" and scope.state == "multi":
            result = replace(result, tier="mid", reason="multi_scope_handoff")

        top_subject = snapshot.records[result.faq_id].subject if result.faq_id in snapshot.records else None

        # 未命中也要给 RAG 留 top3 扩展线索；mid 档还要靠 top1 做"你是不是想问"的追问对象
        candidates = tuple(
            FAQCandidate(
                faq_id=fid,
                question=snapshot.records[fid].question,
                score=round(raw / (raw + self._policy.norm_k), 4),
            )
            for fid, raw in ranked[: self._policy.candidate_n]
            if fid in snapshot.records
        )

        if result.tier == "low" or result.faq_id is None:
            self._stats.candidates_served += len(candidates)
            await self._cache_fill(
                key,
                CacheEntry(
                    faq_id=None,
                    score=result.score,
                    reason=result.reason,
                    tier="low",
                    subject=top_subject,
                    candidates=tuple((c.faq_id, c.score) for c in candidates),
                ),
                self._cache_negative_ttl,
                overwrite=force_fill,
            )
            return self._finish(
                "low",
                started,
                reason=result.reason,
                score=result.score,
                margin=result.margin,
                candidates=candidates,
                subject=top_subject,
                source="index",
                cache_state=cache_state,
                subject_state=scope.state,
                degraded=scope.degraded,
                norm=norm,
            )

        if result.tier == "mid":
            # 中置信：跑 RAG + 候选，**不返回答案**（§7.1.9 边界 2）。
            # 缓存与负缓存同寿（30 s）：mid 是"待确认"的临时结论，不该活 1 小时。
            self._stats.candidates_served += len(candidates)
            await self._cache_fill(
                key,
                CacheEntry(
                    faq_id=result.faq_id,
                    score=result.score,
                    reason=result.reason,
                    tier="mid",
                    subject=top_subject,
                    candidates=tuple((c.faq_id, c.score) for c in candidates),
                ),
                self._cache_negative_ttl,
                overwrite=force_fill,
            )
            return self._finish(
                "mid",
                started,
                record=snapshot.records.get(result.faq_id),
                score=result.score,
                reason=result.reason,
                margin=result.margin,
                candidates=candidates,
                subject=top_subject,
                source="index",
                cache_state=cache_state,
                subject_state=scope.state,
                degraded=scope.degraded,
                norm=norm,
            )

        await self._cache_fill(
            key,
            CacheEntry(
                faq_id=result.faq_id,
                score=result.score,
                reason=result.reason,
                tier="high",
                subject=top_subject,
            ),
            self._cache_ttl,
            overwrite=force_fill,
        )
        return self._finish(
            "high",
            started,
            record=snapshot.records[result.faq_id],
            score=result.score,
            reason=result.reason,
            margin=result.margin,
            source="index",
            cache_state=cache_state,
            subject_state=scope.state,
            degraded=scope.degraded,
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

        # 标定脚本按"学科之内"看分数：非法学科直接全库（评估要的是排序，不是参数校验）
        slots = qa_slots.from_param(subject)
        if slots.subject_source == "invalid":
            slots = SlotResult()
        scope = self._resolve_scope(snapshot, slots)
        if scope.state == "empty":
            return []

        best: dict[int, float] = defaultdict(float)
        for _, index in scope.indexes:
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
                "faq_rebuild",
                version=snapshot.version,
                docs=len(snapshot.records),
                indexed=snapshot.doc_count,
                merged=snapshot.merged,
                cost_ms=round(cost_ms, 1),
            )
            return {
                "version": snapshot.version,
                "docs": len(snapshot.records),
                "indexed": snapshot.doc_count,
                "merged_duplicates": snapshot.merged,
                "cost_ms": round(cost_ms, 1),
            }

    # ------------------------------------------------------------------ 管理读
    async def get(self, faq_id: int) -> FAQRecord | None:
        """按 id 取单条（管理端排障用，不进请求热路径）。"""
        return await self._repo.fetch_by_id(faq_id)

    def canonical_id(self, faq_id: int) -> int:
        """词袋重复合并后的代表 id；未参与合并的直接返回自身。

        合并只发生在"同学科 + token 多重集相同 + 答案逐字相同"的重复问法之间，
        所以代表 id 与自身是**同一条事实**，调用方（如自检脚本）可以用它判断"查回自己"。
        """
        return self._snapshot.merges.get(faq_id, faq_id)

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
            "docs": len(snapshot.records),          # 载入条数（真相源），不含合并
            "indexed": snapshot.doc_count,          # 实际进 BM25 索引的条数
            "merged_duplicates": snapshot.merged,   # 词袋重复并成别名的条数
            "subjects": subjects,
            "unlabeled": unlabeled,
            "built_at": snapshot.built_at,
        }

    def stats(self) -> dict[str, Any]:
        return self._stats.as_dict()

    # ------------------------------------------------------------------ 内部
    def _cache_key(self, version: int, norm: str, scope: str) -> str:
        """key = faq:v{语料版本}:{检索范围}:{归一化问法指纹}。

        用指纹而不是原文：key 长度稳定、不把学生问法写进 Redis 键空间。
        带版本号是刻意的——发布/下线只需让路由到新版本，旧键靠 TTL 自然淘汰。
        带检索范围是必须的——`数学` 与 `*` 是两次不同的检索，共用一个 key 会串答案。
        """
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:20]
        return f"{self._cache_prefix}:faq:v{version}:{scope}:{digest}"

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

    async def _cache_fill(
        self,
        key: str,
        entry: CacheEntry,
        ttl: int,
        *,
        overwrite: bool = False,
    ) -> None:
        """回填热数据（MySQL/索引命中后写入 Redis）。用单飞避免并发重复写。

        `overwrite=True` 用于覆盖"已知不可用"的存量条目（缺 tier 的旧格式、指向已下线 FAQ 的
        脏条目）：否则这些条目会一直活到绝对寿命上限，每次请求都得重算一遍。
        """
        if self._cache is None:
            return
        lock = await self._single_flight.lock(key)
        async with lock:
            try:
                if overwrite or await self._cache.get(key) is None:
                    await self._cache.put(key, entry, ttl)
                    self._stats.cache_fills += 1
                    log.info("faq_cache_write", key=key, ttl=ttl, tier=entry.tier, overwrite=overwrite)
            except Exception as exc:
                self._stats.cache_errors += 1
                log.warning("faq_cache_error", op="fill", error=str(exc))
        await self._single_flight.release(key)

    def _confirmed_record(self, snapshot: _Snapshot, faq_id: int) -> FAQRecord | None:
        """用户确认通道的入参校验：必须是当前快照里存在的一条（含被合并项归一）。"""
        target = snapshot.merges.get(faq_id, faq_id)
        return snapshot.records.get(target)

    async def _fetch_all(self) -> tuple[list[FAQRecord], list[tuple[int, str]]]:
        records = await self._repo.fetch_published()
        aliases = await self._repo.fetch_aliases() if self._alias_enabled else []
        return records, aliases

    def _resolve_scope(self, snapshot: _Snapshot, slots: SlotResult) -> _Scope:
        """槽位 → 检索范围（CLAUDE.md §7.1.8 路由表）。

        这里是"三种情况不再退化成同一种全库检索"的唯一实现点：
        非法 / 无数据 / 冲突各有各的结论，且都写进 `FAQHit.subject_state` + `degraded`。
        """
        every: tuple[tuple[str | None, BM25Index], ...] = tuple(snapshot.indexes.items())
        source = slots.subject_source

        if source == "invalid":
            # 公开 API 已 400；内部编排兜底成全库，但必须让调用方看得见这是降级
            return _Scope(every, "full", ("subject_invalid_fallback",), "*")
        if source == "conflict":
            return _Scope(every, "full", ("subject_conflict_full_scan",), "*")

        names = slots.scope
        if not names:
            # not_provided / unknown：用户没提学科 = 真的全库，不是降级
            return _Scope(every, "full", (), "*")

        if len(names) > 1:
            picked = tuple((name, snapshot.indexes[name]) for name in names if snapshot.indexes.get(name))
            if not picked:
                return _Scope((), "empty", (), f"empty:{'|'.join(names)}")
            if len(picked) == 1:
                return _Scope(picked, "filtered", (), picked[0][0] or "*")
            return _Scope(
                picked, "multi", (), "multi:" + "|".join(sorted(name or "*" for name, _ in picked))
            )

        subject = names[0]
        index = snapshot.indexes.get(subject)
        if index is None or len(index) == 0:
            return _Scope((), "empty", (), f"empty:{subject}")
        if source in _NO_DIRECT:
            # 推断/上下文：本学科索引 + 全库都看，但结论交 RAG（缓存 key 与纯全库分开）
            return _Scope(every, "filtered", (f"subject_{source}_open_scan",), f"open:{subject}")
        return _Scope(((subject, index),), "filtered", (), subject)

    @staticmethod
    def _exact_in_scope(snapshot: _Snapshot, faq_id: int, scope: _Scope) -> bool:
        """精确命中也要服从检索范围：`exact` 是全局表，不校验就能绕过学科过滤。"""
        record = snapshot.records.get(faq_id)
        if record is None:
            return False
        return any(record.subject == name for name, _ in scope.indexes)

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

        # 先算一次 tokens：合并判定与索引构建共用，避免对 2000 条重复分词
        token_map: dict[int, tuple[str, ...]] = {
            rec.faq_id: tuple(normalizer.tokens(rec.question)) for rec in by_id.values()
        }

        # 语料重复合并（CLAUDE.md §7.1.7 语料重复合并契约）。
        # 同学科 + 答案逐字相同 + 去框架词后 Jaccard ≥0.8 → 保留最小 faq_id 作代表，
        # 其余登记为精确命中别名（原话仍可答）但不进 BM25 索引。
        # 答案不同的高相似对**不合并**，只告警交教研（同一件事写了两份答案）。
        canonical: dict[int, int] = {}
        if self._dedupe_near_dup:
            canonical, conflicts = _plan_near_duplicate_merges(by_id, token_map)
            for a, b in conflicts:
                log.warning(
                    "faq_near_dup_answer_conflict",
                    faq_ids=[a, b],
                    questions=[by_id[a].question, by_id[b].question],
                    hint="问法几乎相同但答案不同，不合并，需教研核对",
                )
            if canonical:
                log.info("faq_near_dup_merged", merged=len(canonical), hint="重复问法并成别名，不进 BM25 索引")

        exact: dict[str, int] = {}
        grouped: dict[str | None, list[IndexedDoc]] = defaultdict(list)
        for rec in by_id.values():
            norm = normalizer.normalize(rec.question)
            if not norm:
                continue
            target = canonical.get(rec.faq_id, rec.faq_id)
            exact.setdefault(norm, target)
            if target != rec.faq_id:  # 被合并的重复项只作精确命中别名
                continue
            tokens = token_map[rec.faq_id]
            if not tokens:
                continue
            grouped[rec.subject].append(
                IndexedDoc(faq_id=rec.faq_id, question=rec.question, subject=rec.subject, tokens=tokens)
            )

        indexes = {subject: BM25Index(docs) for subject, docs in grouped.items()}
        # 学科名认不出来只报告、不自动加白：否则一个错别字就成了合法学科（§7.1.8）
        unlabeled = sum(1 for rec in by_id.values() if rec.subject is None)
        if unlabeled:
            log.warning(
                "faq_subject_unknown",
                count=unlabeled,
                total=len(by_id),
                hint="学科名不在 SUBJECTS/SUBJECT_ALIASES，按无学科入库",
            )
        return _Snapshot(
            version=version,
            indexes=indexes,
            exact=exact,
            records=by_id,
            normalizer=normalizer,
            built_at=time.time(),
            merges=canonical,
        )

    def _finish(
        self,
        tier: Tier,
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
        margin: float = 0.0,
        subject_state: SubjectState = "full",
        degraded: tuple[str, ...] = (),
    ) -> FAQHit:
        """统一的出口：`tier` 是唯一入参，`hit` 由它派生（§7.1.9）。"""
        latency_ms = (time.perf_counter() - started) * 1000
        self._stats.searches += 1
        self._stats.total_latency_ms += latency_ms
        self._stats.tiers[tier] = self._stats.tiers.get(tier, 0) + 1
        if tier == "high":
            self._stats.hits += 1
        else:
            self._stats.misses += 1

        # 每次查询一行结论日志：命中来自缓存/索引、命中哪条、档位、分数、耗时。
        # 排查"为什么答错"时这一行是唯一现场；非高置信再补一行"转交 RAG"，便于按行 grep 统计交接率。
        source_cn = _SOURCE_CN.get(source, source or "-")
        q_text = (
            f"问题: [{norm}]"
            if (self._log_query and norm)
            else f"q_hash: {hashlib.sha1(norm.encode('utf-8')).hexdigest()[:12] if norm else '-'}"
        )
        common: dict[str, Any] = {
            "source_cn": source_cn,
            "q_text": q_text,
            "tier": tier,
            "score": round(score, 4),
            "latency_ms": round(latency_ms, 2),
        }
        if tier == "high" and record is not None:
            log.info(
                "faq_search_hit",
                faq_id=record.faq_id,
                subject=record.subject or "-",
                **common,
            )
        else:
            missing = {
                "reason": reason,
                "threshold": self._policy.threshold,
                "margin": round(margin, 4),
                "candidates": [c.faq_id for c in candidates] or "-",
            }
            log.info("faq_search_miss", subject=subject or "-", **common, **missing)
            log.info(
                "faq_handoff_rag",
                q_text=q_text,
                tier=tier,
                candidates=[c.question for c in candidates] or "-",
            )

        if record is not None:
            return FAQHit(
                tier=tier,
                faq_id=record.faq_id,
                # 中置信**不返回答案**：只给候选问题让用户确认，正文必须走 RAG 或确认通道。
                # 把 FAQ 原文漏给生成器，模型会照抄，一条错答案立刻变成"带引用的事实"（§7.1.9）。
                answer=record.answer if tier == "high" else None,
                score=score,
                subject=record.subject,
                matched_question=record.question,
                alias_hit=alias_hit,
                candidates=candidates,
                reason=reason,
                latency_ms=latency_ms,
                source=source,
                cache=cache_state,
                subject_state=subject_state,
                degraded=degraded,
            )
        return FAQHit(
            tier=tier,
            score=score,
            subject=subject,
            candidates=candidates,
            reason=reason,
            latency_ms=latency_ms,
            source=source,
            cache=cache_state,
            subject_state=subject_state,
            degraded=degraded,
        )

