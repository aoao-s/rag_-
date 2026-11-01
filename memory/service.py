"""记忆服务（CLAUDE.md §7.9）：读（recall）/ 写（remember）/ 忘（forget）三件事。

三条边界写在这里，因为它们是本模块唯一会"越界"的地方：

1. **记忆不进事实性回答**：`recall()` 只产出 `MemoryContext`，它是 prompt 的背景段，
   不参与引用对齐、不进 `citations`（§7.3 第 3 条与 §7.7 作用点 7）。
2. **`tenant_id + user_id` 唯一**：所有接口第一个参数都是 `user_id`，仓库层再做一次
   归属校验——不靠调用方自觉（跨用户读写记忆是本模块最严重的事故）。
3. **召回失败必须降级不能阻塞**：`MEM_RECALL_BUDGET_MS` 到点就返回空记忆 + `degraded=True`，
   主链路继续跑检索。记忆是加分项，不是前置条件。

写入是**同步等待**的（`remember()` 在回答之后调用，不在首 token 之前），
没有引入 Redis Stream：本机的吞吐量下多一个中间件只会多一类故障（§7.9 的
"异步消费"是交付形态，先保证正确性）。调用方在 `orchestration/chat.py` 里
`asyncio.create_task` 把它挂到后台，不阻塞响应。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from typing import Any, Literal, Protocol

from base.config import Settings
from base.contracts import MemoryContext
from base.logger import get_logger
from memory.extractor import Candidate, extract
from memory.session import SessionStore
from memory.store import (
    Atom,
    AtomIndex,
    AtomRepo,
    ChatLogRepo,
    Grant,
    GrantRepo,
    Profile,
    ProfileRepo,
    mark_deleted,
    mark_superseded,
    new_atom_id,
    new_grant_id,
    revive,
)

log = get_logger(__name__)

ForgetMode = Literal["revive", "purge_chain"]

# 记忆 token 预算（§3.7 第 3 条）：画像 300 + 摘要 400 + 原子 300，合计 ≤1 000
_PROFILE_KINDS = ("identity",)


class MemoryPermissionError(PermissionError):
    """记忆越权的共同基类。继承 `PermissionError`，不认识的调用方也能 catch 住。"""


class MemoryAccessDenied(MemoryPermissionError):
    """读别人的记忆却没有有效授权 —— 对外一律映射成 **404**。

    不用 403 是有意的：403 等于告诉对方"这条记忆存在，只是你没权看"，
    而记忆的存在本身就是隐私（§3.7 第 7 条）。呼应用户那句话——
    **别人的记忆一律当"不存在"**。
    """


class MemoryReadOnly(MemoryPermissionError):
    """顾问持有只读授权却尝试写入 —— 对外 **403**。

    这里可以用 403 而不是 404：对方已经拿到了读权限，记忆存在这件事他本来就知道，
    403 只是把"只能看"这条边界说出来，不额外泄漏任何东西。
    """


class _Auditable(Protocol):
    def audit(self, atom_id: str, action: str, user_id: str, detail: str = ..., operator: str = ...) -> None: ...


class MemoryService:
    def __init__(
        self,
        settings: Settings,
        repo: AtomRepo,
        session: SessionStore,
        chat_log: ChatLogRepo | None = None,
        grants: GrantRepo | None = None,
        *,
        index: AtomIndex | None = None,
        profiles: ProfileRepo | None = None,
    ) -> None:
        self.settings = settings
        self.repo = repo
        self.session = session
        self.chat_log = chat_log
        # `grants=None` 不是"人人都能读"，是"这个实例不提供任何授权"
        # ——见 `authorize_read`：没有授权仓库时，非本人一律不放行（默认拒绝）。
        self.grants = grants
        # `index=None` 是"没有向量索引"（§7.9 第 1 条的已知未做），召回退回字面排序；
        # `profiles=None` 是"没有画像表"，画像退回从身份类原子记忆推导。
        # 两者都用 None 而不是抛错：内存后端（单测、离线脚本）本来就没有这两样东西。
        self.index = index
        self.profiles = profiles

    # ---- 会话归属（阶段 1 §3.4）----
    # 这两条是**薄转发**，不是新业务：归属存在 `SessionState.owner_id` 里（与窗口同生命周期），
    # 编排层只跟 `MemoryService` 打交道，不直接伸手进 store——少一处能绕过权限的地方。
    async def session_owner(self, sid: str) -> str | None:
        return await self.session.owner_of(sid)

    async def claim_session(self, sid: str, user_id: str) -> str:
        return await self.session.claim(sid, user_id)

    # ---- 顾问只读授权（§3.7 第 7 条 / M4）----
    async def grant_read(
        self,
        owner_id: str,
        grantee_id: str,
        *,
        expires_at: float | None = None,
        granted_by: str = "",
    ) -> dict[str, Any]:
        """学员把**读**权限授予某个顾问。重复授权是覆盖（同一个 id，状态回到 active）。

        为什么"重复授权"必须能覆盖：撤销过的授权如果只能新增一行，`uk_owner_grantee`
        会让第二次插入直接报主键冲突，前端看到的是 500 而不是"已重新授权"。
        """
        if self.grants is None:
            raise RuntimeError("授权仓库未配置（MemoryService(grants=...)）")
        owner = (owner_id or "").strip()
        grantee = (grantee_id or "").strip()
        if not owner or not grantee:
            raise ValueError("owner 与 grantee 都不能为空")
        if owner == grantee:
            # 给自己授权没有意义，而且会让授权表里出现"顾问"的假象：
            # 之后任何人看 `memory_grant` 都会以为有一条真实的委派关系。
            raise ValueError("不能给自己授权")

        existing = await self._grant_of(owner, grantee)
        grant = Grant(
            id=existing.id if existing else new_grant_id(),
            owner_id=owner,
            grantee_id=grantee,
            scope="read",
            status="active",
            granted_by=granted_by or owner,
            granted_at=time.time(),
            expires_at=expires_at,
            revoked_at=None,
        )
        await asyncio.to_thread(self.grants.upsert, grant)
        self._audit("-", "grant_read", owner, f"grantee={grantee} expires_at={expires_at}", operator=granted_by or owner)
        log.info("memory_grant_read", user_id=owner, grantee=grantee, expires_at=expires_at)
        return grant.to_dict()

    async def revoke_read(self, owner_id: str, grantee_id: str) -> dict[str, Any]:
        """撤销授权。撤销是**改状态**而不是删行：删掉之后"曾经授权过给谁"就查不到了。"""
        if self.grants is None:
            raise RuntimeError("授权仓库未配置（MemoryService(grants=...)）")
        grant = await self._grant_of(owner_id, grantee_id)
        if grant is None:
            return {"revoked": False, "reason": "not_found"}
        if grant.revoked_at is not None:
            return {"revoked": False, "reason": "already_revoked"}

        revoked = replace(grant, status="revoked", revoked_at=time.time())
        await asyncio.to_thread(self.grants.upsert, revoked)
        self._audit("-", "revoke_read", owner_id, f"grantee={grantee_id}", operator=owner_id)
        log.info("memory_grant_revoke", user_id=owner_id, grantee=grantee_id)
        return {"revoked": True, "grantee_id": grantee_id, "revoked_at": revoked.revoked_at}

    async def list_grants(self, user_id: str, *, direction: str = "owner") -> dict[str, Any]:
        """`direction="owner"` 看"我把记忆授权给了谁"，`"grantee"` 看"我能看谁的记忆"。"""
        if self.grants is None:
            return {"total": 0, "items": []}
        fetch = self.grants.list_by_owner if direction == "owner" else self.grants.list_by_grantee
        rows = await asyncio.to_thread(fetch, user_id)
        return {"total": len(rows), "items": [g.to_dict() for g in rows]}

    async def authorize_read(self, viewer: str, owner: str, *, now: float | None = None) -> bool:
        """`viewer` 能否读 `owner` 的记忆。**本人永远可以**，其余必须持有效 grant。

        默认拒绝（没有授权仓库、没有 grant、grant 过期或已撤销 —— 全部 False）。
        过期判定放在 `Grant.active_at` 里而不是这里，是为了它单独可测：
        读路径上的日期比较写错，是那种"上线半年后某个顾问还能看到上一届学员记忆"的错。
        """
        if not viewer or not owner:
            return False
        if viewer == owner:
            return True
        grant = await self._grant_of(owner, viewer)
        return bool(grant and grant.active_at(now if now is not None else time.time()))

    # ---- 读 ----
    async def recall(
        self, user_id: str, query: str, *, sid: str | None = None, budget_ms: int | None = None
    ) -> MemoryContext:
        """并行取「摘要 + 原子记忆」，按预算裁剪成 `MemoryContext`。

        超预算不是异常路径，是**正常路径的一种**：记忆缺一段，模型只知道知识库也能答。
        所以这里返回 `degraded=True` 而不是抛错——抛错会让上层要么写 try/except、
        要么整体失败，两种都比"少一段背景"糟糕。
        """
        if not self.settings.feature_memory or not user_id:
            return MemoryContext()

        budget = budget_ms if budget_ms is not None else self.settings.mem_recall_budget_ms
        try:
            return await asyncio.wait_for(
                self._recall_inner(user_id, query, sid), timeout=max(budget, 1) / 1000
            )
        except TimeoutError:
            log.warning("memory_recall_timeout", budget_ms=budget)
            return MemoryContext(degraded=True, extra={"reason": "timeout"})
        except Exception as exc:  # 记忆挂掉不能打挂问答
            log.warning("memory_recall_failed", error=str(exc))
            return MemoryContext(degraded=True, extra={"reason": "error"})

    async def _recall_inner(self, user_id: str, query: str, sid: str | None) -> MemoryContext:
        summary = await self.session.summary(sid) if sid else ""
        atoms = await asyncio.to_thread(self.repo.list_alive, user_id)
        atoms = [a for a in atoms if a.confidence >= self.settings.mem_min_confidence]

        # 画像优先用**存在表里的那一份**（用户/顾问确认过的）；没有才从身份类原子记忆推导。
        # 顺序反过来会让"我改好的画像"被一条记忆覆盖回推导值——页面上看得见、召回里看不见。
        stored = await asyncio.to_thread(self._stored_profile, user_id)
        profile = stored or self._profile_text([a for a in atoms if a.fact_type in _PROFILE_KINDS])
        recalled = await self._recall_order(atoms, user_id, query)

        def _trim(text: str, limit: int) -> str:
            return text[:limit] if len(text) > limit else text

        profile = _trim(profile, self.settings.mem_profile_max_tokens)
        summary = _trim(summary, self.settings.mem_summary_max_tokens)

        picked: list[str] = []
        used = 0
        for atom in recalled:
            if used + len(atom.content) > self.settings.mem_atom_max_tokens:
                break
            picked.append(atom.content)
            used += len(atom.content)
            if len(picked) >= self.settings.mem_recall_top_k:
                break

        return MemoryContext(
            profile=profile,
            summary=summary,
            atoms=tuple(picked),
            tokens=len(profile) + len(summary) + used,
        )

    async def _recall_order(self, atoms: list[Atom], user_id: str, query: str) -> list[Atom]:
        """召回顺序：向量 top_k 在前，其余按字面排序补齐。

        **`None` 与 `[]` 必须分开**（`AtomIndex.search` 的契约）：
        `None` = 索引不可用，整批退化成字面排序；`[]` = 索引是好的、只是没命中，
        那时把字面排序的全部结果排在后面 —— 索引没命中不等于这个用户没有记忆。

        补齐还有第二个作用：索引是副本，写入失败、老数据没回填、刚写完还没落索引
        都会让索引缺行。**缺行不许表现为"这条记忆消失"**——它是用户说的话，
        召回少一条就是模型少知道一件事，而这种缺失在日志里看不出来。
        """
        baseline = self._rank(atoms, query)
        if self.index is None or not atoms:
            return baseline

        ids = await self._index_search(user_id, query)
        if ids is None:
            return baseline

        by_id = {a.id: a for a in atoms}
        head = [by_id[i] for i in ids if i in by_id]
        seen = {a.id for a in head}
        return head + [a for a in baseline if a.id not in seen]

    async def _index_search(self, user_id: str, query: str) -> list[str] | None:
        """向量检索的**唯一**调用点，把"索引抛异常"也收敛成 `None`。

        `MilvusAtomIndex.search` 自己就吞异常返回 `None`，但 Protocol 的契约不该
        靠具体实现自觉：这里再兜一层，坏掉的索引最多让召回退化成字面排序，
        而不是把整次召回（连带画像与摘要）打成一个 `degraded` 的空记忆。
        """
        try:
            return await asyncio.to_thread(
                self.index.search, user_id, query, self.settings.mem_recall_top_k
            )
        except Exception as exc:  # 索引是副本，坏了只该退化，不该连坐
            log.warning("memory_vector_search_failed", error=str(exc)[:200])
            return None

    def _stored_profile(self, user_id: str) -> str:
        if self.profiles is None:
            return ""
        profile = self.profiles.get(user_id)
        return (profile.content if profile else "") or ""

    def _profile_text(self, atoms: list[Atom]) -> str:
        """画像 = 身份类事实的稳定表述，常驻 300 token（§3.7 第 1 条 L3）。"""
        return "；".join(a.content for a in atoms)

    def _rank(self, atoms: list[Atom], query: str) -> list[Atom]:
        """按"与当前问句的字面相关性 → 被提到的次数 → 新旧"排序。

        §3.7 第 8 条的向量召回由 `MilvusAtomIndex` 负责（`_recall_order`），这里是它
        **不可用时的退路**：没有索引、索引报错、内存后端下都要有确定性的顺序。
        保留它还有一层意思——它对同一输入永远给出同一结果，单测能断言；
        向量版做不到"因为我换了个 embedding 版本所以测试挂了"这种噪声。
        """
        if not atoms:
            return []

        def score(atom: Atom) -> tuple[int, int, float]:
            overlap = len(_bigrams(atom.content) & _bigrams(query))
            return (overlap, min(atom.mention_count, 9), atom.valid_from)

        return sorted(atoms, key=score, reverse=True)

    # ---- 写 ----
    async def remember(
        self, user_id: str, sid: str, query: str, *, turn_index: int = 0
    ) -> list[Atom]:
        """门控 → 抽取 → 去重 → 冲突消解 → 落库。返回本轮新增/更新的原子。

        只有 `gated()` 通过的轮次才动数据库：知识型问答里 90% 的轮次不含任何
        用户属性，每次都去查一遍 `memory_atom` 是纯粹的浪费。
        """
        if not self.settings.feature_memory or not user_id:
            return []
        candidates = extract(query)
        if not candidates:
            return []

        saved: list[Atom] = []
        for cand in candidates:
            saved.append(await self._upsert(user_id, sid, cand, turn_index))
        log.info(
            "memory_write",
            user_id=user_id,
            session_id=sid,
            n=len(saved),
            keys=[a.fact_key for a in saved],
        )
        return saved

    async def _upsert(self, user_id: str, sid: str, cand: Candidate, turn_index: int) -> Atom:
        existing = await asyncio.to_thread(self.repo.list_alive, user_id)
        same_key = [a for a in existing if a.fact_key == cand.fact_key]
        now = time.time()

        # ① 去重：同槽位且内容几乎相同 → 合并进已有条目（mention_count+1）
        for old in same_key:
            if _similar(old.content, cand.content) >= self.settings.mem_dedupe_sim:
                merged = _merge(old, cand)
                await asyncio.to_thread(self.repo.update, merged)
                await self._sync_index(merged)
                self._audit(merged.id, "merge", user_id, cand.source_span)
                return merged

        # ② 冲突：同槽位不同值 → 新者胜，旧者 supersede（§7.3.2 冲突行）。
        # 同一槽位的全部旧条目共用一个链根，`revive` 才能顺着链找到前驱
        root = next((a.chain_root_id or a.id for a in same_key), None)
        atom = Atom(
            id=new_atom_id(),
            user_id=user_id,
            fact_type=cand.fact_type,
            fact_key=cand.fact_key,
            content=cand.content,
            confidence=cand.confidence,
            mention_count=1,
            source_sid=sid,
            source_turn=turn_index,
            source_span=cand.source_span,
            valid_from=now,
            chain_root_id=root,
            subject=cand.subject,
            grade=cand.grade,
        )
        for old in same_key:
            superseded = mark_superseded(old, atom.id, now)
            await asyncio.to_thread(self.repo.update, superseded)
            await self._sync_index(superseded)
            self._audit(old.id, "supersede", user_id, f"→ {atom.id}")

        if await asyncio.to_thread(self.repo.count_alive, user_id) >= self.settings.mem_atom_limit_per_user:
            # 上限到了：**不淘汰**，只拒绝新增。旧事实可能正是用户care的，
            # 让用户去记忆页删——静默淘汰会让人"记忆凭空消失"且找不到原因。
            log.warning("memory_atom_limit", user_id=user_id, limit=self.settings.mem_atom_limit_per_user)
            return atom

        await asyncio.to_thread(self.repo.insert, atom)
        await self._sync_index(atom)
        self._audit(atom.id, "insert", user_id, cand.source_span)
        return atom

    # ---- 记忆页 ----
    async def list_atoms(
        self,
        user_id: str,
        *,
        fact_type: str | None = None,
        subject: str | None = None,
        grade: str | None = None,
        q: str | None = None,
        page: int = 1,
        page_size: int = 20,
        include_dead: bool = False,
        viewer: str | None = None,
    ) -> dict[str, Any]:
        viewer = viewer or user_id
        await self._require_read(viewer, user_id)
        rows = await asyncio.to_thread(self.repo.list_alive, user_id)
        if include_dead:
            rows = await asyncio.to_thread(self.repo.list_all, user_id)
        if fact_type:
            rows = [a for a in rows if a.fact_type == fact_type]
        # 学科/年级在内存里筛而不是让 repo 加参数：记忆每人最多 500 条（`MEM_ATOM_LIMIT_PER_USER`），
        # 一次全取回来过滤比在 repo 里长出四个组合 SQL 便宜得多。
        # 传空串是**合法查询**（筛未归类的），所以判据是 `is not None` 而不是 `if subject`。
        if subject is not None:
            rows = [a for a in rows if a.subject == subject]
        if grade is not None:
            rows = [a for a in rows if a.grade == grade]
        if q:
            rows = [a for a in rows if q in a.content]
        rows.sort(key=lambda a: a.valid_from, reverse=True)
        start = max(page - 1, 0) * page_size
        self._note_advisor_read(viewer, user_id, f"scope=atoms n={len(rows)}")
        return {
            "total": len(rows),
            "page": page,
            "page_size": page_size,
            "items": [a.to_dict() for a in rows[start : start + page_size]],
        }

    async def patch(
        self, user_id: str, atom_id: str, *, content: str | None = None, confidence: float | None = None,
        actor: str | None = None,
    ) -> dict[str, Any] | None:
        await self._guard_write(actor or user_id, user_id)
        atom = await asyncio.to_thread(self.repo.get, atom_id)
        if atom is None or atom.user_id != user_id:
            return None  # 别人的记忆一律当"不存在"，不区分 404 与 403
        updated = Atom(
            **{
                **atom.to_dict(),
                "content": (content or atom.content)[:500],
                "confidence": round(min(max(confidence if confidence is not None else atom.confidence, 0.0), 1.0), 2),
            }
        )
        await asyncio.to_thread(self.repo.update, updated)
        await self._sync_index(updated)
        self._audit(atom_id, "update", user_id, updated.content)
        return updated.to_dict()

    async def forget(
        self, user_id: str, atom_id: str, mode: ForgetMode, *, actor: str | None = None
    ) -> dict[str, Any]:
        """删除一条记忆。`mode` 必须显式给，且语义不同（§3.7 第 5 条）：

        - `revive`：删掉链上最新一条，恢复上一条生效（用户说"记错了，之前那条才对"）；
        - `purge_chain`：整链失效（用户说"这件事别再提了"）。

        两种都是**软删**（`deleted_at`），30 天后由清理任务物理删除。
        """
        await self._guard_write(actor or user_id, user_id)
        atom = await asyncio.to_thread(self.repo.get, atom_id)
        if atom is None or atom.user_id != user_id:
            return {"deleted": False, "reason": "not_found", "mode": mode}
        now = time.time()

        if mode == "purge_chain":
            root = atom.chain_root_id or atom.id
            chain = await asyncio.to_thread(self.repo.chain, root)
            for item in chain:
                if item.deleted_at is None:
                    deleted = mark_deleted(item, now)
                    await asyncio.to_thread(self.repo.update, deleted)
                    await self._sync_index(deleted)
            self._audit(atom_id, "purge_chain", user_id, f"root={root} n={len(chain)}")
            return {"deleted": True, "mode": mode, "chain_size": len(chain)}

        # revive：把"被它取代的那条"恢复；没有前驱就是普通删除
        root = atom.chain_root_id or atom.id
        chain = await asyncio.to_thread(self.repo.chain, root)
        restored: list[str] = []
        for item in chain:
            if item.superseded_by == atom.id and item.deleted_at is None:
                revived = revive(item)
                await asyncio.to_thread(self.repo.update, revived)
                await self._sync_index(revived)
                restored.append(item.id)
        deleted = mark_deleted(atom, now)
        await asyncio.to_thread(self.repo.update, deleted)
        await self._sync_index(deleted)
        self._audit(atom_id, "revive", user_id, f"restored={restored}")
        return {"deleted": True, "mode": mode, "restored": restored}

    async def chain(self, user_id: str, atom_id: str, *, viewer: str | None = None) -> dict[str, Any]:
        viewer = viewer or user_id
        await self._require_read(viewer, user_id)
        atom = await asyncio.to_thread(self.repo.get, atom_id)
        if atom is None or atom.user_id != user_id:
            return {"items": []}
        root = atom.chain_root_id or atom.id
        items = await asyncio.to_thread(self.repo.chain, root)
        self._note_advisor_read(viewer, user_id, f"scope=chain n={len(items)}")
        return {"root": root, "items": [a.to_dict() for a in items]}

    async def clear_all(self, user_id: str, *, actor: str | None = None) -> int:
        await self._guard_write(actor or user_id, user_id)
        rows = await asyncio.to_thread(self.repo.list_alive, user_id)
        now = time.time()
        for atom in rows:
            deleted = mark_deleted(atom, now)
            await asyncio.to_thread(self.repo.update, deleted)
            await self._sync_index(deleted)
        # 「清空」是人说的那句话：他要求删掉的是**关于他的全部**，不含例外。
        # 画像单独留一份下来是最容易被忽略的一种"没删干净"。
        if self.profiles is not None:
            await asyncio.to_thread(self.profiles.delete, user_id)
        self._audit("-", "clear_all", user_id, f"n={len(rows)}")
        log.info("memory_clear_all", user_id=user_id, n=len(rows))
        return len(rows)

    async def export(self, user_id: str, *, viewer: str | None = None) -> dict[str, Any]:
        viewer = viewer or user_id
        await self._require_read(viewer, user_id)
        rows = await asyncio.to_thread(self.repo.list_alive, user_id)
        self._note_advisor_read(viewer, user_id, f"scope=export n={len(rows)}")
        return {
            "user_id": user_id,
            "exported_at": time.time(),
            # 画像是记忆的一部分，导出时不能只导原子记忆——用户拿走的是"关于我的全部"。
            "profile": await self._profile_payload(user_id),
            "atoms": [a.to_dict() for a in rows],
        }

    # ---- L3 用户画像（§3.7 第 1 条 / §7.9.1）----
    async def get_profile(self, user_id: str, *, viewer: str | None = None) -> dict[str, Any]:
        viewer = viewer or user_id
        await self._require_read(viewer, user_id)
        self._note_advisor_read(viewer, user_id, "scope=profile")
        return await self._profile_payload(user_id)

    async def _profile_payload(self, user_id: str) -> dict[str, Any]:
        """取画像。**有存表的一份就用它**，否则从身份类原子记忆推导（`source="derived"`）。

        推导的那份不落库：它随原子记忆变，落库之后就出现"改了一条记忆、画像没跟着变"
        的假象；而两处都要读的一致性包袱，比每次拼三个字符串贵得多。

        读权限不在这里判（`get_profile` / `export` 各自已经判过）——同一份数据
        两个入口各判一次，第二次判的就只是一个"看起来更安全"的重复。
        """
        stored = await asyncio.to_thread(self._stored, user_id)
        if stored is not None and stored.content.strip():
            return {**stored.to_dict(), "tokens": len(stored.content)}
        atoms = await asyncio.to_thread(self.repo.list_alive, user_id)
        derived = self._profile_text(
            [
                a
                for a in atoms
                if a.fact_type in _PROFILE_KINDS
                and a.confidence >= self.settings.mem_min_confidence
            ]
        )
        return {
            "user_id": user_id,
            "content": derived,
            "source": "derived",
            "updated_at": 0.0,
            "updated_by": "",
            "tokens": len(derived),
        }

    async def set_profile(
        self, user_id: str, content: str | None, *, actor: str | None = None
    ) -> dict[str, Any]:
        """人工写画像（用户本人或经授权的人）。内容超预算就截断——**不报错**。

        截断而不是拒绝，是因为画像是"常驻提示词"的一段，多写几个字没必要让用户
        去算 token；而超过 300 的部分本来也进不了 prompt，留在库里只会让页面显示的
        和模型看到的不是同一段。空内容等于清除（用户把框清空再保存的唯一合理语义）。
        """
        actor_id = actor or user_id
        await self._guard_write(actor_id, user_id)
        text = (content or "").strip()
        if not text:
            return await self.clear_profile(user_id, actor=actor_id)
        if self.profiles is None:
            raise RuntimeError("画像仓库未配置（MemoryService(profiles=...)）")
        capped = text[: self.settings.mem_profile_max_tokens]
        profile = Profile(
            user_id=user_id,
            content=capped,
            source="manual",
            updated_at=time.time(),
            updated_by=actor_id,
        )
        await asyncio.to_thread(self.profiles.upsert, profile)
        self._audit("-", "profile_set", user_id, capped[:200], operator=actor_id)
        log.info("memory_profile_set", user_id=user_id, chars=len(capped), operator=actor_id)
        return {**profile.to_dict(), "tokens": len(capped)}

    async def clear_profile(self, user_id: str, *, actor: str | None = None) -> dict[str, Any]:
        """删掉人工画像，召回自动退回"从身份类原子记忆推导"那一档。

        它删的是**画像这一张卡**，不动原子记忆——用户想删记忆本身走 `forget` 或 `clear_all`。
        两者混成一个动作，会让"我只想改画像"变成一次不可撤销的数据删除。
        """
        actor_id = actor or user_id
        await self._guard_write(actor_id, user_id)
        if self.profiles is not None:
            await asyncio.to_thread(self.profiles.delete, user_id)
        self._audit("-", "profile_clear", user_id, operator=actor_id)
        log.info("memory_profile_clear", user_id=user_id, operator=actor_id)
        return {"cleared": True, "user_id": user_id}

    def _stored(self, user_id: str) -> Profile | None:
        return self.profiles.get(user_id) if self.profiles is not None else None

    # ---- 内部 ----
    async def _sync_index(self, atom: Atom) -> None:
        """把一次写入同步到向量索引：活着的更新，失效/删除的**从索引里删掉**。

        删而不是把 `valid_to` 改成"现在"：索引是副本，少一行比留一行靠过滤条件
        挡着更难出错——过滤条件写错一次，已删除的记忆就会重新出现在回答里。

        **异常就地吞掉**：MySQL 才是真相源，索引挂了不该让一次成功的记忆写入变成失败。
        `MilvusAtomIndex` 自己就吞，但这里再兜一层——Protocol 的契约不靠具体实现自觉。
        """
        if self.index is None:
            return
        try:
            if atom.alive:
                await asyncio.to_thread(self.index.upsert, atom)
            else:
                await asyncio.to_thread(self.index.delete, atom.id)
        except Exception as exc:
            event = "memory_vector_upsert_failed" if atom.alive else "memory_vector_delete_failed"
            log.warning(event, atom_id=atom.id, error=str(exc)[:200])

    async def _grant_of(self, owner_id: str, grantee_id: str) -> Grant | None:
        if self.grants is None:
            return None
        return await asyncio.to_thread(self.grants.get, owner_id, grantee_id)

    async def _require_read(self, viewer: str, owner: str) -> None:
        """读别人的记忆前的唯一闸门。本人直接放行（`authorize_read` 的第一条）。"""
        if await self.authorize_read(viewer, owner):
            return
        # 拒绝也要留痕：被拒的读取尝试和成功的读取一样，都是"谁在碰这条记忆"的信息。
        self._audit("-", "advisor_read_denied", owner, f"viewer={viewer}", operator=viewer)
        log.warning("memory_grant_denied", user_id=owner, viewer=viewer, action="read")
        raise MemoryAccessDenied(f"{viewer} 没有读取 {owner} 记忆的授权")

    async def _guard_write(self, actor: str, owner: str) -> None:
        """写别人的记忆前的唯一闸门：无 grant → 404，有 grant 也只能看 → 403。"""
        if actor == owner:
            return
        grant = await self._grant_of(owner, actor)
        if grant is None or not grant.active_at(time.time()):
            self._audit("-", "advisor_write_denied", owner, f"viewer={actor}", operator=actor)
            log.warning("memory_grant_denied", user_id=owner, viewer=actor, action="write")
            raise MemoryAccessDenied(f"{actor} 没有访问 {owner} 记忆的权限")
        self._audit("-", "advisor_write_denied", owner, f"viewer={actor} scope={grant.scope}", operator=actor)
        log.warning("memory_grant_denied", user_id=owner, viewer=actor, action="write_readonly")
        raise MemoryReadOnly("顾问只有只读授权，不能修改或删除学员记忆")

    def _note_advisor_read(self, viewer: str, owner: str, detail: str) -> None:
        """顾问成功读到记忆 —— 学员必须能在审计里看到"谁看过我的记忆"。"""
        if viewer == owner:
            return
        self._audit("-", "advisor_read", owner, f"viewer={viewer} {detail}", operator=viewer)
        log.info("memory_advisor_read", user_id=owner, viewer=viewer, detail=detail)

    def _audit(
        self, atom_id: str, action: str, user_id: str, detail: str = "", operator: str = ""
    ) -> None:
        audit = getattr(self.repo, "audit", None)
        if callable(audit):
            audit(atom_id, action, user_id, detail, operator)


def _bigrams(text: str) -> set[str]:
    text = "".join(text.split())
    return {text[i : i + 2] for i in range(len(text) - 1)} if len(text) > 1 else {text}


def _similar(a: str, b: str) -> float:
    """字符二元组 Jaccard。是"向量相似度 >0.92"的确定性替代（同 `_rank` 的理由）。

    刻意留一个已知缺口：改写幅度大的同义事实（"数学不好" vs "数学是我的弱项"）
    相似度只有 0.3 左右，会被当成两条新事实各插一行。这比反向错误（把两件不同的
    事合并成一条）安全——合并错了是**丢信息**，多一条只是画像里啰嗦。
    """
    x, y = _bigrams(a), _bigrams(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def _merge(old: Atom, cand: Candidate) -> Atom:
    """合并：保留信息量大的表述（更长的那个），被提到次数 +1，置信取高。

    学科/年级用 `old.x or cand.x` 而不是"新的覆盖旧的"：`我上初三` 和 `我数学不好`
    是同一个人说的两句话，合并后该同时带着年级与学科；任一侧为空就用另一侧补上。
    **不做冲突判定**（"我上初三" vs "我上高一" 是不同槽位的事实，本就走 supersede 那条路）。
    """
    longer = old.content if len(old.content) >= len(cand.content) else cand.content
    return Atom(
        **{
            **old.to_dict(),
            "content": longer,
            "mention_count": old.mention_count + 1,
            "confidence": max(old.confidence, cand.confidence),
            "subject": old.subject or cand.subject,
            "grade": old.grade or cand.grade,
        }
    )
