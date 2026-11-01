"""记忆模块单测（CLAUDE.md §7.9 / §7.3）。

四条最容易出事的性质，逐条对应一条用例族：
1. **不污染**：知识型问句不产生记忆，助手回答不产生记忆；
2. **不串用户**：跨用户读写一律当"不存在"，不区分 404 与 403；
3. **不消失**：删除必须显式选 `revive` 还是 `purge_chain`，两种语义分开验；
4. **不阻塞**：召回超时降级为空记忆，而不是把问答打挂。

用内存仓库（`InMemoryAtomRepo`）+ 进程内会话（`SessionStore(redis=None)`）：
这一层证明的是"规则对不对"，不是"MySQL 写没写进去"。
"""

from __future__ import annotations

import asyncio
import time

from base.config import Settings
from memory.extractor import Candidate, extract
from memory.service import MemoryService, _merge
from memory.session import SessionStore
from memory.store import Atom, InMemoryAtomRepo, MySQLAtomRepo, new_atom_id


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "feature_memory": True,
        "mem_min_confidence": 0.6,
        "mem_recall_top_k": 5,
        "mem_dedupe_sim": 0.92,
        "mem_atom_limit_per_user": 500,
        "mem_profile_max_tokens": 300,
        "mem_summary_max_tokens": 400,
        "mem_atom_max_tokens": 300,
        "mem_recall_budget_ms": 120,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _service(**overrides: object) -> tuple[MemoryService, InMemoryAtomRepo, SessionStore]:
    settings = _settings(**overrides)
    repo = InMemoryAtomRepo()
    session = SessionStore(settings)
    return MemoryService(settings, repo, session), repo, session


def _atom(user_id: str = "u1", *, content: str = "我数学不好", **extra: object) -> Atom:
    base: dict[str, object] = {
        "id": new_atom_id(),
        "user_id": user_id,
        "fact_type": "constraint",
        "fact_key": "subject_weak:数学",
        "content": content,
        "confidence": 0.85,
        "valid_from": time.time(),
    }
    base.update(extra)
    return Atom(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------ 1. 不污染
def test_knowledge_question_produces_no_memory() -> None:
    """知识类问句占了 90% 的轮次，把它们抽成记忆是记忆系统最常见的污染源。"""
    service, repo, _ = _service()

    async def run() -> None:
        for q in ("勾股定理怎么证明", "什么是复分解反应", "谢谢老师", "二次函数顶点式怎么写"):
            assert await service.remember("u1", "s1", q) == []
        assert repo.list_all("u1") == []

    asyncio.run(run())


def test_user_fact_is_extracted_and_recalled() -> None:
    service, _, _ = _service()

    async def run() -> None:
        saved = await service.remember("u1", "s1", "我上初三，数学不太好")
        assert {a.fact_key for a in saved} == {"grade", "subject_weak:数学"}

        ctx = await service.recall("u1", "函数图像怎么画", sid="s1")
        assert "初三" in ctx.profile  # 身份类进画像（常驻 300 token）
        assert any("数学" in a for a in ctx.atoms)
        assert ctx.tokens > 0 and not ctx.degraded
        # 记忆是"背景"，不是"依据"：渲染出来必须带段落标记，供 prompt 明确区分
        assert "【学员画像】" in ctx.render()

    asyncio.run(run())


def test_same_fact_is_merged_not_duplicated() -> None:
    service, repo, _ = _service()

    async def run() -> None:
        await service.remember("u1", "s1", "我数学不好")
        await service.remember("u2", "s2", "我数学不好")  # 另一个用户，互不影响
        await service.remember("u1", "s3", "我数学不好")

        mine = repo.list_alive("u1")
        assert len(mine) == 1 and mine[0].mention_count == 2

    asyncio.run(run())


# ------------------------------------------------------------------ 2. 冲突与链
def test_new_value_supersedes_old_and_keeps_one_chain() -> None:
    service, repo, _ = _service()

    async def run() -> None:
        (old,) = await service.remember("u1", "s1", "我上初三")
        (new,) = await service.remember("u1", "s2", "我上高一")

        alive = repo.list_alive("u1")
        assert [a.content for a in alive] == ["我上高一"]  # 画像里不能同时留着两个年级
        dead = repo.get(old.id)
        assert dead is not None and dead.valid_to is not None and dead.superseded_by == new.id
        assert new.chain_root_id == old.id

    asyncio.run(run())


def test_revive_restores_previous_value() -> None:
    """用户说"记错了，之前那条才对"：删掉新的，旧的要**复活**。"""
    service, repo, _ = _service()

    async def run() -> None:
        (old,) = await service.remember("u1", "s1", "我上初三")
        (new,) = await service.remember("u1", "s2", "我上高一")

        result = await service.forget("u1", new.id, "revive")
        assert result["restored"] == [old.id]
        assert [a.content for a in repo.list_alive("u1")] == ["我上初三"]

        chain = await service.chain("u1", old.id)
        assert {item["content"] for item in chain["items"]} == {"我上初三", "我上高一"}

    asyncio.run(run())


def test_purge_chain_removes_the_whole_history() -> None:
    """用户说"这件事别再提了"：整条链失效，不留任何可召回版本。"""
    service, repo, _ = _service()

    async def run() -> None:
        await service.remember("u1", "s1", "我上初三")
        (new,) = await service.remember("u1", "s2", "我上高一")

        result = await service.forget("u1", new.id, "purge_chain")
        assert result["chain_size"] == 2
        assert repo.list_alive("u1") == []
        assert all(a.deleted_at is not None for a in repo.list_all("u1"))

    asyncio.run(run())


# ------------------------------------------------------------------ 3. 不串用户
def test_other_users_memory_is_invisible_and_untouchable() -> None:
    service, repo, _ = _service()

    async def run() -> None:
        (mine,) = await service.remember("u1", "s1", "我数学不好")

        # 读写一律当"不存在"：区分 404 与 403 等于告诉攻击者"这条记忆存在"
        assert await service.patch("u2", mine.id, content="改你的") is None
        assert (await service.forget("u2", mine.id, "purge_chain"))["deleted"] is False
        assert await service.chain("u2", mine.id) == {"items": []}
        assert await service.recall("u2", "数学题") == await service.recall("u2", "数学题")
        assert repo.get(mine.id).content == "我数学不好"  # type: ignore[union-attr]

        # 清空只清自己的
        await service.clear_all("u2")
        assert len(repo.list_alive("u1")) == 1

    asyncio.run(run())


def test_low_confidence_atom_is_not_recalled() -> None:
    """`confidence < MEM_MIN_CONFIDENCE` 的条目留在库里（可下钻、可人工修），但不进 prompt。"""
    service, repo, _ = _service()

    async def run() -> None:
        repo.insert(_atom("u1", content="我可能想考师范", confidence=0.5))
        repo.insert(_atom("u1", content="我数学不好", confidence=0.85))

        ctx = await service.recall("u1", "想考什么")
        assert ctx.atoms == ("我数学不好",)
        # 记忆页仍看得到（列表接口分页返回，权威计数在 `total`）
        page = await service.list_atoms("u1")
        assert page["total"] == 2 and len(page["items"]) == 2

    asyncio.run(run())


# ------------------------------------------------------------------ 4. 不阻塞
def test_recall_timeout_degrades_instead_of_raising() -> None:
    class SlowRepo(InMemoryAtomRepo):
        def list_alive(self, user_id: str) -> list[Atom]:
            time.sleep(0.2)  # 模拟慢查询（真的跑在 to_thread 里）
            return super().list_alive(user_id)

    settings = _settings(mem_recall_budget_ms=10)
    service = MemoryService(settings, SlowRepo([_atom()]), SessionStore(settings))

    async def run() -> None:
        started = time.perf_counter()
        ctx = await service.recall("u1", "数学题")
        assert ctx.degraded and ctx.tokens == 0 and ctx.render() == ""
        assert time.perf_counter() - started < 0.2  # 到点就返回，没有等慢查询

    asyncio.run(run())


def test_recall_exception_degrades() -> None:
    class BoomRepo(InMemoryAtomRepo):
        def list_alive(self, user_id: str) -> list[Atom]:
            raise RuntimeError("milvus down")

    settings = _settings()
    service = MemoryService(settings, BoomRepo(), SessionStore(settings))

    async def run() -> None:
        ctx = await service.recall("u1", "数学题")
        assert ctx.degraded and ctx.extra.get("reason") == "error"

    asyncio.run(run())


def test_token_budget_trims_atoms() -> None:
    # 每条内容 6 个字符，6 字预算只装得下一条——第二条宁可丢，也不能撑破 prompt
    service, repo, _ = _service(mem_atom_max_tokens=6, mem_recall_top_k=5)

    async def run() -> None:
        for text in ("我数学不好啊", "我英语也不好", "我物理听不懂"):
            repo.insert(_atom("u1", content=text, fact_key=f"k{text}"))

        ctx = await service.recall("u1", "数学英语物理")
        assert len(ctx.atoms) == 1
        assert ctx.tokens <= 300

    asyncio.run(run())


def test_atom_limit_refuses_new_instead_of_evicting() -> None:
    """上限到了**拒绝新增**而不是静默淘汰：旧事实可能正是用户care的。"""
    service, repo, _ = _service(mem_atom_limit_per_user=1)

    async def run() -> None:
        await service.remember("u1", "s1", "我数学不好")
        await service.remember("u1", "s2", "我没时间")  # 不同槽位

        alive = repo.list_alive("u1")
        assert len(alive) == 1 and alive[0].content == "我数学不好"

    asyncio.run(run())


def test_feature_flag_off_disables_everything() -> None:
    service, repo, _ = _service(feature_memory=False)

    async def run() -> None:
        assert await service.remember("u1", "s1", "我数学不好") == []
        ctx = await service.recall("u1", "数学题")
        assert ctx.tokens == 0 and repo.list_all("u1") == []

    asyncio.run(run())


# ------------------------------------------------- 5. 结构化坐标（M0 坐标收口）
def test_coordinates_use_the_canonical_vocabulary() -> None:
    """`我上初三，数学不太好` 落库时：年级写 `九年级`、学科写 `数学`。

    这不是洁癖——Milvus 的 `grade` 字段、`Atom.grade`、`qa_slots` 三处必须是同一套写法，
    否则按年级的等值比较永远不成立，而且**不会报错**。
    """
    service, repo, _ = _service()

    async def run() -> None:
        saved = await service.remember("u1", "s1", "我上初三，数学不太好")
        by_key = {a.fact_key: a for a in saved}
        # 一年级说一次就够：两条事实都带年级，不能只在第一句生效
        assert {a.grade for a in saved} == {"九年级"}
        assert by_key["subject_weak:数学"].subject == "数学"
        # 年级事实说的是学段，不该被标成数学
        assert by_key["grade"].subject == ""

        # 存进去的和读回来的是同一份（`to_dict` 展开给 `Atom(**...)` 用）
        stored = {a.fact_key: a for a in repo.list_alive("u1")}
        assert stored["grade"].grade == "九年级"

    asyncio.run(run())


def test_unknown_grade_still_records_the_fact() -> None:
    """`我是初中生`：`GRADES` 里没有"初中"，grade 留空但事实照样进画像（知道多少记多少）。

    空串是**如实表达"这句话没给出可用的结构化年级"**，硬塞一个不在词表里的值
    会让后续所有按年级的等值比较永远不成立——那正是 M0 要消灭的静默不一致。
    """
    service, _, _ = _service()

    async def run() -> None:
        saved = await service.remember("u1", "s1", "我是初中生")
        assert [a.fact_key for a in saved] == ["grade"]
        assert saved[0].grade == ""  # 认得出学段，认不出年级
        assert saved[0].content == "我是初中生"  # 原话照旧进画像

    asyncio.run(run())


def test_merge_keeps_coordinates_from_either_side() -> None:
    """同一槽位的两次说法合并时，任一侧缺的坐标由另一侧补，不能被空值冲掉。"""
    old = _atom("u1", subject="数学", grade="九年级")
    cand = Candidate("constraint", "subject_weak:数学", "数学不太好", 0.85, "数学不太好")
    merged = _merge(old, cand)
    assert (merged.subject, merged.grade) == ("数学", "九年级")
    assert merged.mention_count == 2

    bare = _atom("u1")  # 旧条目没有坐标，新候选带了，则取新的
    rich = Candidate("constraint", "subject_weak:数学", "数学不好", 0.85, "数学不好", subject="数学", grade="八年级")
    assert (_merge(bare, rich).subject, _merge(bare, rich).grade) == ("数学", "八年级")


def test_atom_list_filters_by_coordinates() -> None:
    service, _, _ = _service()

    async def run() -> None:
        await service.remember("u1", "s1", "我上初三，数学不太好")
        await service.remember("u1", "s2", "我英语不好")  # 这轮没提年级

        assert (await service.list_atoms("u1", grade="九年级"))["total"] == 2
        assert (await service.list_atoms("u1", grade="八年级"))["total"] == 0
        assert [i["fact_key"] for i in (await service.list_atoms("u1", subject="数学"))["items"]] == [
            "subject_weak:数学"
        ]
        # 空串是**合法查询**（筛未归类的），不能被当成"不过滤"而返回全部
        unclassified = await service.list_atoms("u1", subject="")
        assert [i["fact_key"] for i in unclassified["items"]] == ["grade"]

    asyncio.run(run())


class _RecordingMySQL:
    """只记录 SQL 与参数：这一层要证明的是"坐标真的到达了 SQL"，不是"MySQL 写进去了"。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
        self.calls.append((sql, params))

    def query(self, sql: str, params: tuple[object, ...] = ()) -> list[dict[str, object]]:
        self.calls.append((sql, params))
        return []


def test_mysql_repo_sql_carries_coordinates() -> None:
    """`_COLS` 里有的列，`INSERT` / `UPDATE` 也必须带——否则每次读写都是 Unknown column。

    这正是 §7.2.7-G 那类缺陷的形态：SQL 字符串看着没错，错在它与参数列表不是同一件事。
    """
    mysql = _RecordingMySQL()
    repo = MySQLAtomRepo(mysql)  # type: ignore[arg-type]
    atom = _atom("u1", subject="数学", grade="九年级")

    repo.insert(atom)
    sql, params = mysql.calls[-1]
    assert "INSERT INTO memory_atom" in sql and "subject" in sql and "grade" in sql
    assert params[-2:] == ("数学", "九年级")

    repo.update(atom)
    sql, params = mysql.calls[-1]
    assert "UPDATE memory_atom" in sql and "subject=%s" in sql and "grade=%s" in sql
    assert params[-3:] == ("数学", "九年级", atom.id)

    repo.list_alive("u1")
    assert "subject, grade" in mysql.calls[-1][0]


def test_memory_atom_ddl_declares_coordinates() -> None:
    """建表语句与 `_COLS` 必须是同一套列，否则新库和老库的形态会分叉。"""
    from scripts.init_db import DDL, MEMORY_COLUMNS

    memory_ddl = next(d for d in DDL if "CREATE TABLE IF NOT EXISTS memory_atom" in d)
    for column, _ in MEMORY_COLUMNS:
        assert f" {column} " in memory_ddl
    assert "idx_user_subject_grade" in memory_ddl


def test_render_uses_the_documented_section_markers() -> None:
    """段落标记与 CLAUDE.md §7.9 逐字一致：它是模型区分"背景"与"依据"的唯一线索。"""
    from base.contracts import MemoryContext

    text = MemoryContext(profile="初三学生", summary="问了二次函数", atoms=("数学不好",)).render()
    assert "【学员画像】" in text and "【会话摘要】" in text and "【历史记忆】" in text
    assert "历史摘要" not in text and "长期记忆" not in text


# ------------------------------------- 6. L2 向量召回 / L3 画像（§3.7 第 1、8 条）
class _FakeIndex:
    """`AtomIndex` 的替身：只回答"该返回哪些 id / 该不该炸"，向量本身不参与断言。

    真向量召回的质量（MemRecall@5）要连真 Milvus + 真 bge-m3 才能量，
    这一层证明的是**接线**：头部顺序来自索引、剩余补齐不丢、坏了只退化不连坐。
    """

    def __init__(self, ids: list[str] | None = None, *, error: Exception | None = None) -> None:
        self.ids = ids
        self.error = error
        self.upserted: list[str] = []
        self.deleted: list[str] = []
        self.searches: list[tuple[str, str, int]] = []

    def search(self, user_id: str, query: str, top_k: int = 5) -> list[str] | None:
        self.searches.append((user_id, query, top_k))
        if self.error is not None:
            raise self.error
        return self.ids

    def upsert(self, atom: Atom) -> None:
        if self.error is not None:
            raise self.error
        self.upserted.append(atom.id)

    def delete(self, atom_id: str) -> None:
        if self.error is not None:
            raise self.error
        self.deleted.append(atom_id)


def _service_with(**kwargs: object) -> tuple[MemoryService, InMemoryAtomRepo, _FakeIndex]:
    from memory.store import InMemoryProfileRepo

    settings = _settings()
    repo = InMemoryAtomRepo()
    session = SessionStore(settings)
    index = kwargs.pop("index", None) or _FakeIndex()
    profiles = kwargs.pop("profiles", None) or InMemoryProfileRepo()
    service = MemoryService(
        settings, repo, session, index=index, profiles=profiles, **kwargs  # type: ignore[arg-type]
    )
    return service, repo, index


def test_recall_puts_index_hits_first_and_keeps_the_rest() -> None:
    """索引给的 top_k 排前面，**剩下的一条都不能丢**（索引是副本，会缺行）。"""
    service, repo, index = _service_with()
    later = _atom("u1", content="我喜欢物理")
    first = _atom("u1", content="我数学不好")
    second = _atom("u1", content="我想考重点高中")
    for atom in (later, first, second):
        repo.insert(atom)
    index.ids = [second.id, first.id]

    async def run() -> None:
        ctx = await service.recall("u1", "数学怎么提高")
        # 头部按索引顺序；`later` 不在索引里，仍排在后面而不是消失
        assert ctx.atoms == ("我想考重点高中", "我数学不好", "我喜欢物理")
        assert index.searches == [("u1", "数学怎么提高", 5)]

    asyncio.run(run())


def test_index_unavailable_falls_back_to_literal_order() -> None:
    """`None` = 索引不可用 → 整批按字面排序，与没有索引时逐条一致。"""
    service, repo, index = _service_with()
    plain = _atom("u1", content="我数学不好")
    repo.insert(plain)
    index.ids = None

    async def run() -> None:
        with_index = await service.recall("u1", "数学")
        service.index = None
        without_index = await service.recall("u1", "数学")
        assert with_index.atoms == without_index.atoms == ("我数学不好",)

    asyncio.run(run())


def test_broken_index_degrades_to_ranking_not_to_empty_memory() -> None:
    """索引抛异常**不许**把整次召回打成降级的空记忆——那会让画像和摘要一起丢。

    `recall()` 外面有一个兜底 `except`，返回的是 `MemoryContext()`；
    如果索引异常漏到那里，表现就是"这个用户没有任何记忆"，而日志里只看得到
    `memory_recall_failed`。这条用例钉住的是"在更近的地方接住它"。
    """
    service, repo, index = _service_with()
    repo.insert(_atom("u1", content="我数学不好"))
    index.error = RuntimeError("milvus down")

    async def run() -> None:
        ctx = await service.recall("u1", "数学")
        assert ctx.degraded is False
        assert ctx.atoms == ("我数学不好",)
        assert ctx.extra.get("reason") != "error"

    asyncio.run(run())


def test_writes_and_deletes_are_pushed_to_the_index() -> None:
    """写穿：新记忆进索引，删除的记忆**从索引里删掉**（不是改成失效时间）。"""
    service, _, index = _service_with()

    async def run() -> None:
        saved = await service.remember("u1", "s1", "我数学不好")
        assert [a.id for a in saved] == index.upserted
        atom = saved[0]
        await service.forget("u1", atom.id, "purge_chain")
        assert atom.id in index.deleted

    asyncio.run(run())


def test_memory_write_survives_a_broken_index() -> None:
    """索引写失败只记日志：MySQL 才是真相源，一次成功的记忆写入不该因此变成失败。"""
    service, repo, index = _service_with()
    index.error = RuntimeError("milvus down")

    async def run() -> None:
        saved = await service.remember("u1", "s1", "我数学不好")
        assert [a.fact_key for a in saved] == ["subject_weak:数学"]
        assert repo.list_alive("u1")  # 库里确实写进去了

    asyncio.run(run())


class _FakeMilvus:
    """只抓 `search` 的入参：这一层要证明的是**过滤表达式真的到达了请求**（§7.2.7-G）。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def search(self, **kwargs: object) -> list[list[dict[str, str]]]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("collection not loaded")
        return [[{"id": "a1"}, {"id": "a2"}]]


class _FakeEmbedder:
    name = "fake:bge-m3"
    dim = 4

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[dict[int, float]]]:
        return ([[0.1, 0.2, 0.3, 0.4] for _ in texts], [{} for _ in texts])


def test_milvus_atom_index_filter_reaches_the_request() -> None:
    """可见条件（租户/用户/未失效/未删除/置信度）必须逐条出现在 `filter` 里。

    少一条就是"已删除的记忆还在回答"或"读到别人的记忆"。§7.2.7-G 的教训：
    表达式字符串看着对不算数，得断言它到达了请求对象。
    """
    from memory.store import MilvusAtomIndex

    client = _FakeMilvus()
    index = MilvusAtomIndex(client, _FakeEmbedder(), _settings(), tenant_id="t1")

    assert index.search("u1", "数学怎么提高") == ["a1", "a2"]
    expr = str(client.calls[-1]["filter"])
    assert 'tenant_id == "t1"' in expr
    assert 'user_id == "u1"' in expr
    assert "valid_to == 0" in expr and "deleted_at == 0" in expr
    assert "confidence >= 0.6" in expr
    assert index.search("u1", "   ") == []  # 空问句不值得一次网络往返


def test_milvus_atom_index_returns_none_when_search_fails() -> None:
    """索引失败返回 `None` 而不是 `[]`：两者混同会让"索引挂了"表现为"这人没有记忆"。"""
    from memory.store import MilvusAtomIndex

    index = MilvusAtomIndex(_FakeMilvus(fail=True), _FakeEmbedder(), _settings())
    assert index.search("u1", "数学") is None


def test_stored_profile_wins_over_derived() -> None:
    """人工写过画像就以它为召回用的那一份——否则"我改好的画像"页面上看得见、召回里看不见。"""
    service, repo, _ = _service_with()
    repo.insert(
        _atom("u1", content="初三学生", fact_type="identity", fact_key="grade", grade="九年级")
    )

    async def run() -> None:
        derived = await service.get_profile("u1")
        assert derived["source"] == "derived" and derived["content"] == "初三学生"

        await service.set_profile("u1", "初三学生，数学薄弱，想冲重点高中")
        stored = await service.get_profile("u1")
        assert stored["source"] == "manual"

        ctx = await service.recall("u1", "数学")
        assert ctx.profile.startswith("初三学生，数学薄弱")
        # 画像是一张卡：人的话盖住推导值，但原子记忆本身一条不动
        assert len(repo.list_alive("u1")) == 1

    asyncio.run(run())


def test_set_profile_caps_length_and_leaves_an_audit_row() -> None:
    """超预算截断而不是报错（多写几个字不该让用户去算 token），但审计必须留痕。"""
    service, repo, _ = _service_with()

    async def run() -> None:
        body = await service.set_profile("u1", "啊" * 400)
        assert len(body["content"]) == 300  # mem_profile_max_tokens
        assert body["tokens"] == 300
        assert body["updated_by"] == "u1"
        assert any(a["action"] == "profile_set" for a in repo.audits)

    asyncio.run(run())


def test_clear_profile_returns_to_the_derived_one() -> None:
    """清画像只删画像这一张卡，不动原子记忆（想删记忆本身走 forget / clear_all）。"""
    service, repo, _ = _service_with()
    repo.insert(
        _atom("u1", content="初三学生", fact_type="identity", fact_key="grade", grade="九年级")
    )

    async def run() -> None:
        await service.set_profile("u1", "手写的画像")
        assert (await service.get_profile("u1"))["source"] == "manual"

        await service.clear_profile("u1")
        back = await service.get_profile("u1")
        assert back["source"] == "derived" and back["content"] == "初三学生"
        assert len(repo.list_alive("u1")) == 1
        assert any(a["action"] == "profile_clear" for a in repo.audits)
        # 空内容等于清除：用户把框清空再保存，唯一的合理语义
        assert (await service.set_profile("u1", "  "))["cleared"] is True

    asyncio.run(run())


def test_clear_all_also_drops_the_profile() -> None:
    """「清空」是用户说的那句话：他要删的是关于他的全部，画像不含例外。"""
    service, repo, _ = _service_with()

    async def run() -> None:
        await service.remember("u1", "s1", "我数学不好")
        await service.set_profile("u1", "手写画像")
        await service.clear_all("u1")
        assert (await service.get_profile("u1"))["source"] == "derived"
        assert (await service.get_profile("u1"))["content"] == ""

    asyncio.run(run())


def test_user_profile_ddl_and_drop_order() -> None:
    """`user_profile` 既要在建表语句里，也要在 `--drop` 的删除清单里。

    漏了后者，`--drop` 之后这张表会带着上一个环境的画像活下来——
    而"重建开发库"正是最该把用户数据清干净的时刻。
    """
    from scripts.init_db import DDL, DROP_ORDER

    assert any("CREATE TABLE IF NOT EXISTS user_profile" in d for d in DDL)
    assert "user_profile" in DROP_ORDER
