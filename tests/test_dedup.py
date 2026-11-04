"""近重复文档检测（CLAUDE.md §7.2.6「未做」项）。

要挡的具体一例：`datasets/09_textbooks` 里同一本教材的两次下载
（`人教版-数学-九年级上册.pdf` / `人教版数学九年级上册.pdf`）。md5 不同、内容一致，
两份都入库会让同一段证据互为对手，证据门的跨文档 margin 直接判 `ambiguous`。

同时要证明**没有误杀**：同一学科的两本不同教材、同一篇文档重跑入库，都不该被拦。
"多拦一份"比"漏拦一份"更糟——漏拦只是证据差点，多拦是书没了。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from base.config import Settings
from rag_system.document_loaders.base import LoadedDoc
from rag_system.document_loaders.dedup import DedupIndex, hamming, normalize_name, simhash
from rag_system.ingest import IngestPipeline

# 同一本教材的两份"内容一致、排版/页眉略不同"的正文
BOOK = (
    "第二十二章 二次函数。一般地，形如 y=ax²+bx+c（a≠0）的函数叫做二次函数。"
    "二次函数的图象是一条抛物线，其顶点坐标可以通过配方法求得。"
    "当 a>0 时开口向上，当 a<0 时开口向下。函数在对称轴处取得最值。"
)
OTHER_BOOK = (
    "第二十三章 旋转。把一个图形绕着某一点转动一个角度的图形变换叫做旋转。"
    "旋转前后的图形全等，对应点到旋转中心的距离相等。"
    "中心对称图形是旋转 180 度后与自身重合的图形。"
)


# ---- 判据本身 ----
def test_download_variants_share_normalized_name() -> None:
    assert normalize_name("F:/docs/人教版-数学-九年级上册.pdf") == normalize_name(
        "F:/docs/人教版数学九年级上册.pdf"
    )


def test_normalize_name_keeps_parent_dirs() -> None:
    # 同名不同学科目录是两本书，不是同一份
    assert normalize_name("F:/d/01_textbook/初中/化学/教材.pdf") != normalize_name(
        "F:/d/01_textbook/初中/历史/教材.pdf"
    )


def test_generic_stem_in_different_subject_dirs_is_not_duplicate() -> None:
    """新语料是 `学科目录/教材.pdf`：只归一文件名会把整库并成一本书。

    2026-10-29 实测（`normalize_name` 只取 stem 时）：52 篇文档 → 1 篇入库 + 49 份"重复"。
    """
    index = DedupIndex()
    assert index.check_and_add("F:/datasets/01_textbook/初中/化学/教材.pdf", BOOK) is None
    assert index.check_and_add("F:/datasets/01_textbook/初中/历史/教材.pdf", OTHER_BOOK) is None
    assert len(index) == 2


def test_simhash_survives_punctuation_and_linebreak_changes() -> None:
    repacked = BOOK.replace("。", "。\n").replace("，", " ，")
    assert hamming(simhash(BOOK), simhash(repacked)) <= 3


def test_different_books_are_far_apart() -> None:
    assert hamming(simhash(BOOK), simhash(OTHER_BOOK)) > 3


# 2026-10-29 回归：按频次加权时，模板/公式里的高频二元组会把 64 位压成同一个值
_TEMPLATED = "$\\mathrm{Al} \\mathrm{Cu} \\mathrm{Fe}$ 三种金属元素组成，回答下列问题：" * 60
_PAPER_A = _TEMPLATED + "第一份文档讲的是化学实验的基本操作与安全规范，包括量筒读数与酒精灯使用。"
_PAPER_B = _TEMPLATED + "第二份文档讲的是电磁感应与法拉第定律的推导，包括楞次定律的判据与涡流。"


def test_simhash_not_collapsed_by_high_frequency_tokens() -> None:
    """两份共享长模板、正文完全不同的文档不能被判成重复。

    频次加权时这一对的汉明距离是 **0**（模板 3 000 字里高频二元组权重盖过正文），
    真实语料里因此把 `高中/化学/高三/试卷.html` 判成 `高中/化学/高一/教案.html` 的重复而拒收。
    每种二元组只记一次后是 17，稳在阈值之外。
    """
    assert hamming(simhash(_PAPER_A), simhash(_PAPER_B)) > DedupIndex().distance


def test_simhash_still_catches_identical_content() -> None:
    """收紧权重不能把真重复也放过去：同内容只在页眉/空白上不同，仍要拦得住。"""
    index = DedupIndex()
    index.check_and_add("F:/d/试卷.html", _PAPER_A)
    hit = index.check_and_add("F:/d/复习卷.html", _PAPER_A.replace("。", "。\n"))
    assert hit is not None and hit.reason == "simhash" and hit.distance <= 3


# ---- 索引行为 ----
def test_same_content_different_name_is_duplicate() -> None:
    # 真实那一例：文件名只差分隔符，归一化后相同 → 走便宜的 name 判据
    index = DedupIndex()
    assert index.check_and_add("F:/d/人教版-数学-九年级上册.pdf", BOOK) is None
    hit = index.check_and_add("F:/d/人教版数学九年级上册.pdf", BOOK.replace("。", "。\n"))
    assert hit is not None and hit.reason == "name"
    assert hit.kept_source.endswith("人教版-数学-九年级上册.pdf")


def test_same_content_unrelated_name_falls_through_to_simhash() -> None:
    # 文件名完全不像（另存为 / 重命名）时，靠正文指纹兜住
    index = DedupIndex()
    index.check_and_add("F:/d/九年级数学.pdf", BOOK)
    hit = index.check_and_add("F:/d/九年级数学-副本.pdf", BOOK.replace("。", "。\n"))
    assert hit is not None and hit.reason == "simhash" and hit.distance <= 3


def test_different_books_both_kept() -> None:
    index = DedupIndex()
    index.check_and_add("F:/d/九年级上册.pdf", BOOK)
    assert index.check_and_add("F:/d/九年级下册.pdf", OTHER_BOOK) is None
    assert len(index) == 2


def test_rejected_document_is_not_registered() -> None:
    # 被拒的那份不能进表，否则第三种命名会认错"先到者"
    index = DedupIndex()
    index.check_and_add("F:/d/a-数学.pdf", BOOK)
    index.check_and_add("F:/d/a数学.pdf", BOOK)
    hit = index.check_and_add("F:/d/a_数学.pdf", BOOK)
    assert hit is not None and hit.kept_source == "F:/d/a-数学.pdf"


def test_reingest_same_source_is_not_self_duplicate() -> None:
    index = DedupIndex()
    index.check_and_add("F:/d/数学.pdf", BOOK)
    assert index.check_and_add("F:/d/数学.pdf", BOOK + "（修订版新增一段）") is None


def test_disabled_index_accepts_everything() -> None:
    index = DedupIndex(enabled=False)
    assert index.check_and_add("F:/d/数学.pdf", BOOK) is None
    assert index.check_and_add("F:/d/数学.pdf", BOOK) is None
    assert len(index) == 0


def test_fingerprint_file_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "fp.json"
    first = DedupIndex()
    first.check_and_add("F:/d/人教版-数学-九年级上册.pdf", BOOK)
    first.save(path)

    # 增量入库时新建的索引必须能拦住上一批进过的那份
    second = DedupIndex()
    assert second.load(path) == 1
    assert second.check_and_add("F:/d/人教版数学九年级上册.pdf", BOOK) is not None


def test_broken_fingerprint_file_is_treated_as_empty(tmp_path: Path) -> None:
    path = tmp_path / "fp.json"
    path.write_text("{ 不是 json", encoding="utf-8")
    assert DedupIndex().load(path) == 0


def test_short_text_is_not_fingerprinted(tmp_path: Path) -> None:
    # 太短的正文 simhash 不稳定，宁可放过也不误杀
    index = DedupIndex()
    assert index.check_and_add("F:/d/短.md", "很短") is None
    assert index.check_and_add("F:/d/短2.md", "很短") is None


# ---- 接进入库流水线 ----
class _Store:
    def __init__(self) -> None:
        self.upserted: list[str] = []

    def ensure_collection(self, *, recreate: bool = False) -> None:
        return None

    def delete_by_source(self, source: str, *, tenant_id: str, kb_version: str) -> int:
        return 0

    def upsert(self, chunks, dense, sparse, **meta) -> int:
        self.upserted.append(meta["source"])
        return len(chunks)


class _Embedder:
    name = "fake-bge-m3"

    def encode(self, texts: list[str]):
        n = len(texts)
        return np.zeros((n, 1024), dtype=np.float32), [{"1": 0.5} for _ in range(n)]


def _pipeline(tmp_path: Path, docs: list[LoadedDoc]) -> tuple[IngestPipeline, _Store]:
    store = _Store()
    factory = type(
        "F",
        (),
        {"iter_documents": staticmethod(lambda targets: iter([(Path(d.source), d) for d in docs]))},
    )()
    settings = Settings(
        mysql_password="123456",
        minio_secret_key="minioadmin",
        rag_docs_dir=str(tmp_path),
        rag_parent_size=200,
        rag_parent_overlap=20,
        rag_child_size=80,
        rag_child_overlap=10,
        rag_max_batch_size=4,
    )
    pipe = IngestPipeline(
        settings,
        store=store,
        embedder=_Embedder(),
        factory=factory,
        fingerprint_path=tmp_path / "doc_fingerprints.json",
    )
    return pipe, store


def _doc(source: str, body: str) -> LoadedDoc:
    return LoadedDoc(text=body * 20, source=source, doc_type="textbook", subject="数学", title=Path(source).stem)


def test_pipeline_skips_second_download_of_same_textbook(tmp_path: Path) -> None:
    pipe, store = _pipeline(
        tmp_path,
        [
            _doc("F:/d/人教版-数学-九年级上册.pdf", BOOK),
            _doc("F:/d/人教版数学九年级上册.pdf", BOOK),
        ],
    )
    report = pipe.run([tmp_path])

    assert report.docs_ok == 1 and report.duplicates == 1
    assert store.upserted == ["F:/d/人教版-数学-九年级上册.pdf"]  # 先到的那份留下
    dup = [r for r in report.results if r.status == "duplicate"]
    assert len(dup) == 1 and "doc_duplicate" in dup[0].error
    # 报告里必须能查到"谁被拦了、跟谁重复"——静默跳过等于书没了没人知道
    assert report.summary()["skipped_duplicates"][0]["path"].endswith("人教版数学九年级上册.pdf")
    assert (tmp_path / "doc_fingerprints.json").exists()


def test_allow_duplicate_bypasses_the_check(tmp_path: Path) -> None:
    docs = [
        _doc("F:/d/人教版-数学-九年级上册.pdf", BOOK),
        _doc("F:/d/人教版数学九年级上册.pdf", BOOK),
    ]
    pipe, store = _pipeline(tmp_path, docs)
    report = pipe.run([tmp_path], allow_duplicate=True)
    assert report.duplicates == 0 and report.docs_ok == 2
    assert len(store.upserted) == 2


def test_fingerprints_persist_across_runs(tmp_path: Path) -> None:
    pipe, _ = _pipeline(tmp_path, [_doc("F:/d/人教版-数学-九年级上册.pdf", BOOK)])
    pipe.run([tmp_path])

    # 第二次进程只会跑另一个文件；没有持久化指纹就拦不住它
    second, store = _pipeline(tmp_path, [_doc("F:/d/人教版数学九年级上册.pdf", BOOK)])
    report = second.run([tmp_path])
    assert report.duplicates == 1 and store.upserted == []


def test_same_source_can_be_ingested_in_different_tenant_and_version_scopes() -> None:
    index = DedupIndex()
    source = "F:/datasets/初中/数学/教材.pdf"
    assert index.check_and_add(source, BOOK, scope="school_a\x1fv1") is None
    # 同一物理文件可以被另一个租户、另一个 KB 版本分别索引。
    assert index.check_and_add(source, BOOK, scope="school_b\x1fv1") is None
    assert index.check_and_add(source, BOOK, scope="school_a\x1fv2") is None
    # 同一个 scope 内仍然能拦住不同路径的重复文档。
    duplicate = index.check_and_add("F:/datasets/初中/数学/教材-副本.pdf", BOOK, scope="school_a\x1fv1")
    assert duplicate is not None


def test_fingerprint_roundtrip_preserves_scope(tmp_path: Path) -> None:
    path = tmp_path / "scoped-fp.json"
    first = DedupIndex()
    source = "F:/datasets/初中/数学/教材.pdf"
    first.check_and_add(source, BOOK, scope="school_a\x1fv2")
    first.save(path)

    second = DedupIndex()
    second.load(path)
    assert second.check_and_add(source, BOOK, scope="school_a\x1fv2") is None
    assert second.check_and_add(source, BOOK, scope="school_b\x1fv2") is None
