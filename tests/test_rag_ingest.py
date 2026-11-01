"""入库流水线的编排逻辑单测：用假 store / 假 embedder，不连 Milvus、不载模型。

真实向量化的正确性靠 BGE-M3 本身，这里只保证：失败隔离、幂等删除、块数统计、
dry-run 不落库、清单可写。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from base.config import Settings
from rag_system.document_loaders.base import LoadedDoc
from rag_system.ingest import IngestPipeline, IngestReport


class FakeStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.upserted: list[dict] = []
        self.ensured = 0
        self.content: dict[str, list[str]] = {}

    def ensure_collection(self, *, recreate: bool = False) -> None:
        self.ensured += 1
        if recreate:
            self.content.clear()

    def delete_by_source(self, source: str) -> int:
        self.deleted.append(source)
        return len(self.content.pop(source, []))

    def upsert(self, chunks, dense, sparse, **meta) -> int:
        self.upserted.append({"n": len(chunks), "meta": meta, "dense_shape": dense.shape})
        self.content[meta["source"]] = [c.text for c in chunks]
        return len(chunks)


class FakeEmbedder:
    name = "fake-bge-m3"

    def encode(self, texts: list[str]):
        n = len(texts)
        dense = np.zeros((n, 1024), dtype=np.float32)
        # 稀疏用 dict 列表模拟（真实实现返回 csr/嵌套 dict）
        sparse = [{"1": 0.5} for _ in range(n)]
        return dense, sparse


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        mysql_password="123456",
        rag_docs_dir=str(tmp_path),
        minio_secret_key="minioadmin",
        rag_parent_size=200,
        rag_parent_overlap=20,
        rag_child_size=80,
        rag_child_overlap=10,
        rag_max_batch_size=4,
    )


def _pipeline(tmp_path: Path, docs: list[LoadedDoc]) -> tuple[IngestPipeline, FakeStore]:
    store = FakeStore()
    factory = type(
        "F",
        (),
        {
            "iter_documents": staticmethod(
                lambda targets: iter([(Path(d.source), d) for d in docs] + [(Path("F:/x/broken.md"), ValueError("坏文件"))])
            )
        },
    )()
    pipe = IngestPipeline(
        _settings(tmp_path),
        store=store,
        embedder=FakeEmbedder(),
        factory=factory,
        # 指纹文件指到 tmp：否则单测会往仓库的 data/ 里写真实指纹，并跨用例互相干扰
        fingerprint_path=tmp_path / "doc_fingerprints.json",
    )
    return pipe, store


def _doc(source: str, body: str = "函数是描述变量关系的数学对象。\n") -> LoadedDoc:
    return LoadedDoc(text=body * 40, source=source, doc_type="textbook", subject="数学", title=Path(source).stem)


def test_run_writes_chunks_and_isolates_failure(tmp_path: Path) -> None:
    pipe, store = _pipeline(tmp_path, [_doc("F:/x/数学-a.md")])
    report = pipe.run([tmp_path])

    assert store.ensured == 1
    assert report.files == 2  # 1 个正常 + 1 个抛异常的
    assert report.docs_ok == 1
    assert report.docs_failed == 1
    assert report.chunks > 0
    assert report.written == report.chunks
    assert store.deleted == ["F:/x/数学-a.md"]
    assert report.results[0].status == "ok"


def test_rerun_deletes_old_chunks(tmp_path: Path) -> None:
    pipe, store = _pipeline(tmp_path, [_doc("F:/x/数学-a.md")])
    pipe.run([tmp_path])
    pipe.run([tmp_path])
    # 两次入库各删一次，证明是"先删后写"而不是叠加
    assert store.deleted == ["F:/x/数学-a.md", "F:/x/数学-a.md"]
    assert len(store.upserted) == 2


def test_dry_run_does_not_touch_store(tmp_path: Path) -> None:
    pipe, store = _pipeline(tmp_path, [_doc("F:/x/数学-a.md")])
    report = pipe.run([tmp_path], dry_run=True)
    assert store.ensured == 0
    assert store.upserted == []
    assert report.chunks > 0
    assert report.written == 0


def test_limit_stops_early(tmp_path: Path) -> None:
    pipe, _ = _pipeline(tmp_path, [_doc("F:/x/数学-a.md"), _doc("F:/x/数学-b.md")])
    report = pipe.run([tmp_path], limit=1)
    assert report.files == 1


def test_manifest_written(tmp_path: Path) -> None:
    pipe, _ = _pipeline(tmp_path, [_doc("F:/x/数学-a.md")])
    report = pipe.run([tmp_path])
    out = pipe.write_manifest(report, tmp_path / "manifest.json")
    text = out.read_text(encoding="utf-8")
    assert "docs_ok" in text and "chunks" in text


def test_empty_doc_skipped(tmp_path: Path) -> None:
    pipe, store = _pipeline(tmp_path, [LoadedDoc(text="", source="F:/x/数学-empty.md", title="empty")])
    report = pipe.run([tmp_path])
    assert report.docs_ok == 1
    assert report.written == 0
    assert store.upserted == []
    assert report.results[0].status == "skipped"
