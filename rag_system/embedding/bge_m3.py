"""BGE-M3 向量化（稠密 + 稀疏一次出）。

为什么用 BGE-M3 而不是"向量 + 另算 BM25"：
1. 一次前向同时给出 dense 与 sparse（lexical weights），省一遍分词与统计；
2. 稀疏侧来自同一套语义表示，跨语言/同义词比 BM25 的词面匹配更稳；
3. 与 Milvus 的 `hybrid_search` + `WeightedRanker` 天然配套（旧项目已用这套）。

没有 GPU 或没装 FlagEmbedding 时退化为 `HashEmbedder`：它只保证"同文本同向量"，
用来跑单测和离线链路（此时检索质量无意义，但流程、删除、幂等都能验证）。
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Protocol

from base.config import Settings, get_settings
from base.logger import get_logger

log = get_logger(__name__)

DENSE = list[float]
SPARSE = dict[int, float]


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> tuple[list[DENSE], list[SPARSE]]:
        ...


class HashEmbedder:
    """确定性哈希向量：同文本同向量，维度固定 1024。**仅用于测试/离线**。"""

    name = "hash"
    dim = 1024

    def encode(self, texts: list[str]) -> tuple[list[DENSE], list[SPARSE]]:
        dense: list[DENSE] = []
        sparse: list[SPARSE] = []
        for text in texts:
            vec = [0.0] * self.dim
            bucket: dict[int, float] = {}
            for token in text.split():
                h = hashlib.sha1(token.encode()).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                vec[idx] += 1.0
                bucket[idx] = bucket.get(idx, 0.0) + 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            dense.append([v / norm for v in vec])
            sparse.append(bucket)
        return dense, sparse

    def encode_query(self, text: str) -> tuple[DENSE, SPARSE]:
        dense, sparse = self.encode([text])
        return dense[0], sparse[0]


class BGEM3Embedder:
    """真模型：CUDA 上 fp16，CPU 上 fp32。"""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "auto",
        fp16: bool = True,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel  # 延迟导入：没装也不影响其它模块
        import torch

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = f"bge-m3:{device}"
        self.dim = 1024
        self._model = BGEM3FlagModel(
            str(model_path),
            use_fp16=(fp16 and device == "cuda"),
            device=device,
        )
        log.info("embedder_ready", model=str(model_path), device=device, fp16=fp16 and device == "cuda")

    def encode(self, texts: list[str]) -> tuple[list[DENSE], list[SPARSE]]:
        if not texts:
            return [], []
        out = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [list(map(float, v)) for v in out["dense_vecs"]]
        sparse = [{int(k): float(w) for k, w in item.items()} for item in out["lexical_weights"]]
        return dense, sparse

    def encode_query(self, text: str) -> tuple[DENSE, SPARSE]:
        dense, sparse = self.encode([text])
        return dense[0], sparse[0]


def build_embedder(settings: Settings | None = None) -> Embedder:
    """auto：能用真模型就用，否则退化并**明确打日志**（不静默降级）。"""
    s = settings or get_settings()
    backend = s.rag_embed_backend.lower()
    if backend == "hash":
        return HashEmbedder()
    model_path = s.embed_model_path
    if not model_path.exists():
        if backend == "bge":
            raise RuntimeError(f"RAG_EMBED_MODEL_PATH 不存在：{model_path}")
        log.warning("embedder_fallback", reason="model_missing", path=str(model_path))
        return HashEmbedder()
    try:
        return BGEM3Embedder(
            model_path,
            device=s.rag_embed_device,
            fp16=s.rag_embed_fp16,
            batch_size=s.rag_embed_batch_size,
            max_length=s.rag_embed_max_length,
        )
    except ImportError as exc:
        if backend == "bge":
            raise RuntimeError("FlagEmbedding 未安装，无法使用 BGE-M3") from exc
        log.warning("embedder_fallback", reason="flagembedding_missing", error=str(exc))
        return HashEmbedder()
