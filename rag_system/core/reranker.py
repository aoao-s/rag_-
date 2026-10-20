"""BGE-reranker-v2-m3 封装。

只做一件事：给 (query, passage) 打分。模型加载慢（约 1.5 GB）、必须复用，
所以这里是个长生命周期对象，由 Retriever 惰性创建后一直持有。
"""

from __future__ import annotations

import threading
from pathlib import Path

from base.logger import get_logger

log = get_logger(__name__)


class BgeReranker:
    def __init__(self, model_path: str | Path, *, use_fp16: bool = False, max_length: int = 512) -> None:
        self.model_path = str(model_path)
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._model = None
        self._lock = threading.Lock()

    def _ensure(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(self.model_path, use_fp16=self.use_fp16)
            log.info("reranker_loaded", model=self.model_path, fp16=self.use_fp16)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """返回与 passages 等长的分数；空输入直接返回空列表，不触发模型加载。"""
        if not passages:
            return []
        self._ensure()
        pairs = [[query, p[: self.max_length * 4]] for p in passages]
        raw = self._model.compute_score(pairs, normalize=True)
        if isinstance(raw, (int, float)):
            return [float(raw)]
        return [float(x) for x in raw]


def build_reranker(settings) -> BgeReranker:
    model_path = Path(settings.rag_reranker_path)
    if not model_path.exists():
        raise FileNotFoundError(f"重排模型不存在：{model_path}（设置 RAG_RERANKER_PATH 指向 bge-reranker-v2-m3）")
    return BgeReranker(
        model_path,
        use_fp16=settings.rag_use_fp16,
        max_length=settings.rag_rerank_max_length,
    )
