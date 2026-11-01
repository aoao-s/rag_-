"""BERT 精调意图分类器（任务 2.2 / CLAUDE.md §7.7）。

## 为什么有这一层

七类业务意图是**封闭标签集**，而 `datasets/06_intent` 已经把这类样本喂给过 BERT 精调
（`rag_system/core/bert_train_model.py`）。在封闭集上，精调模型比通用大模型更稳、更便宜、
更快：单条 CPU 前向是毫秒级，大模型是一次几百毫秒的往返，还只能搭在"改写"那次调用里
顺带问一句。任务 2.2 的口径就是一个级联：

    BERT 置信   → 直接定档（`method="bert"`）
    BERT 弃权/不可用 → 大模型辅助判断（`method="model"`）
    大模型也弃权/不可用 → 规则分类器（`method="rule"`）
    都不成立 → `unknown`（`method="fallback"`）→ `generic_clarify`

**弃权是正常结论，不是故障**：`classify()` 返回 `None` 表示"我没有可靠意见"，
调用方接着往下问。低置信时强行挑一类，会让错误档位把正确答案挡在 `doc_type` 过滤外
（§7.7 硬规则 4），而且那条答案还会被写进缓存——比"多问一次大模型"贵得多。

## 三条约束

1. **永不抛异常**：它挂在检索之前，抛一次这条请求就全挂。
2. **不在事件循环里做 torch 前向**（§5.2）：`classify()` 走 `asyncio.to_thread`。
3. **检查点自带的标签映射必须与 `base/labels.py` 完全一致**，不一致就整条链路自禁用
   ——不是退出进程。下标错位时模型照跑、不报错，只是把"超出范围"静默错分进"闲聊"，
   这种错比"没有模型"难查得多。`bert_test_model.py` 在这里是 `SystemExit`，那是评测
   脚本该有的行为；在线服务不能因为一个**可选**模型而拒绝起服务（§7.7 降级表）。
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from pathlib import Path
from typing import Any

from base.config import BASE_DIR, Settings, get_settings
from base.contracts import IntentResult
from base.labels import ID2INTENT, INTENT_MIN_CONFIDENCE
from base.logger import get_logger
from understanding.profiles import profile_for

log = get_logger(__name__)

CHECKPOINT_NAME = "bert-intent-ft"
# 与训练脚本的 `max_len=24` 一致。改这里必须重训：截断点不同，长问句喂给模型的
# 后半段就有了、没了，分布与训练时不一致
MAX_LEN = 24

# `rag_system/core/bert_train_model.py::handle_command` 的**逐字副本**。
# 为什么是副本而不是 import：`understanding/` 不许 import `rag_system`（§5.1 分层），
# 而且那一条 import 会把 pandas / sklearn / torch 一起拖进请求路径。
# 为什么不能"改得更合理"：训练时把标点之外的非中文/字母/数字字符全删掉了，
# 推理时少删或多删一个字符，训练与推理的输入分布就不一致。
_NORM_RE = re.compile(r"[^一-龥a-zA-Z0-9,.!:;\s]")


def normalize(text: str) -> str:
    """训练集用的同一套清洗（见 `_NORM_RE` 的注释）。"""
    return _NORM_RE.sub("", text)


def default_checkpoint_path(settings: Settings | None = None) -> Path:
    """检查点目录。`MODELS_DIR` 没配时回落到仓库内的固定位置。

    `Settings.MODELS_DIR` 的默认值是空串 → `Path("")` 就是**当前工作目录**。
    那会让检查点随"从哪个目录启动"漂移，而失败形态是静默自禁用（只有一行 warning），
    正是最难查的那类问题。所以空值 / "." 一律回落到 `BASE_DIR/rag_system/models`。
    """
    s = settings or get_settings()
    raw = str(s.MODELS_DIR or "").strip()
    if raw in ("", "."):
        root = BASE_DIR / "rag_system" / "models"
    else:
        root = Path(raw)
        root = root if root.is_absolute() else (BASE_DIR / root)
    return root / CHECKPOINT_NAME


class BertIntentClassifier:
    """精调意图分类器。构造是**零开销**的，加载发生在 `warmup()` 或首次分类时。"""

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        min_confidence: float = INTENT_MIN_CONFIDENCE,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else default_checkpoint_path()
        # 与 `llm_understanding` 同一个闸门（`base/labels.INTENT_MIN_CONFIDENCE`）。
        # 两个模型用不同的门槛，"A 说 0.58 弃权、B 说 0.58 定档"在日志里看不出差别
        self.min_confidence = min_confidence
        self._lock = threading.Lock()
        self._loaded = False  # 已尝试过加载（成功或失败都算），避免每条请求重试一遍
        self._ready = False
        self._reason = ""
        self._torch: Any = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    # ---- 状态 ----
    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def reason(self) -> str:
        """不可用原因（启动日志用）。可用时是空串。"""
        return self._reason

    def disable(self, reason: str) -> None:
        """整条链路自禁用（标签映射不一致、依赖缺失、检查点加载失败）。"""
        self._ready = False
        self._loaded = True
        self._reason = reason
        self._model = self._tokenizer = self._torch = self._device = None

    # ---- 加载 ----
    def warmup(self) -> bool:
        """启动期加载（§4.2：模型不进请求路径）。返回是否可用。"""
        return self._ensure()

    def _ensure(self) -> bool:
        if self._loaded:
            return self._ready
        with self._lock:  # 并发首调用只加载一次
            if self._loaded:
                return self._ready
            t0 = time.perf_counter()
            try:
                # 延迟导入：没装 torch 的纯 FAQ 部署不必为这份 import 付时间
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except Exception as exc:  # noqa: BLE001
                self.disable(f"依赖缺失: {exc}")
                log.warning("bert_intent_unavailable", reason=self._reason)
                return False
            if not self.checkpoint.is_dir():
                self.disable(f"检查点不存在: {self.checkpoint}")
                log.warning("bert_intent_unavailable", reason=self._reason)
                return False
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint))
                model = AutoModelForSequenceClassification.from_pretrained(str(self.checkpoint))
            except Exception as exc:  # noqa: BLE001
                self.disable(f"检查点加载失败: {exc}")
                log.warning("bert_intent_unavailable", reason=self._reason)
                return False

            # 与 `bert_test_model.py` 同一条校验，只是失败后果不同（见模块注释 3）
            labels = {int(k): v for k, v in model.config.id2label.items()}
            if labels != dict(ID2INTENT):
                self.disable(
                    f"检查点标签映射与 base/labels.py 不一致（检查点: {labels}；"
                    f"期望: {dict(ID2INTENT)}）——先跑 rag_system/core/bert_train_model.py 重训"
                )
                log.warning("bert_intent_unavailable", reason=self._reason)
                return False

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            self._torch, self._model, self._tokenizer, self._device = torch, model, tokenizer, device
            self._ready = True
            self._loaded = True
            log.info(
                "bert_intent_ready",
                path=str(self.checkpoint),
                labels=len(labels),
                device=str(device),
                cost_ms=round((time.perf_counter() - t0) * 1000, 1),
            )
            return True

    # ---- 推理 ----
    def _forward(self, text: str) -> tuple[int, float]:
        """一次前向 → `(类别下标, 置信度)`。单测覆盖这个缝，不必伪造一个 torch。"""
        torch = self._torch
        enc = self._tokenizer(
            [text],
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.inference_mode():
            logits = self._model(
                enc["input_ids"].to(self._device), enc["attention_mask"].to(self._device)
            ).logits
            probs = torch.softmax(logits, dim=1).ravel()
        confidence, index = torch.max(probs, dim=0)
        return int(index.item()), float(confidence.item())

    def classify_sync(self, query: str) -> IntentResult | None:
        """同步分类。不可用 / 弃权一律返回 `None`（见模块注释）。"""
        if not self._ensure():
            return None
        text = normalize(query or "").strip()
        if not text:
            return None
        try:
            index, confidence = self._forward(text)
        except Exception as exc:  # noqa: BLE001
            # 推理异常不连坐整条请求：意图只是"挑一条先验"（§7.7 降级表）。
            # 但要把原因留下——CUDA OOM 与"输入太长"在日志里长得完全不一样
            log.warning("bert_intent_failed", error=str(exc))
            return None
        label = ID2INTENT.get(index)
        if label is None or confidence < self.min_confidence:
            # 弃权。**返回 None 而不是 unknown**：`unknown` 是整条级联的结论，
            # 不是第一段的结论——第一段就把话说死，大模型与规则再没有机会
            log.debug(
                "bert_intent_abstained", confidence=round(confidence, 3),
                threshold=self.min_confidence, label=label,
            )
            return None
        return IntentResult(
            label=label,
            confidence=round(confidence, 4),
            method="bert",
            profile=profile_for(label).name,
        )

    async def classify(self, query: str, *, timeout_ms: int = 200) -> IntentResult | None:
        """异步入口：torch 前向放进线程池（§5.2 禁止在协程里做阻塞调用）。"""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.classify_sync, query),
                timeout=max(int(timeout_ms), 1) / 1000,
            )
        except asyncio.TimeoutError:
            # 超时说明线程池被别的活占着。这时宁可放弃这条结论让大模型接手，
            # 也不要拖住首 token（§6.1）
            log.warning("bert_intent_timeout", timeout_ms=timeout_ms)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("bert_intent_failed", error=str(exc))
            return None


__all__ = [
    "BertIntentClassifier",
    "CHECKPOINT_NAME",
    "MAX_LEN",
    "default_checkpoint_path",
    "normalize",
]
