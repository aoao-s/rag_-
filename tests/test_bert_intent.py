"""BERT 初始意图分类器（任务 2.2）的契约。

盯的是**级联的第一段到底给什么结论**，不是"模型准不准"（那是 `bert_test_model.py` 的事）：

- 高置信 → `IntentResult(method="bert")`，标签取自 `base/labels.ID2INTENT`；
- 低置信 / 空输入 / 下标越界 / 前向异常 / 超时 → **`None`**（弃权），**不是 `unknown`**。
  `unknown` 是整条级联的结论（大模型与规则都弃权），第一段就把话说死会让后两段再没有机会；
- 检查点标签映射与 `base/labels.py` 不一致 → 整条链路**自禁用**而不是退出进程，也不许照跑。
  下标错位时模型照跑、不报错，只会把"超出范围"静默错分进"闲聊"——这种错比"没有模型"难查得多。

全部用替身，不加载真实权重：真检查点现在是旧的 6 类（10-22 训的），
按设计它必须被标签校验拦下，把这一点写进用例比人工确认靠谱。
"""

from __future__ import annotations

import asyncio
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest

from base.config import BASE_DIR, Settings
from base.labels import INTENT2ID, ID2INTENT, INTENT_MIN_CONFIDENCE
from understanding.bert_intent import (
    CHECKPOINT_NAME,
    BertIntentClassifier,
    default_checkpoint_path,
    normalize,
)
from understanding.profiles import profile_for


# ---------------------------------------------------------------- 替身
def _install_fake_runtime(monkeypatch: pytest.MonkeyPatch, id2label: dict[str, str]) -> None:
    """把 torch / transformers 换成够用的假模块，让 `_ensure()` 不碰真权重。

    只实现被测代码真正用到的那几个名字（`import` 会优先命中 `sys.modules`，
    所以即使环境里装着真 torch，这里也不会去加载它）。
    """
    torch = types.ModuleType("torch")
    torch.device = lambda name: name  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]

    class _Tokenizer:
        @staticmethod
        def from_pretrained(_path: str) -> object:
            return object()

    class _Model:
        def __init__(self) -> None:
            self.config = types.SimpleNamespace(id2label=id2label)

        @classmethod
        def from_pretrained(cls, _path: str) -> _Model:
            return cls()

        def to(self, _device: Any) -> _Model:
            return self

        def eval(self) -> _Model:
            return self

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _Tokenizer  # type: ignore[attr-defined]
    transformers.AutoModelForSequenceClassification = _Model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


class StubBert(BertIntentClassifier):
    """跳过加载，只替换"一次前向"这一帧。"""

    def __init__(
        self,
        *,
        index: int = 0,
        confidence: float = 0.94,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        super().__init__(checkpoint="<stub>")
        self.index = index
        self.confidence = confidence
        self.delay = delay
        self.error = error
        self.forwarded: list[str] = []
        # 冒充"已加载且可用"，`_ensure()` 直接放行
        self._loaded = True
        self._ready = True

    def _forward(self, text: str) -> tuple[int, float]:
        self.forwarded.append(text)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.index, self.confidence


# ---------------------------------------------------------------- ① 清洗口径
def test_normalize_is_byte_identical_to_the_training_script() -> None:
    """推理端的清洗必须与训练端逐字节一致（改一个字，训练/推理分布就分叉）。

    `understanding/` 不许 import `rag_system`（§5.1 分层），所以这里是**副本**；
    副本的唯一防线就是这条用例——把训练脚本里的 `handle_command` 拉进来做逐条比对。
    """
    from rag_system.core.bert_train_model import handle_command

    probes = [
        "什么是勾股定理？",
        "老师，我想知道一次函数的图像怎么画🙂",
        "x²-6x+9 因式分解（八年级）",
        "三角函数 sin(30°) = 0.5?",
        "abcXYZ123,.:;!?",
        "",
        "ＡＢＣ　全角空格",
    ]
    for probe in probes:
        assert normalize(probe) == handle_command(probe), probe


# ---------------------------------------------------------------- ② 检查点路径
def test_default_checkpoint_falls_back_to_the_repo_models_dir() -> None:
    """`MODELS_DIR` 的默认值是空串 → `Path("")` 就是 CWD。不特判就会随启动目录漂移。"""
    empty = Settings(MODELS_DIR="")
    assert default_checkpoint_path(empty) == BASE_DIR / "rag_system" / "models" / CHECKPOINT_NAME
    dot = Settings(MODELS_DIR=".")
    assert default_checkpoint_path(dot) == BASE_DIR / "rag_system" / "models" / CHECKPOINT_NAME


def test_default_checkpoint_honours_absolute_and_relative_models_dir(tmp_path: Path) -> None:
    absolute = default_checkpoint_path(Settings(MODELS_DIR=str(tmp_path)))
    assert absolute == tmp_path / CHECKPOINT_NAME
    # 相对路径按 BASE_DIR 解析：换工作目录不会找不到（与 `RAG_EMBED_MODEL_PATH` 同一口径）
    relative = default_checkpoint_path(Settings(MODELS_DIR="rag_system/models"))
    assert relative == BASE_DIR / "rag_system" / "models" / CHECKPOINT_NAME


# ---------------------------------------------------------------- ③ 不可用 = 弃权
def test_missing_checkpoint_is_unavailable_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_runtime(monkeypatch, dict(ID2INTENT))
    classifier = BertIntentClassifier(checkpoint=tmp_path / "does-not-exist")
    assert classifier.warmup() is False
    assert classifier.ready is False
    assert "检查点不存在" in classifier.reason
    assert classifier.classify_sync("什么是勾股定理") is None
    assert asyncio.run(classifier.classify("什么是勾股定理")) is None


def test_label_mapping_mismatch_disables_the_whole_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """旧 6 类的检查点（`学习方法/教案素材/...`）必须被拦下——不是照跑、也不是退出进程。

    这正是当前磁盘上那个 2026-10-22 训练产物的形态：`ID2INTENT` 已换成 7 类，
    旧检查点的下标全错位。拦下之后的线上行为是"意图改由大模型/规则给出"，服务照常。
    """
    checkpoint = tmp_path / CHECKPOINT_NAME
    checkpoint.mkdir()
    stale = {"0": "学习方法", "1": "教案素材", "2": "备课辅助", "3": "知识点查询", "4": "题目讲解", "5": "闲聊"}
    _install_fake_runtime(monkeypatch, stale)

    classifier = BertIntentClassifier(checkpoint=checkpoint)
    assert classifier.warmup() is False
    assert classifier.ready is False
    assert "不一致" in classifier.reason and "重训" in classifier.reason
    assert classifier.classify_sync("什么是勾股定理") is None


def test_matching_checkpoint_becomes_ready(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """正向对照：标签映射对得上时必须真的可用，否则上面那条拦的是所有情况。"""
    checkpoint = tmp_path / CHECKPOINT_NAME
    checkpoint.mkdir()
    _install_fake_runtime(monkeypatch, {str(k): v for k, v in ID2INTENT.items()})

    classifier = BertIntentClassifier(checkpoint=checkpoint)
    assert classifier.warmup() is True
    assert classifier.ready is True
    assert classifier.reason == ""


# ---------------------------------------------------------------- ④ 高置信 → bert
def test_high_confidence_returns_a_bert_intent_result() -> None:
    """任务 2.2 要求的形状：`{"intent": "EXAM_GENERATION", "confidence": 0.94, "method": "bert"}`。"""
    index = INTENT2ID["考试出题"]
    classifier = StubBert(index=index, confidence=0.94)
    result = classifier.classify_sync("帮我出一套二次函数的卷子")
    assert result is not None
    assert result.label == "考试出题"
    assert result.confidence == 0.94
    assert result.method == "bert"
    assert result.profile == profile_for("考试出题").name
    # 对外形状：中文规范名进 label，英文代号进 intent（`base/labels.INTENT_CODES`）
    payload = result.as_dict()
    assert payload["intent"] == "EXAM_GENERATION"
    assert payload["label"] == "考试出题"
    assert payload["method"] == "bert"
    assert payload["confidence"] == 0.94


def test_query_is_normalized_before_the_forward_pass() -> None:
    """前向收到的必须是清洗过的串：训练时就是这么喂的。"""
    classifier = StubBert(index=INTENT2ID["学习讲解"], confidence=0.9)
    classifier.classify_sync("什么是勾股定理🙂？")
    # `🙂` 与全角 `？` 都不在训练集的字符集里，两者都该被删掉
    assert classifier.forwarded == ["什么是勾股定理"]


# ---------------------------------------------------------------- ⑤ 弃权 = None
def test_low_confidence_abstains_with_none_not_unknown() -> None:
    """< 0.6 不猜（§3.2 第 2 条）。返回 `None` 让下一段接手，**不是** `unknown`。"""
    classifier = StubBert(index=INTENT2ID["学习讲解"], confidence=INTENT_MIN_CONFIDENCE - 0.01)
    assert classifier.classify_sync("这个怎么弄") is None


def test_confidence_exactly_at_the_threshold_is_accepted() -> None:
    """边界取闭区间：`confidence < min` 才弃权。"""
    classifier = StubBert(index=INTENT2ID["学习讲解"], confidence=INTENT_MIN_CONFIDENCE)
    result = classifier.classify_sync("什么是勾股定理")
    assert result is not None and result.label == "学习讲解"


def test_unknown_index_abstains() -> None:
    """模型吐出一个词表外的下标（配置被改坏）时弃权，而不是抛 IndexError。"""
    classifier = StubBert(index=99, confidence=0.99)
    assert classifier.classify_sync("什么是勾股定理") is None


def test_empty_query_abstains() -> None:
    classifier = StubBert()
    assert classifier.classify_sync("") is None
    assert classifier.classify_sync("   ") is None


def test_forward_failure_abstains_instead_of_propagating() -> None:
    classifier = StubBert(error=RuntimeError("CUDA out of memory"))
    assert classifier.classify_sync("什么是勾股定理") is None
    assert asyncio.run(classifier.classify("什么是勾股定理")) is None


def test_slow_forward_times_out_to_none() -> None:
    """超时就放弃这条结论：意图只是挑一条先验，不许拖住首 token（§6.1）。"""
    classifier = StubBert(delay=0.3, confidence=0.99)
    assert asyncio.run(classifier.classify("什么是勾股定理", timeout_ms=50)) is None


def test_disabled_classifier_returns_none_from_both_entries(tmp_path: Path) -> None:
    """没装 torch / 检查点缺失时 `_ensure()` 返回 False：同步与异步入口都必须是 `None`。"""
    classifier = BertIntentClassifier(checkpoint=tmp_path / "nope")
    classifier.disable("依赖缺失: stub")
    assert asyncio.run(classifier.classify("什么是勾股定理")) is None
