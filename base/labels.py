"""标签注册表：意图、学科。

为什么单独成文件（见 CLAUDE.md §3.2 / §7.7）：
意图标签集合可能随产品口径变化，全项目只允许在这里定义，
业务代码一律 import 常量，禁止在分支里写字符串字面量——否则换口径要改几十处。

当前口径（2026-10-26 定案）：K12 教学 6 类，与 `datasets/06_intent` 一致；
招生 5 类已废弃。改口径只改本文件与 `understanding/profiles.py` 的档位表。
"""

from __future__ import annotations

from typing import Final

# 意图：K12 教学 6 类（数据集口径，5 100 条、六类均衡）
INTENT_LABELS: Final[tuple[str, ...]] = (
    "知识点查询",
    "题目讲解",
    "教案素材",
    "学习方法",
    "备课辅助",
    "闲聊",
)
# 标签 ↔ 模型输出下标的唯一映射：训练脚本、推理服务、评测都从这里取。
# 顺序由 INTENT_LABELS 决定，**不要**在别处再写一份 label2id。
INTENT2ID: Final[dict[str, int]] = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2INTENT: Final[dict[int, str]] = {i: label for i, label in enumerate(INTENT_LABELS)}

# 兜底意图：分类模型不可用或置信度不足时使用，绝不硬路由到专业检索。
# 过渡值 = "闲聊"（效果等于"不检索"）；PR5 接真实分类器时改为 "unknown"，
# 走 generic_clarify（全库检索 + 结尾追问），见 CLAUDE.md §7.7 硬规则 2。
INTENT_FALLBACK: Final[str] = "闲聊"

# 不需要检索的意图（直接走生成或兜底话术）
INTENT_NO_RETRIEVAL: Final[frozenset[str]] = frozenset({"闲聊"})

# 学科：与 datasets/04_faq 的 subject_name 保持一致
# **权威配置**，不靠语料扫描自动扩展——否则一个错别字就成了合法学科（见 CLAUDE.md §7.1.8）。
SUBJECTS: Final[tuple[str, ...]] = (
    "语文",
    "数学",
    "英语",
    "物理",
    "化学",
    "生物",
    "道德与法治",  # 现行教材口径；语料 97 条即用此名
    "历史",
    "地理",
)

# 写法归一：教材改名/口语简称 → canonical 名。
# "政治" 是旧口径（标签表原先只有它），现行语料统一写"道德与法治"；两者是同一门课的
# 不同叫法，映射错会让那 97 条永远召不回来（见 CLAUDE.md §7.1.8）。
# 这是**配置**不是代码逻辑：口径若由项目方改回"政治"，只改这张表。
SUBJECT_ALIASES: Final[dict[str, str]] = {
    "政治": "道德与法治",
    "思想政治": "道德与法治",
    "思想品德": "道德与法治",
    "道法": "道德与法治",
    "品德": "道德与法治",
}


def validate_intent(label: str) -> str:
    """校验意图标签，非法值返回兜底意图（不抛异常，避免污染请求路径）。"""
    return label if label in INTENT_LABELS else INTENT_FALLBACK


def normalize_subject(raw: str | None) -> str | None:
    """把学科写法归一（去空白/后缀"学科"/别名表），未知返回 None。"""
    if not raw:
        return None
    value = raw.strip().removesuffix("学科").strip()
    value = SUBJECT_ALIASES.get(value, value)
    return value if value in SUBJECTS else None


def is_valid_subject(raw: str | None) -> bool:
    """是否是已知学科写法（含别名）。公开 API 用它决定 400 还是放行。"""
    return normalize_subject(raw) is not None
