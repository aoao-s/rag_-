"""标签注册表：意图、学科。

为什么单独成文件（见 CLAUDE.md §3.2 待拍板项）：
意图标签集合可能随产品口径变化（教学 6 类 / 招生 5 类），全项目只允许在这里定义，
业务代码一律 import 常量，禁止在分支里写字符串字面量——否则换口径要改几十处。

当前口径：`datasets/06_intent` 与 `datasets/README.md` 一致的 K12 教学 6 类。
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

# 兜底意图：分类模型不可用或置信度不足时使用，绝不硬路由到专业检索
INTENT_FALLBACK: Final[str] = "闲聊"

# 不需要检索的意图（直接走生成或兜底话术）
INTENT_NO_RETRIEVAL: Final[frozenset[str]] = frozenset({"闲聊"})

# 学科：与 datasets/04_faq 的 subject_name 保持一致
SUBJECTS: Final[tuple[str, ...]] = (
    "语文",
    "数学",
    "英语",
    "物理",
    "化学",
    "生物",
    "政治",
    "历史",
    "地理",
)


def validate_intent(label: str) -> str:
    """校验意图标签，非法值返回兜底意图（不抛异常，避免污染请求路径）。"""
    return label if label in INTENT_LABELS else INTENT_FALLBACK


def normalize_subject(raw: str | None) -> str | None:
    """把学科写法归一（去空白/后缀"学科"），未知返回 None。"""
    if not raw:
        return None
    value = raw.strip().removesuffix("学科").strip()
    return value if value in SUBJECTS else None
