"""理解层（CLAUDE.md §7.7）：意图分类 → 路由档位。

三层分工，改行为只改对应那一层：

- `intent.py`：**怎么分类**（规则快路径，模型最终替换这一层）；
- `profiles.py`：**标签 → 档位的数据表**（不是 if/else，加档位 = 改表）；
- 编排层：只读 `IntentProfile`，不读标签字符串。

边界（§7.7 三条红线）：本包不检索、不生成、不写库；`IntentResult` 没有 `subject` 字段，
学科只由 `SlotResult` 决定。
"""

from understanding.intent import IntentClassifier
from understanding.profiles import PROFILES, profile_for

__all__ = ["IntentClassifier", "PROFILES", "profile_for"]
