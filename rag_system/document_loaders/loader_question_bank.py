"""题库加载器：`题库.json` → 一题一段的 Markdown。

为什么单独一个加载器：`datasets/初高中教材教案试卷-按年级/**/题库.json` 共 46 个文件、
一万多道题，是"题目讲解"最直接的证据——题干、答案、解析齐全
（不像 `datasets/03_question_bank`，那里的 answer 多数只有一个选项字母、analysis 为空）。
没有加载器，这批语料连入库都进不去（`.json` 不在任何既有加载器的 `extensions` 里）。

**两套 schema 都要认**（2026-10-29 实测，同一批语料里并存，不能只认一套）：

| 学段 | 字段 | 例子 |
|---|---|---|
| 初中 | `subject / ques_type / ques_difficulty / ques_content / ques_answer / ques_analyze / ques_knowledges` | `{"ques_content": "题目内容: （1）计算：|√3−3| + …"}` |
| 高中 | `year / category / question / answer / analysis / source` | `{"year": "2010", "question": "1. （5 分) 已知集合 $A=…", "answer": ["D"]}` |

所以按**字段别名**取，不按学段分支：`question` 与 `ques_content` 是同一个东西的两种写法，
再加一个学段的第三条别名也不会改渲染逻辑。只认一种写法时，26 个高中题库会整体
`LoaderError` 掉——那是语料里题量最大的一块，掉了等于"题目讲解"这一类问题没有证据。

渲染成 Markdown 的标题层级（每道题一个 `###`）是刻意的：切分器 `_sections()` 按标题切段，
一题一段 → **一道题就是一个父块**。若把整份 JSON 拍平成连续正文，`_pack()` 会把
四五道题装进同一个父块，"引用到第 3 题"时会连同第 4、5 题一起塞进上下文。

字段口径：

- `ques_content` 有的带 `题目内容: ` 前缀，去掉——它会把几百道题的题首都变成同一个词，
  BM25 上等于给每道题加了一份共享噪声。
- **选项行必须留着**（`A. …` / `B. …`）：高中题库的 `answer` 只有一个字母 `["D"]`，
  把选项当"格式噪声"删掉，答案就变成一个无所指的字母，检索到的证据也答不了"这题选什么"。
- `ques_analyze` / `analysis` 为空时整段不渲染，不留"解析：（空）"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from base.labels import STAGES, normalize_subject
from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError, guess_subject

_QUESTION_PREFIXES = ("题目内容:", "题目内容：", "题目:", "题目：")

# 字段别名：同一个语义在不同来源里可能叫三个名字。顺序即优先级
_STEM_KEYS = ("ques_content", "question", "题目内容", "stem")
_ANSWER_KEYS = ("ques_answer", "answer", "答案")
_ANALYZE_KEYS = ("ques_analyze", "analysis", "解析")
_POINTS_KEYS = ("ques_knowledges", "knowledge", "知识点")

# 元信息行（`题型：X｜难度：Y｜年份：2010｜来源：…`）：只放有值的，空值不占位
_META_KEYS = (
    ("题型：", ("ques_type", "question_type", "type")),
    ("难度：", ("ques_difficulty", "difficulty")),
    ("年份：", ("year",)),
    ("分类：", ("category",)),
    ("来源：", ("source",)),
)


class QuestionBankLoader(BaseDocumentLoader):
    """`.json` 题库 → Markdown（每题一个 `###` 小节）。"""

    extensions = (".json",)

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path).resolve()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LoaderError(f"题库 JSON 读取失败 {p}：{exc}") from exc
        if not isinstance(raw, list) or not raw:
            raise LoaderError(f"题库 JSON 不是非空数组：{p}")
        items = [q for q in raw if isinstance(q, dict)]
        if not items or not _looks_like_bank(items):
            # 宁可报错也不要静默产出空文档：`.json` 是很常见的格式，
            # 走到这里说明这份不是题库，应该由人决定要不要支持它
            raise LoaderError(f"不是题库结构（既无 ques_content 也无 question）：{p}")

        text, rendered = render_questions(items)
        return LoadedDoc(
            text=text,
            source=str(p),
            doc_type="question_bank",
            subject=guess_subject(str(p)) or _subject_from_items(items),
            title="",  # 交给工厂拼 `初中七年级化学题库`（stem 是通用词）
            meta={"questions": len(items), "rendered": rendered, "parser": "json"},
        )


def _looks_like_bank(items: list[dict[str, Any]]) -> bool:
    """过半条目有题干就当题库。

    不用 `all()`：真实题库里总有空题/占位行，一条不合规就整份拒收，
    等于让一行的脏数据把 800 道好题挡在库外。全部为空才算"不是题库"。
    """
    with_stem = sum(1 for q in items if _first(q, _STEM_KEYS))
    return with_stem * 2 >= len(items) and with_stem > 0


def render_questions(items: list[dict[str, Any]]) -> tuple[str, int]:
    """把题目列表渲染成 Markdown；返回 `(正文, 实际渲染条数)`。空解析不占位。"""
    blocks: list[str] = []
    for q in items:
        stem = _clean(str(_first(q, _STEM_KEYS) or ""))
        if not stem:
            continue
        lines = [f"### 第 {len(blocks) + 1} 题"]
        meta = "｜".join(value for value in (_meta(q, label, keys) for label, keys in _META_KEYS) if value)
        if meta:
            lines.append(meta)
        lines.append(stem)
        answer = _joined(_first(q, _ANSWER_KEYS))
        if answer:
            lines.append(f"答案：{answer}")
        analyze = _clean(str(_first(q, _ANALYZE_KEYS) or ""))
        if analyze:
            lines.append(f"解析：{analyze}")
        points = _joined(_first(q, _POINTS_KEYS))
        if points:
            lines.append(f"知识点：{points}")
        blocks.append("\n".join(lines))
    if not blocks:
        raise LoaderError("题库渲染后为空")
    return "\n\n".join(blocks), len(blocks)


def _first(q: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按别名顺序取第一个非空字段。"""
    for key in keys:
        value = q.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _meta(q: dict[str, Any], label: str, keys: tuple[str, ...]) -> str:
    value = _first(q, keys)
    return f"{label}{value}" if value not in (None, "") else ""


def _clean(text: str) -> str:
    value = text.strip()
    for prefix in _QUESTION_PREFIXES:
        if value.startswith(prefix):
            return value[len(prefix) :].strip()
    return value


def _joined(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        return "；".join(str(x).strip() for x in raw if str(x).strip())
    return ""


def _subject_from_items(items: list[dict[str, Any]]) -> str | None:
    """JSON 内的 `subject` 写作 `初中化学`：去掉学段前缀再归一。"""
    for q in items:
        raw = str(q.get("subject") or "").strip()
        if not raw:
            continue
        bare = raw
        for stage in STAGES:
            if bare.startswith(stage):
                bare = bare[len(stage) :]
                break
        subject = normalize_subject(bare)
        if subject:
            return subject
    return None
