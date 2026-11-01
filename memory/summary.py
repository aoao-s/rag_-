"""会话摘要 L1（CLAUDE.md §7.9）。

**抽取式**，不是生成式：把被压掉的轮次缩成"用户问过什么 / 结论是什么"的骨架，
而不是再调一次模型重写。理由有三条：

1. 摘要要参与**每一次**请求的 prompt 装配（§3.7 第 3 条的 400 token 硬预算），
   生成式摘要既贵又慢，而信息增益在"记住聊过什么"这件事上很小；
2. 本机没有可用的 LLM key（`LLM_API_KEY` 为空时 chat 直接兜底），
   生成式摘要会让整个记忆模块在无 key 环境里根本跑不起来——降级路径不该是"半残"；
3. 抽取式是**确定性的**：同一段对话压出来的摘要逐字相同，单测能断言，
   线上也能复现"为什么模型记得这件事、不记得那件事"。

有 key 时的生成式摘要（`MEM_EXTRACT_MODEL`）是后续增强，不是当前的必需项：
接口留在 `summarize(turns, llm=...)`，传了 LLM 才走模型，否则走 `digest`。
"""

from __future__ import annotations

from typing import Any, Protocol

from memory.session import Turn

# 摘要里保留的骨架标记：模型读 prompt 时靠它们分清"用户问过"与"已经答过"
_USER_MARK = "问"
_ASSIST_MARK = "答"


class _Completer(Protocol):
    enabled: bool

    async def complete(self, messages: list[dict[str, str]], *, budget_ms: int, model: str | None = ...) -> Any: ...


def digest(summary: str, turns: list[Turn], *, max_chars: int = 400) -> str:
    """把 `turns` 压成摘要骨架，接在旧摘要后面，按 `max_chars` 截断（保留最近的）。

    只保留用户问句与助手回答的首句——后者常有"知识库暂无该内容"这类兜底，
    保留它能避免模型在下一轮重复推荐已经被拒的内容。
    """
    parts: list[str] = []
    for turn in turns:
        text = " ".join((turn.content or "").split())
        if not text:
            continue
        if turn.role == "user":
            parts.append(f"{_USER_MARK}:{_clip(text, 60)}")
        else:
            parts.append(f"{_ASSIST_MARK}:{_clip(text, 40)}")
    merged = "；".join([p for p in [summary.strip(), *parts] if p])
    # 从尾部保留：越近的对话越重要；旧内容先被挤掉
    return merged[-max_chars:] if len(merged) > max_chars else merged


async def summarize(
    turns: list[Turn], *, llm: _Completer | None = None, max_chars: int = 400
) -> str:
    """有可用 LLM 时用模型压，否则用 `digest`。两者都保证长度上限。"""
    if llm is not None and getattr(llm, "enabled", False):
        try:
            result = await llm.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "把下面的对话压成不超过 200 字的中文摘要，只保留：用户问过什么主题、"
                            "已经确认的结论、未解决的问题。不要编造对话里没有的信息。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n".join(f"{t.role}:{t.content}" for t in turns)[:4000],
                    },
                ],
                budget_ms=2000,
            )
            text = " ".join((getattr(result, "text", "") or "").split())
            if text:
                return text[:max_chars]
        except Exception:  # noqa: BLE001 - 摘要失败只能退回抽取式，不能拖垮请求
            pass
    return digest("", turns, max_chars=max_chars)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"
