"""HyDE 生成网关测试。

验证：
1. FEATURE_HYDE=false 时不创建生成器；
2. HyDE 显式传入 LLM_QA_MODEL；
3. HyDE 显式传入 HYDE_BUDGET_MS。
"""

from __future__ import annotations

import asyncio
from typing import Any

from base.config import Settings
from main import Container


class StubLLM:
    """模拟云端 LLM 网关。"""

    def __init__(self) -> None:
        self.enabled = True
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        budget_ms: int,
        model: str | None = None,
    ) -> Any:
        self.calls.append(
            {
                "messages": messages,
                "budget_ms": budget_ms,
                "model": model,
            }
        )

        class Result:
            text = "这是用于检索的假设性答案"

        return Result()


def test_hyde_gateway_uses_explicit_model_and_budget() -> None:
    """HyDE 必须使用配置中的生成模型和独立预算。"""

    settings = Settings(
        feature_hyde=True,
        hyde_budget_ms=1500,
        llm_qa_model="deepseek-chat",
    )

    container = Container(settings)

    stub = StubLLM()
    container.llm = stub

    generator = container._hyde_gateway(settings)

    assert generator is not None

    result = asyncio.run(
        generator("什么是勾股定理")
    )

    assert result == "这是用于检索的假设性答案"

    assert len(stub.calls) == 1

    call = stub.calls[0]

    # 必须显式使用正式回答模型
    assert call["model"] == "deepseek-chat"

    # 不能使用 LLM_TIMEOUT_S=60 作为 HyDE 预算
    assert call["budget_ms"] == 1500


def test_hyde_disabled_returns_none() -> None:
    """关闭 FEATURE_HYDE 后，不应该创建生成器。"""

    settings = Settings(
        feature_hyde=False,
    )

    container = Container(settings)

    generator = container._hyde_gateway(settings)

    assert generator is None