"""阶段 1 / 阶段 2 入口联调演示（只打入口，不做全链路验收）。

阶段 1 = 统一请求入口 + 准入校验：所有请求都从 `POST /v1/chat` 进，
         非法请求在这里就回统一形状的 400/413，不进 FAQ、不进检索。
阶段 2 = 问题理解：入口放行之后，归一化 / 改写 / 意图 / 槽位 的结论
         直接体现在同一个响应体的 `rewritten / rewrite_method / intent /
         intent_method / intent_confidence / profile / strategy` 上。

用法：
    python scripts/demo_stage12_entry.py            # 输出到 docs/analysis/stage12_entry_demo.md
    python scripts/demo_stage12_entry.py --stdout   # 打到控制台（需 UTF-8 终端）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import httpx

BASE = "http://127.0.0.1:8000"
OUT = Path(__file__).resolve().parent.parent / "docs" / "analysis" / "stage12_entry_demo.md"

# 响应体里属于"阶段 2 理解层"的字段：入口放行与否看不出来，这一层才看得出来
UNDERSTANDING_KEYS = (
    "rewritten",
    "rewrite_method",
    "intent",
    "intent_method",
    "intent_confidence",
    "profile",
    "strategy",
)


def _chat(client: httpx.Client, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    resp = client.post(f"{BASE}/v1/chat", json=payload, headers=headers or {})
    cost = int((time.perf_counter() - started) * 1000)
    try:
        body = resp.json()
    except ValueError:
        body = {"_raw": resp.text[:200]}
    return {"http": resp.status_code, "cost_ms": cost, "body": body}


def _pick_fast_question(client: httpx.Client) -> str:
    """挑一条 FAQ 高置信命中的问句：走 FAQ 直返，几毫秒返回，不占演示时间。"""
    for q in (
        "什么是勾股定理？",
        "什么是二次函数？",
        "三角形的内角和是多少度？",
        "什么是光合作用？",
        "什么是牛顿第一定律？",
    ):
        resp = client.get(f"{BASE}/v1/faq/search", params={"q": q})
        if resp.status_code == 200 and resp.json().get("tier") == "high":
            return q
    return "什么是勾股定理？"


def _fmt(body: dict[str, Any]) -> str:
    keep = (
        "request_id",
        "status",
        "source",
        "error_code",
        "session_id",
        "role",
        "intent",
        "intent_method",
        "intent_confidence",
        "profile",
        "strategy",
        "rewritten",
        "rewrite_method",
        "degraded",
        "detail",
    )
    picked = {k: body[k] for k in keep if k in body}
    if "answer" in body and body["answer"]:
        picked["answer"] = (body["answer"][:60] + "…") if len(body["answer"]) > 60 else body["answer"]
    return json.dumps(picked, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()

    lines: list[str] = ["# 阶段 1 / 阶段 2 入口演示（实测输出）", ""]
    lines.append(f"服务：`{BASE}`　生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    with httpx.Client(timeout=180.0) as client:
        health = client.get(f"{BASE}/readyz").json()
        lines += ["## 0. 服务就绪", "", "```json", json.dumps(health, ensure_ascii=False), "```", ""]

        fast_q = _pick_fast_question(client)

        lines += [
            "## 阶段 1：统一请求入口 + 准入校验",
            "",
            "入口只有一个：`POST /v1/chat`。下面每条都是**真实的 HTTP 往返**，",
            "非法的请求在准入层就返回，FAQ 与检索一次都不会被调用。",
            "",
        ]

        cases: list[tuple[str, dict[str, Any], dict[str, str] | None]] = [
            ("① 空问句 → 400 empty_query", {"query": "   "}, None),
            ("② 401 字（上限 400）→ 400 query_too_long", {"query": "数" * 401}, None),
            (
                "③ 非法学科 → 400 subject_invalid",
                {"query": "什么是勾股定理？", "subject": "天文"},
                None,
            ),
            ("④ 未知字段 qurey → 400 invalid_request", {"qurey": "什么是勾股定理？"}, None),
            (
                "⑤ 请求体超 64 KB → 413 request_too_large",
                {"query": "什么是勾股定理？", "pad": "x" * 70000},
                None,
            ),
            (
                f"⑥ 合法请求（FAQ 高置信直返）→ 200：「{fast_q}」",
                {"query": fast_q},
                {"X-User-Id": "demo-stu-001"},
            ),
        ]

        sid: str | None = None
        for title, payload, headers in cases:
            result = _chat(client, payload, headers)
            body = result["body"]
            if title.startswith("⑥"):
                sid = body.get("session_id")
            lines += [
                f"### {title}",
                "",
                f"HTTP **{result['http']}**　耗时 {result['cost_ms']} ms",
                "",
                "```json",
                _fmt(body),
                "```",
                "",
            ]

        # ⑦ 会话沿用 + 越权
        if sid:
            own = _chat(
                client,
                {"query": fast_q, "session_id": sid},
                {"X-User-Id": "demo-stu-001"},
            )
            other = _chat(
                client,
                {"query": fast_q, "session_id": sid},
                {"X-User-Id": "demo-stu-002"},
            )
            lines += [
                "### ⑦ 同一个 X-User-Id 带回 session_id → 沿用；换个人带同一个 → 400 invalid_session",
                "",
                f"原 session_id：`{sid}`",
                "",
                f"- 本人带回：HTTP **{own['http']}**，返回 session_id `{own['body'].get('session_id')}`"
                f"（{'沿用' if own['body'].get('session_id') == sid else '未沿用'}）",
                f"- 他人带回：HTTP **{other['http']}**，error_code `{other['body'].get('error_code')}`",
                "",
                "```json",
                _fmt(other["body"]),
                "```",
                "",
            ]

        lines += [
            "## 阶段 2：问题理解（归一化 / 改写 / 意图 / 槽位 / 复合问句）",
            "",
            "入口放行之后，理解的结论直接写在**同一个响应体**里，不需要另开接口：",
            "",
            "- `rewritten` / `rewrite_method`：改写成什么、用哪条路（rule / llm / none）",
            "- `intent` / `intent_method` / `intent_confidence`：6 类意图 + 谁给的结论",
            "- `profile`：意图翻译成的路由档位（下游只读它，不读标签）",
            "- `strategy`：检索策略（direct / sub_query / backtrack / hyde）",
            "",
        ]

        u1 = _chat(
            client,
            {"query": "二次函数　的顶点式　怎么求？"},  # 全角空格，看归一化
            {"X-User-Id": "demo-stu-001"},
        )
        lines += [
            "### ⑧ 归一化 + 意图 + 档位（同一响应体）",
            "",
            f"问句（含全角空格）：`二次函数　的顶点式　怎么求？`　HTTP **{u1['http']}**"
            f"　耗时 {u1['cost_ms']} ms",
            "",
            "```json",
            json.dumps(
                {k: u1["body"].get(k) for k in UNDERSTANDING_KEYS}, ensure_ascii=False, indent=2
            ),
            "```",
            "",
        ]

        # ⑨ 指代：同一 session 追问
        if sid:
            _chat(
                client,
                {"query": "二次函数的顶点式怎么求", "session_id": sid},
                {"X-User-Id": "demo-stu-001"},
            )
            u2 = _chat(
                client,
                {"query": "它的对称轴呢", "session_id": sid},
                {"X-User-Id": "demo-stu-001"},
            )
            lines += [
                "### ⑨ 指代消解：同一会话追问「它的对称轴呢」",
                "",
                f"上一轮：`二次函数的顶点式怎么求`　本轮：`它的对称轴呢`"
                f"　HTTP **{u2['http']}**　耗时 {u2['cost_ms']} ms",
                "",
                "```json",
                json.dumps(
                    {
                        k: u2["body"].get(k)
                        for k in ("rewritten", "rewrite_method", "intent", "profile", "strategy")
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]

        # ⑩ 复合问句
        u3 = _chat(
            client,
            {"query": "比较一次函数和二次函数的区别，并分别给我出三道练习题。"},
            {"X-User-Id": "demo-stu-001"},
        )
        lines += [
            "### ⑩ 复合问句拆分 → strategy = sub_query",
            "",
            f"HTTP **{u3['http']}**　耗时 {u3['cost_ms']} ms",
            "",
            "```json",
            json.dumps(
                {
                    "strategy": u3["body"].get("strategy"),
                    "rewritten": u3["body"].get("rewritten"),
                    "intent": u3["body"].get("intent"),
                    "profile": u3["body"].get("profile"),
                    "stages": u3["body"].get("stages"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]

        # ⑪ FAQ 三档（入口处的分流依据）
        lines += [
            "## 附：FAQ 三档（阶段 1 放行后的第一道分流，`/v1/faq/search` 可直接看）",
            "",
            "| 问句 | tier | hit | score | reason |",
            "|---|---|---|---|---|",
        ]
        for q in (
            "什么是二次函数？",
            "牛顿第一定律是什么？",
            "等腰直角三角形的三边关系是哪些？",
            "勾股定理",
            "今天天气怎么样",
        ):
            resp = client.get(f"{BASE}/v1/faq/search", params={"q": q})
            if resp.status_code != 200:
                continue
            d = resp.json()
            lines.append(
                f"| {q} | `{d.get('tier')}` | {d.get('hit')} | {d.get('score'):.4f} "
                f"| `{d.get('reason')}` |"
            )
        lines += [
            "",
            "`high` 直接作答且**跳过检索**；`mid` 跑 RAG、不过证据门就追问，",
            "其 top3 问答对作为辅助上下文进 prompt（`FEATURE_FAQ_CANDIDATES`）；`low` 候选不进 prompt。",
            "",
        ]

    text = "\n".join(lines)
    if args.stdout:
        print(text)
    else:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text, encoding="utf-8")
        print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
