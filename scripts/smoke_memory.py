"""会话与记忆冒烟（CLAUDE.md §7.9）。

为什么不用 pytest：这一条验的是**装配**，不是单元逻辑——真起服务、真发 HTTP、
真写 MySQL + Redis，验完把这家临时用户的记忆删干净。单测能证明"规则对不对"，
证明不了「`Container.build_memory` 把四个东西接上了」。

四条断言，对应 §7.9 的四条边界：
1. 同一 `session_id` 的指代句**改写出了上一轮的实体**（L0 窗口生效）；
2. 用户自述被抽成**两条**原子记忆（`我上初三` + `数学不太好`）；
3. 知识型问句**不产生**记忆（不污染）；
4. 别人的记忆一律当不存在（`X-User-Id` 隔离）。

用法：`python scripts/smoke_memory.py --base http://127.0.0.1:8000`
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

import httpx


def _chat(client: httpx.Client, user: str, query: str, session_id: str | None) -> dict:
    resp = client.post(
        "/v1/chat",
        json={"query": query, "session_id": session_id},
        headers={"X-User-Id": user},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


def _atoms(client: httpx.Client, user: str) -> dict:
    resp = client.get("/v1/memory/atoms", headers={"X-User-Id": user}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    user = f"smoke-{uuid.uuid4().hex[:8]}"
    other = f"smoke-{uuid.uuid4().hex[:8]}"
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  | {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    with httpx.Client(base_url=args.base) as client:
        ready = client.get("/readyz", timeout=30).json()
        check("readyz", ready.get("status") == "ready" and ready.get("memory") is True,
              f"memory={ready.get('memory')} backend={ready.get('memory_backend')}")
        if ready.get("status") != "ready":
            return 1

        turn1 = _chat(client, user, "我上初三，数学不太好", None)
        sid = turn1.get("session_id")
        check("turn1 签发 session_id", bool(sid), f"source={turn1.get('source')}")

        turn2 = _chat(client, user, "它的公式是什么", sid)
        rewritten = turn2.get("rewritten") or ""
        check("turn2 指代改写带上上文", "数学" in rewritten or "初三" in rewritten, f"rewritten={rewritten!r}")

        # 记忆写入是异步的（`_persist_later`），给它一点时间，但不要 sleep 到掩盖问题
        for _ in range(20):
            page = _atoms(client, user)
            if page.get("total", 0) >= 2:
                break
            time.sleep(0.2)
        keys = sorted(item["fact_key"] for item in page.get("items", []))
        check("用户自述抽出两条原子记忆", page.get("total") == 2, f"fact_keys={keys}")

        before = page.get("total", 0)
        _chat(client, user, "勾股定理怎么证明", sid)
        time.sleep(1.5)
        after = _atoms(client, user).get("total", 0)
        check("知识型问句不产生记忆", after == before, f"{before} -> {after}")

        check("别的用户看不到这些记忆", _atoms(client, other).get("total") == 0)
        code = client.get("/v1/memory/atoms", timeout=10).status_code
        check("缺少 X-User-Id 一律 401", code == 401, f"status={code}")

        # 清理：删光这家临时用户的记忆，冒烟不留在库里。
        # `confirm=true` 是 §7.3.4 的二次确认，缺了就 400——这条本身也顺带验了。
        cleared = client.delete(
            "/v1/memory/all", params={"confirm": "true"}, headers={"X-User-Id": user}, timeout=30
        )
        check("清空临时用户记忆", cleared.status_code == 200 and _atoms(client, user)["total"] == 0,
              f"status={cleared.status_code}")

    print(f"\n{'全部通过' if not failures else '失败: ' + ', '.join(failures)}")
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
