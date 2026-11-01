"""多路结果融合（CLAUDE.md §7.2.7-B：多路召回 → 融合）。

**为什么不用 RRF**：证据门的阈值（0.74 / 0.01 / 0.20）是在 **Milvus 融合分**空间标定的。
RRF 会把分数压到 `1/(k+rank)` 量级，同一批证据过门时"分数够不够"完全没有可比性，
线上要么全过要么全不过。所以这里按"同分空间"融合：

    score = max(各任务的分数 × 任务权重)

多路都召回到同一父块时，取最高分并把命中片段合并（互相印证体现在 `child_hits` 变多，
由 `evidence_builder` 换算成"多路支持"并在排序里做平手裁决）。
这条路子牺牲了 RRF 对"名次靠前但绝对值低"的偏好，换来的是**阈值仍然可解释**——
宁可少一个花活，也不要一个说不清的门槛。
"""

from __future__ import annotations

from typing import Sequence

from base.contracts import Evidence, RetrievalResult


def fuse(results: Sequence[RetrievalResult], *, weights: Sequence[float] | None = None) -> list[Evidence]:
    """把多路检索结果合并成一份证据列表（按父块去重）。"""
    if not results:
        return []
    if len(results) == 1:
        return list(results[0].evidences)

    w = list(weights or [])
    if len(w) != len(results):
        w = [1.0] * len(results)

    merged: dict[str, Evidence] = {}
    support: dict[str, set[str]] = {}
    for weight, res in zip(w, results):
        for ev in res.evidences:
            scored = ev.score * weight
            key = ev.parent_id
            cur = merged.get(key)
            if cur is None:
                ev.score = scored
                merged[key] = ev
                support[key] = {ev.task_id} if ev.task_id else set()
                continue
            for child in ev.child_hits:
                if child and child not in cur.child_hits:
                    cur.child_hits.append(child)
            if ev.task_id:
                support[key].add(ev.task_id)
            if scored > cur.score:
                # 保留分高的那一条作为主证据（它决定了引用卡片里的页码与标题）
                best_child_hits = cur.child_hits
                cur.parent_id = ev.parent_id
                cur.text = ev.text
                cur.score = scored
                cur.rerank_score = ev.rerank_score
                cur.title = ev.title
                cur.source = ev.source
                cur.subject = ev.subject
                cur.doc_type = ev.doc_type
                cur.heading = ev.heading
                cur.page = ev.page
                cur.task_id = ev.task_id
                cur.child_id = ev.child_id
                cur.child_hits = best_child_hits

    # 平手时"被多路同时召回"的排在前面：单路偶然高分不如两路都找到可信
    return sorted(
        merged.values(),
        key=lambda e: (round(e.score, 6), len(support.get(e.parent_id, ()))),
        reverse=True,
    )
