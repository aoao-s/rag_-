"""证据门（CLAUDE.md §3.3.5 / §7.2.7-B 第 6 步）。

门只回答一个问题：**这批证据够不够支撑一次作答？** 它不做重试、不改参数、不调模型。
把重试决策放在这里，等于让"判定"和"行动"耦合——四轮标定的教训是：一旦门会自己调阈值，
就再也说不清某条命中是"召回找到的"还是"门槛被降到它脚下"。

四条判据并列（任一不过就不生成）：

1. `top1 融合分 < threshold`；
2. 与**另一篇文档**最好证据的分差 `< margin`（同文档多块算互相印证）；
3. 只有 1 块孤证时要求 `top1 ≥ threshold + single_doc_bonus`；
4. 问句实词在 top1 证据里的覆盖率 `< min_qcov`（挡"同主题不同知识点"）。

第五条（阶段 4.2）是**多路覆盖度**：`SUB_QUERY` 计划了几路，就要求每一路都产出过证据。
前四条全都只看 top1，而多路分解最典型的失败恰恰是"其中两路一条都没召回，第三路拿到了
一条高分"——此时 top1 与 margin 都很漂亮，门会说 sufficient，模型拿着三分之一的信息
答一个问了三件事的问题。这条**只在计划任务数 ≥2 时才判**：单任务时"某一路没证据"
就是"整体没证据"，前四条已经判过。

判据顺序写死（与 `reasons` 的归因顺序一致）：空结果（按过滤条件分诊）→ `below_threshold`
→ `single_weak_evidence` → `conflicting` → `ambiguous` → `low_query_coverage` →
`missing_task_evidence` → `sufficient`。**顺序不能乱**：同一份观测既算"分数不够"又算
"某一路缺失"，两张诊断表加起来就会超过总数。

默认 0.74 / 0.01 / 0.20 由黄金集 763 条标定（`docs/analysis/rag_threshold_report.md`）。
`conflicts` 由 `evidence_builder.detect_conflicts()` 提供：两篇文档对同一个量给出了不同的数，
这种情况**必须并列返回或转人工**，禁止静默挑高分那条（§7.2.7-C）。
"""

from __future__ import annotations

from typing import Sequence

from base.contracts import (
    EvidenceDecision,
    GateStatus,
    RetrievalFilters,
    RetrievalResult,
    RetryAction,
)
from base.logger import get_logger

log = get_logger(__name__)

# 兼容旧导入名：`GateDecision` 曾经是本模块自己定义的布尔结构，现在是契约里的 `EvidenceDecision`
GateDecision = EvidenceDecision

# 虚词表按**整词**匹配，不做前缀匹配：`是否` / `不等式` / `三角形` 都是实词，
# 一律用前缀会把它们当虚词删掉，覆盖率因此失真。
# 多字虚词要逐个列出——jieba 会把"怎么办"切成一个词，只删"怎么"等于没删，
# 纯虚词的问句就会算出覆盖率 0 而不是"判不了"（`tests/test_rag_retriever.py` 守着这条）。
_STOP_WORDS = frozenset(
    "的 了 是 在 和 有 就 不 也 都 而 与 及 或 这 那 我 你 他 它 们 个 中 上 下 为 以 对 从 到 把 被 让 "
    "什么 怎么 怎样 如何 为什么 哪些 哪个 多少 请问 一下 "
    "怎么办 怎么样 咋办 咋样 为何 干什么 是什么 有没有 能不能 可不可以".split()
)


def query_coverage(query: str, text: str) -> float:
    """问句实词在证据里的覆盖率。全是虚词时返回 1.0（判不了就不挡人）。"""
    words = [w for w in _content_words(query) if len(w) >= 2]
    if not words:
        return 1.0
    hit = sum(1 for w in words if w in text)
    return hit / len(words)


def _content_words(text: str) -> list[str]:
    try:
        import jieba

        raw = list(jieba.cut(text))
    except Exception:  # noqa: BLE001 - 没装 jieba 时退化为按字切，不影响判定方向
        raw = list(text)
    out: list[str] = []
    for w in raw:
        w = w.strip()
        if not w or all(ch in "，。？！；：、（）()【】《》 \t\n\r?.,!;:+-*/=" for ch in w):
            continue
        if w in _STOP_WORDS:
            continue
        out.append(w)
    return out


def evaluate(
    result: RetrievalResult,
    *,
    threshold: float = 0.35,
    margin: float = 0.03,
    min_docs: int = 1,
    single_doc_bonus: float = 0.10,
    min_qcov: float = 0.20,
    filters: RetrievalFilters | None = None,
    conflicts: Sequence[str] = (),
) -> EvidenceDecision:
    evidences = result.evidences
    if not evidences:
        # 有业务过滤时先怀疑"过滤太窄"，而不是"知识库没有"——两者的处置完全不同。
        # 归因再细分一层：只有 doc_type 空 vs 只有 subject 空，处置方向其实不同
        # （前者多半是意图档位收窄过头，后者多半是学科判错或该学科没入库），
        # 合成一句 `no_candidates_with_filter` 时排障的人得自己再去猜是哪一项。
        reason = _empty_reason(filters)
        if reason != "no_candidates":
            return _decision("insufficient", (reason,), "relax_business_filter", result, 0.0, 0.0, 0, 0.0)
        return _decision("insufficient", ("no_candidates",), "retry_direct", result, 0.0, 0.0, 0, 0.0)

    top = evidences[0]
    top_score = top.score
    best_by_doc = _best_by_doc(evidences)
    n_docs = len(best_by_doc)
    others = [s for doc, s in best_by_doc.items() if doc != top.doc_id]
    margin_value = top_score - max(others) if others else top_score
    qcov = query_coverage(result.query, top.text)
    missing = _missing_tasks(result)
    common = (top_score, margin_value, n_docs, qcov)

    if top_score < threshold:
        return _decision("insufficient", ("below_threshold",), "retry_original_query", result, *common)

    if len(evidences) < max(min_docs, 2) and top_score < threshold + single_doc_bonus:
        # 孤证：不是"歧义"，是"只有一条弱证据"，处置是扩同文档邻域而不是追问
        return _decision(
            "insufficient", ("single_weak_evidence",), "expand_neighborhood", result, *common
        )

    if conflicts:
        return _decision("conflicting", tuple(conflicts), "return_conflict", result, *common)

    if others and margin_value < margin:
        return _decision("ambiguous", ("ambiguous",), "ask_disambiguate", result, *common)

    if qcov < min_qcov:
        return _decision("insufficient", ("low_query_coverage",), "rewrite_query", result, *common)

    # 阶段 4.2：**多路分解必须每一路都有证据**，不能只看整体 top1。
    #
    # 只看 top1 会漏掉最典型的一种失败：「比较一次函数和二次函数，并各出三道练习题」
    # 拆成三路，其中两路一条都没召回，但第三路拿到一条 0.95 的证据——此时 top1 很高、
    # margin 也很漂亮，门会说 sufficient，然后模型拿着三分之一的信息答一个"问了三件事"
    # 的问题。用户读到的是一段自信的、只回答了三分之一的文字。
    #
    # 处置是 `retry_direct`（不追问、不换阈值）：分解这条路走不通时，退回"把整句当一个问题"
    # 再检一次，比继续用残缺的多路结果生成要诚实。`retry_direct` 在
    # `RagPipeline._REPAIRABLE` 里，**修复仍由 pipeline 统一执行**（至多一次、修完重新过门，
    # 阶段 4.4），门只给建议，不自己重试。
    #
    # `len(result.tasks) >= 2` 这条前置是必要的：单任务时"某一路没证据"与"整体没证据"
    # 是同一件事，上面的空结果/below_threshold 已经判过了；而单任务的检索器实现里
    # `task_id` 可能是空的（诊断接口、单测替身都这样），拿它去比 `missing` 会把
    # 一个正常结果判成"某一路缺失"。
    if len([t for t in result.tasks if t]) >= 2 and missing:
        return _decision("insufficient", ("missing_task_evidence",), "retry_direct", result, *common)

    return _decision("sufficient", ("ok",), "none", result, *common, missing=())


def _empty_reason(filters: RetrievalFilters | None) -> str:
    """空结果归因（实施手册 §5.2 的原因表）。

    | 约束 | reason |
    |---|---|
    | 无业务过滤 | `no_candidates`（知识库真的没有 → 换 DIRECT 重检一次） |
    | 只有 `doc_types` | `doc_type_too_narrow`（意图档位收窄过头） |
    | 只有 `subjects` | `subject_filter_empty`（学科判错 / 该学科没入库） |
    | 两项都有，或只有 stages/grades | `no_candidates_with_filter`（说不清是哪一项，一并放宽） |

    为什么只到"一两项"这一级：再往下细分（比如 stages 与 grades 各一个原因）就会
    产生一批**永远为 0 的桶**，报告里看着很细，实际没有任何判据在用它。三条与手册对齐、
    多出来的合并成兜底，比反过来好——少一个 reason 只会让日志粗一点，
    多一个没人处理的 reason 会让下游的 `if` 永远走不到。
    """
    if filters is None or not filters.business_restricted:
        return "no_candidates"
    only_doc_types = bool(filters.doc_types) and not (
        filters.subjects or filters.stages or filters.grades
    )
    if only_doc_types:
        return "doc_type_too_narrow"
    only_subjects = bool(filters.subjects) and not (
        filters.doc_types or filters.stages or filters.grades
    )
    if only_subjects:
        return "subject_filter_empty"
    return "no_candidates_with_filter"


def _decision(
    status: GateStatus,
    reasons: tuple[str, ...],
    action: RetryAction,
    result: RetrievalResult,
    top_score: float,
    margin: float,
    n_docs: int,
    qcov: float,
    *,
    missing: tuple[str, ...] | None = None,
) -> EvidenceDecision:
    d = EvidenceDecision(
        status=status,
        reasons=reasons,
        missing_tasks=missing if missing is not None else _missing_tasks(result),
        retry_action=action,
        top_score=top_score,
        margin=margin,
        n_docs=n_docs,
        qcov=qcov,
    )
    log.info(
        "rag_gate",
        status=d.status,
        reasons=",".join(d.reasons),
        action=d.retry_action,
        top_score=round(d.top_score, 4),
        margin=round(d.margin, 4),
        n_docs=d.n_docs,
        qcov=round(d.qcov, 4),
    )
    return d


def _best_by_doc(evidences) -> dict[str, float]:
    best: dict[str, float] = {}
    for e in evidences:
        doc = e.doc_id
        if e.score > best.get(doc, float("-inf")):
            best[doc] = e.score
    return best


def _missing_tasks(result: RetrievalResult) -> tuple[str, ...]:
    """计划里有、但一条证据都没产出的检索任务（SUB_QUERY 时用来判断"哪一路没找到"）。

    判据是 `RetrievalResult.task_hits`（召回期记录，见 `contracts.RetrievalResult`），
    **不是** `evidences` 里的 `task_id`：融合只给每个父块留一个 `task_id`，`final_n`
    又只留前几条，用证据反推会把"两路都命中、只是没进最终 top_n"的那一路判成没产出，
    白跑一次 `retry_direct` 并把 SUB_QUERY 计划换成单路。

    与证据里的 `task_id` 取并集，是为了兼容没有填 `task_hits` 的调用方（单测与
    诊断接口自己拼 `RetrievalResult`）：任一来源说"产出过"就不算缺失。
    """
    planned = [t for t in result.tasks if t]
    if not planned:
        return ()
    seen = set(result.task_hits) | {e.task_id for e in result.evidences if e.task_id}
    return tuple(t for t in planned if t not in seen)
