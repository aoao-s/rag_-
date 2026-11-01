"""写库字段截断与分区路由。

两件事放在同一个文件里，因为都是 `milvus_store` 的**纯函数**、都不需要连库：
`_clip` 决定"写进去会不会整份失败"，`partition_of` / `search_partitions` 决定"写进去还查不查得到"。
后者更隐蔽：分区写错不报错，只是检索静默少结果。
"""

from __future__ import annotations

from rag_system.store.milvus_store import (
    DOC_TYPE_PARTITIONS,
    EXCLUDED_BY_DEFAULT,
    _clip,
    partition_of,
    search_partitions,
)


def test_clip_keeps_short_text() -> None:
    assert _clip("初中数学教材", 4096) == "初中数学教材"


def test_clip_counts_bytes_not_chars() -> None:
    text = "数" * 1401  # 4 203 字节
    out = _clip(text, 4096)
    assert out != text
    assert len(out.encode("utf-8")) <= 4096
    assert out == "数" * 1365  # 4096-1 字节 = 1365 字，不切半个字符


def test_clip_empty_is_safe() -> None:
    assert _clip("", 64) == ""
    assert _clip(None, 64) == ""  # type: ignore[arg-type]


# ---- 分区路由 -------------------------------------------------------------


def test_partition_of_uses_doc_type() -> None:
    assert partition_of("textbook") == "textbook"
    assert partition_of("question_bank") == "question_bank"


def test_partition_of_unknown_type_gets_own_partition() -> None:
    # 并进 `_default` 就再也查不到（传了 partition_names 不再搜 `_default`），所以各占一个同名分区
    assert partition_of(None) == "other"
    assert partition_of("") == "other"
    assert partition_of("  ") == "other"
    assert partition_of("syllabus") == "syllabus"


def test_default_search_excludes_question_bank() -> None:
    """这是"默认排除题库"的唯一落点：不点名 doc_type 时题库分区根本不进 `partition_names`。"""
    picked = search_partitions(None, DOC_TYPE_PARTITIONS)
    assert picked is not None
    assert "question_bank" not in picked
    assert set(picked) == set(DOC_TYPE_PARTITIONS) - set(EXCLUDED_BY_DEFAULT)


def test_explicit_doc_types_can_reach_question_bank() -> None:
    """`kb_stepwise` 点名题库时必须真的搜得到，否则那 1.8 万块等于白入。"""
    picked = search_partitions(("question_bank", "exam"), DOC_TYPE_PARTITIONS)
    assert set(picked or []) == {"question_bank", "exam"}


def test_unknown_doc_types_fall_back_to_unrestricted() -> None:
    # 拼错类型名要退化成全库检索，而不是搜一个必然为空的分区集合
    assert search_partitions(("物理",), DOC_TYPE_PARTITIONS) is None


def test_no_partitions_yet_is_unrestricted() -> None:
    # 建集合与建分区之间有窗口期；空列表不能变成"搜 0 个分区"
    assert search_partitions(None, []) is None
    assert search_partitions(("textbook",), []) is None
