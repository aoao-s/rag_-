"""rag_system.core：在线检索与生成链路（PR3 待实现）。

计划模块（与 CLAUDE.md §7.2 对应）：
- rag_pipeline.py   一次提问的编排：意图/策略 → 检索 → 重排 → 证据门 → 生成 → 引用
- retriever.py      Milvus 混合检索（BGE-M3 稠密 + 稀疏），父块去重
- reranker.py       bge-reranker-v2-m3，GPU 常驻 + 批量 + 512 token 截断
- evidence_gate.py  证据充足性判定（不过则明说不知道，不硬编）
- citation.py       引用对齐（parent_id → 文件名 + 页码）
"""
