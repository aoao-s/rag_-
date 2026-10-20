"""memory：四层记忆模块（PR4，尚未实现）。

设计见 docs/EduRAG-Pro-记忆系统设计.md，目录规划：
- service.py     MemoryService（召回：画像 / 原子记忆 / 会话摘要 / 逐字历史，RRF 融合）
- writer.py      写入链路（Redis Stream 异步消费：门控 → 抽取 → 去重 → 冲突 → 摘要 → 落库）
- extractor.py / dedupe.py / conflict.py / summarizer.py / policy.py
- store/         mysql_repo / redis_repo / vector_repo（Milvus）
"""
