"""qa_system：FAQ 精准问答（BM25 精确检索 + 别名表 + 双缓冲热更新）。

对外只暴露 `qa_system.res.qa_search.FAQService` 与四个 HTTP 接口（qa_system/api.py），
rag_system 未命中承接时只依赖 `base.contracts.FAQHit`，不 import 本包内部。
"""
