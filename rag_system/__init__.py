"""rag_system：知识库检索与生成（PR3，尚未实现）。

目录约定与旧项目一致：
- core/       检索编排、重排、证据门、生成（rag_pipeline / retriever / reranker / evidence_gate）
- data/       离线产物（切块清单、embedding 缓存）
- docs/       原始文档（不入库，按需同步）
- document_loaders/  各类文档解析器（pdf/docx/md/html）
- text_splitters/    父块/子块切分策略
- models/     本地模型权重（BGE-M3、bge-reranker-v2-m3、BERT 意图模型），已在 .gitignore
"""
