"""qa_system.res：检索实现（对标旧项目 qa_system/res 的 qa_search / qa_db / qa_cache）。

模块分工：
- qa_normalize.py  文本归一化（全半角、标点、语气词、数字、口语同义、别名）
- qa_corpus.py     BM25 索引与倒排（分学科、jieba 分词、IDF 非负）
- qa_matcher.py    命中三条件判定（阈值 / top1-top2 间隔 / 学科一致）
- qa_db.py         FAQ 仓库（MySQL 实现 + 内存实现，供测试与降级）
- qa_cache.py      Redis 连接与分布式锁、缓存工具
- qa_search.py     FAQService 门面：装载 / 检索 / 发布 / 下线 / 重建 / 预热
"""

from qa_system.res.qa_search import FAQService

__all__ = ["FAQService"]
