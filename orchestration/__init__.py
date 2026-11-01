"""orchestration：唯一同时依赖 qa_system 与 rag_system 的包（CLAUDE.md §5.1 / §7.8）。

- state.py  ChatState（一次请求的唯一载体）
- chat.py   ChatPipeline（同步最小闭环）
- api.py    POST /v1/chat
"""
