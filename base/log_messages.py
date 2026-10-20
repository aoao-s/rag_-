"""日志文案表：把结构化事件翻译成人能读的中文行。

为什么要有这一层：
- 事件名（`faq_cache_hit`）适合 grep 与告警规则，不适合人看；直接打英文事件名 + k=v，
  运维凌晨排查时要先做一次心算。
- 文案集中在一处，改口径不用翻十个文件；同时天然统一了"命中来源"的措辞
  （Redis 缓存 / MySQL 索引 / RAG 检索），避免同一个意思三种写法。

模板用 `str.format_map` 渲染，缺字段时显示 `-`，不会因为少传一个字段把日志打崩。
"""

from __future__ import annotations

MESSAGES: dict[str, str] = {
    # ---- 启动 / 生命周期 -------------------------------------------------
    "redis_ready": "Redis 连接就绪 [host: {host}, port: {port}, db: {db}]",
    "redis_unavailable": "Redis 连接失败，缓存降级为进程内 [错误: {error}]",
    "faq_cache_ready": "FAQ 二级缓存就绪 [正缓存 TTL: {ttl}s, 绝对寿命: {max_ttl}s, 负缓存: {negative_ttl}s]",
    "faq_cache_disabled": "FAQ 二级缓存未启用 [原因: {reason}]",
    "faq_cache_from_csv": "FAQ 语料来自 CSV（无库模式）[文件: {path}, 条数: {docs}]",
    "faq_loaded": "FAQ 语料装载完成 [条数: {docs}, 学科: {subjects}, 别名: {aliases}, 耗时: {cost_ms}ms]",
    "faq_ready": "FAQ 模块就绪 [条数: {docs}, 版本: {version}, 降级: {degraded}, 耗时: {cost_ms}ms]",
    "faq_load_failed_degraded": "FAQ 装载失败，以降级状态启动 [错误: {error}]",
    "config_problem": "配置自检发现问题 [详情: {detail}]",
    "app_ready": "服务启动完成 [版本: {version}, 环境: {env}]",
    "app_stopped": "服务已停止",
    "unhandled_error": "未捕获异常 [错误: {error}]",
    "http_request": "HTTP {method} {path} -> {status} [{latency_ms}ms]",

    # ---- 检索：一行结论，必须能看出命中来源 ------------------------------
    "faq_search_hit": "检索命中 [来源: {source_cn}] {q_text} | faq_id: {faq_id} | 学科: {subject} | 得分: {score} | 耗时: {latency_ms}ms",
    "faq_search_miss": "检索未命中 [来源: {source_cn}] {q_text} | 原因: {reason} | 最高得分: {score} (阈值: {threshold}) | 分差: {margin} | 候选: {candidates} | 耗时: {latency_ms}ms",
    "faq_handoff_rag": "FAQ 未命中，转交 RAG 检索 {q_text} | 候选问题: {candidates}",
    "faq_search_query": "查询原文: [{q}]",
    "faq_no_evidence": "无可用语料，直接转交 RAG 检索 {q_text}",

    # ---- 缓存（DEBUG 级，落文件；控制台只看结论） -------------------------
    "faq_cache_hit": "Redis 缓存命中 [Key: {key}]",
    "faq_cache_miss": "Redis 缓存未命中 [Key: {key}]",
    "faq_cache_write": "数据成功写入 Redis [Key: {key}], TTL: {ttl}s",
    "faq_cache_slide": "Redis 缓存续命 [Key: {key}], 剩余 {remaining_s}s → {new_ttl}s",
    "faq_cache_stale": "Redis 缓存指向的 FAQ 已下线，按未命中重算 [faq_id: {faq_id}, 版本: {version}]",
    "faq_cache_expired_absolute": "Redis 缓存超过绝对寿命，强制回源 [Key: {key}, 存活: {age_s}s]",
    "faq_cache_bad_value": "Redis 缓存值损坏，按未命中处理 [Key: {key}]",
    "faq_cache_get_failed": "Redis 读取失败，降级走索引 [错误: {error}]",
    "faq_cache_put_failed": "Redis 写入失败，忽略本次回填 [错误: {error}]",
    "faq_cache_slide_failed": "Redis 续命失败，忽略 [错误: {error}]",
    "faq_cache_error": "Redis 缓存异常（fail-open，继续走索引）[操作: {op}, 错误: {error}]",

    # ---- MySQL ----------------------------------------------------------
    "mysql_pool_ready": "MySQL 连接池就绪 [host: {host}, db: {db}]",
    "mysql_ping_failed": "MySQL 心跳失败 [错误: {error}]",
    "faq_db_query": "MySQL 命中 [操作: {op}, 行数: {rows}, 耗时: {cost_ms}ms]",
    "faq_db_publish": "MySQL 已发布 FAQ [faq_id: {faq_id}, 操作人: {operator}, 影响行数: {affected}]",
    "faq_db_deprecate": "MySQL 已下线 FAQ [faq_id: {faq_id}, 操作人: {operator}, 影响行数: {affected}]",
    "faq_db_failed": "MySQL 操作失败 [操作: {op}, 错误: {error}]",

    # ---- 索引 / 预热 -----------------------------------------------------
    "faq_rebuild": "FAQ 索引重建完成 [版本: {version}, 条数: {docs}, 耗时: {cost_ms}ms]",
    "faq_warmup_done": "FAQ 预热完成 [样本: {samples}, 耗时: {cost_ms}ms]",
    "faq_alias_loaded": "FAQ 别名装载完成 [条数: {aliases}]",

    # ---- RAG（PR3 接入后启用） -------------------------------------------
    "rag_retrieval_hit": "RAG 检索命中 [块数: {chunks}, 最高分: {score}, 策略: {strategy}, 耗时: {latency_ms}ms]",
    "rag_retrieval_miss": "RAG 检索未命中 [最高分: {score} (证据门: {threshold}), 耗时: {latency_ms}ms]",
    "rag_rerank_done": "RAG 重排完成 [候选: {before}, 保留: {after}, 耗时: {latency_ms}ms]",
    "rag_answer_done": "RAG 回答完成 [引用: {citations}, 耗时: {latency_ms}ms]",
    "rag_evidence_gate_blocked": "证据门拦截，不生成答案 [最高分: {score} < 阈值 {threshold}]",

    # ---- 脚本 -----------------------------------------------------------
    "csv_parsed": "CSV 解析完成 [文件: {path}, 行数: {rows}, 去重后: {unique}]",
    "skip_incomplete_row": "跳过不完整行 [行号: {line}, 问题: {question}]",
    "import_failed": "导入失败 [错误: {error}]",
    "import_done": "导入完成 [新增: {inserted}, 更新: {updated}, 状态: {status}]",
    "column_added": "补齐表字段 [表: {table}, 字段: {column}]",
    "faq_hash_duplicated": "存在重复问题，content_hash 只建普通索引 [说明: {detail}]",
    "table_dropped": "删除旧表 [表: {table}]",
    "legacy_subjects_deprecated": "旧学科历史数据已置为下线 [行数: {rows}]",
    "db_ready": "数据库初始化完成 [库: {database}, 表: {tables}]",
}


def render(event: str, fields: dict[str, object]) -> str:
    """把事件名 + 字段渲染成一行中文；未登记的事件退化为 `事件 | k: v`。"""
    template = MESSAGES.get(event)
    if template is None:
        extras = " | ".join(f"{k}: {v}" for k, v in fields.items())
        return f"{event} | {extras}" if extras else event

    class _Safe(dict):
        def __missing__(self, key: str) -> str:  # 缺字段显示 '-'，绝不因日志抛异常
            return "-"

    try:
        return template.format_map(_Safe(fields))
    except Exception:
        extras = " | ".join(f"{k}: {v}" for k, v in fields.items())
        return f"{event} | {extras}"
