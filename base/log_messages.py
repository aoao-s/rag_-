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
    # 事件名以**调用点实际传的**为准：这里原先写的是 `faq_cache_from_csv`，而
    # `main.py` 发的是 `faq_corpus_from_csv`，于是这条日志一直退化成裸 `k: v`。
    # 模板表靠名字对齐，名字错一个字符就等于没登记——改事件名时两边一起改。
    "faq_corpus_from_csv": "FAQ 语料来自 CSV（无库模式）[文件: {path}, 条数: {docs}]",
    # `docs`（装载条数）、`indexed`（进 BM25 索引条数）、`merged`（并成别名的条数）
    # 三个数必须一起出现：只给 `docs` 时，看到"2000 条 + 别名 0"根本推不出
    # "1374 进索引 / 626 并成内存别名"——`aliases` 是 MySQL `faq_alias` 的行数，不是同一件事。
    "faq_loaded": "FAQ 语料装载完成 [条数: {docs}, 进索引: {indexed}, 合并重复: {merged}, 学科: {subjects}, 别名: {aliases}, 耗时: {cost_ms}ms]",
    "faq_ready": "FAQ 模块就绪 [条数: {docs}, 版本: {version}, 降级: {degraded}, 耗时: {cost_ms}ms]",
    "faq_load_failed_degraded": "FAQ 装载失败，以降级状态启动 [错误: {error}]",
    "faq_near_dup_scan": "近重复扫描完成 [可合并对: {pairs_merged}, 答案冲突: {conflicts}]",
    "faq_near_dup_merged": "近重复问法已合并为别名（不进 BM25 索引）[合并: {merged} 条]",
    "faq_near_dup_answer_conflict": "问法几乎相同但答案不同，不合并，交教研核对 [faq_id: {faq_ids}, 问题: {questions}]",
    "faq_subject_unknown": "存在未识别学科，按无学科入库 [条数: {count}/{total}]",
    "faq_selected_invalid": "确认通道的 faq_id 无效（已下线或不存在），按正常链路处理 [faq_id: {faq_id}]",
    "rag_ready": "RAG 检索就绪 [块数: {docs}, 向量模型: {embedder}, 重排: {rerank}, 集合: {collection}]",
    "rag_disabled": "RAG 检索未启用（未配置或功能关闭）",
    "rag_unavailable": "RAG 检索不可用，检索接口将返回 503 [错误: {error}]",
    "memory_ready": "记忆模块就绪 [仓库: {backend}, Redis: {redis}, 窗口: {window_turns} 轮 / {window_tokens} token]",
    "memory_disabled": "记忆模块未启用 [原因: {reason}]",
    "memory_backend_in_memory": "记忆使用进程内仓库，重启即丢 [依据: {reason}]",
    "config_problem": "配置自检发现问题 [详情: {detail}]",
    "config_warning": "配置提醒（不阻断启动）[详情: {detail}]",
    "app_ready": "服务启动完成 [版本: {version}, 环境: {env}]",
    "app_stopped": "服务已停止",
    "unhandled_error": "未捕获异常 [错误: {error}]",
    "http_request": "HTTP {method} {path} -> {status} [{latency_ms}ms]",

    # ---- 准入层（阶段 1 §3.2–§3.5）---------------------------------------
    # 一行结论 + 一行拒绝：拒绝也必须留痕，否则"为什么这个请求 400"只能靠复现。
    # 记录里**不含问句原文**——`q_len` 就够排障，原文受 `LOG_QUERY_TEXT` 管
    "admission_ok": "准入通过 [request_id: {request_id}, 身份: {user_id} ({identity_source}), 角色: {role}, 会话: {session_id}, 学科: {subject}, 问句长度: {q_len}, 降级: {degraded}]",
    "admission_rejected": "准入拒绝 [request_id: {request_id}, 规则: {rule}, 原因: {reason}]",
    "request_id_rejected": "请求头 X-Request-Id 不合法，已重新生成 [长度: {length}, 原因: {reason}]",
    # 归属写入失败只降级不阻断：本进程的快表仍然记着（`ChatPipeline.claim_session`），
    # 重启后这条会话会回到"未认领"，那是可接受的——比让人问不了问题好
    "session_claim_failed": "会话归属写入失败（fail-open，仅本进程内有效）[错误: {error}]",
    # 业务权限（阶段 3）：拒绝也要留一行，且带上"判成了什么意图"——否则线上看到一批
    # permission_denied 却不知道是哪条规则挡的，只能靠复现去猜。
    # **不记被拒的意图之外的东西**，也不记问句原文（受 `LOG_QUERY_TEXT` 管）
    "permission_denied": "权限不足，已在检索前拒绝 [request_id: {request_id}, 角色: {role}, 意图: {intent} ({intent_method}), 档位: {profile}]",

    # ---- 检索：一行结论，必须能看出命中来源 ------------------------------
    "faq_search_hit": "检索命中 [来源: {source_cn}] {q_text} | 档位: {tier} | faq_id: {faq_id} | 学科: {subject} | 得分: {score} | 耗时: {latency_ms}ms",
    "faq_search_miss": "检索未命中 [来源: {source_cn}] {q_text} | 档位: {tier} | 原因: {reason} | 最高得分: {score} (阈值: {threshold}) | 分差: {margin} | 候选: {candidates} | 耗时: {latency_ms}ms",
    "faq_handoff_rag": "FAQ 未命中，转交 RAG 检索 {q_text} | 档位: {tier} | 候选问题: {candidates}",
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
    "faq_cache_no_tier": "Redis 缓存条目缺置信档（旧版本写入），按低置信重算并覆盖写 [faq_id: {faq_id}]",
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
    "rag_search": "RAG 检索完成 [问句长度: {q_len}, 召回: {raw_hits} → 父块: {parents} → 保留: {kept}, 最高分: {top_score}, 重排: {reranked}, 降级: {degraded}, 耗时: {cost_ms}ms]",
    "rag_search_empty": "RAG 检索为空 [问句长度: {q_len}, top_k: {top_k}, 耗时: {cost_ms}ms]",
    "rag_rerank_failed": "重排失败，按融合分排序 [错误: {error}]",
    "rag_recall_timeout": "检索超过时间预算，已关闭重排重检 [预算: {budget_ms}ms, 原重排: {rerank}, 处理: {retry}]",
    "rag_pipeline": "检索链路完成 [策略: {strategy}, 任务数: {tasks}, 证据门: {status}, 尝试: {attempts}, 修复: {repairs}, 耗时: {cost_ms}ms]",
    "rag_repair": "证据门建议修复，已重检一次 [动作: {action}, 说明: {note}, 第 {attempt} 次]",
    "rag_repair_skipped": "证据门建议修复但未执行 [动作: {action}, 原因: {reason}]",
    "rag_rerank_done": "RAG 重排完成 [候选: {before}, 保留: {after}, 耗时: {latency_ms}ms]",
    "rag_answer_done": "RAG 回答完成 [引用: {citations}, 耗时: {latency_ms}ms]",
    "rag_evidence_gate_blocked": "证据门拦截，不生成答案 [最高分: {score} < 阈值 {threshold}]",
    "ingest_duplicate": "近重复文档，拒绝入库 [文件: {path}, 与 {kept_source} 重复, 判据: {reason}, 距离: {distance}]",
    "ingest_fingerprints_saved": "文档指纹已保存 [条数: {count}, 路径: {path}]",

    # ---- 编排 / 生成（§7.8）---------------------------------------------
    "llm_ready": "生成模型就绪 [模型: {model}, 网关: {base_url}, 超时: {timeout_s}s]",
    "llm_not_configured": "生成模型未配置，聊天接口只走兜底话术 [提示: {hint}]",
    # 本地小模型（闲聊档，§7.7）。与主网关各一条：这两行并排看才知道
    # "闲聊到底有没有本地引擎可用"，只看 llm_ready 会以为两条链路都在云端。
    "llm_local_ready": "本地小模型就绪（闲聊档专用） [模型: {model}, 网关: {base_url}, 超时: {timeout_s}s, reasoning_effort: {reasoning_effort}]",
    "llm_local_not_configured": "本地小模型未配置，闲聊档回落主网关 [提示: {hint}]",
    "llm_model_switch": "主模型不可用，切换兜底模型 [主: {primary}, 兜底: {fallback}, 错误: {error}]",
    "llm_retry": "生成请求发生传输层故障，重发一次 [模型: {model}, 第 {attempt}/{max_attempts} 次, 错误: {error}]",
    "chat_ready": "编排层就绪 [RAG: {rag}, LLM: {llm}, BERT 意图: {bert}]",
    "chat_done": "问答完成 [request_id: {request_id}, 出口: {source}, 引擎: {llm}, FAQ 档: {tier}, 引用: {citations}, 降级: {degraded}, 耗时: {latency_ms}ms]",
    "chat_stream_cancelled": "流式回答被客户端中断，已停止生成 [request_id: {request_id}, 阶段: {stage}]",
    "retrieve_failed": "RAG 检索异常，转兜底/追问 [错误: {error}]",
    "generate_failed": "生成失败，转兜底话术 [错误: {error}]",
    "faq_search_failed": "FAQ 检索异常，本次直接走知识库 [request_id: {request_id}, 错误: {error}]",
    "intent_failed": "意图识别异常，按通用档位处理 [request_id: {request_id}, 错误: {error}]",
    # BERT 初始分类（任务 2.2）。`chat_ready` 只报"有没有"，原因在这几条里——
    # 静默自禁用是本模块最容易踩的坑（标签映射不一致时模型照跑、只是答非所问）
    "bert_intent_ready": "BERT 意图分类器就绪 [检查点: {path}, 标签: {labels}, 设备: {device}, 加载: {cost_ms}ms]",
    "bert_intent_unavailable": "BERT 意图分类器不可用，意图改由大模型/规则给出 [原因: {reason}]",
    "bert_intent_disabled": "BERT 意图分类器已关闭 [原因: {reason}]",
    "bert_intent_hit": "BERT 意图分类命中 [request_id: {request_id}, 意图: {label}, 置信: {confidence}, 耗时: {latency_ms}ms]",
    "bert_intent_abstained": "BERT 意图分类弃权，交给下一段 [置信: {confidence}, 门槛: {threshold}, 最可能: {label}]",
    "bert_intent_timeout": "BERT 意图分类超时，交给下一段 [上限: {timeout_ms}ms]",
    "bert_intent_failed": "BERT 意图分类异常，交给下一段 [错误: {error}]",
    "session_load_failed": "会话历史读取异常，本轮按无上下文处理 [错误: {error}]",
    "memory_persist_failed": "记忆写库失败（不影响本次回答）[request_id: {request_id}, 错误: {error}]",
    "query_rewritten": "会话改写完成 [request_id: {request_id}, 方式: {method}, 长度: {q_len} → {rewritten_len}]",
    # 槽位（任务 2.3）。一行看全"规则给了什么、模型补了什么、哪些被丢"——
    # 分三行写会出现"只看到模型那条、没看到降级那条"的半截现场
    "slots_ready": "槽位就绪 [学科: {subject}({subject_source}), 年级: {grade}, 章节: {chapter}, 知识点: {knowledge_points}, 任务: {task_kind}, 来源: {slot_source}, 降级: {degraded}]",
    # 统一理解结果（任务 2.5）。一行看全"这次理解出了什么"：成功与否、意图与来源、
    # 是不是复合问句/追问。分几行写的话，排障时要靠 request_id 手工拼回去
    "understanding_ready": "理解完成 [request_id: {request_id}, 成功: {success}, 意图: {intent}({intent_method}), 置信: {confidence}, 档位: {profile}, 复合: {is_complex}({sub_queries} 路), 追问: {is_follow_up}]",
    # 指代没解开（任务 2.4）。这是"降级但照常检索"的一条：代词原样进了检索问句，
    # 答案可能跑偏，但**不猜主语**——编一个没出现过的限定词去收窄检索才是真故障
    "coreference_unresolved": "指代未解开，按原问句检索 [request_id: {request_id}, 历史: {history_turns} 轮, 改写方式: {rewrite_method}]",
    # 复合问句拆分（任务 2.4，2026-11-01 改由本地小模型拆）。`source` 一格说清"这次是谁拆的"：
    # `model` 是本地小模型、其余是规则各分支——排障时先看这一格，就不用去猜模型到底有没有被调到
    "compound_split": "复合问句拆分 [request_id: {request_id}, 来源: {source}, 子问题: {sub_queries} 路, 丢弃: {dropped}, 耗时: {latency_ms}ms]",
    # 中置信 FAQ 候选进生成 prompt（§7.1.9 修订）。记条数是为了能分清
    # "模型答差了是因为没给候选"与"给了候选反而被带偏"——两种结论的处置相反
    "faq_candidates_injected": "中置信 FAQ 候选作为辅助上下文注入生成 [request_id: {request_id}, 条数: {count}, faq_id: {faq_ids}]",

    # ---- 记忆（§7.9）----------------------------------------------------
    "memory_write": "记忆写入完成 [用户: {user_id}, 会话: {session_id}, 新增: {n}, 槽位: {keys}]",
    "memory_recall_timeout": "记忆召回超时，本轮按无记忆处理 [预算: {budget_ms}ms]",
    "memory_recall_failed": "记忆召回失败，本轮按无记忆处理 [错误: {error}]",
    "memory_atom_limit": "原子记忆已达上限，拒绝新增 [用户: {user_id}, 上限: {limit}]",
    "memory_clear_all": "已清空用户记忆 [用户: {user_id}, 条数: {n}]",
    # 审计与授权（M4）。审计失败单独一条：它不影响用户操作，但"审计写不进去"
    # 直接意味着合规追溯断档，不能和普通写库失败混成同一行。
    "memory_audit_failed": "记忆审计落库失败（用户操作已完成）[动作: {action}, 错误: {error}]",
    "memory_grant_read": "已授权顾问读取记忆 [学员: {user_id}, 顾问: {grantee}, 到期: {expires_at}]",
    "memory_grant_revoke": "已撤销顾问读取授权 [学员: {user_id}, 顾问: {grantee}]",
    "memory_grant_denied": "越权访问记忆被拒 [学员: {user_id}, 访问者: {viewer}, 动作: {action}]",
    "memory_advisor_read": "顾问读取学员记忆 [学员: {user_id}, 顾问: {viewer}, 范围: {detail}]",
    "memory_export": "导出用户记忆 [用户: {user_id}, 操作者: {viewer}, 条数: {n}]",
    # 物理清理（scripts/purge_memory.py）。预演与真删用两条不同的事件名：
    # 运维看到"记忆清理完成"时必须能一眼分清这次到底删没删
    "memory_purge_planned": "记忆清理预演（未删除）[表: {table}, 待删: {rows} 条, 窗口: {days} 天, 口径: {reason}]",
    "memory_purge_done": "记忆清理完成 [表: {table}, 已删: {rows} 条, 窗口: {days} 天, 口径: {reason}]",
    "memory_purge_failed": "记忆清理失败（其余表继续）[表: {table}, 错误: {error}]",
    # L2 向量召回（M1）与 L3 画像。**索引故障一律只告警不失败**：召回退化成字面排序
    # 是功能降级（用户仍能问问题），而"记忆模块起不来"才是故障。启动期的三行
    # （ready / unavailable / disabled）分开写，是为了让"为什么召回没走向量"
    # 在日志里只有一个答案，不用去猜是开关关了还是 RAG 没起来。
    "memory_vector_collection_created": "已创建原子记忆向量集合 [集合: {collection}, 维度: {dim}]",
    "memory_vector_ready": "原子记忆向量召回就绪 [集合: {collection}, 嵌入: {embedder}, 现有: {rows} 条]",
    "memory_vector_unavailable": "原子记忆向量召回不可用，本轮退回字面排序 [原因: {reason}]",
    "memory_vector_disabled": "原子记忆向量召回已关闭 [原因: {reason}]",
    "memory_vector_search_failed": "原子记忆向量检索失败，本次退回字面排序 [错误: {error}]",
    "memory_vector_upsert_failed": "原子记忆向量写入失败（MySQL 已是真相源，不影响记忆可用）[记忆: {atom_id}, 错误: {error}]",
    "memory_vector_delete_failed": "原子记忆向量删除失败（该条可能仍被检索到，下次可见性过滤会挡住）[记忆: {atom_id}, 错误: {error}]",
    "memory_vector_count_failed": "原子记忆向量集合计数失败（只影响启动日志）[错误: {error}]",
    "memory_profile_set": "更新用户画像 [用户: {user_id}, 操作者: {operator}, 字数: {chars}]",
    "memory_profile_clear": "清除用户画像，回到由原子记忆推导 [用户: {user_id}, 操作者: {operator}]",
    "session_read_failed": "会话窗口读取异常，按空窗口处理 [错误: {error}]",
    "session_bad_value": "会话窗口缓存值损坏，按空窗口处理 [会话: {session_id}]",
    "session_write_failed": "会话窗口写入失败（不影响本次回答）[错误: {error}]",
    "session_clear_failed": "会话窗口清理失败 [错误: {error}]",
    "chat_message_persist_failed": "逐字历史落库失败（不影响本次回答）[错误: {error}]",
    "session_summary_persist_failed": "会话摘要落库失败 [错误: {error}]",
    "citation_out_of_range": "引用编号越界，已从正文剔除 [编号: {refs}, request_id: {request_id}]",

    # ---- 认证（auth/）----------------------------------------------------
    # 登录成功与失败用两个事件名，不靠一个 `success` 字段区分：
    # 运维要统计"登录失败率"时，直接 grep `auth_login_failed` 就是那一批行。
    "auth_register": "注册成功并签发令牌 [用户: {user_id}, 租户: {tenant_id}, 角色: {role}]",
    "auth_login": "登录成功 [用户: {user_id}, 租户: {tenant_id}, 角色: {role}]",
    "auth_login_failed": "登录失败 [原因: {reason}, 用户: {username}]",
    "auth_refresh": "刷新令牌成功（旧 refresh 已作废）[用户: {user_id}, 租户: {tenant_id}]",
    "auth_logout": "登出，已作废该用户全部 refresh [用户: {user_id}, 租户: {tenant_id}, 作废: {revoked} 条]",
    # 令牌被拒的三个原因（unknown_jti / revoked_or_expired / wrong_type / decode）分开记：
    # "已吊销却被再次使用"是泄漏信号，"过期"只是用户掉线，两种处置完全不同。
    "auth_token_rejected": "令牌被拒 [原因: {reason}, 用户: {user_id}]",
    "auth_rejected": "认证接口拒绝请求 [错误码: {code}, HTTP: {http_status}]",
    "auth_internal_error": "认证接口内部错误 [类型: {error}]",

    # ---- 脚本 -----------------------------------------------------------
    "csv_parsed": "CSV 解析完成 [文件: {path}, 行数: {rows}, 去重后: {unique}]",
    "skip_incomplete_row": "跳过不完整行 [行号: {line}, 问题: {question}]",
    "import_failed": "导入失败 [错误: {error}]",
    "import_done": "导入完成 [新增: {inserted}, 更新: {updated}, 状态: {status}]",
    "column_added": "补齐表字段 [表: {table}, 字段: {column}]",
    "faq_hash_duplicated": "存在重复问题，content_hash 只建普通索引 [说明: {detail}]",
    "table_dropped": "删除旧表 [表: {table}]",
    "legacy_subjects_deprecated": "旧学科历史数据已置为下线 [行数: {rows}]",
    "faq_deprecated_missing": "FAQ 语料换代的旧行已下线 [行数: {rows}, 依据: 本次 CSV 里没有]",
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


# ---- 降级原因的中文说明（进 /v1/chat 的 `degraded_reasons`）------------------
# `degraded` 里是机器码（grep / 告警 / 指标都读它），人读要再翻译一次。
# 这张表是唯一出处：**不要**在编排层手写中文降级文案，否则同一个原因
# 在响应体、日志、前端提示里会变成三种说法。
DEGRADED_LABELS: dict[str, str] = {
    "faq_unavailable": "FAQ 服务异常，本次直接走知识库检索",
    "faq_selected_invalid": "确认的 FAQ 已下线或不存在，已按正常链路重新查询",
    "intent_unavailable": "意图识别失败，按通用档位处理",
    "intent_no_retrieval": "该问题被判为无需检索，已跳过知识库",
    "intent_no_retrieval_guarded": "输入带疑问但被判为无需检索，已强制走检索",
    # 与上一条刻意分开：被"疑问形态安全阀"改成检索的是一个**误解**（本来该查），
    # 而这一条是**正常拒答**（本来就不该答）。合在一起会让"误判率"永远查不清。
    "intent_out_of_scope": "该问题超出教育领域，已按服务范围拒答",
    "intent_out_of_scope_guarded": "输入带教学场景词却被判为越界，已改为检索并追问",
    "understanding_timeout": "问题理解超时，已用归一化后的原问句检索",
    "understanding_failed": "问题理解模型异常，已用归一化后的原问句检索",
    "understanding_bad_json": "问题理解返回格式不符，已用归一化后的原问句检索",
    "rewrite_rejected": "大模型改写不合格（超长/换行/丢失限定），已退回归一化结果",
    "intent_rejected": "大模型给出的意图不可用，已退回规则分类",
    "strategy_rejected": "大模型没给出可用的检索策略，已退回规则规划",
    # 槽位（任务 2.3）：四条都是"模型那一半被丢掉，规则那一半照常"。
    # 与 `rewrite_rejected` 同一口径——合并的是一次调用，不是三项的成败。
    "slots_rejected": "大模型返回的槽位不是对象，已只用规则抽到的槽位",
    "slot_model_invalid": "大模型槽位格式不符，已只用规则抽到的槽位",
    "slot_model_filter_dropped": "大模型试图指定学科/年级/教材版本/学期，已丢弃（这四项只认规则）",
    "slot_model_value_rejected": "大模型给出的难度或考试类型不在词表内，已丢弃该字段",
    "slot_clarify": "必要槽位缺失，已先追问再执行",
    "doc_type_relaxed": "按文档类型收窄后无结果，已放宽并重检",
    "rag_unavailable": "知识库检索未就绪，本次没有检索",
    "retrieve_failed": "知识库检索异常，本次没有拿到证据",
    "rerank_failed": "重排不可用，已按融合分排序",
    "partial_recall_failed": "部分检索通道失败，已用剩余结果作答",
    "retrieval_timeout": "检索超过时间预算，已关闭重排后重检",
    "retry_failed": "修复那次检索没能完成，已用第一次的结果作答",
    "retry_exhausted": "检索修复后证据仍不足",
    "backtrack_without_history": "问题含指代但没有上下文，已改为追问",
    # 指代未解开（任务 2.4）。与上一条的区别：上面是"没有历史，改成追问"，
    # 这条是"有历史但改写没落地，代词原样拿去检索了"——结论不同，处置也不同
    "coreference_unresolved": "问题含指代但未能改写，已按原问句检索",
    # 复合问句拆分（任务 2.4）。四条都是"**这次是规则拆的**"——结果照样能用，
    # 所以它们不会把 `status` 推到 degraded（见 `state._DEGRADED_ANSWER` 刻意不含它们）。
    # 分开记是因为处置不同：超时该调预算，格式不符该改提示词，被拒说明模型在编词
    "compound_timeout": "本地小模型拆分超时，已用规则拆分",
    "compound_failed": "本地小模型拆分异常，已用规则拆分",
    "compound_bad_json": "本地小模型拆分返回格式不符，已用规则拆分",
    "compound_rejected": "本地小模型拆出的子问题不是原句片段，已用规则拆分",
    "budget_exceeded": "本次请求预算耗尽",
    "llm_unavailable": "生成模型不可用",
    # 本地引擎不可用**不等于**这一轮生成不了：已回落主网关照常回答。
    # 所以它进 degraded 是为了让人看见"闲聊多付了一次云端往返"，不是错误告警。
    "llm_local_unavailable": "本地小模型端点不可用，闲聊已回落主模型",
    "llm_fallback_model": "主模型不可用，已切换兜底模型",
    "llm_retry": "生成请求发生过重试",
    "citation_out_of_range": "模型引用了不存在的来源，已剔除",
    "citation_unsupported": "存在没有证据支撑的句子，已剔除",
    "citation_missing": "答案没有任何可对齐的引用",
    "citation_repaired": "已剔除无证据支撑的句子后重新校验通过",
    "session_unavailable": "会话历史读写异常，本轮按无上下文处理",
    # 准入层（阶段 1）：会话归属查不了 ≠ 查不到。前者只能 fail-open 放行，
    # 但用户与运维都该知道"这次没校验成"，否则一次 Redis 抖动会悄悄变成
    # "任何人都能接管别人的会话"而不留痕迹
    "session_owner_unchecked": "会话归属未能校验（存储异常），本轮按首次使用放行",
    "memory_degraded": "记忆召回超时或异常，本轮按无记忆处理",
    # 业务权限（阶段 3）：与 `intent_out_of_scope` 刻意分开——那条是"这类问题不该答"，
    # 这条是"问题可以答，但这个账号没有这项功能"。处置完全不同（补语料 vs 开通权限）
    "permission_denied": "当前角色的权限不含该意图，已在检索前拒绝",
    "internal_error": "服务内部异常，本次没有完成回答",
}

# 带参数的降级码：`retry:<动作>` / `gate:<判据>`
_DEGRADED_PREFIX: dict[str, str] = {
    "retry:": "已按证据门建议修复一次检索（{rest}）",
    "gate:": "证据门未通过（{rest}）",
}


def degraded_reason(code: str) -> str:
    """降级码 → 一句中文。未登记的原样返回，**不编造**一个看起来像解释的说法。"""
    label = DEGRADED_LABELS.get(code)
    if label is not None:
        return label
    for prefix, template in _DEGRADED_PREFIX.items():
        if code.startswith(prefix):
            return template.format(rest=code[len(prefix):] or "-")
    return code


def degraded_reasons(codes: list[str]) -> list[str]:
    """批量翻译并去重（同一原因只出现一次，顺序按首次出现）。"""
    return list(dict.fromkeys(degraded_reason(code) for code in codes if code))
