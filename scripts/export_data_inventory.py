"""导出项目数据清单（多 sheet Excel）。

用途：把「这个系统到底需要哪些数据」一次性摊开，便于项目方按表补齐。
数据来源：datasets/ 实际文件清点 + scripts/init_db.py 的 DDL +
rag_system/store/milvus_store.py 的 schema + base/config.py 与 .env.example。

用法：
    python scripts/export_data_inventory.py                 # 输出 docs/数据清单-EduRAG-Pro.xlsx
    python scripts/export_data_inventory.py --out out/x.xlsx

改表结构 / 加数据目录后重跑本脚本即可，不要手工改 Excel（下次导出会被覆盖）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xlsxwriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- 主题
HEADER_BG = "#1F3864"
TITLE_BG = "#D9E2F3"
ZEBRA = "#F2F2F2"
OK_BG = "#E2EFDA"     # 已有 / 已建
WARN_BG = "#FFF2CC"   # 规划中 / 部分
BAD_BG = "#FCE4E4"    # 失效 / 缺失

OK_PREFIX = ("已建", "已有", "已实现", "在线", "完成", "已标定")
WARN_PREFIX = ("规划中", "部分", "待建", "未接入", "未使用", "拟定")
BAD_PREFIX = ("失效", "缺失", "未建", "不适用")


def _status_bg(value: str) -> str | None:
    text = str(value).strip()
    if text.startswith(OK_PREFIX):
        return OK_BG
    if text.startswith(BAD_PREFIX):
        return BAD_BG
    if text.startswith(WARN_PREFIX):
        return WARN_BG
    return None


def add_sheet(
    wb: xlsxwriter.Workbook,
    name: str,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    widths: list[int],
    status_col: int | None = None,
    note: str | None = None,
) -> None:
    ws = wb.add_worksheet(name)
    ws.hide_gridlines(2)
    ws.set_tab_color(HEADER_BG)

    fmt_title = wb.add_format(
        {"bold": True, "font_size": 13, "bg_color": TITLE_BG, "font_color": "#1F3864",
         "align": "left", "valign": "vcenter", "border": 1}
    )
    fmt_note = wb.add_format({"font_size": 9, "font_color": "#595959", "italic": True, "valign": "vcenter"})
    fmt_head = wb.add_format(
        {"bold": True, "font_color": "white", "bg_color": HEADER_BG, "align": "center",
         "valign": "vcenter", "text_wrap": True, "border": 1}
    )
    fmt_cell = wb.add_format({"valign": "top", "text_wrap": True, "border": 1, "font_size": 10})
    fmt_cell_zebra = wb.add_format(
        {"valign": "top", "text_wrap": True, "border": 1, "font_size": 10, "bg_color": ZEBRA}
    )
    fmt_first = wb.add_format(
        {"valign": "top", "text_wrap": True, "border": 1, "font_size": 10, "bold": True}
    )
    fmt_status = {
        bg: wb.add_format(
            {"valign": "top", "text_wrap": True, "border": 1, "font_size": 10, "bg_color": bg}
        )
        for bg in (OK_BG, WARN_BG, BAD_BG)
    }

    ncol = len(headers)
    ws.merge_range(0, 0, 0, ncol - 1, title, fmt_title)
    ws.set_row(0, 26)
    if note:
        ws.merge_range(1, 0, 1, ncol - 1, note, fmt_note)
        ws.set_row(1, 18)
        header_row = 2
    else:
        header_row = 1

    for i, head in enumerate(headers):
        ws.write(header_row, i, head, fmt_head)
    ws.set_row(header_row, 30)

    for r, row in enumerate(rows):
        excel_row = header_row + 1 + r
        zebra = r % 2 == 1
        for c, value in enumerate(row):
            text = "" if value is None else str(value)
            cell_fmt = fmt_cell_zebra if zebra else fmt_cell
            if c == 0:
                cell_fmt = fmt_first
            if status_col is not None and c == status_col:
                bg = _status_bg(text)
                if bg:
                    cell_fmt = fmt_status[bg]
            ws.write(excel_row, c, text, cell_fmt)

    for i, width in enumerate(widths):
        ws.set_column(i, i, width)
    ws.freeze_panes(header_row + 1, 1)
    if rows:
        ws.autofilter(header_row, 0, header_row + len(rows), ncol - 1)


# --------------------------------------------------------------------------- 各表数据
def sheet_overview() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["序号", "数据域", "具体数据", "现状", "现有规模", "落地位置", "说明"]
    rows = [
        ["1", "原始语料", "教材章节知识点（Markdown）", "已有", "6 份 / 6 学科", "datasets/01_textbook → Milvus", "doc_type=textbook，摄入后进向量库"],
        ["2", "原始语料", "优秀教案（Markdown）", "已有", "6 份", "datasets/02_lesson_plans → Milvus", "doc_type=lesson_plan"],
        ["3", "原始语料", "教学视频字幕（SRT）", "已有", "2 个 / 18 个字幕块", "datasets/08_subtitles → Milvus", "doc_type=subtitle"],
        ["4", "原始语料", "真实电子教材（PDF）", "已有", "1 本 / 8.51 MB", "datasets/09_textbooks", "人教版数学九上；MinerU 云解析或本地 PyMuPDF"],
        ["5", "题库", "种子题 + 开源转换题（JSONL）", "已有但未接入", "80 + 15 785 = 15 865 道", "datasets/03_question_bank", "12 字段；在线链路目前不读题库"],
        ["6", "FAQ 语料", "高频常见问答（CSV）", "已建", "2 000 条 / 9 学科", "datasets/04_faq → MySQL `faq`", "生产真相源是 MySQL；CSV 为导入源"],
        ["7", "FAQ 别名", "同一问题的其他问法", "未建（表已就绪）", "0 条", "MySQL `faq_alias`", "目前靠归一化 + BM25 兜；别名可显著提召回"],
        ["8", "微调语料", "教育指令数据（JSONL）", "已有但未使用", "50 条", "datasets/05_sft", "目标 6 万条；当前未接训练"],
        ["9", "意图数据", "意图分类训练/测试（JSONL）", "已有", "4 080 + 1 020 条", "datasets/06_intent", "6 类 ×680 / ×170；2026-10-26 定案采用 K12 6 类（招生 5 类废弃）"],
        ["10", "评估数据", "RAGAS 评估集（JSONL）", "已标定", "1 000 条（763 可判定）", "datasets/07_eval", "证据门 0.74/0.01/0.20 由它标定"],
        ["11", "评估数据", "FAQ 同义问法正样本（JSONL）", "失效", "480 条", "docs/analysis/faq_paraphrases.jsonl", "480/480 的 faq_id 与当前正文不一致，必须归档重建"],
        ["12", "业务数据", "MySQL 已建 3 张表", "已建", "faq / faq_alias / faq_feedback", "scripts/init_db.py", "端口 3307；库名 rag_sbc"],
        ["13", "业务数据", "MySQL 规划 17 张表", "规划中", "会话/日志/反馈/文档/审计/记忆/QAF", "设计文档 §6.4 / §7.3", "详见「03-MySQL表结构」"],
        ["14", "向量数据", "Milvus 父子块集合", "已建", "15 文件 / 622 子块（2026-10-20 实测）", "MILVUS_URI 的 edurag.edurag_chunk", "1024 维稠密 + 稀疏倒排，HNSW"],
        ["15", "热数据", "Redis FAQ 二级缓存", "已建", "正缓存 + 30 s 负缓存", "edurag:{env}:faq:v{版本}:{范围}:{摘要}", "TTL 1 h，绝对寿命 6 h，命中续期"],
        ["16", "热数据", "Redis 规划键族", "规划中", "语义缓存/会话/限流/锁/记忆/事件流", "CLAUDE.md §6.4", "详见「05-Redis键」"],
        ["17", "模型权重", "BGE-M3 / Reranker / 意图 BERT", "部分已有", "约 4–5 GB 显存", "../Ragpro/rag_system/models + MODELS_DIR", "意图模型需先用 06_intent 训练"],
        ["18", "外部服务", "DeepSeek / MinerU / MinIO", "部分接入", "LLM 与 MinerU 已接，MinIO 未接", ".env", "密钥只从环境注入，不入库"],
        ["19", "标定产出", "阈值 / 敏感度报告", "部分失效", "3 份报告 + 2 份跑批", "docs/analysis/", "RAG 报告有效；FAQ 阈值结论已失效"],
        ["20", "运行留档", "结构化日志", "在线", "50 MB × 10 份轮转", "logs/app.log + stdout", "生产 LOG_QUERY_TEXT=false"],
    ]
    return headers, rows, [6, 12, 30, 16, 26, 40, 44], 3


def sheet_corpus() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["目录", "内容", "格式", "现有规模", "字段 / 结构", "系统用途", "状态"]
    rows = [
        ["datasets/01_textbook", "教材章节知识点", "Markdown", "6 份（语文/数学/英语/物理/化学/生物）", "标题层级 + 正文", "进 Milvus（doc_type=textbook），RAG 主证据源", "已有"],
        ["datasets/02_lesson_plans", "优秀教案", "Markdown", "6 份", "教学目标/重难点/教学过程", "进 Milvus（doc_type=lesson_plan）", "已有"],
        ["datasets/08_subtitles", "教学视频字幕", "SRT", "2 个 / 18 块", "时间轴 + 文本（入库前剥时间轴）", "进 Milvus（doc_type=subtitle）", "已有"],
        ["datasets/09_textbooks", "电子教材 PDF", "PDF", "1 本 / 8.51 MB", "扫描/版式 PDF", "MinerU 云解析或本地 PyMuPDF 后入库", "已有（覆盖不足）"],
        ["datasets/03_question_bank", "题库", "JSONL", "80 种子 + 15 785 开源", "question_id/subject/grade/chapter/knowledge_points/question_type/difficulty/question/answer/analysis/source/textbook_version", "题库型问答（规划中）", "已有但未接入"],
        ["datasets/04_faq", "高频问答", "CSV（UTF-8 BOM）", "2 000 条 / 9 学科", "学科名称,问题,答案", "导入 MySQL `faq` → 分学科 BM25", "已建"],
        ["datasets/05_sft", "教育指令微调", "JSONL", "50 条", "instruction/input/output", "LoRA 微调（未接入）", "已有但未使用"],
        ["datasets/06_intent", "意图分类", "JSONL", "训练 4 080 + 测试 1 020", "query/label（6 类均衡）", "训练 BERT-small 意图分类器", "已有"],
        ["datasets/07_eval", "RAGAS 评估集", "JSONL", "1 000 条", "question/ground_truth/contexts/intent", "检索与证据门回归、忠实度抽检", "已标定"],
        ["datasets/raw_opensource", "开源原始数据快照", "目录", "ceval-exam / gaokao-benchmark / gaokao_snapshot", "上游原始格式", "由转换脚本产出 03_question_bank", "已有"],
        ["datasets/scripts", "数据生成/下载/转换/校验脚本", "Python", "9 个脚本", "gen_seed_* / download_opensource / convert_to_project_format / validate_datasets", "数据可复现", "已有"],
        ["docs/analysis", "标定与跑批产出", "Markdown / JSONL", "3 报告 + 2 跑批", "见「06-评估与标定数据」", "阈值标定与回归基线", "部分失效"],
    ]
    return headers, rows, [28, 20, 16, 34, 56, 34, 14], 6


def sheet_faq() -> tuple[list[str], list[list[str]], list[int], int | None]:
    headers = ["项目", "类型", "必填", "约束 / 规则", "示例", "说明"]
    rows = [
        ["CSV 列：学科名称", "字符串", "是", "必须 ∈ SUBJECTS（9 学科）或其别名；别名经 SUBJECT_ALIASES 归一", "数学", "学科是检索范围第一层约束，缺失会退化成全库"],
        ["CSV 列：问题", "字符串", "是", "≤1000 字；归一化后必须唯一（去重后重复即丢弃）", "勾股定理怎么证明？", "content_hash = md5(normalized_question)"],
        ["CSV 列：答案", "字符串", "是", "非空；≤10 000 字；人工审核后才可 published", "两直角边平方和等于斜边平方……", "FAQ 答案逐字返回，不允许模型改写"],
        ["MySQL `faq`.status", "枚举", "是", "draft / pending / published / deprecated；只有 published 进索引", "published", "发布即建索引，下线即移出索引"],
        ["MySQL `faq_alias`.alias", "字符串", "否", "≤255 字；全局唯一；整串等价映射到 faq_id", "学费多少钱", "替代方案：靠 BM25 近义；不做子串替换"],
        ["质量规则 1", "—", "—", "发布时必须有 subject 标签，否则拒绝发布", "—", "P1 硬门槛（QAF §3.8）"],
        ["质量规则 2", "—", "—", "同一锚点问题至少 3 条同义问法用于标定", "—", "阈值标定的前提，当前样本集已失效"],
        ["质量规则 3", "—", "—", "不允许混入非当前学段的旧口径学科名（如旧「政治」）", "—", "旧库 97 行 subject_name 为空即为历史遗留"],
        ["质量规则 4", "—", "—", "跨学科不得共用同一问题原文", "—", "分学科索引下会双命中，margin 判据失效"],
    ]
    return headers, rows, [26, 12, 8, 60, 34, 50], None


def sheet_mysql() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["表名", "中文名", "字段", "类型", "允许空", "默认值", "说明", "状态", "出处"]
    E = "已建（init_db.py）"
    P = "规划中（设计文档）"
    rows: list[list[str]] = [
        # faq
        ["faq", "FAQ 主表", "id", "BIGINT PK AUTO_INCREMENT", "否", "—", "主键", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "subject_name", "VARCHAR(32)", "否", "''", "学科（canonical，见 SUBJECTS）", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "question", "VARCHAR(1000)", "否", "—", "锚点问题原文", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "answer", "TEXT", "否", "—", "人工审核过的答案，逐字返回", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "status", "ENUM(draft,pending,published,deprecated)", "否", "draft", "只有 published 进 BM25 索引", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "content_hash", "CHAR(32)", "否", "''", "md5(归一化问题)；唯一键，幂等导入", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "source", "VARCHAR(64)", "否", "import", "数据来源（import/legacy/manual）", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "hit_count", "INT", "否", "0", "命中计数（运营排序用）", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "创建时间", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "updated_at", "DATETIME", "否", "CURRENT_TIMESTAMP ON UPDATE", "更新时间", E, "scripts/init_db.py"],
        ["faq", "FAQ 主表", "updated_by", "VARCHAR(64)", "否", "''", "最后操作人", E, "scripts/init_db.py"],
        # faq_alias
        ["faq_alias", "FAQ 别名", "id", "BIGINT PK AUTO_INCREMENT", "否", "—", "主键", E, "scripts/init_db.py"],
        ["faq_alias", "FAQ 别名", "faq_id", "BIGINT", "否", "—", "指向 faq.id", E, "scripts/init_db.py"],
        ["faq_alias", "FAQ 别名", "alias", "VARCHAR(255)", "否", "—", "同义问法，全局唯一", E, "scripts/init_db.py"],
        ["faq_alias", "FAQ 别名", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "创建时间", E, "scripts/init_db.py"],
        # faq_feedback
        ["faq_feedback", "FAQ 反馈", "id", "BIGINT PK AUTO_INCREMENT", "否", "—", "主键", E, "scripts/init_db.py"],
        ["faq_feedback", "FAQ 反馈", "request_id", "VARCHAR(64)", "否", "—", "请求幂等键，必须带", E, "scripts/init_db.py"],
        ["faq_feedback", "FAQ 反馈", "faq_id", "BIGINT", "是", "NULL", "命中的 FAQ（未命中为 NULL）", E, "scripts/init_db.py"],
        ["faq_feedback", "FAQ 反馈", "verdict", "ENUM(up,down)", "否", "—", "点赞/点踩", E, "scripts/init_db.py"],
        ["faq_feedback", "FAQ 反馈", "tags", "JSON", "是", "NULL", "错误归因标签（固定枚举）", E, "scripts/init_db.py"],
        ["faq_feedback", "FAQ 反馈", "comment", "VARCHAR(1000)", "是", "NULL", "自由文本纠错", E, "scripts/init_db.py"],
        ["faq_feedback", "FAQ 反馈", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "创建时间", E, "scripts/init_db.py"],
        # chat_message
        ["chat_message", "会话消息", "id", "BIGINT PK AI", "否", "—", "主键（L0 逐字历史的持久化）", P, "§3.7 / §6.4"],
        ["chat_message", "会话消息", "session_id", "VARCHAR(40)", "否", "—", "会话 id（=sid）", P, "拟定字段"],
        ["chat_message", "会话消息", "user_id", "VARCHAR(32)", "否", "—", "用户 id（租户内唯一）", P, "拟定字段"],
        ["chat_message", "会话消息", "role", "ENUM(user,assistant,system)", "否", "—", "消息角色", P, "拟定字段"],
        ["chat_message", "会话消息", "content", "TEXT", "否", "—", "消息正文", P, "拟定字段"],
        ["chat_message", "会话消息", "request_id", "VARCHAR(64)", "是", "NULL", "关联 request_log，幂等重放", P, "拟定字段"],
        ["chat_message", "会话消息", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "拟定字段"],
        # request_log
        ["request_log", "请求日志", "request_id", "VARCHAR(64) PK", "否", "—", "幂等键，贯穿缓存/历史/计费", P, "§6.2"],
        ["request_log", "请求日志", "user_id", "VARCHAR(32)", "否", "—", "调用者", P, "拟定字段"],
        ["request_log", "请求日志", "session_id", "VARCHAR(40)", "是", "NULL", "会话 id", P, "拟定字段"],
        ["request_log", "请求日志", "query_hash", "CHAR(32)", "否", "—", "默认只记哈希，不落原文", P, "§5.2"],
        ["request_log", "请求日志", "rewritten", "VARCHAR(1000)", "是", "NULL", "会话改写后的 query", P, "§3.2"],
        ["request_log", "请求日志", "intent", "VARCHAR(32)", "是", "NULL", "意图分类结果", P, "§3.2"],
        ["request_log", "请求日志", "strategy", "VARCHAR(16)", "是", "NULL", "DIRECT/HyDE/SUB_QUERY/BACKTRACK", P, "§3.3"],
        ["request_log", "请求日志", "source", "VARCHAR(16)", "是", "NULL", "faq / rag / cache / fallback", P, "§3.1"],
        ["request_log", "请求日志", "cached", "TINYINT", "否", "0", "是否缓存命中", P, "拟定字段"],
        ["request_log", "请求日志", "latency_ms", "INT", "是", "NULL", "端到端耗时", P, "§6.1"],
        ["request_log", "请求日志", "first_token_ms", "INT", "是", "NULL", "首 token 耗时（SLO ≤1.2 s）", P, "§1.4"],
        ["request_log", "请求日志", "tokens_in", "INT", "是", "NULL", "输入 token（成本核算）", P, "§3.6"],
        ["request_log", "请求日志", "tokens_out", "INT", "是", "NULL", "输出 token", P, "§3.6"],
        ["request_log", "请求日志", "status", "VARCHAR(16)", "否", "ok", "ok / degraded / error / cancelled", P, "§3.1"],
        ["request_log", "请求日志", "error_code", "VARCHAR(32)", "是", "NULL", "错误分类计数用", P, "§3.6"],
        ["request_log", "请求日志", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "拟定字段"],
        # feedback
        ["feedback", "答案反馈", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.6"],
        ["feedback", "答案反馈", "request_id", "VARCHAR(64)", "否", "—", "必须能反查到 query 与档位", P, "§3.8"],
        ["feedback", "答案反馈", "user_id", "VARCHAR(32)", "否", "—", "反馈人", P, "拟定字段"],
        ["feedback", "答案反馈", "verdict", "ENUM(up,down)", "否", "—", "赞/踩", P, "§3.6"],
        ["feedback", "答案反馈", "tags", "JSON", "是", "NULL", "固定枚举的归因标签", P, "§3.8"],
        ["feedback", "答案反馈", "comment", "VARCHAR(1000)", "是", "NULL", "纠错自由文本", P, "§3.6"],
        ["feedback", "答案反馈", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "拟定字段"],
        # doc_meta
        ["doc_meta", "文档元数据", "doc_id", "VARCHAR(64) PK", "否", "—", "与 Milvus doc_id 对应", P, "§6.4"],
        ["doc_meta", "文档元数据", "file_name", "VARCHAR(512)", "否", "—", "原始文件名（引用卡片展示）", P, "§3.4"],
        ["doc_meta", "文档元数据", "source", "VARCHAR(512)", "否", "—", "MinIO 路径/来源 URL", P, "§3.5"],
        ["doc_meta", "文档元数据", "doc_type", "VARCHAR(32)", "否", "other", "textbook/lesson_plan/subtitle/exam/other", P, "document_loaders/base.py"],
        ["doc_meta", "文档元数据", "subject", "VARCHAR(16)", "是", "NULL", "学科；认不出留 NULL（不猜）", P, "§7.2.6"],
        ["doc_meta", "文档元数据", "content_hash", "CHAR(32)", "否", "—", "未变则跳过摄入（增量幂等）", P, "§3.5"],
        ["doc_meta", "文档元数据", "kb_version", "VARCHAR(16)", "否", "v1", "所属索引版本，回滚用", P, "§3.5"],
        ["doc_meta", "文档元数据", "chunk_count", "INT", "否", "0", "产出的子块数", P, "拟定字段"],
        ["doc_meta", "文档元数据", "status", "VARCHAR(16)", "否", "pending", "pending/parsed/indexed/failed", P, "拟定字段"],
        ["doc_meta", "文档元数据", "uploaded_by", "VARCHAR(64)", "是", "NULL", "上传人", P, "拟定字段"],
        ["doc_meta", "文档元数据", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "拟定字段"],
        ["doc_meta", "文档元数据", "updated_at", "DATETIME", "否", "CURRENT_TIMESTAMP ON UPDATE", "时间", P, "拟定字段"],
        # audit_log
        ["audit_log", "审计日志", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.5"],
        ["audit_log", "审计日志", "actor", "VARCHAR(64)", "否", "—", "操作人", P, "拟定字段"],
        ["audit_log", "审计日志", "action", "VARCHAR(64)", "否", "—", "publish/deprecate/delete_memory/...", P, "拟定字段"],
        ["audit_log", "审计日志", "target", "VARCHAR(128)", "是", "NULL", "操作对象（faq_id/atom_id/...）", P, "拟定字段"],
        ["audit_log", "审计日志", "detail", "JSON", "是", "NULL", "变更前后值", P, "拟定字段"],
        ["audit_log", "审计日志", "trace_id", "VARCHAR(64)", "是", "NULL", "链路追踪", P, "§5.2"],
        ["audit_log", "审计日志", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "拟定字段"],
        # memory
        ["user_profile", "用户画像（L3）", "user_id", "VARCHAR(32) PK", "否", "—", "租户内唯一", P, "§3.7"],
        ["user_profile", "用户画像（L3）", "tenant_id", "VARCHAR(32)", "否", "default", "租户隔离", P, "§3.7"],
        ["user_profile", "用户画像（L3）", "profile", "JSON", "否", "—", "稳定属性，常驻 ≤300 token", P, "§3.7"],
        ["user_profile", "用户画像（L3）", "updated_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "§3.7"],
        ["session_summary", "会话摘要（L1）", "sid", "VARCHAR(40) PK", "否", "—", "会话 id", P, "§3.7"],
        ["session_summary", "会话摘要（L1）", "user_id", "VARCHAR(32)", "否", "—", "归属用户", P, "§3.7"],
        ["session_summary", "会话摘要（L1）", "summary", "TEXT", "否", "—", "≤400 token，含 topics/open_items/decisions", P, "§7.3.2"],
        ["session_summary", "会话摘要（L1）", "turns", "INT", "否", "0", "已滚动轮数", P, "§3.7"],
        ["session_summary", "会话摘要（L1）", "updated_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "§3.7"],
        ["memory_atom", "原子记忆（L2）", "id", "VARCHAR(40) PK", "否", "—", "原子记忆 id", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "tenant_id", "VARCHAR(32)", "否", "—", "租户隔离", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "user_id", "VARCHAR(32)", "否", "—", "归属用户（默认私有）", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "fact_type", "ENUM(preference,identity,constraint,decision,progress)", "否", "—", "五类事实", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "fact_key", "VARCHAR(64)", "否", "—", "归一化槽位，如 goal_exam / budget", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "content", "VARCHAR(500)", "否", "—", "一句事实", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "confidence", "DECIMAL(3,2)", "否", "—", "召回门槛 ≥0.6", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "mention_count", "INT", "否", "1", "被提及次数（去重合并时 +1）", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "source_sid", "VARCHAR(40)", "是", "NULL", "来源会话", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "source_turn", "INT", "是", "NULL", "来源轮次（可下钻）", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "source_span", "VARCHAR(200)", "是", "NULL", "原文片段", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "valid_from", "DATETIME", "否", "CURRENT_TIMESTAMP", "生效时间", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "valid_to", "DATETIME", "是", "NULL", "失效时间（NULL = 生效中）", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "superseded_by", "VARCHAR(40)", "是", "NULL", "被哪条新记忆取代", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "chain_root_id", "VARCHAR(40)", "是", "NULL", "冲突链根，purge_chain 用", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "deleted_at", "DATETIME", "是", "NULL", "软删；30 天后物理清理", P, "§7.3.3"],
        ["memory_atom", "原子记忆（L2）", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "§7.3.3"],
        ["memory_grant", "记忆授权", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.7"],
        ["memory_grant", "记忆授权", "owner_user_id", "VARCHAR(32)", "否", "—", "被授权记忆的属主", P, "§3.7"],
        ["memory_grant", "记忆授权", "grantee_user_id", "VARCHAR(32)", "否", "—", "只读方（顾问）", P, "§3.7"],
        ["memory_grant", "记忆授权", "scope", "VARCHAR(64)", "否", "read", "只读授权（写永不放行）", P, "§3.7"],
        ["memory_grant", "记忆授权", "granted_by", "VARCHAR(64)", "否", "—", "授权人", P, "拟定字段"],
        ["memory_grant", "记忆授权", "revoked_at", "DATETIME", "是", "NULL", "撤销时间", P, "拟定字段"],
        ["memory_audit", "记忆审计", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.7"],
        ["memory_audit", "记忆审计", "actor", "VARCHAR(64)", "否", "—", "操作人", P, "§3.7"],
        ["memory_audit", "记忆审计", "user_id", "VARCHAR(32)", "否", "—", "记忆属主", P, "§3.7"],
        ["memory_audit", "记忆审计", "atom_id", "VARCHAR(40)", "是", "NULL", "被操作记忆", P, "§3.7"],
        ["memory_audit", "记忆审计", "action", "VARCHAR(32)", "否", "—", "edit/delete(revive|purge_chain)/export/all", P, "§3.7"],
        ["memory_audit", "记忆审计", "trace_id", "VARCHAR(64)", "是", "NULL", "链路追踪", P, "§5.2"],
        ["memory_audit", "记忆审计", "created_at", "DATETIME", "否", "CURRENT_TIMESTAMP", "时间", P, "§3.7"],
        # QAF
        ["qa_pair", "QA 对（QAF）", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.8"],
        ["qa_pair", "QA 对（QAF）", "question", "VARCHAR(1000)", "否", "—", "问题", P, "§3.8"],
        ["qa_pair", "QA 对（QAF）", "answer", "TEXT", "否", "—", "答案", P, "§3.8"],
        ["qa_pair", "QA 对（QAF）", "status", "ENUM(draft,pending,published,deprecated)", "否", "draft", "生命周期四态", P, "§3.8"],
        ["qa_pair", "QA 对（QAF）", "subject", "VARCHAR(16)", "否", "—", "发布必须有 subject 标签", P, "§3.8"],
        ["qa_pair", "QA 对（QAF）", "published_at", "DATETIME", "是", "NULL", "发布即建索引，5 min 内可查", P, "§3.8"],
        ["qa_version", "QA 版本", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.8"],
        ["qa_version", "QA 版本", "qa_pair_id", "BIGINT", "否", "—", "指向 qa_pair", P, "§3.8"],
        ["qa_version", "QA 版本", "version", "INT", "否", "1", "版本号", P, "§3.8"],
        ["qa_version", "QA 版本", "question", "VARCHAR(1000)", "否", "—", "该版本问题", P, "拟定字段"],
        ["qa_version", "QA 版本", "answer", "TEXT", "否", "—", "该版本答案", P, "拟定字段"],
        ["qa_version", "QA 版本", "operator", "VARCHAR(64)", "否", "—", "操作人", P, "拟定字段"],
        ["miss_pool", "未命中池", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.8"],
        ["miss_pool", "未命中池", "query_hash", "CHAR(32)", "否", "—", "归一化问法哈希", P, "§3.8"],
        ["miss_pool", "未命中池", "query_text", "VARCHAR(1000)", "是", "NULL", "原文（仅内部，按隐私策略）", P, "§3.8"],
        ["miss_pool", "未命中池", "request_id", "VARCHAR(64)", "是", "NULL", "溯源", P, "§3.8"],
        ["miss_pool", "未命中池", "top_candidates", "JSON", "是", "NULL", "top3 候选（生成 FAQ 候选的输入）", P, "§7.1.2"],
        ["miss_pool", "未命中池", "hit_count", "INT", "否", "1", "重复出现次数，用于聚类排序", P, "§3.8"],
        ["tag", "标签字典", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.8"],
        ["tag", "标签字典", "namespace", "VARCHAR(16)", "否", "—", "intent/subject/course/level/scenario/audience/doc_type/risk/acl", P, "§3.8"],
        ["tag", "标签字典", "name", "VARCHAR(64)", "否", "—", "标签名", P, "§3.8"],
        ["tag", "标签字典", "parent_id", "BIGINT", "是", "NULL", "树形结构", P, "§3.8"],
        ["tag_synonym", "标签同义词", "id", "BIGINT PK AI", "否", "—", "主键", P, "§3.8"],
        ["tag_synonym", "标签同义词", "tag_id", "BIGINT", "否", "—", "指向 tag", P, "§3.8"],
        ["tag_synonym", "标签同义词", "synonym", "VARCHAR(64)", "否", "—", "同义写法（AC 自动机词表）", P, "§3.8"],
        ["qa_tag", "QA-标签关联", "qa_pair_id", "BIGINT", "否", "—", "联合主键", P, "§3.8"],
        ["qa_tag", "QA-标签关联", "tag_id", "BIGINT", "否", "—", "联合主键", P, "§3.8"],
        ["qa_tag", "QA-标签关联", "confidence", "DECIMAL(3,2)", "否", "1.0", "<0.6 需人工确认", P, "§3.8"],
        ["qa_tag", "QA-标签关联", "source", "VARCHAR(16)", "否", "rule", "rule / model / manual", P, "§3.8"],
        ["doc_tag", "文档-标签关联", "doc_id", "VARCHAR(64)", "否", "—", "联合主键", P, "§3.8"],
        ["doc_tag", "文档-标签关联", "tag_id", "BIGINT", "否", "—", "联合主键", P, "§3.8"],
        ["doc_tag", "文档-标签关联", "confidence", "DECIMAL(3,2)", "否", "1.0", "置信度", P, "§3.8"],
        ["doc_tag", "文档-标签关联", "source", "VARCHAR(16)", "否", "rule", "rule / model / manual", P, "§3.8"],
    ]
    return headers, rows, [16, 18, 18, 40, 8, 22, 44, 20, 26], 7


def sheet_milvus() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["字段", "类型", "长度 / 维度", "索引 / 主键", "用途", "检索过滤", "状态"]
    E = "已建（milvus_store.py）"
    P = "规划中（§6.4）"
    rows = [
        ["chunk_id", "VARCHAR", "160", "主键", "子块唯一 id", "—", E],
        ["doc_id", "VARCHAR", "64", "—", "所属文档，按文档删/重跑", "按 doc_id 先删后写", E],
        ["parent_id", "VARCHAR", "160", "—", "父块 id（{doc_id}:p{n}）", "父块去重与回填", E],
        ["text", "VARCHAR", "4096", "—", "子块正文（被检索命中）", "—", E],
        ["parent_text", "VARCHAR", "8192", "—", "父块正文（进 prompt 的证据）", "—", E],
        ["source", "VARCHAR", "512", "—", "原始文件路径/来源", "按 source 删旧块", E],
        ["title", "VARCHAR", "256", "—", "文档标题", "—", E],
        ["subject", "VARCHAR", "16", "—", "学科（canonical）", "标量前置过滤", E],
        ["doc_type", "VARCHAR", "32", "—", "textbook / lesson_plan / subtitle / exam / image / other", "标量前置过滤", E],
        ["heading", "VARCHAR", "512", "—", "所在章节标题（引用卡片展示）", "—", E],
        ["page", "INT32", "—", "—", "页码（PDF 逐页标记）", "引用定位", E],
        ["chunk_index", "INT32", "—", "—", "块序号（同文档内）", "—", E],
        ["ts", "INT64", "—", "—", "写入时间戳", "—", E],
        ["embedder", "VARCHAR", "32", "—", "产出向量的模型名（如 bge-m3 / hash）", "防混模型", E],
        ["dense", "FLOAT_VECTOR", "1024", "HNSW / COSINE / M=16 / efConstruction=200 / ef=64", "稠密语义召回", "—", E],
        ["sparse", "SPARSE_FLOAT_VECTOR", "—", "SPARSE_INVERTED_INDEX / IP / drop_ratio_build=0.2", "稀疏关键词召回", "—", E],
        ["tenant_id", "VARCHAR", "32", "—", "租户隔离（多租户前置过滤的前提）", "标量前置过滤", P],
        ["acl_tags", "ARRAY / VARCHAR", "—", "—", "文档访问控制标签", "标量前置过滤", P],
        ["kb_version", "VARCHAR", "16", "—", "索引版本，切版本即失效缓存", "标量前置过滤", P],
        ["content_hash", "VARCHAR", "32", "—", "文档内容哈希，增量幂等", "—", P],
    ]
    return headers, rows, [16, 22, 12, 50, 44, 24, 26], 6


def sheet_redis() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["Key 模式", "类型", "TTL", "存放内容", "用途", "降级行为", "状态"]
    E = "已建（qa_cache.py）"
    P = "规划中（§6.4）"
    rows = [
        ["edurag:{env}:faq:v{语料版本}:{检索范围}:{sha1(norm)[:20]}", "String(JSON)", "正缓存 3600 s / 绝对 21600 s", "FAQHit：hit/faq_id/answer/score/subject/subject_state/candidates/born", "FAQ 命中结果二级缓存", "Redis 异常 fail-open，只记 cache_errors 指标", E],
        ["edurag:{env}:faq:v{语料版本}:{检索范围}:{sha1(norm)[:20]}", "String(JSON)", "30 s（负缓存）", "hit=false + top3 候选问法", "挡住重复脏查询反复打分；候选交给 RAG 做扩展", "同上；损坏值按未命中覆盖写", E],
        ["edurag:{env}:sem:{kb_version}:{tenant}:{hash}", "String(JSON)", "24 h", "语义缓存答案", "L2 语义缓存（阈值 0.95 才开）", "视为未命中", P],
        ["edurag:{env}:session:{sid}", "LIST / HASH", "会话存活期", "逐字对话窗口（40 轮 / 28 K token）", "L0 工作记忆（MySQL 为真相源）", "直读 MySQL，p95 允许到 300 ms", P],
        ["edurag:{env}:faq:corpus:{subject}", "String(INT)", "永久", "该学科语料版本号", "热更新后让缓存自然失效", "发布后 30 s 全量校验兜底", P],
        ["edurag:{env}:rl:{user_id}:{bucket}", "String(INT)", "窗口长度", "限流计数", "接入层限流", "放行（fail-open）并告警", P],
        ["edurag:{env}:lock:{md5(rewritten+kb_version+tenant)}", "String(SET NX)", "≤10 s", "单飞锁（singleflight）", "同 key 并发只跑一次，防缓存击穿", "拿不到锁直接走正常链路", P],
        ["edurag:{env}:mem:hot:{user_id}", "String / HASH", "数分钟", "画像 + 近期原子记忆热副本", "记忆召回加速；删除时同步失效", "回源 MySQL + Milvus", P],
        ["edurag:{env}:mem:sum:{sid}", "String", "会话存活期", "L1 会话摘要", "省一次 MySQL 读", "回源 MySQL", P],
        ["edurag:{env}:tag:tree:{ns}", "String(JSON)", "永久", "某命名空间的标签树", "打标时避免反复查表", "回源 MySQL", P],
        ["edurag:faq:events", "Stream", "—", "FAQ 发布/下线事件", "热更新事件流（语料 >5 万条才启用）", "退回定时全量重建", P],
    ]
    return headers, rows, [58, 16, 26, 46, 38, 34, 22], 6


def sheet_eval() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["数据集 / 产出", "用途", "现有规模", "要求规模", "字段", "状态", "结论 / 备注"]
    rows = [
        ["datasets/07_eval/ragas_eval.jsonl", "RAGAS 评估 + 证据门标定", "1 000 条", "≥300 题（覆盖 6 类意图）", "question / ground_truth / contexts / intent", "已有", "6 类分布：知识点 351、题目讲解 136、教案素材 106、备课 90、学习方法 80、闲聊 237"],
        ["docs/analysis/rag_eval_runs.jsonl", "证据门标定原始跑批", "763 条可判定", "可复跑", "含每题 top-10 证据与 qcov", "已标定", "→ 阈值 0.74 / margin 0.01 / qcov 0.20；可答放行 34%、误放 3%"],
        ["docs/analysis/rag_threshold_report.md", "证据门标定报告", "1 份", "—", "—", "已标定", "黄金集覆盖 7–9 年级，库内仅 1 本九年级数学，Recall 上限受语料限制"],
        ["docs/analysis/faq_paraphrases.jsonl", "FAQ 阈值标定正/负样本", "480 条", "正 ≥600 + 负 ≥300", "faq_id / subject / original / paraphrase", "失效", "480/480 的 faq_id 与当前 2 000 条正文不一致，按现文件跑 precision=0"],
        ["docs/analysis/faq_threshold_report.md", "FAQ 阈值报告", "1 份", "重跑后更新", "—", "失效", "0.25/0.10 仅在 160 条语料上标定，只能当保守兜底"],
        ["docs/analysis/norm_k_sensitivity.md", "FAQ_NORM_K 敏感度分析", "1 份", "—", "—", "已有", "K∈[6,36] 判定一致率 95%–100%，K 是刻度不是判据"],
        ["槽位评估集（待建）", "学科槽位路由评估", "0", "7 类 × ≥20 条", "query / subject / grade / 期望档位", "缺失", "明确/简称/口语/缺槽位/多学科/冲突/误导性关键词；看错误过滤率"],
        ["记忆回归集（待建）", "MemRecall@5 与事实准确率", "0", "100 题", "user_id / 历史轮次 / 问题 / 期望记忆", "缺失", "验收：MemRecall@5 ≥0.85、事实准确率 ≥0.95"],
        ["datasets/06_intent/intent_test.jsonl", "意图分类评测", "1 020 条", "—", "query / label", "已有", "6 类各 170；验收 F1 ≥0.90"],
        ["黄金集回归（CI）", "合并门禁", "0（未接 CI）", "≥300 题 + 阈值触发阻断", "同 ragas_eval", "未接入", "§3.6：回归超阈值阻断合并"],
        ["忠实度抽检样本", "每日抽 100 条打分", "0", "每日 100 条", "question / answer / citations", "缺失", "编造率 ≤2%、忠实度 ≥0.95"],
    ]
    return headers, rows, [38, 30, 18, 26, 40, 12, 62], 5


def sheet_models() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["名称", "类型", "路径 / 端点", "规模", "设备 / 部署", "加载或调用时机", "状态"]
    rows = [
        ["BAAI/bge-m3", "向量模型（稠密 1024 维 + 稀疏）", "RAG_EMBED_MODEL_PATH（默认 ../Ragpro/rag_system/models/BAAI/bge-m3）", "约 2.5 GB 显存（FP16）", "GPU 常驻", "启动时加载并预热；请求路径零加载", "已有（CPU 可跑）"],
        ["BAAI/bge-reranker-v2-m3", "精排模型（568 M）", "RAG_RERANKER_PATH", "约 1.5 GB 显存（FP16）", "GPU 批量；CPU 上 40 候选约 4.5 s", "启动时加载（有 CUDA 才启用）", "已有"],
        ["意图分类 BERT-small", "分类模型（K12 6 类）", "MODELS_DIR / ONNX Runtime", "<0.3 GB", "GPU 或 CPU", "启动时加载；失败降级为规则", "待训练（数据已备）"],
        ["DeepSeek LLM", "生成 / 路由 / 抽取 / 摘要", "LLM_BASE_URL + LLM_API_KEY", "外部 API，无显存占用", "外部服务", "请求路径流式调用；首 token ≤600 ms", "未接入"],
        ["MinerU 云解析", "PDF → Markdown", "MINERU_BASE + MINERU_TOKEN", "外部 API（按额度）", "外部服务", "离线摄入期调用；结果按内容 sha256 缓存", "已接入"],
        ["PyMuPDF 本地解析", "PDF 兜底解析", "stdlib 包 fitz", "—", "本地进程", "云解析失败自动降级", "已接入"],
        ["RapidOCR", "图片/扫描页 OCR", "rapidocr_paddle → rapidocr_onnxruntime", "—", "GPU 优先，CPU 兜底", "摄入期调用（RAG_OCR_IMAGES 控制）", "已接入"],
        ["jieba + 领域词典", "中文分词（FAQ BM25）", "qa_system/res/qa_normalize.py", "内存索引", "进程内", "启动时建一次，请求期不加载", "已接入"],
        ["Milvus 2.4", "向量数据库", "MILVUS_URI（默认 http://127.0.0.1:19531）", "—", "docker（etcd + minio + milvus + attu）", "启动时建库建集合并 load", "已接入"],
        ["MySQL 8.0", "关系库（真相源）", "MYSQL_HOST:MYSQL_PORT（默认 3307）", "—", "docker", "启动时建连接池", "已接入"],
        ["Redis 7/8", "缓存 / 会话 / 限流", "REDIS_HOST:REDIS_PORT（默认 6380）", "—", "docker", "启动 ping 失败即降级", "已接入"],
        ["MinIO", "对象存储（原始文档）", "MINIO_ENDPOINT / ACCESS_KEY / SECRET_KEY / BUCKET", "—", "docker", "摄入期存原文与解析中间件", "未接入"],
    ]
    return headers, rows, [26, 30, 54, 26, 26, 40, 26], 6


def sheet_config() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["变量", "默认值", "含义", "分类", "备注"]
    rows = [
        ["APP_ENV / APP_VERSION", "dev / 0.1.0", "运行环境与版本号", "应用", "进日志与缓存 key 前缀"],
        ["HTTP_HOST / HTTP_PORT", "127.0.0.1 / 8000", "服务监听地址", "应用", "—"],
        ["MYSQL_HOST/PORT/USER/PASSWORD/DATABASE", "127.0.0.1 / 3307 / root / — / rag_sbc", "MySQL 连接（同时是 compose 启动参数）", "存储", "密码为空则启动自检失败"],
        ["MYSQL_POOL_MAX", "16", "连接池上限", "存储", "—"],
        ["REDIS_HOST/PORT/PASSWORD/DB", "127.0.0.1 / 6380 / — / 0", "Redis 连接（密码同时用于容器 requirepass）", "存储", "不可用时降级为进程内缓存"],
        ["MILVUS_URI / DATABASE / COLLECTION", "http://127.0.0.1:19531 / edurag / edurag_chunk", "向量库连接与集合", "存储", "集合不存在则 RAG 不就绪（FAQ 不受影响）"],
        ["MILVUS_DIM", "1024", "稠密向量维度", "存储", "换嵌入模型必须同步改"],
        ["MILVUS_TOKEN", "空", "Milvus 鉴权 token", "存储", "本地 standalone 为空"],
        ["FAQ_THRESHOLD", "0.25", "归一化分数绝对门槛（0.25 ⟺ 原始分 4.0）", "qa_system", "重标定前仅作保守兜底"],
        ["FAQ_MARGIN", "0.10", "top1 与 top2 的最小间隔；含义随 FAQ_MARGIN_MODE 变", "qa_system", "改 norm_k 必须重扫"],
        ["FAQ_MARGIN_MODE", "normalized", "margin 所在空间：relative / normalized / raw", "qa_system", "relative 为优先候选，需重标定"],
        ["FAQ_MIN_CANDIDATES", "2", "候选数下限；只有 1 条候选不算命中", "qa_system", "堵住 top1=top2 的假命中"],
        ["FAQ_CROSS_SUBJECT_DELTA", "0.05", "多学科时主学科与另一学科 best 的归一化分差下限", "qa_system", "跨学科只比归一化分"],
        ["FAQ_TOP_K", "20", "单学科 BM25 召回条数", "qa_system", "—"],
        ["FAQ_NORM_K", "12.0", "分数归一化刻度：score = raw/(raw+K)", "qa_system", "刻度不是判据；改它要重标 threshold"],
        ["FAQ_CANDIDATE_N", "3", "未命中时回给 RAG 的候选问法条数", "qa_system", "负缓存必须一并存"],
        ["FAQ_ALIAS_ENABLED", "true", "是否启用别名精确映射", "qa_system", "—"],
        ["FAQ_CORPUS_CSV", "空", "无库模式：从 CSV 装载语料", "qa_system", "生产必须留空，MySQL 是真相源"],
        ["FAQ_ADMIN_TOKEN", "空", "写接口（发布/下线）令牌", "qa_system", "为空时只允许本机调试"],
        ["FAQ_CACHE_ENABLED", "true", "FAQ 二级缓存开关", "缓存", "—"],
        ["FAQ_CACHE_TTL", "3600", "正缓存 TTL（滑动续命）", "缓存", "—"],
        ["FAQ_CACHE_MAX_TTL", "21600", "绝对寿命上限", "缓存", "防止下线答案被无限续命"],
        ["FAQ_CACHE_SLIDE_RATIO", "0.5", "剩余寿命 ≤ 该比例才续命", "缓存", "省 Redis 写"],
        ["FAQ_CACHE_NEGATIVE_TTL", "30", "未命中负缓存 TTL", "缓存", "带 top3 候选"],
        ["ALLOW_DEGRADED_START", "true（dev）", "DB 不可用时是否允许空语料启动", "启动", "生产必须 false"],
        ["RAG_ENABLED", "true", "检索链开关", "rag_system", "关闭则 /v1/rag/* 返回 503"],
        ["RAG_DOCS_DIR", "01_textbook,02_lesson_plans,08_subtitles", "待入库原始资料目录（逗号分隔）", "rag_system", "相对仓库根"],
        ["RAG_PARENT_SIZE / OVERLAP", "1200 / 100", "父块大小与重叠", "rag_system", "进 prompt 的证据单位"],
        ["RAG_CHILD_SIZE / OVERLAP", "300 / 20", "子块大小与重叠", "rag_system", "被检索命中的单位"],
        ["RAG_MAX_CHUNKS_PER_DOC", "2000", "单文档子块上限", "rag_system", "防止大 PDF 撑爆库"],
        ["RAG_TOP_K", "40", "每路 ANN 召回条数", "rag_system", "dense/sparse 各 40"],
        ["RAG_FINAL_N", "6", "最终进 prompt 的父块数", "rag_system", "—"],
        ["RAG_RERANK_ENABLED", "auto", "重排开关；auto=仅 CUDA 可用时开", "rag_system", "CPU 上 40 候选约 4.5 s"],
        ["RAG_RERANK_CANDIDATES", "40", "送入重排的候选数", "rag_system", "—"],
        ["RAG_RERANK_CHARS / MAX_LENGTH", "2000 / 512", "重排输入截断", "rag_system", "约 512 token"],
        ["RAG_EMBED_BACKEND", "auto", "auto=真模型 / hash=确定性哈希（只验证管线）", "rag_system", "—"],
        ["RAG_EMBED_MODEL_PATH", "../Ragpro/.../bge-m3", "稠密+稀疏向量模型路径", "rag_system", "—"],
        ["RAG_RERANKER_PATH", "../Ragpro/.../bge-reranker-v2-m3", "精排模型路径", "rag_system", "—"],
        ["RAG_EMBED_DEVICE", "auto", "cuda / cpu / auto", "rag_system", "—"],
        ["RAG_EMBED_FP16 / RAG_USE_FP16", "true / true", "半精度（CPU 上 fp16 会报错）", "rag_system", "—"],
        ["RAG_EMBED_BATCH_SIZE / MAX_LENGTH", "8 / 512", "向量化批大小与截断", "rag_system", "OOM 时下调"],
        ["RAG_EVIDENCE_THRESHOLD", "0.74", "证据门分数门槛 τ", "证据门", "由 763 条黄金集标定"],
        ["RAG_EVIDENCE_MARGIN", "0.01", "跨文档分差（同文档多块算互相印证）", "证据门", "0 = 关闭歧义判定"],
        ["RAG_EVIDENCE_MIN_DOCS", "1", "最少证据文档数", "证据门", "—"],
        ["RAG_EVIDENCE_QCOV", "0.20", "问句实词在 top1 证据里的最低覆盖率", "证据门", "挡同主题不同知识点"],
        ["RAG_PDF_PARSER", "auto", "auto / mineru / local", "摄入", "auto=云失败退本地"],
        ["RAG_OCR_IMAGES", "false", "是否 OCR Office/图片内嵌图", "摄入", "慢且贵"],
        ["MINERU_TOKEN / BASE / MODEL", "空 / https://mineru.net/api/v4 / vlm", "MinerU 云解析", "摄入", "无 token 时 PDF 直接报错，不静默跳过"],
        ["MINERU_POLL_INTERVAL / TIMEOUT_S", "5 / 900", "轮询间隔与超时", "摄入", "—"],
        ["MINERU_CACHE_DIR", "data/mineru", "解析结果缓存目录", "摄入", "同一 PDF 只花一次额度"],
        ["MODELS_DIR", "空", "本地模型目录（意图分类器等）", "模型", "代码默认值，模板未列"],
        ["LOG_LEVEL / LOG_FILE_LEVEL", "INFO / DEBUG", "控制台/文件日志级别", "日志", "双出口"],
        ["LOG_DIR / LOG_FILE", "logs / app.log", "日志落盘位置", "日志", "—"],
        ["LOG_MAX_BYTES / LOG_BACKUP_COUNT", "52 428 800 / 10", "单文件 50 MB × 10 份轮转", "日志", "—"],
        ["LOG_QUERY_TEXT", "false", "是否把原始 query 写进日志", "日志", "生产必须 false（学生隐私）"],
    ]
    return headers, rows, [40, 34, 52, 14, 48], None


def sheet_gaps() -> tuple[list[str], list[list[str]], list[int], int]:
    headers = ["序号", "缺口", "现状", "目标", "影响", "处置动作", "优先级"]
    rows = [
        ["1", "FAQ 同义问法样本集失效", "480 条，faq_id 与当前正文 480/480 不一致", "正样本 ≥600 + 负样本 ≥300，逐条校验 original", "阈值无法重标定，误命中/漏命中都不可控", "归档旧文件 → gen → calibrate 三步重跑（CLAUDE.md §7.1.7）", "P0"],
        ["2", "FAQ 阈值仍是旧结论", "0.25 / 0.10（160 条语料标的）", "2 000 条语料上 precision ≥0.95、recall ≥0.637", "线上命中质量未知", "依赖第 1 项；产出新报告并写语料指纹", "P0"],
        ["3", "槽位评估集缺失", "0 条", "7 类 × ≥20 条人工标注", "错误过滤率（用户彻底拿不到答案）无法量化", "按 §7.1.8 采样并标注 subject/grade", "P0"],
        ["4", "MySQL 历史行学科为空", "旧导入的 97 行 subject_name=''", "全部归一为「道德与法治」", "这 97 条只能全库检索时才可能命中", "重跑 import_faq 或 UPDATE subject_name", "P1"],
        ["5", "Milvus 缺多租户前置字段", "无 tenant_id / acl_tags / kb_version", "四字段齐全，过滤前置", "多租户上线时只能 Python 后过滤（明令禁止）", "改 schema + 重建集合 + 重灌", "P1"],
        ["6", "索引版本化未实现", "单集合 edurag_chunk", "edurag_v{N} + alias 切换 + 上一版回滚", "无法灰度切库、无法回滚、缓存不随版本失效", "按 §3.5 建 alias 与切换脚本", "P1"],
        ["7", "MinIO 未接入", "0", "原始文档 + 解析中间件入库", "文档原件无处留存，引用无法回溯原文", "compose 加 minio，摄入期先存原文", "P1"],
        ["8", "记忆模块数据面未建", "0（10 张表全部规划中）", "user_profile/session_summary/memory_atom/... 落库", "PR4 无法开工", "DDL + repository 先落地", "P1"],
        ["9", "QAF 数据面未建", "0（7 张表全部规划中）", "qa_pair/qa_version/miss_pool/tag/... 落库", "反馈闭环与标签体系无处落", "DDL + 打标流水线", "P2"],
        ["10", "意图口径已定案（原缺口关闭）", "数据集 6 类（K12 教学）vs 文档 5 类（招生）", "2026-10-26 统一定为 K12 6 类（CLAUDE.md §3.2 / §7.7）", "文档已同步，`INTENT_LABELS` 本就是 6 类，无需改代码", "保持 `INTENT_LABELS` 为唯一出处；待训练分类器", "已关闭"],
        ["11", "教材语料覆盖不足", "6 份章节 + 1 本 PDF", "黄金集覆盖的 7–9 年级各学科都有", "可答题里只有约 20% 能召回正确文档", "按黄金集学科/年级补齐语料后重标证据门", "P0"],
        ["12", "黄金集未接 CI", "手工跑 eval_rag.py", "合并前自动跑 Recall/MRR/忠实度，超阈值阻断", "回归退化无人拦", "CI 里加 eval 任务 + 基线文件", "P2"],
        ["13", "题库未接入在线链路", "15 865 道只在 datasets", "题库 collection / MySQL", "题目讲解类问题只能走文档检索", "定义题库 schema 与检索接口", "P2"],
        ["14", "SFT 语料过少", "50 条", "6 万条（目标）", "无法微调出可用模型", "用 download_opensource 扩充", "P2"],
        ["15", "忠实度抽检未运行", "0", "每日 100 条，编造率 ≤2%", "无法证明「不编造」这条 P0 承诺", "生成链路接通后加离线抽检任务", "P1"],
    ]
    return headers, rows, [6, 26, 40, 38, 40, 48, 10], 6


# --------------------------------------------------------------------------- 主流程
def build(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    wb = xlsxwriter.Workbook(str(out), {"default_date_format": "yyyy-mm-dd", "strings_to_numbers": False})
    wb.set_properties(
        {
            "title": "EduRAG-Pro 数据清单",
            "subject": "系统所需全部数据（语料 / 表结构 / 向量库 / 缓存 / 评估 / 模型 / 参数）",
            "author": "EduRAG-Pro",
            "comments": "由 scripts/export_data_inventory.py 生成，改结构后重跑，不要手改",
        }
    )

    def add(func, name: str, title: str, note: str | None = None) -> None:
        headers, rows, widths, status_col = func()
        add_sheet(wb, name, title, headers, rows, widths, status_col=status_col, note=note)

    add(sheet_overview, "00-总览", "EduRAG-Pro 数据总览", "口径：本表回答「系统跑起来需要哪些数据、现在有什么、还缺什么」。状态色：绿=已有 / 黄=规划中 / 红=失效或缺失。")
    add(sheet_corpus, "01-原始语料", "datasets/ 原始语料清单", "全部为只读数据；摄入后落到 Milvus 或 MySQL，原始文件不参与在线请求。")
    add(sheet_faq, "02-FAQ语料规范", "FAQ 语料（04_faq/edu_faq.csv → MySQL faq）字段与质量规则")
    add(sheet_mysql, "03-MySQL表结构", "MySQL 表结构清单（3 张已建 + 17 张规划）", "已建字段以 scripts/init_db.py 为准；规划字段以 CLAUDE.md §6.4 / §7.3 为准，标注「拟定」的为按契约推导、实现前需再确认。")
    add(sheet_milvus, "04-Milvus集合", "Milvus 集合 schema（edurag.edurag_chunk）", "过滤必须前置到 Milvus expr，不允许检索后 Python 过滤。")
    add(sheet_redis, "05-Redis键", "Redis 键族与 TTL", "Redis 只放可重算的数据；读写全部 fail-open。")
    add(sheet_eval, "06-评估与标定数据", "评估集、标定跑批与待建评估集")
    add(sheet_models, "07-模型与外部服务", "模型权重与外部依赖", "关键模型加载失败 → fail-fast 退出；意图模型失败 → 降级为规则。")
    add(sheet_config, "08-配置参数", "配置基线（.env / base/config.py）", "所有阈值只从这里取，业务代码禁止硬编码。")
    add(sheet_gaps, "09-数据缺口", "数据缺口与处置动作（按优先级）", "P0 = 阻断交付；P1 = 交付前完成；P2 = 下一版本。")

    # 学科分布单独画在 02 表下方，避免再多一个 sheet
    ws = wb.get_worksheet_by_name("02-FAQ语料规范")
    if ws is not None:
        fmt_t = wb.add_format({"bold": True, "font_size": 11, "font_color": "#1F3864"})
        fmt_h = wb.add_format({"bold": True, "bg_color": HEADER_BG, "font_color": "white", "border": 1, "align": "center"})
        fmt_c = wb.add_format({"border": 1, "font_size": 10})
        base = 14
        ws.write(base, 0, "现行语料学科分布（2 000 条）", fmt_t)
        ws.write(base + 1, 0, "学科", fmt_h)
        ws.write(base + 1, 1, "条数", fmt_h)
        dist = [
            ("数学", 605), ("物理", 310), ("化学", 237), ("生物", 184), ("英语", 163),
            ("语文", 162), ("历史", 134), ("地理", 108), ("道德与法治", 97),
        ]
        for i, (subject, count) in enumerate(dist, start=base + 2):
            ws.write(i, 0, subject, fmt_c)
            ws.write(i, 1, count, fmt_c)
        ws.write(base + len(dist) + 2, 0, "合计", fmt_c)
        ws.write(base + len(dist) + 2, 1, sum(c for _, c in dist), fmt_c)

    wb.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="导出 EduRAG-Pro 数据清单 Excel")
    parser.add_argument("--out", default=str(ROOT / "docs" / "数据清单-EduRAG-Pro.xlsx"))
    args = parser.parse_args()
    path = build(Path(args.out))
    print(f"written: {path}")


if __name__ == "__main__":
    main()
