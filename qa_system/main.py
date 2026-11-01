"""qa_system 查询测试台（交互式 CLI）。

用途：不经过 HTTP、不启动 FastAPI，直接把 FAQService 拉起来问问题，用来判断
"这条问题到底该由 FAQ 回答，还是该交给 RAG"。

运行（三种都行，内部会自己把仓库根加进 sys.path）：

    python -m qa_system.main                 # 读 MySQL 里 published 的语料
    python qa_system\\main.py                 # 同上
    python -m qa_system.main --csv           # 不连库，直接从 datasets CSV 装载
    python -m qa_system.main -q "勾股定理咋证明啊"   # 单条查询，打印后退出（可脚本化）

交互命令：
    /help            帮助
    /stats           语料规模与命中统计
    /rebuild         重建索引（等价于后台发布后重建）
    /rank <问题>     只打分不判定，看 top5 分数与 top1-top2 分差（调阈值时用）
    /subject <学科>  只在该学科内检索；/subject off 取消
    exit / quit      退出

为什么要有这个文件：`/v1/faq/search` 只回最终结论，排障时看不出"差多少分"。
`/rank` 直接暴露分数，这正是标定 `FAQ_THRESHOLD` / `FAQ_MARGIN` 时最需要的信息。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# 允许 `python qa_system/main.py` 直接运行：把仓库根（本文件的父目录的父目录）放进 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from base.config import BASE_DIR, ENV_FILE, get_settings  # noqa: E402
from base.contracts import FAQHit  # noqa: E402
from base.logger import configure, get_logger  # noqa: E402
from qa_system.res.qa_db import FAQRepository, InMemoryFAQRepository, MySQLFAQRepository  # noqa: E402
from qa_system.res.qa_matcher import MatchPolicy  # noqa: E402
from qa_system.res.qa_search import FAQService  # noqa: E402

log = get_logger(__name__)

REASON_ZH = {
    "exact": "精确命中（与原问题归一化后一致）",
    "alias_exact": "别名命中（同义词表折算后一致）",
    "ok": "相似命中（分数与领先幅度均达标）",
    "mid_confidence": "中置信：过阈值但领先幅度不够，只给候选与追问，不直接作答",
    "multi_scope_handoff": "中置信：多学科打平，交 RAG + 追问（跨学科 delta 未标定）",
    "cross_subject_ambiguous": "中置信：两个学科分数太接近，交 RAG + 追问",
    "subject_inferred_handoff": "中置信：学科是推断的，交 RAG + 追问",
    "subject_context_handoff": "中置信：学科来自上下文，交 RAG + 追问",
    "subject_conflict_handoff": "中置信：规则与模型学科结论冲突，交 RAG",
    "subject_invalid_handoff": "中置信：学科参数非法（已全线兜底），交 RAG + 追问",
    "user_confirmed": "用户确认命中（追问被采纳）",
    "single_candidate": "未命中：只有一个候选，谈不上领先",
    "subject_empty": "未命中：该学科没有语料（不兜底成全库）",
    "below_threshold": "未命中：最高分低于阈值",
    "ambiguous": "未命中：top1 与 top2 太接近（同题族，不能猜）",
    "no_match": "未命中：一个词都没对上，整句交给 RAG",
    "empty_tokens": "未命中：分词结果为空",
    "empty_query": "未命中：空问题",
    "not_loaded": "未命中：语料未装载",
}

# 命中来源必须显示：同一句问两遍，第一遍走索引、第二遍走缓存，耗时差一个量级，
# 不标出来会让人误以为"服务忽快忽慢"。
SOURCE_ZH = {"cache": "Redis 缓存", "index": "MySQL 索引", "rag": "RAG 检索", "none": "无"}

BANNER = """\
================================================================
 EduRAG-Pro · qa_system 查询测试台
 语料：MySQL 中 status=published 的 FAQ
 判定：score ≥ {threshold} 且 top1-top2 ≥ {margin} 才算命中
 输入问题回车查询；/help 看命令；exit 退出
================================================================"""


def build_policy(settings) -> MatchPolicy:
    return MatchPolicy(
        threshold=settings.faq_threshold,
        margin=settings.faq_margin,
        candidate_n=settings.faq_candidate_n,
        norm_k=settings.faq_norm_k,
        high_margin=settings.faq_high_margin,  # §7.1.9 三档；等于 margin 即单档
    )


async def build_service(*, use_csv: bool, limit: int | None) -> FAQService:
    settings = get_settings()
    policy = build_policy(settings)

    # 配置问题先喊出来：最常见的失败是"换个工作目录启动 → 读不到 .env → 密码为空 → 1045"
    problems = settings.self_check()
    if problems:
        print(f"[!] 配置自检未通过（配置文件：{ENV_FILE}，存在={ENV_FILE.exists()}）")
        for problem in problems:
            print(f"    - {problem}")
        print()

    repository: FAQRepository
    if use_csv:
        from qa_system.res.qa_csv import load_csv_records

        csv_path = Path("datasets/04_faq/edu_faq.csv")
        if not csv_path.is_absolute():
            csv_path = BASE_DIR / csv_path
        records = load_csv_records(csv_path, limit=limit)
        repository = InMemoryFAQRepository(records)
        print(f"[i] 语料来源：{csv_path}（{len(records)} 条）")
    else:
        from base.db import MySQL

        mysql = MySQL(settings)
        if not mysql.ping():
            raise SystemExit(
                f"[x] 连不上 MySQL：{settings.mysql_user}@{settings.mysql_host}:{settings.mysql_port}"
                f"/{settings.mysql_database}\n"
                f"    配置文件：{ENV_FILE}（存在={ENV_FILE.exists()}）\n"
                f"    1) 容器是否在跑：docker ps --filter name=edurag-\n"
                f"    2) 端口/密码是否与 .env 一致（默认 3307 / 见 MYSQL_PASSWORD）\n"
                f"    3) 只想离线试：加 --csv 走本地语料"
            )
        repository = MySQLFAQRepository(mysql)
        print(f"[i] 语料来源：MySQL {settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")

    # 与 API 用同一套 Redis 二级缓存：CLI 里连问同一句两次，第二次就会显示「来源：Redis 缓存」，
    # 这是验证续命/回填逻辑最直接的方式。Redis 不可用时退化为进程内缓存，不影响查询。
    faq_cache = None
    if settings.faq_cache_enabled:
        from base.logger import get_logger as _get_logger

        try:
            from qa_system.res.qa_cache import RedisFAQCache, create_redis

            redis_client = create_redis(settings)
            redis_client.ping()
            faq_cache = RedisFAQCache(
                redis_client,
                max_ttl=settings.faq_cache_max_ttl,
                slide_ratio=settings.faq_cache_slide_ratio,
            )
            print(f"[i] 缓存：Redis {settings.redis_host}:{settings.redis_port}/{settings.redis_db}")
        except Exception as exc:
            from qa_system.res.qa_cache import InMemoryFAQCache

            faq_cache = InMemoryFAQCache(max_ttl=settings.faq_cache_max_ttl)
            _get_logger(__name__).warning("faq_cache_disabled", reason="redis_unavailable", detail=str(exc))
            print(f"[!] Redis 不可用，退化为进程内缓存：{exc}")

    service = FAQService(
        repository,
        policy=policy,
        alias_enabled=settings.faq_alias_enabled,
        top_k=settings.faq_top_k,
        extra_words=settings.subjects,
        cache=faq_cache,
        cache_ttl=settings.faq_cache_ttl,
        cache_negative_ttl=settings.faq_cache_negative_ttl,
        cache_prefix=f"edurag:{settings.app_env}",
        log_query=settings.log_query_text,
        dedupe_near_dup=settings.faq_dedupe_near_dup,
        tenant_id=settings.tenant_id,
    )
    info = await service.load()
    await service.warmup()
    print(f"[i] 装载完成：{info.get('docs')} 条，耗时 {info.get('cost_ms')} ms")
    return service


def reason_text(reason: str) -> str:
    """把 `cache_ok` 这类带前缀的原因翻译成人话（缓存命中时 reason 会多一层前缀）。"""
    if reason.startswith("cache_"):
        return REASON_ZH.get(reason[6:], reason[6:]) + "（结果直接取自缓存）"
    return REASON_ZH.get(reason, reason)


def print_hit(query: str, hit: FAQHit, subject: str | None) -> None:
    flag = {"high": "命中", "mid": "中置信", "low": "未命中"}.get(hit.tier, hit.tier)
    source = SOURCE_ZH.get(hit.source, hit.source or "未知")
    cache_note = {
        "hit": "缓存命中，未回源",
        "miss": "缓存未命中，已回源",
        "disabled": "缓存未启用",
        "error": "缓存异常，已降级回源",
    }.get(hit.cache, "")
    print()
    print(f"问题：{query}" + (f"   （限定学科：{subject}）" if subject else ""))
    print(
        f"结论：{flag}   来源：{source}"
        + (f"（{cache_note}）" if cache_note else "")
        + f"   分数 {hit.score:.4f}   耗时 {hit.latency_ms:.2f} ms"
    )
    print(f"原因：{reason_text(hit.reason)}")

    if hit.hit:
        print(f"原问：{hit.matched_question}")
        print(f"学科：{hit.subject}   faq_id={hit.faq_id}" + ("   （经别名）" if hit.alias_hit else ""))
        print("-" * 60)
        print(f"答案：{hit.answer}")
        print("-" * 60)
    elif hit.candidates:
        if hit.tier == "mid" and hit.matched_question:
            # 中置信的唯一"好处"：有一个可信的追问对象（top1），答案仍必须走 RAG 或用户确认
            print(f"追问：你是不是想问「{hit.matched_question}」？(faq_id={hit.faq_id})")
        print("候选（交给 RAG 做 query 扩展）：")
        for idx, cand in enumerate(hit.candidates, start=1):
            print(f"  {idx}. [{cand.score:.4f}] {cand.question}  (faq_id={cand.faq_id})")
    print()


async def print_rank(service: FAQService, query: str, subject: str | None, top: int) -> None:
    ranked = await service.rank(query, subject=subject, top_k=top)
    settings = get_settings()
    print()
    print(f"排障打分：{query}" + (f"   （限定学科：{subject}）" if subject else ""))
    if not ranked:
        print("  没有任何候选（空问题 / 分词为空 / 语料未装载）")
        print()
        return

    for idx, (faq_id, score) in enumerate(ranked, start=1):
        record = await service.get(faq_id)
        question = record.question if record else "?"
        print(f"  {idx}. [{score:.4f}] {question}  (faq_id={faq_id})")

    top1 = ranked[0][1]
    top2 = ranked[1][1] if len(ranked) > 1 else 0.0
    gap = top1 - top2
    passed_threshold = top1 >= settings.faq_threshold
    passed_margin = gap >= settings.faq_margin
    print()
    print(f"  top1={top1:.4f}  top2={top2:.4f}  分差={gap:.4f}")
    print(
        f"  阈值 {settings.faq_threshold}：{'通过' if passed_threshold else '不通过'}"
        f"   ｜   领先幅度 {settings.faq_margin}：{'通过' if passed_margin else '不通过'}"
    )
    print()


def print_stats(service: FAQService) -> None:
    state = service.state()
    stats = service.stats()
    print()
    print(f"语料：{state['docs']} 条，版本 v{state['version']}，学科 {len(state['subjects'])} 个"
          + (f"，未标注学科 {state['unlabeled']} 条" if state.get("unlabeled") else ""))
    print(f"学科：{'、'.join(state['subjects'])}")
    print(
        f"累计：{stats['searches']} 次查询，命中 {stats['hits']}（{stats['hit_rate'] * 100:.1f}%），"
        f"精确 {stats['exact_hits']}，别名 {stats['alias_hits']}，平均 {stats['avg_latency_ms']} ms"
    )
    print()


async def interactive(service: FAQService) -> None:
    settings = get_settings()
    print(BANNER.format(threshold=settings.faq_threshold, margin=settings.faq_margin))
    subject: str | None = None

    while True:
        try:
            raw = input("问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if not raw:
            continue
        lowered = raw.lower()
        if lowered in {"exit", "quit", "q"}:
            print("再见。")
            return

        if raw.startswith("/"):
            command, _, argument = raw[1:].partition(" ")
            argument = argument.strip()
            if command == "help":
                print("  /help            帮助")
                print("  /stats           语料规模与命中统计")
                print("  /rebuild         重建索引")
                print("  /rank <问题>     只打分不判定（调阈值用）")
                print("  /subject <学科>  限定学科；/subject off 取消")
                print("  exit / quit      退出")
            elif command == "stats":
                print_stats(service)
            elif command == "rebuild":
                info = await service.rebuild()
                print(f"[i] 重建完成：{info}\n")
            elif command == "rank":
                if not argument:
                    print("[!] 用法：/rank 你的问题\n")
                    continue
                await print_rank(service, argument, subject, top=5)
            elif command == "subject":
                if not argument or argument.lower() in {"off", "none", "clear"}:
                    subject = None
                    print("[i] 已取消学科限定\n")
                else:
                    subject = argument
                    print(f"[i] 之后的查询限定学科：{subject}（输入 /subject off 取消）\n")
            else:
                print(f"[!] 未知命令 /{command}，输入 /help 查看\n")
            continue

        hit = await service.search(raw, subject=subject)
        print_hit(raw, hit, subject)


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    # CLI 与 API 写同一份 logs/app.log：在测试台里问的问题，事后能在日志里对上。
    # 控制台只留 WARNING，避免业务日志把"问题/结论/答案"的阅读节奏打散——
    # 命中来源已经由 print_hit 打在自己那一行里了。
    configure(
        settings.log_level,
        console_level="WARNING",
        log_path=settings.log_path,
        max_bytes=settings.log_max_bytes,
        backup_count=settings.log_backup_count,
        file_level=settings.log_file_level,
    )
    # jieba 自带 DEBUG 日志（"Building prefix dict…"），在测试台里只会干扰输出
    import logging

    logging.getLogger("jieba").setLevel(logging.WARNING)

    # 依赖自检：用错解释器是最高频的坑（比如系统 python / 另一个 conda 环境）
    missing = []
    for module in ("jieba", "pymysql", "dbutils", "pydantic_settings"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise SystemExit(
            f"[x] 当前解释器缺依赖：{', '.join(missing)}\n"
            f"    解释器：{sys.executable}\n"
            f"    解决：用项目虚拟环境 .\\.venv\\Scripts\\python.exe -m qa_system.main，\n"
            f"          或给这个环境装依赖：{sys.executable} -m pip install -r requirements.txt"
        )

    print(f"[i] 解释器：{sys.executable}")
    service = await build_service(use_csv=args.csv, limit=args.limit)

    try:
        if args.query:
            for query in args.query:
                if args.rank:
                    await print_rank(service, query, args.subject, top=args.top)
                else:
                    hit = await service.search(query, subject=args.subject)
                    print_hit(query, hit, args.subject)
            return 0
        await interactive(service)
        return 0
    finally:
        # 语料服务本身不持有连接池，连接由外部 MySQL 对象管理；CLI 退出即回收
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="qa_system 查询测试台")
    parser.add_argument("--csv", action="store_true", help="从 datasets/04_faq/edu_faq.csv 装载，不连 MySQL")
    parser.add_argument("--limit", type=int, default=None, help="配合 --csv：只装载前 N 条")
    parser.add_argument("-q", "--query", action="append", help="单条查询（可重复），打印后退出")
    parser.add_argument("--rank", action="store_true", help="配合 -q：只打分不判定")
    parser.add_argument("--subject", default=None, help="限定学科")
    parser.add_argument("--top", type=int, default=5, help="配合 --rank：显示前 N 条分数")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
