"""预热 MinerU 解析缓存：把 PDF 云端解析与入库解耦。

为什么要单独一个脚本：两条链路卡在不同的资源上——MinerU 是**网络**瓶颈
（一本教材约 6–7 分钟），入库是 **CPU** 瓶颈（BGE-M3 在 CPU 上约 1.4 块/秒）。
串行跑时后者排队等前者，全量 20 本教材要 2 小时以上，一条后台命令就会撞到时限被杀。
这里只做解析、并发 N 路，结果写进 `data/mineru/<digest>/content.md`——与
`MinerUClient` 是**同一份缓存**，之后的 `python -m rag_system.ingest` 全部命中缓存、零网络等待。

它只读 `datasets/`，不加载模型、不写 Milvus、不写 `doc_fingerprints.json` / `ingest_manifest.json`，
所以**可以和入库进程同时跑**而不会争抢状态文件。

用法：

    python scripts/prewarm_mineru.py                 # 扫 RAG_DOCS_DIR 下所有未缓存的 PDF
    python scripts/prewarm_mineru.py --paths datasets/01_textbook --workers 3
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from base.config import BASE_DIR, Settings, get_settings  # noqa: E402
from base.logger import configure, get_logger  # noqa: E402
from rag_system.document_loaders.loader_pdf_mineru import MinerUClient, content_hash  # noqa: E402

log = get_logger(__name__)


def _iter_pdfs(roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".pdf":
            out.append(root)
            continue
        if root.is_dir():
            out.extend(sorted(p for p in root.rglob("*.pdf") if p.is_file()))
    # 去重并保序：同一份 PDF 被 --paths 和 RAG_DOCS_DIR 同时点到时不解析两遍
    return list(dict.fromkeys(out))


def _resolve_docs(settings: Settings) -> list[Path]:
    roots = [BASE_DIR / part.strip() for part in settings.rag_docs_dir.split(",") if part.strip()]
    return [r for r in roots if r.exists()]


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    configure(settings.log_level, log_path=settings.log_path)

    p = argparse.ArgumentParser(description="预热 MinerU 解析缓存（只解析，不入库）")
    p.add_argument("--paths", nargs="*", default=[], help="文件或目录；默认用 RAG_DOCS_DIR")
    p.add_argument("--workers", type=int, default=3, help="并发解析数（云端限流时下调）")
    args = p.parse_args(argv)

    roots = [Path(x).resolve() for x in args.paths] or _resolve_docs(settings)
    pdfs = _iter_pdfs(roots)
    if not pdfs:
        print("没有找到 PDF")
        return 0

    client = MinerUClient(
        settings.mineru_token,
        base=settings.mineru_base,
        model=settings.mineru_model,
        poll_interval=settings.mineru_poll_interval,
        timeout_s=settings.mineru_timeout_s,
        upload_timeout_s=settings.mineru_upload_timeout_s,
        cache_dir=settings.mineru_cache_path,
    )

    pending: list[Path] = []
    for pdf in pdfs:
        if client._load_cache(content_hash(pdf)) is not None:  # noqa: SLF001 - 同包内部接口
            print(f"[cached] {pdf.relative_to(BASE_DIR)}", flush=True)
        else:
            pending.append(pdf)

    if not pending:
        print(f"全部命中缓存（{len(pdfs)} 个 PDF）", flush=True)
        return 0

    print(f"待解析 {len(pending)} 个 PDF，并发 {args.workers}", flush=True)
    started = time.time()
    failed = 0

    def _one(pdf: Path) -> tuple[Path, int, float]:
        t0 = time.time()
        markdown, _meta = client.parse(pdf)  # 命中缓存时瞬间返回，重复调用安全
        return pdf, len(markdown), time.time() - t0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_one, pdf): pdf for pdf in pending}
        for done, fut in enumerate(as_completed(futures), start=1):
            pdf = futures[fut]
            try:
                _, chars, cost = fut.result()
                print(
                    f"[{done}/{len(pending)}] ok {pdf.relative_to(BASE_DIR)} "
                    f"chars={chars} cost={cost:.0f}s",
                    flush=True,
                )
                log.info("mineru_prewarm_ok", path=str(pdf), chars=chars, cost_s=int(cost))
            except Exception as exc:  # noqa: BLE001 - 单个文件失败不该中断整批
                failed += 1
                print(f"[{done}/{len(pending)}] FAIL {pdf.relative_to(BASE_DIR)}: {exc}", flush=True)
                log.warning("mineru_prewarm_error", path=str(pdf), error=str(exc))

    print(f"完成：成功 {len(pending) - failed} / 失败 {failed}，耗时 {time.time() - started:.0f}s", flush=True)
    print("接着跑：python -m rag_system.ingest --defaults", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
