"""PDF 加载器：交给 MinerU 云端解析，本地只做缓存与结构还原。

为什么不用 PyMuPDF 直接抽文字：
- 教材 PDF 大量是扫描件/双栏/公式/表格，`page.get_text()` 出来是断行与乱码，
  旧项目的补救办法是"整页渲染 + OCR 识别大图"，代价高且丢版面；
- MinerU 的 vlm 模式直接产出 Markdown（标题层级、表格 HTML、公式 LaTeX 都在），
  切分器可以按标题切，公式也不会被拆成一行行碎片。

额度很贵（一页一次），所以这里做了内容级缓存：同一份 PDF 解析一次，结果落在
`data/mineru/<content_hash>/`，重跑入库不再调云端。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import tempfile
import time
import zipfile
from pathlib import Path

import httpx

from base.logger import get_logger
from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError, guess_subject

log = get_logger(__name__)

_IMAGE_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_IMAGE_HTML = re.compile(r"<img\b[^>]*/?>", re.IGNORECASE)

# MinerU 单文件页数上限。交付语料里 `高中/数学/教材.pdf` 270 页、`高中/历史/教材.pdf` 222 页，
# 直接送会被拒（2026-10-29 实测 `number of pages exceeds limit (200 pages)`），
# 而 `RAG_PDF_PARSER=auto` 会把它当成"云解析失败"静默退成本地 PyMuPDF——质量掉一半还看不出来。
MAX_PAGES_PER_REQUEST = 200

# 拆卷后每卷的 data_id 后缀分隔符：既用于上传名，也用于识别"这份是第几卷"
_PART_SEP = "._part"


def strip_images(markdown: str) -> str:
    """去掉图片引用。

    MinerU 的图片链接是 `images/<64 位哈希>.jpg`，直接进切分会把哈希串当正文词，
    既占 token 又把 BM25/向量带偏；图片本身对"查知识点"没有检索价值
    （需要看图时由前端按 page 定位原 PDF）。表格 HTML 与 LaTeX 公式**保留**。
    """
    text = _IMAGE_MD.sub("", markdown)
    text = _IMAGE_HTML.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_hash(path: Path) -> str:
    """按内容算哈希（不按 mtime）：文件重新下载但内容没变时不该重新解析。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _is_page_limit_error(exc: BaseException) -> bool:
    """识别"页数超限"这一类拒收。

    按文案匹配是没办法的办法（API 没有给错误码），所以只认两个稳定片段：
    `exceeds limit` 与 `pages`。认错方向的代价是"多拆了一次"，比"整本退成本地解析"轻得多。
    """
    text = str(exc).lower()
    return "exceeds limit" in text and "page" in text


class MinerUClient:
    """MinerU v4 API 的最小客户端：申请上传链接 → PUT 上传 → 轮询 → 下载 zip。"""

    def __init__(
        self,
        token: str,
        *,
        base: str = "https://mineru.net/api/v4",
        model: str = "vlm",
        poll_interval: int = 5,
        timeout_s: int = 900,
        cache_dir: Path | None = None,
        http_timeout: float = 120.0,
        upload_timeout_s: float = 900.0,
        max_pages: int = MAX_PAGES_PER_REQUEST,
    ) -> None:
        if not token:
            raise LoaderError("MINERU_TOKEN 未配置，无法解析 PDF（在 .env 里填 token）")
        self._token = token
        self._base = base.rstrip("/")
        self._model = model
        self._poll_interval = max(1, poll_interval)
        self._timeout = timeout_s
        self._cache_dir = cache_dir
        self._max_pages = max(1, max_pages)
        self._http = httpx.Client(timeout=http_timeout)
        # 上传单独一条超时：`httpx` 的 `timeout=` 是**整条请求**（含写 body）的上限，
        # 120 s 对 6–66 MB 的教材在上行带宽下必然写超时——2026-10-29 实测大量
        # `WriteTimeout: The write operation timed out` 后静默退成本地 PyMuPDF（质量更低）。
        # 只放宽写/读，连接与等池仍保持短超时：连不上就该立刻失败，而不是干等 15 分钟。
        self._upload_timeout = httpx.Timeout(
            connect=30.0, read=float(http_timeout), write=float(upload_timeout_s), pool=30.0
        )

    # ---- 对外 ----
    def parse(self, pdf_path: Path) -> tuple[str, dict]:
        digest = content_hash(pdf_path)
        cached = self._load_cache(digest)
        if cached is not None:
            markdown, meta = cached
            meta["from_cache"] = True
            return markdown, meta

        try:
            markdown, meta = self._parse_remote(pdf_path, digest)
        except LoaderError as exc:
            if not _is_page_limit_error(exc):
                raise
            # 超页数**不是失败**，是"这份要拆开发"：拆卷后每卷都不超限，
            # 按顺序拼回同一份 markdown，缓存仍记在原文件的 digest 上
            markdown, meta = self._parse_split(pdf_path, digest, str(exc))
        self._save_cache(digest, markdown, meta)
        meta["from_cache"] = False
        return markdown, meta

    # ---- 云端流程 ----
    def _parse_remote(self, pdf_path: Path, digest: str) -> tuple[str, dict]:
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        payload = {
            "enable_formula": True,  # 公式转 LaTeX，问"求根公式"时靠它
            "enable_table": True,    # 表格转 HTML，教材里的对照表全靠它
            "language": "ch",
            "model_version": self._model,
            "files": [{"name": pdf_path.name, "data_id": digest}],
        }
        resp = self._http.post(f"{self._base}/file-urls/batch", headers=headers, json=payload)
        data = resp.json()
        if data.get("code") != 0:
            raise LoaderError(f"MinerU 申请上传链接失败：{data}")
        batch_id = data["data"]["batch_id"]
        upload_url = data["data"]["file_urls"][0]

        # 注意：
        # 1. 这个 PUT 是 OSS 预签名 URL，**带 Authorization 会 403**；
        # 2. **也不要带 `Content-Type`**——签名把 content-type 算进去了，改成 application/pdf
        #    会让服务端直接断开连接（2026-10-29 实测：POST 200 → PUT `RemoteProtocolError`）。
        #    只显式给 `Content-Length`：预签名 URL 走对象存储，chunked 上传会被判签名不符。
        # 3. 传**文件对象**而不是 `read()` 出来的 bytes：httpx 会流式发送，
        #    66 MB 的教材不用先整体读进内存（同时也让 write 超时真正按"写"计时）。
        with pdf_path.open("rb") as fh:
            put = self._http.put(
                upload_url,
                content=fh,
                headers={"Content-Length": str(pdf_path.stat().st_size)},
                timeout=self._upload_timeout,
            )
        if put.status_code >= 400:
            raise LoaderError(f"MinerU 上传失败 HTTP {put.status_code}：{put.text[:200]}")

        items = self._poll(batch_id, headers)
        done = [i for i in items if i.get("state") == "done"]
        if not done:
            reason = items[0].get("err_msg") if items else "无结果"
            raise LoaderError(f"MinerU 解析失败：{reason}")

        zip_bytes = self._http.get(done[0]["full_zip_url"]).content
        markdown, pages, images = self._unpack(zip_bytes, digest)
        return markdown, {"batch_id": batch_id, "pages": pages, "images": images, "digest": digest}

    def _parse_split(self, pdf_path: Path, digest: str, reason: str) -> tuple[str, dict]:
        """把超页数的 PDF 拆成 ≤`max_pages` 的若干卷，逐卷云端解析后按序拼接。

        为什么在客户端里做而不是让调用方处理：调用方（工厂）只知道"这份 PDF 要用哪个解析器"，
        不知道云端的上限；把拆卷藏在这里，`auto` 模式就不会再因为页数把教材降级成本地解析。

        分卷用 PyMuPDF 写临时文件（`datasets/` 只读，绝不能就地改）。临时目录用完即删；
        每卷的抽取目录也一并清掉，缓存目录里只留最终合并的那一份，免得 `data/mineru/` 里
        多出一堆 `.*._partN` 目录、把"已缓存多少本"数错。
        """
        import fitz  # PyMuPDF：拆卷只用到它，没装就不该走到这条路

        doc = fitz.open(str(pdf_path))
        try:
            total = doc.page_count
            parts = math.ceil(total / self._max_pages)
            if parts <= 1:
                raise LoaderError(f"MinerU 拒收但页数只有 {total}：{reason}")

            datas: list[dict] = []
            chunks: list[str] = []
            with tempfile.TemporaryDirectory(prefix="mineru_split_") as tmp:
                tmp_dir = Path(tmp)
                for i in range(parts):
                    start = i * self._max_pages
                    end = min(start + self._max_pages, total)
                    part_pdf = tmp_dir / f"{pdf_path.stem}{_PART_SEP}{i + 1}.pdf"
                    part = fitz.open()
                    part.insert_pdf(doc, from_page=start, to_page=end - 1)
                    part.save(str(part_pdf))
                    part.close()

                    part_key = f"{digest}{_PART_SEP}{i + 1}"
                    markdown, meta = self._parse_remote(part_pdf, part_key)
                    chunks.append(markdown)
                    datas.append(meta)
                    log.info(
                        "mineru_part_ok",
                        file=pdf_path.name,
                        part=i + 1,
                        parts=parts,
                        pages=end - start,
                    )
                    self._drop_extract(part_key)
        finally:
            doc.close()

        markdown = "\n\n---\n\n".join(chunks)
        return markdown, {
            "split": parts,
            "pages": sum(int(d.get("pages", 0)) for d in datas),
            "images": sum(int(d.get("images", 0)) for d in datas),
            "digest": digest,
            "batches": [d.get("batch_id") for d in datas],
            "split_reason": reason,
        }

    def _drop_extract(self, key: str) -> None:
        """删掉某一卷的抽取目录（只删自己刚建的，缓存根目录不动）。"""
        if self._cache_dir is None:
            return
        target = self._cache_dir / key
        if not target.is_dir():
            return
        for child in sorted(target.rglob("*"), reverse=True):
            try:
                child.rmdir() if child.is_dir() else child.unlink()
            except OSError:
                pass
        try:
            target.rmdir()
        except OSError:
            pass

    def _poll(self, batch_id: str, headers: dict) -> list[dict]:
        deadline = time.time() + self._timeout
        while True:
            resp = self._http.get(f"{self._base}/extract-results/batch/{batch_id}", headers=headers)
            items = resp.json()["data"]["extract_result"]
            if all(i["state"] in ("done", "failed") for i in items):
                return items
            if time.time() > deadline:
                raise LoaderError(f"MinerU 轮询超时（{self._timeout}s）：batch={batch_id}")
            time.sleep(self._poll_interval)

    def _unpack(self, zip_bytes: bytes, digest: str) -> tuple[str, int, int]:
        """解压并返回 (markdown, 页数, 图片数)。

        优先用 `content_list.json` 的 page_idx 统计页数；找不到就退回"按二级标题数估算"。
        """
        target = (self._cache_dir or Path.cwd()) / digest
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(target)

        md_files = sorted(target.rglob("*.md"))
        if not md_files:
            raise LoaderError(f"MinerU 结果里没有 .md：{target}")
        # all.md 是本地合并产物，排除掉，避免自己吃自己
        md_files = [m for m in md_files if m.name != "all.md"]
        markdown = "\n\n---\n\n".join(m.read_text(encoding="utf-8") for m in md_files)

        pages = 0
        for cl in target.rglob("*content_list.json"):
            try:
                blocks = json.loads(cl.read_text(encoding="utf-8"))
                pages = max(pages, max((b.get("page_idx", 0) for b in blocks), default=0) + 1)
            except (json.JSONDecodeError, TypeError):
                continue
        images = sum(1 for _ in target.rglob("*.jpg")) + sum(1 for _ in target.rglob("*.png"))
        return markdown, pages, images

    # ---- 缓存 ----
    def _cache_path(self, digest: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / digest / "content.md"

    def _load_cache(self, digest: str) -> tuple[str, dict] | None:
        p = self._cache_path(digest)
        if p is None or not p.exists():
            return None
        meta_path = p.with_name("meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return p.read_text(encoding="utf-8"), meta

    def _save_cache(self, digest: str, markdown: str, meta: dict) -> None:
        p = self._cache_path(digest)
        if p is None:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(markdown, encoding="utf-8")
        p.with_name("meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


class PdfLoader(BaseDocumentLoader):
    """`.pdf` → MinerU → Markdown。"""

    extensions = (".pdf",)

    def __init__(self, client: MinerUClient) -> None:
        self._client = client

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path).resolve()
        markdown, meta = self._client.parse(p)
        markdown = strip_images(markdown)
        if not markdown.strip():
            raise LoaderError(f"MinerU 返回空正文：{p}")
        return LoadedDoc(
            text=markdown,
            source=str(p),
            doc_type="textbook",
            subject=guess_subject(str(p)),
            title=p.stem,
            meta=meta,
        )