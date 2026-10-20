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
import time
import zipfile
from pathlib import Path

import httpx

from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError, guess_subject

_SUBJECTS = ("语文", "数学", "英语", "物理", "化学", "生物", "历史", "地理", "政治")


def content_hash(path: Path) -> str:
    """按内容算哈希（不按 mtime）：文件重新下载但内容没变时不该重新解析。"""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


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
    ) -> None:
        if not token:
            raise LoaderError("MINERU_TOKEN 未配置，无法解析 PDF（在 .env 里填 token）")
        self._token = token
        self._base = base.rstrip("/")
        self._model = model
        self._poll = max(1, poll_interval)
        self._timeout = timeout_s
        self._cache_dir = cache_dir
        self._http = httpx.Client(timeout=http_timeout)

    # ---- 对外 ----
    def parse(self, pdf_path: Path) -> tuple[str, dict]:
        digest = content_hash(pdf_path)
        cached = self._load_cache(digest)
        if cached is not None:
            markdown, meta = cached
            meta["from_cache"] = True
            return markdown, meta

        markdown, meta = self._parse_remote(pdf_path, digest)
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

        # 注意：这个 PUT 是预签名 URL，带 Authorization 反而会 403
        with pdf_path.open("rb") as fh:
            put = self._http.put(upload_url, content=fh.read())
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

    def _poll(self, batch_id: str, headers: dict) -> list[dict]:
        deadline = time.time() + self._timeout
        while True:
            resp = self._http.get(f"{self._base}/extract-results/batch/{batch_id}", headers=headers)
            items = resp.json()["data"]["extract_result"]
            if all(i["state"] in ("done", "failed") for i in items):
                return items
            if time.time() > deadline:
                raise LoaderError(f"MinerU 轮询超时（{self._timeout}s）：batch={batch_id}")
            time.sleep(self._poll)

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
        if not markdown.strip():
            raise LoaderError(f"MinerU 返回空正文：{p}")
        return LoadedDoc(
            text=markdown,
            source=str(p),
            doc_type="textbook",
            subject=guess_subject(p.name, _SUBJECTS),
            title=p.stem,
            meta=meta,
        )
