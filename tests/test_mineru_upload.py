"""MinerU 上传契约：PUT 的**请求头**与上传超时是两条独立的坑，各钉一条。

2026-10-29 实测（全量教材入库）：
- 上传沿用通用 120 s 超时 → 6–66 MB 的教材在上行带宽下必然 `WriteTimeout`，
  `_MinerUWithLocalFallback` 静默退成本地 PyMuPDF（10 MB 的初中数学只剩 88 259 字，
  云端解析是 155 599 字）；
- 给 PUT 补 `Content-Type: application/pdf` → 预签名 URL 的签名里含 content-type，
  服务端直接断开连接（`RemoteProtocolError`），连重试都进不去。

所以这里断言两件事：**只带 Content-Length**、**write 超时来自独立配置**。
用 `httpx.MockTransport` 拦请求，不依赖网络与 MinerU 额度。
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from rag_system.document_loaders.loader_pdf_mineru import MinerUClient


def _zip_with_markdown() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("full.md", "# 二次函数\n\n顶点式 y=a(x-h)²+k。")
        zf.writestr("content_list.json", json.dumps([{"page_idx": 0}, {"page_idx": 4}]))
    return buf.getvalue()


@pytest.fixture()
def recorded():
    """挂上 MockTransport，返回 (client, 记录到的请求列表)。"""
    import httpx

    seen: list[httpx.Request] = []
    payload = _zip_with_markdown()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        url = str(request.url)
        if url.endswith("/file-urls/batch"):
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {"batch_id": "b1", "file_urls": ["https://oss.example.com/up/a.pdf?sig=x"]},
                },
            )
        if "/up/" in url:  # 预签名上传
            return httpx.Response(200)
        if "/extract-results/batch/" in url:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "extract_result": [
                            {"state": "done", "full_zip_url": "https://oss.example.com/down/r.zip"}
                        ]
                    }
                },
            )
        return httpx.Response(200, content=payload)  # 结果 zip

    client = MinerUClient("token", poll_interval=1, timeout_s=5, upload_timeout_s=600)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))  # noqa: SLF001
    return client, seen


def test_upload_put_has_no_content_type(tmp_path, recorded):
    """预签名 PUT 只能带 Content-Length：带上 Content-Type 会被服务端断连。"""
    client, seen = recorded
    pdf = tmp_path / "教材.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    client.parse(pdf)

    puts = [r for r in seen if r.method == "PUT"]
    assert len(puts) == 1
    put = puts[0]
    assert "content-type" not in {k.lower() for k in put.headers}
    assert put.headers["content-length"] == str(pdf.stat().st_size)
    # 同样不能带 Authorization（也会破坏 OSS 预签名）
    assert "authorization" not in {k.lower() for k in put.headers}


def test_upload_timeout_is_separate_from_poll_timeout():
    """上传超时独立于通用 HTTP 超时，否则 66 MB 教材必然 WriteTimeout。"""
    client = MinerUClient("token", timeout_s=900, upload_timeout_s=1800, http_timeout=120.0)
    assert client._upload_timeout.write == 1800.0  # noqa: SLF001
    assert client._upload_timeout.connect <= 30.0  # noqa: SLF001
    # 轮询/下载仍走短超时：连不上该立刻失败，不干等半小时
    assert client._http.timeout.read == 120.0  # noqa: SLF001


def test_parse_writes_cache(tmp_path, recorded):
    """解析结果落进 `data/mineru/<digest>/content.md`，第二次不再走网络。"""
    client, seen = recorded
    client._cache_dir = tmp_path / "mineru"  # noqa: SLF001
    pdf = tmp_path / "教材.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    markdown, meta = client.parse(pdf)
    assert "顶点式" in markdown
    assert meta["pages"] == 5

    before = len(seen)
    markdown2, meta2 = client.parse(pdf)
    assert markdown2 == markdown
    assert meta2["from_cache"] is True
    assert len(seen) == before, "第二次解析不该再发请求"


def test_oversized_pdf_is_split_and_merged(tmp_path):
    """超过 200 页的教材要拆卷送云，而不是静默退成本地 PyMuPDF。

    真实触发：`高中/数学/教材.pdf` 270 页、`高中/历史/教材.pdf` 222 页，
    MinerU 回 `number of pages exceeds limit (200 pages)`。
    """
    import fitz
    import httpx

    from rag_system.document_loaders.loader_pdf_mineru import _PART_SEP

    # 5 页 PDF，max_pages=2 → 应拆成 3 卷
    pdf = tmp_path / "教材.pdf"
    src = fitz.open()
    for _ in range(5):
        src.new_page()
    src.save(str(pdf))
    src.close()

    payload = _zip_with_markdown()
    batches: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/file-urls/batch"):
            name = json.loads(request.content)["files"][0]["name"]
            bid = f"b{len(batches) + 1}"
            batches[bid] = name
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": {"batch_id": bid, "file_urls": [f"https://oss.example.com/up/{bid}.pdf?sig=x"]},
                },
            )
        if "/up/" in url:
            return httpx.Response(200)
        if "/extract-results/batch/" in url:
            bid = url.rsplit("/", 1)[-1]
            if _PART_SEP not in batches[bid]:  # 整本上传：云端按页数拒收
                return httpx.Response(
                    200,
                    json={
                        "data": {
                            "extract_result": [
                                {
                                    "state": "failed",
                                    "err_msg": "number of pages exceeds limit (200 pages), "
                                    "please split the file and try again",
                                }
                            ]
                        }
                    },
                )
            return httpx.Response(
                200,
                json={
                    "data": {
                        "extract_result": [
                            {"state": "done", "full_zip_url": "https://oss.example.com/down/r.zip"}
                        ]
                    }
                },
            )
        return httpx.Response(200, content=payload)

    cache = tmp_path / "mineru"
    client = MinerUClient("token", poll_interval=1, timeout_s=5, upload_timeout_s=600, max_pages=2)
    client._http = httpx.Client(transport=httpx.MockTransport(handler))  # noqa: SLF001
    client._cache_dir = cache  # noqa: SLF001

    markdown, meta = client.parse(pdf)

    assert meta["split"] == 3, "5 页 / 每卷 2 页 → 3 卷"
    assert markdown.count("顶点式") == 3, "三卷按序拼回同一份 markdown"
    assert "split_reason" in meta
    # 分卷的抽取目录用完要清掉，缓存根目录只留最终合并的那一份
    leftover = [p.name for p in cache.iterdir() if _PART_SEP in p.name]
    assert leftover == [], f"分卷目录没清干净：{leftover}"
    assert (cache / meta["digest"] / "content.md").exists()

    # 再次解析走缓存，不再发请求
    before = len(batches)
    client.parse(pdf)
    assert len(batches) == before
