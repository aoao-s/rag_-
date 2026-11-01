"""HTML 加载器（教案 / 试卷 语料）。

为什么不直接正则去标签：教案的"教学目标 / 知识点梳理 / 典型例题"和试卷的题号、选项、
参考答案就是检索要用的结构骨架。一律 `<[^>]+>` 拍平之后，切分器只能按字数硬切，
一个父块会横跨两道题，引用卡片也就定位不到"第 3 题"。

所以走一遍标准库 `HTMLParser`，把文档流映射成 Markdown（不引第三方依赖）：

- `script` / `style` / `svg` / `noscript` 整段丢弃（MathJax、样式表都在这里，留着就是脏 chunk）；
- `h1..h6` → `#…`，`li` → `- `，`p/div/tr/br` → 换行，`td/th` → `| ` 分隔；
- 其余标签只丢标签、留文字，因此行内 LaTeX（`$x^2$`）与强调都能原样保留；
- 源码里的缩进换行统一折叠成单个空格，避免"每行都是空白"的假结构。

空正文**抛错**而不是返回空串：一个只有导航栏的 HTML 进库，等于往索引里塞噪音。
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from rag_system.document_loaders.base import BaseDocumentLoader, LoadedDoc, LoaderError, guess_subject, read_text

# 内容本身没有语义的标签：整段吃掉
# 只放**有闭合标签**的容器：`meta`/`link` 是空元素，放进来自增不自减，
# 会把后面的正文整段吞掉（第一次跑就是这么空的）。
_SKIP_TAGS = frozenset({"script", "style", "noscript", "svg", "head"})
# 块级标签：前后各留一个换行，别让两段黏成一句
_BLOCK_TAGS = frozenset(
    {
        "p", "div", "section", "article", "header", "footer", "main", "aside",
        "ul", "ol", "dl", "dt", "dd", "li", "table", "blockquote", "pre",
        "figure", "figcaption", "hr", "form", "nav",
    }
)
_HEADING_TAGS = {f"h{n}": n for n in range(1, 7)}
_VOID_TAGS = frozenset({"br", "img", "hr", "input", "meta", "link"})

_SPACES = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


class _MarkdownExtractor(HTMLParser):
    """HTML → 线性 Markdown 文本。只做"结构分行"，不做样式还原。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._title = ""
        self._in_title = False

    # ---- 输出 ----
    def _emit(self, text: str) -> None:
        if text:
            self._parts.append(text)

    def _newline(self) -> None:
        self._parts.append("\n")

    @property
    def text(self) -> str:
        body = "".join(self._parts)
        body = "\n".join(_SPACES.sub(" ", line).strip() for line in body.split("\n"))
        body = _BLANK_LINES.sub("\n\n", body)
        return body.strip()

    @property
    def title(self) -> str:
        return _SPACES.sub(" ", self._title).strip()

    # ---- HTMLParser 回调 ----
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if self._skip_depth:
            return
        if tag in _HEADING_TAGS:
            self._newline()
            self._emit("#" * _HEADING_TAGS[tag] + " ")
        elif tag == "li":
            self._newline()
            self._emit("- ")
        elif tag in ("td", "th"):
            self._emit("| ")
        elif tag == "br":
            self._newline()
        elif tag in _BLOCK_TAGS:
            self._newline()
        elif tag == "pre":
            self._pre_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _VOID_TAGS and tag != "hr" and not self._skip_depth:
            self._newline()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if self._skip_depth:
            return
        if tag in ("td", "th"):
            self._emit(" ")
        elif tag == "tr":
            # 补上收尾的竖线，一行表格才是完整的一行
            self._emit("|\n")
        elif tag in _HEADING_TAGS or tag in _BLOCK_TAGS:
            if tag == "pre":
                self._pre_depth = max(0, self._pre_depth - 1)
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title += data
            return
        if self._skip_depth:
            return
        # <pre> 里的换行是有意义的（代码/排版题），其余按空白折叠
        if self._pre_depth:
            self._emit(data)
        else:
            self._emit(_SPACES.sub(" ", data))


class HtmlLoader(BaseDocumentLoader):
    """`.html` / `.htm` → Markdown 风格正文 + 学科/标题推断。"""

    extensions = (".html", ".htm")

    def load(self, path: str | Path) -> LoadedDoc:
        p = Path(path)
        raw = read_text(p)
        extractor = _MarkdownExtractor()
        try:
            extractor.feed(raw)
            extractor.close()
        except Exception as exc:  # noqa: BLE001 - 解析器错误也要带上文件名
            raise LoaderError(f"HTML 解析失败：{p.name}：{type(exc).__name__}: {exc}") from exc

        body = extractor.text
        if not body:
            raise LoaderError(f"HTML 没有任何可提取正文：{p.name}")

        return LoadedDoc(
            text=body,
            source=str(p.resolve()),
            doc_type="other",  # 交给工厂按目录/文件名判定（教案 / 试卷 / 教材）
            subject=guess_subject(str(p)),
            title=extractor.title or p.stem,
            meta={"chars": len(body), "parser": "htmlparser"},
        )
