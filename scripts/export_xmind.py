"""把 docs/architecture_outline.txt 转成 XMind 文件（另附 Markdown 大纲）。

为什么要多这一层：XMind 的树里全是中文和引号，直接把内容写进 Python 字符串很容易
被引号/转义搞坏（本项目已经踩过一次）。所以内容放在纯文本大纲里，脚本只负责解析和打包。

大纲格式（缩进 2 空格一级）：

    EduRAG-Pro 1.41 智能教育知识库
      ① 定位与边界
        一句话 <<可选的备注，会折叠成节点备注>>

输出（.xmind 是 zip）：
- 新版 XMind（2020+/Zen）：content.json + metadata.json + manifest.json
- 旧版 XMind 8：content.xml + meta.xml + styles.xml + META-INF/manifest.xml
两种都写，谁打开都能读。

用法：python scripts/export_xmind.py [大纲路径] [输出路径]
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "docs" / "architecture_outline.txt"
DEFAULT_OUT = ROOT / "docs" / "EduRAG-Pro-架构全景.xmind"
NOTE_OPEN = "<<"
NOTE_CLOSE = ">>"

_counter = 0


def new_id(prefix: str = "t") -> str:
    global _counter
    _counter += 1
    return f"{prefix}{_counter:04d}"


def parse_outline(path: Path) -> dict:
    root: dict | None = None
    stack: list[dict] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        level = indent // 2
        text = raw.strip()

        note = ""
        if NOTE_OPEN in text:
            head, _, rest = text.partition(NOTE_OPEN)
            note = rest.rsplit(NOTE_CLOSE, 1)[0].strip()
            text = head.strip()

        node = {"id": new_id(), "title": text, "note": note, "kids": []}
        if level == 0 or root is None:
            root = node
            stack = [node]
            continue
        # 缩进跳级（比如从 0 直接到 2）时，挂到最近的已有祖先
        level = min(level, len(stack))
        stack = stack[:level]
        stack[-1]["kids"].append(node)
        stack.append(node)

    if root is None:
        raise SystemExit(f"大纲是空的：{path}")
    return root


def to_json_topic(node: dict) -> dict:
    topic: dict = {"id": node["id"], "class": "topic", "title": node["title"]}
    if node["note"]:
        topic["notes"] = {"plain": {"content": node["note"]}}
    if node["kids"]:
        topic["children"] = {"attached": [to_json_topic(k) for k in node["kids"]]}
    return topic


def to_xml_topic(node: dict) -> str:
    parts = [f'<topic id="{node["id"]}">', f"<title>{escape(node['title'])}</title>"]
    if node["note"]:
        parts.append(f"<notes><plain>{escape(node['note'])}</plain></notes>")
    if node["kids"]:
        parts.append('<children><topics type="attached">')
        parts.extend(to_xml_topic(k) for k in node["kids"])
        parts.append("</topics></children>")
    parts.append("</topic>")
    return "".join(parts)


def to_markdown(node: dict, depth: int = 0) -> list[str]:
    lines = [f"{'  ' * depth}- {node['title']}"]
    if node["note"]:
        lines.append(f"{'  ' * depth}  > {node['note']}")
    for kid in node["kids"]:
        lines.extend(to_markdown(kid, depth + 1))
    return lines


def count_topics(node: dict) -> int:
    return 1 + sum(count_topics(k) for k in node["kids"])


def main() -> int:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    root = parse_outline(src)
    sheet_id = new_id("s")
    title = "EduRAG-Pro 1.41 架构全景"

    content_json = [
        {"id": sheet_id, "class": "sheet", "title": title, "rootTopic": to_json_topic(root)}
    ]
    metadata = {
        "creator": {"name": "EduRAG-Pro xmind exporter", "version": "1.41"},
        "activeSheetId": sheet_id,
        "layoutEngineVersion": "3",
    }
    manifest_json = {
        "file-entries": {
            "content.json": {},
            "metadata.json": {},
            "content.xml": {},
            "meta.xml": {},
            "styles.xml": {},
            "META-INF/manifest.xml": {},
        }
    }

    root_xml = to_xml_topic(root).replace(
        f'<topic id="{root["id"]}">',
        f'<topic id="{root["id"]}" structure-class="org.xmind.ui.map.unbalanced">',
        1,
    )
    content_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
        '<xmap-content xmlns="urn:xmind:xmap:xmlns:content:2.0" '
        'xmlns:fo="http://www.w3.org/1999/XSL/Format" '
        'xmlns:svg="http://www.w3.org/2000/svg" '
        'xmlns:xhtml="http://www.w3.org/1999/xhtml" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'timestamp="{int(time.time() * 1000)}" version="2.0">'
        f'<sheet id="{sheet_id}">{root_xml}<title>{escape(title)}</title></sheet>'
        "</xmap-content>"
    )
    meta_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
        '<meta xmlns="urn:xmind:xmap:xmlns:meta:2.0" version="2.0">'
        "<Author><Name>EduRAG-Pro</Name></Author></meta>"
    )
    styles_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
        '<xmap-styles xmlns="urn:xmind:xmap:xmlns:style:2.0" version="2.0">'
        "<styles/><themes/></xmap-styles>"
    )
    manifest_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>'
        '<manifest xmlns="urn:xmind:xmap:xmlns:manifest:1.0">'
        '<file-entry full-path="content.xml" media-type="text/xml"/>'
        '<file-entry full-path="meta.xml" media-type="text/xml"/>'
        '<file-entry full-path="styles.xml" media-type="text/xml"/>'
        '<file-entry full-path="META-INF/" media-type=""/>'
        '<file-entry full-path="META-INF/manifest.xml" media-type="text/xml"/>'
        "</manifest>"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("content.json", json.dumps(content_json, ensure_ascii=False))
        zf.writestr("metadata.json", json.dumps(metadata, ensure_ascii=False))
        zf.writestr("manifest.json", json.dumps(manifest_json, ensure_ascii=False))
        zf.writestr("content.xml", content_xml)
        zf.writestr("meta.xml", meta_xml)
        zf.writestr("styles.xml", styles_xml)
        zf.writestr("META-INF/manifest.xml", manifest_xml)

    md = out.with_suffix(".md")
    md.write_text("\n".join(to_markdown(root)) + "\n", encoding="utf-8")

    print(f"saved {out}")
    print(f"  topics={count_topics(root)} size={out.stat().st_size} B")
    print(f"saved {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
