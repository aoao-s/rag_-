"""近重复文档检测（CLAUDE.md §7.2.6「未做」项的落地）。

**为什么需要**：`datasets/09_textbooks` 里同一本教材有两次下载
（`人教版-数学-九年级上册.pdf` 与 `人教版数学九年级上册.pdf`），md5 不同、内容一致。
`content_hash` 抓不到它，两份都入库会让同一段证据以"两篇文档"的身份互为对手——
证据门的跨文档 margin（§3.3.5 判据②）直接判 `ambiguous`，明明能找到的题被判成不可答。

**两道判据，先便宜后昂贵**：

1. **归一化路径**：**最后两级目录 + 文件名**归一后相同 → 视为同一份；
2. **文本 simhash**：取前 `prefix_chars` 个字符的字符二元组做 64 位 simhash，
   汉明距离 ≤ `distance` 视为同一份。用前 N 字符而不是全文：同一本教材两次下载的差异
   常在页眉、水印、下载戳上，正文开头足以覆盖。

**只留先到的那份**（后到被拒）：入库顺序由文件列表决定，可复现；反过来"较新的覆盖旧的"
会让重跑同一批文件时结果随 mtime 漂移，同一份数据跑两次进两份不同的库。

`datasets/` 只读（§2.5），所以这里**不改语料**，只做装载期拒绝，并把拒绝原因写进报告。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

# 文件名里的分隔符/装饰：`人教版-数学-九年级上册` 与 `人教版数学九年级上册` 归一后相同
_NAME_NOISE = re.compile(r"[\s\-_·、,，.。()（）\[\]【】<>《》]+")
# 正文里的空白与常见标点：两次下载的差异常常只是换行/空格重排
_TEXT_NOISE = re.compile(r"[\s\-—_·、,，.。!！?？:：;；\"'“”‘’()（）\[\]【】<>《》/\\|]+")

SIMHASH_BITS = 64


def normalize_name(source: str | Path) -> str:
    """文档归一标识：**最后两级目录 + 文件名**，去扩展名与分隔符、转小写。

    为什么不能只取文件名：新语料（§7.2.6）是 `01_textbook/初中/数学/教材.pdf` 这种
    "学科与学段在目录、文件名是通用词"的结构，只归一 stem 会让**所有**教材都得到
    `教材` 这一个 key——2026-10-29 实测 52 篇文档被判成 1 篇 + 49 份重复，整库只剩一本书。
    多拦一份比漏拦一份糟得多（§7.2.6「诚实结论」同理：宁可漏拦）。

    带上级目录之后，两个方向都对：同目录下的两次下载（`人教版-数学-九年级上册.pdf`
    / `人教版数学九年级上册.pdf`）仍然相等；不同学科的同名 `教材.pdf` 不再撞车。
    连目录名一起归一，是因为中文目录里混着 `01_textbook` 这种前缀与 `-`/`_`，
    不归一等于把"语料目录改名"变成"整库重录"。
    """
    p = Path(str(source))
    parts = [*p.parts[max(0, len(p.parts) - 3) : -1], p.stem] if len(p.parts) > 1 else [p.stem]
    return "/".join(_NAME_NOISE.sub("", part).lower() for part in parts)


def _tokens(text: str) -> list[str]:
    """去噪后的字符二元组，**每种只记一次**（按出现顺序去重）。

    刻意不保留频次：simhash 里的权重表达"这个 token 有多重要"，而"它重复了多少遍"
    是排版噪声，不是重要性。2026-10-29 实测按频次加权会**把不同文档压成同一份**——
    教育语料里 LaTeX 公式（`\\mathrm`、`\\cdot`、`{}{` 等）和题库模板两三千字里高频重复，
    十几个高频二元组的权重盖过几千个低频二元组，64 位里近 40 位被压成同号。
    结果是 `高中/化学/高一/教案.html`（7 134 字）与 `高中/化学/高三/试卷.html`（33 270 字）
    汉明距离只有 **2**，被误判成重复而拒收，且"先到者"是那份毫不相干的教案。
    改成每种二元组记 1 次后同一对是 **18**；真正重复的那些（题库三份同内容、`试卷` 与 `复习卷`）
    仍然是 0–2，拦得住。

    宁可漏拦不能误杀（§7.2.6）：漏掉一份重复只是多一份证据，误杀一份等于知识库里少一整个文档，
    而且库里留下的是错的那份。
    """
    cleaned = _TEXT_NOISE.sub("", text)
    return list(dict.fromkeys(cleaned[i : i + 2] for i in range(len(cleaned) - 1)))


def simhash(text: str, *, bits: int = SIMHASH_BITS) -> int:
    """字符二元组 simhash（每种二元组记 1 次，不加频次权重）。中文没有空格，
    word 级 simhash 不适用；字符二元组对"同一段文字换了标点/换行"稳定。"""
    vector = [0] * bits
    for token in _tokens(text):
        digest = int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(bits):
            vector[i] += 1 if (digest >> i) & 1 else -1
    value = 0
    for i in range(bits):
        if vector[i] > 0:
            value |= 1 << i
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


@dataclass(frozen=True, slots=True)
class Duplicate:
    """命中的近重复。`kept_source` 是先前登记的那份（本次将被拒绝）。"""

    kept_source: str
    reason: str  # name / simhash
    distance: int = 0

    def to_dict(self) -> dict[str, object]:
        return {"kept_source": self.kept_source, "reason": self.reason, "distance": self.distance}


class DedupIndex:
    """一次入库任务的查重表。`check_and_add` 是唯一入口：命中就不登记，
    避免"两份互为对手"变成"第二份顶替第一份"。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        distance: int = 3,
        prefix_chars: int = 8000,
        min_chars: int = 80,
    ) -> None:
        self.enabled = enabled
        self.distance = distance
        self.prefix_chars = prefix_chars
        self.min_chars = min_chars
        # 查重范围必须与知识库隔离范围一致：同一路径可以分别进入不同租户、不同版本。
        # 旧指纹文件没有 scope 字段时，按 default:v1 兼容，避免默认库重跑时突然漏拦。
        self._by_name: dict[tuple[str, str], str] = {}
        self._prints: list[tuple[str, str, int]] = []

    # ---- 查询与登记 ----
    def check(self, source: str, text: str, *, scope: str = "default:v1") -> Duplicate | None:
        """在指定隔离范围内查重。**自己不算重复**，不同租户/版本互不挡路。"""
        if not self.enabled:
            return None
        key = (scope, normalize_name(source))
        kept = self._by_name.get(key)
        if kept is not None and kept != source:
            return Duplicate(kept_source=kept, reason="name")

        body = (text or "")[: self.prefix_chars]
        if len(body) < self.min_chars:
            return None  # 太短，simhash 不稳定，宁可放过
        fingerprint = simhash(body)
        best: Duplicate | None = None
        for kept_scope, kept_source, kept_print in self._prints:
            if kept_scope != scope or kept_source == source:
                continue
            dist = hamming(fingerprint, kept_print)
            if dist <= self.distance and (best is None or dist < best.distance):
                best = Duplicate(kept_source=kept_source, reason="simhash", distance=dist)
        return best

    def add(self, source: str, text: str, *, scope: str = "default:v1") -> None:
        """登记。同一隔离范围内的同一 source 重跑时覆盖旧指纹。"""
        if not self.enabled:
            return
        key = (scope, normalize_name(source))
        self._by_name[key] = source
        self._prints = [
            (sco, src, fp) for sco, src, fp in self._prints
            if not (sco == scope and src == source)
        ]
        body = (text or "")[: self.prefix_chars]
        if len(body) >= self.min_chars:
            self._prints.append((scope, source, simhash(body)))

    def check_and_add(self, source: str, text: str, *, scope: str = "default:v1") -> Duplicate | None:
        hit = self.check(source, text, scope=scope)
        if hit is None:
            self.add(source, text, scope=scope)
        return hit

    def __len__(self) -> int:
        return len(self._by_name)

    # ---- 跨任务持久化（同一批文件重跑、增量入库都要能拦住）----
    def load(self, path: Path) -> int:
        if not path.exists():
            return 0
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return 0  # 指纹文件坏了当空表：宁可漏拦，也不能让入库整体失败
        for item in payload.get("docs", []):
            source = str(item.get("source", ""))
            if not source:
                continue
            # 旧格式没有 scope，归属历史默认知识库，不能把它扩散到其他租户。
            scope = str(item.get("scope") or "default:v1")
            self._by_name[(scope, normalize_name(source))] = source
            fingerprint = item.get("simhash")
            if isinstance(fingerprint, int):
                self._prints.append((scope, source, fingerprint))
        return len(self._by_name)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "docs": [
                {
                    "scope": scope,
                    "source": source,
                    "name": normalize_name(source),
                    "simhash": fingerprint,
                }
                for scope, source, fingerprint in self._prints
            ]
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
