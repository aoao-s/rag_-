"""查询归一化与分词。

为什么需要它：FAQ 命中率的差距几乎全在这一层。旧项目直接拿原始 query 做 BM25，
"学费多少钱？" 和 "学费多少" 被当成两个串，命中率因此损失一大截。

处理顺序：全半角 → 大小写 → 去标点空白 → 中文数字 → 语气词/礼貌语 → 别名归一。
"""

from __future__ import annotations

import logging
import re
import string
import unicodedata
from collections.abc import Iterable, Mapping

# 只去句首礼貌语与句尾语气词，绝不做全文删字（"一下午"不能被削成"午"）
_LEADING_FILLERS = (
    "麻烦问一下",
    "想问一下",
    "请问一下",
    "我想问",
    "我想知道",
    "请问",
    "问一下",
    "想问",
    "你好",
    "您好",
)
_TRAILING_PARTICLES = ("谢谢啦", "谢谢", "呢", "吗", "啊", "呀", "吧", "呗", "嘛", "了", "哈")

_PUNCT = set(string.punctuation) | set("，。！？、；：“”‘’（）【】《》…—～·〈〉「」『』〔〕")

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
_UNITS = ("个", "年", "月", "周", "天", "小时", "分钟", "次", "期", "门", "种", "节", "人")
_CN_NUM_RE = re.compile(rf"([{''.join(_CN_DIGITS)}])(?=(?:{'|'.join(_UNITS)}))")

# 极简停用词：只去真正的语法词，学科名词一个都不能去
_STOPWORDS = frozenset({"的", "是", "我", "你", "他", "她", "它", "吧", "呢", "吗", "啊", "和", "与", "及"})

# 口语同义词：单字替换安全（不碰学科名词），必须在 jieba 前做，
# 否则"咋证明"会被切成生僻块，BM25 直接丢词。
_COLLOQUIAL = (
    ("咋样", "怎么样"),
    ("咋回事", "怎么回事"),
    ("咋", "怎么"),
    ("啥", "什么"),
    ("甭", "不用"),
)

try:  # pragma: no cover - 取决于环境是否装 jieba
    import jieba

    # jieba 默认把自己挂在 DEBUG 并打印"Building prefix dict…"，会把启动日志刷满
    jieba.setLogLevel(logging.WARNING)
    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    _HAS_JIEBA = False


def to_halfwidth(text: str) -> str:
    """全角转半角（NFKC 会顺手合并一些异体，但对中文查询无害）。"""
    return unicodedata.normalize("NFKC", text)


def strip_punct(text: str) -> str:
    return "".join(ch for ch in text if ch not in _PUNCT and not ch.isspace())


def normalize_numbers(text: str) -> str:
    """中文数字 + 量词 归一为阿拉伯数字："三个月" → "3个月"。"""

    def _sub(match: re.Match[str]) -> str:
        return str(_CN_DIGITS[match.group(1)])

    return _CN_NUM_RE.sub(_sub, text)


def strip_fillers(text: str) -> str:
    changed = True
    while changed and text:
        changed = False
        for prefix in _LEADING_FILLERS:
            if text.startswith(prefix):
                text = text[len(prefix):]
                changed = True
        for suffix in _TRAILING_PARTICLES:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text


def _clean(text: str) -> str:
    """不含别名的净化（模块级纯函数：Normalizer.__init__ 构造别名词典时不能依赖实例）。"""
    out = strip_punct(to_halfwidth(text)).lower()
    out = normalize_numbers(out)
    for colloquial, formal in _COLLOQUIAL:
        out = out.replace(colloquial, formal)
    return strip_fillers(out)


class Normalizer:
    """线程内复用的无状态归一器；别名词典由 FAQService 在重建时注入。"""

    def __init__(self, aliases: Mapping[str, str] | None = None, extra_words: Iterable[str] = ()) -> None:
        self._aliases = {
            _clean(k): _clean(v)
            for k, v in (aliases or {}).items()
            if k and v
        }
        # 长别名优先（保留排序，便于将来做前缀匹配时行为稳定）
        self._alias_items = sorted(self._aliases.items(), key=lambda kv: len(kv[0]), reverse=True)
        self._extra_words = tuple(extra_words)
        self._dict_loaded = False

    # ---- 对外 ----
    def preload(self) -> None:
        """启动期调用：加载 jieba 词典并预热，避免首请求多花 1–2 s。"""
        if not _HAS_JIEBA:
            return
        for word in self._extra_words:
            if word:
                jieba.add_word(word, freq=20000)
        jieba.initialize()
        jieba.lcut("预热分词")
        self._dict_loaded = True

    @property
    def dict_loaded(self) -> bool:
        return self._dict_loaded

    def normalize(self, query: str, *, apply_alias: bool = True) -> str:
        """归一化后的规范串：用于精确哈希命中与 BM25 语料。

        `apply_alias=False` 用于判定"这次命中是否靠别名"，统计需要区分。
        """
        return self._normalize_text(query, apply_alias=apply_alias)

    def tokens(self, text: str) -> list[str]:
        """分词结果（已去停用词与空白）。"""
        norm = self._normalize_text(text)
        if not norm:
            return []
        if _HAS_JIEBA:
            raw = jieba.lcut(norm, cut_all=False)
        else:  # 无 jieba 时的退化：英文/数字按词，中文按二元组
            raw = []
            for piece in re.findall(r"[a-zA-Z]+|\d+", norm):
                raw.append(piece.lower())
            cjk = re.sub(r"[a-zA-Z\d]+", "", norm)
            raw.extend(cjk[i : i + 2] for i in range(max(0, len(cjk) - 1)))
            if len(cjk) == 1:
                raw.append(cjk)
        return [tok for tok in raw if tok.strip() and tok not in _STOPWORDS]

    # ---- 内部 ----
    def _normalize_text(self, text: str, *, apply_alias: bool = True) -> str:
        if not text:
            return ""
        out = _clean(text)
        if apply_alias:
            # 别名只做"整串等价"映射：子串替换会让"学费多少钱"变成"学费学费多少"
            # （canonical 自身含别名片段时必然重复）；近似问法交给 BM25，不在这里硬拼。
            out = self._aliases.get(out, out)
        return out

