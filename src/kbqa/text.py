"""文本规范化与分词：向量路与关键词路共用同一套切词规则。

**为什么要共用**：如果向量侧用字符 bigram、BM25 侧用 jieba 词，两路召回的「相似」
定义就不是一回事，RRF 融合时两边的分数根本没有可比性——融合会退化成「谁排第一听谁的」。
共用一套 token 至少保证两边讨论的是同一批语义单元。

中文检索的两个坑：

* **全角/半角与大小写**：``ＰＤＦ`` 与 ``pdf`` 是同一个词。不做 NFKC + casefold，
  用户搜 "pdf" 就搜不到库里写的 "PDF"。
* **纯标点与停用词**：它们对 BM25 有 IDF 贡献（罕见标点 IDF 很高），但对语义没有。
  过滤掉能显著减少噪声命中。

## 领域词典不是可选项

jieba 的通用词典没有收录制度类文本的常用词，切分会直接切错：``年假`` → ``年``+``假``，
``年假能不能留到明年`` → ``年``、``假能``、``不能``…… 再加上下面「单字一律过滤」的
规则，查询里最关键的 ``年假`` 会被整个丢掉。

这不是分词精度问题，而是**召回归零**：``tokenize("年假有几天")`` 只剩 ``('几天',)``，
BM25 对所有文档都算出 0 分，关键词路返回空，双路融合退化成单路向量。实测里
``kbqa search "年假有几天"`` 的诊断是「关键词 0 条」——一个看起来正常的空结果，
背后是一条路完全失效。

所以 ``data/lexicon.txt`` 是必读的：从文档标题与正文里人工挑出的领域词，
启动时 ``load_userdict`` 载入。词表来自语料本身，不是从通用词表里抄的——语料换了，
词表要跟着换，这跟索引要重建是同一件事。
"""

from __future__ import annotations

import os
import re
import unicodedata
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import jieba

jieba.setLogLevel(60)  # 关掉 jieba 的启动日志，它会污染 CLI 的 JSON 输出

LEXICON_ENV = "KBQA_LEXICON"
DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "data" / "lexicon.txt"
SYNONYM_ENV = "KBQA_SYNONYMS"
DEFAULT_SYNONYMS = Path(__file__).resolve().parents[2] / "data" / "synonyms.txt"

_ASCII_WORD = re.compile(r"[a-zA-Z][a-zA-Z0-9_.+-]*|\d+(?:\.\d+)?%?")
_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)

#: 高频虚词：对语义几乎无贡献，却会在 BM25 里制造大量假匹配
STOPWORDS = frozenset(
    [
        "的",
        "了",
        "和",
        "与",
        "及",
        "或",
        "是",
        "在",
        "有",
        "为",
        "对",
        "从",
        "到",
        "由",
        "按",
        "就",
        "也",
        "都",
        "而",
        "其",
        "之",
        "这",
        "那",
        "个",
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "请问",
        "一下",
        "什么",
        "怎么",
        "如何",
        "哪些",
        "多少",
        "是否",
        "可以",
        "需要",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "哦",
        "嗯",
        "请",
        "帮我",
        "告诉",
    ]
)


def normalize(text: str) -> str:
    """NFKC + 空白压缩 + 大小写折叠。用于缓存 key 与去重判断。"""
    text = unicodedata.normalize("NFKC", text or "")
    return " ".join(text.split()).casefold()


@lru_cache(maxsize=1)
def load_lexicon() -> str:
    """载入领域词表，返回实际使用的路径（空串表示没找到）。

    由 ``tokenize`` 首次调用时惰性触发：词典必须在第一次分词**之前**进 jieba，
    否则进程内已经算过的 token 会永远停在旧切分上——``lru_cache`` 不会帮你重算。
    """
    path = Path(os.environ.get(LEXICON_ENV) or DEFAULT_LEXICON)
    if not path.is_file():
        return ""
    jieba.load_userdict(str(path))
    return str(path)


@lru_cache(maxsize=1)
def load_synonyms() -> Mapping[str, str]:
    """同义词表：``标准词 别名 别名``，返回「别名 → 标准词」。

    领域词典解决了「切错」，同义词解决「切对了但两边用不同的词」：
    问句说「年假」，正文写「年休假」，两边都是好词，BM25 却一个都不匹配。
    这类词族在制度文本里很密集（年假/年休假、餐补/出差补贴），
    与其在查询侧猜，不如显式列出来。

    返回只读映射：它来自 ``lru_cache``，是可被多处共享的同一份对象。
    """
    path = Path(os.environ.get(SYNONYM_ENV) or DEFAULT_SYNONYMS)
    if not path.is_file():
        return MappingProxyType({})
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        canonical, aliases = parts[0], parts[1:]
        for alias in aliases:
            mapping[alias] = canonical
    return MappingProxyType(mapping)


@lru_cache(maxsize=8192)
def tokenize(text: str) -> tuple[str, ...]:
    """jieba 分词 + 英文/数字词典保留，去停用词与纯标点。

    返回 tuple 而不是 list：结果会被放进 ``lru_cache``，可变对象一旦被调用方改掉，
    下一次命中就是错的。
    """
    load_lexicon()
    text = unicodedata.normalize("NFKC", text or "").casefold()
    tokens: list[str] = []

    # 先把英文/数字整词捞出来，避免 jieba 把 "iso9001" 切成 "iso" + "9001"
    for match in _ASCII_WORD.finditer(text):
        tokens.append(match.group(0))

    raw: list[str] = []
    for word in jieba.cut(text, cut_all=False):
        word = word.strip()
        if not word or _PUNCT.match(word):
            continue
        raw.append(word)
        if word in STOPWORDS or len(word) == 1 and not word.isdigit():
            # 单字只在是数字时保留（"3年" 里的 3 有意义），其余单字噪声太大
            continue
        tokens.append(word)

    if not tokens:
        # 全是单字或虚词（"怎么办""能不能"）：退到未过滤的单字。
        # 宁可噪声大一点，也不要让关键词路直接返回 0 条——空结果会把双路打成单路，
        # 而且是静默的，从诊断里只能看到「关键词 0」。
        tokens = [w for w in raw if w not in STOPWORDS]

    # 同义词扩展：只加不改，原始 token 全部保留（它们是用户真实用词，IDF 有价值）
    synonyms = load_synonyms()
    if synonyms:
        extra = [synonyms[t] for t in tokens if t in synonyms and synonyms[t] not in tokens]
        tokens.extend(dict.fromkeys(extra))

    return tuple(tokens)


def char_ngrams(text: str, n: int = 2) -> tuple[str, ...]:
    """字符 n-gram：中文没有空格，bigram 是不依赖分词的兜底表示。"""
    clean = "".join(ch for ch in normalize(text) if not _PUNCT.match(ch))
    if len(clean) <= n:
        return (clean,) if clean else ()
    return tuple(clean[i : i + n] for i in range(len(clean) - n + 1))


def dedupe(tokens: tuple[str, ...]) -> tuple[str, ...]:
    """保序去重。BM25 的 TF 用得到重复，所以这个只在需要「词表」时用。"""
    return tuple(dict.fromkeys(tokens))
