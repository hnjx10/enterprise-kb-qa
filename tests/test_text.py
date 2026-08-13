"""分词与词表：领域词典、同义词、单字回退。

这一层出问题的方式很隐蔽：不报错、不抛异常，只是某个查询的关键词路返回 0 条，
双路融合静默退化成单路。所以每条规则都单独钉一个用例。
"""

from __future__ import annotations

from kbqa.text import (
    STOPWORDS,
    char_ngrams,
    dedupe,
    load_lexicon,
    load_synonyms,
    normalize,
    tokenize,
)


def test_lexicon_keeps_domain_word() -> None:
    """jieba 默认把「年假」切成「年」+「假」，两字又会被单字规则丢掉。

    没有领域词典时 ``tokenize("年假有几天") == ("几天",)``——查询的核心词整个消失。
    """
    tokens = tokenize("年假有几天")
    assert "年假" in tokens


def test_synonym_expands_to_canonical() -> None:
    """正文写「年休假」、问句写「年假」：两边都要产出标准词，BM25 才匹配得上。"""
    tokens = tokenize("年休假怎么计算")
    assert "年假" in tokens
    # 原词保留：用户真实用词本身带 IDF 信息，不该被同义词替换掉
    assert "年休假" in tokens


def test_synonym_does_not_duplicate() -> None:
    """标准词已经在结果里时，不再重复追加（重复会虚增 BM25 的词频）。"""
    tokens = tokenize("年假")
    assert tokens.count("年假") == 1


def test_single_char_fallback_keeps_keyword_route_alive() -> None:
    """全是虚词/单字的查询不能返回空 tuple。

    返回空会让 BM25 一路返回 0 条候选：不报错，只是悄悄从双路变单路。
    """
    assert tokenize("怎么办") != ()
    assert all(t not in STOPWORDS for t in tokenize("怎么办"))


def test_digits_survive_single_char_filter() -> None:
    """单字过滤放过数字：「3 天」「600 元」里的数字是有意义的检索条件。"""
    assert "600" in tokenize("住宿费 600 元")


def test_punctuation_dropped() -> None:
    assert tokenize("住宿费，标准。") == tokenize("住宿费 标准")


def test_normalize_handles_fullwidth_and_case() -> None:
    """全角与大小写不归一，「ＰＤＦ」和「pdf」就是两个词。"""
    assert normalize("  ＰＤＦ   文档 ") == "pdf 文档"


def test_char_ngrams() -> None:
    assert char_ngrams("报销", n=2) == ("报销",)
    assert "报销" in char_ngrams("差旅报销", n=2)


def test_dedupe_preserves_order() -> None:
    assert dedupe(("a", "b", "a", "c", "b")) == ("a", "b", "c")


def test_lexicon_and_synonyms_are_loaded_from_data_dir() -> None:
    """两个词表都要真的从 data/ 读到了，否则上面的用例只是因为碰巧才过。"""
    assert load_lexicon().endswith("lexicon.txt")
    assert len(load_synonyms()) > 0
