"""分块：三级结构、长度约束、重叠、确定性。

这组用例守的都是「改一个参数就悄悄失效」的性质——分块错了不会报错，
只会让检索慢慢变差，所以只能靠断言。
"""

from __future__ import annotations

from kbqa.chunking import Chunk, content_digest, split_document

DOC = """# 差旅管理办法

## 住宿费标准

一线城市每晚上限 600 元。其他城市每晚上限 350 元。

## 报销时限

出差结束后 15 个自然日内提交报销单。超过 30 天需要双签。超过 90 天不予受理。
"""


def test_heading_hierarchy_becomes_section_path() -> None:
    chunks = split_document(DOC, doc_id="travel")
    assert chunks, "非空文档必须切出块"
    assert any(c.section_path == ("差旅管理办法", "住宿费标准") for c in chunks)
    labels = {c.section_label for c in chunks}
    assert "差旅管理办法 > 报销时限" in labels


def test_chunk_ids_are_deterministic_and_ordered() -> None:
    first = split_document(DOC, doc_id="travel")
    second = split_document(DOC, doc_id="travel")
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.chunk_id for c in first] == [f"travel::{i:04d}" for i in range(len(first))]


def test_digest_changes_when_content_changes() -> None:
    a = split_document(DOC, doc_id="travel")
    b = split_document(DOC.replace("600 元", "800 元"), doc_id="travel")
    assert content_digest(a) != content_digest(b), "内容变了指纹必须变，否则缓存不会失效"


def test_long_paragraph_is_split_by_sentence() -> None:
    """单段超过 max_chars 时要退到句子级，否则会切出一个超长块。"""
    from kbqa.config import ChunkSettings

    cfg = ChunkSettings(target_chars=120, max_chars=150, min_chars=0, overlap_sentences=0)
    long_doc = "## 条款\n\n" + "这是一句测试用的说明文字，用于验证长段落切分。" * 12
    chunks = split_document(long_doc, doc_id="long", settings=cfg)
    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= cfg.max_chars + 60  # 允许句号与换行带来的少量超出


def test_overlap_is_shared_between_neighbours() -> None:
    from kbqa.config import ChunkSettings

    cfg = ChunkSettings(target_chars=80, max_chars=200, min_chars=0, overlap_sentences=1)
    text = "## 条款\n\n" + "。".join(f"第{i}条说明内容" for i in range(12)) + "。"
    chunks = split_document(text, doc_id="ov", settings=cfg)
    assert len(chunks) >= 2
    # 相邻块共享一句：前一块的末句应出现在后一块开头
    tail = chunks[0].text.split("\n")[-1]
    assert tail in chunks[1].text


def test_short_tail_is_merged_back() -> None:
    """尾巴太短就并回上一块，避免出现只有几个字的碎片块。"""
    from kbqa.config import ChunkSettings

    cfg = ChunkSettings(target_chars=200, max_chars=400, min_chars=80, overlap_sentences=0)
    text = "## 条款\n\n" + "甲" * 200 + "\n\n乙。"
    chunks = split_document(text, doc_id="tail", settings=cfg)
    assert len(chunks) == 1
    assert chunks[0].text.endswith("乙。")


def test_empty_document_returns_no_chunks() -> None:
    assert split_document("   \n\n ", doc_id="empty") == []


def test_document_without_headings_still_works() -> None:
    chunks = split_document("没有任何标题的一段正文。第二句也在这里。", doc_id="plain")
    assert len(chunks) == 1
    assert chunks[0].section_path == ()
    assert chunks[0].section_label == "（正文）"


def test_dept_is_carried_on_every_chunk() -> None:
    chunks = split_document(DOC, doc_id="travel", dept="finance")
    assert all(c.dept == "finance" for c in chunks)


def test_citation_contains_title_and_section() -> None:
    chunk = Chunk(
        chunk_id="d::0001",
        doc_id="d",
        text="x",
        section_path=("总则", "适用范围"),
        ordinal=1,
        char_start=0,
        char_end=1,
    )
    assert chunk.citation("差旅办法") == "差旅办法 · 总则 > 适用范围"
