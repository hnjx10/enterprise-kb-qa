"""检索：RRF 数学性质、权限过滤位置、单路口径。

重点不是「能搜到」，而是几条容易被改坏的性质：

* RRF 只吃名次，两路换分数尺度不影响结果；
* 权限过滤发生在截断之前——否则低权限用户的召回数量会随权限缩水；
* 关掉一路后，结果必须真的只来自另一路（评测的基线就靠这个）。
"""

from __future__ import annotations

import numpy as np
import pytest

from kbqa.chunking import Chunk
from kbqa.config import RetrievalSettings
from kbqa.embeddings import HashingEmbedder
from kbqa.index import Hit, HybridIndex, rrf_fuse


def chunk(cid: str, text: str, dept: str = "") -> Chunk:
    doc_id, ordinal = cid.split("::")
    return Chunk(
        chunk_id=cid,
        doc_id=doc_id,
        text=text,
        section_path=("章节",),
        ordinal=int(ordinal),
        char_start=0,
        char_end=len(text),
        dept=dept,
    )


CORPUS = [
    chunk("travel::0000", "一线城市住宿费每晚上限六百元，超出部分个人承担。", "finance"),
    chunk("travel::0001", "出差结束后十五个自然日内提交报销单。", "finance"),
    chunk("hr::0000", "入职满一年享有五天年假，满十年享有十天年假。", "hr"),
    chunk("hr::0001", "请假三天以上需要在 OA 填写工作交接单。", "hr"),
    chunk("public::0000", "标准工作时间为上午九点至下午六点。", ""),
]


@pytest.fixture()
def index() -> HybridIndex:
    idx = HybridIndex(HashingEmbedder(dim=256))
    idx.build(CORPUS)
    return idx


# ------------------------------------------------------------------ RRF 本体


def test_rrf_uses_rank_not_raw_score() -> None:
    """把向量分数整体乘 1000，融合结果不能变——RRF 只看名次。"""
    a1 = chunk("a::0000", "甲")
    a2 = chunk("b::0000", "乙")
    vec = [Hit(chunk=a1, score=0.9), Hit(chunk=a2, score=0.8)]
    kw = [Hit(chunk=a2, score=5.0)]
    fused_a = rrf_fuse(vec, kw, k=60)
    vec_scaled = [Hit(chunk=a1, score=900), Hit(chunk=a2, score=800)]
    fused_b = rrf_fuse(vec_scaled, [Hit(chunk=a2, score=5000)], k=60)
    assert [h.chunk.chunk_id for h in fused_a] == [h.chunk.chunk_id for h in fused_b]


def test_rrf_rewards_agreement() -> None:
    """两路都命中的块应排在只有一路命中的块前面。"""
    both = chunk("both::0000", "甲乙")
    only = chunk("only::0000", "甲")
    fused = rrf_fuse(
        [Hit(chunk=both, score=1.0), Hit(chunk=only, score=0.9)], [Hit(chunk=both, score=3.0)]
    )
    assert fused[0].chunk.chunk_id == "both::0000"
    assert fused[0].sources == ("vector", "keyword")


def test_rrf_score_matches_formula() -> None:
    c = chunk("x::0000", "甲")
    fused = rrf_fuse([Hit(chunk=c, score=1.0)], [Hit(chunk=c, score=1.0)], k=60)
    assert fused[0].score == pytest.approx(2 / 61)


# ------------------------------------------------------------------ 索引行为


def test_search_returns_hits_with_source_labels(index: HybridIndex) -> None:
    hits, diag = index.search("住宿费上限", allowed_depts={"finance"})
    assert hits
    assert hits[0].chunk.doc_id == "travel"
    assert diag.returned == len(hits)


def test_permission_filter_hides_other_departments(index: HybridIndex) -> None:
    """人事问报销，看不到财务文档——哪怕词面完全匹配。"""
    hits, _ = index.search("住宿费上限", allowed_depts={"hr"})
    assert all(h.chunk.dept != "finance" for h in hits)


def test_public_document_is_visible_to_everyone(index: HybridIndex) -> None:
    hits, _ = index.search("标准工作时间", allowed_depts={"finance"})
    assert any(h.chunk.dept == "" for h in hits)


def test_filtering_happens_before_truncation(index: HybridIndex) -> None:
    """低权限用户不该因为过滤而少拿结果。"""
    settings = RetrievalSettings(vector_top_k=5, keyword_top_k=5, final_top_k=3, oversample=4)
    hits, diag = index.search("年假 报销 工作时间", allowed_depts={""}, settings=settings)
    assert diag.filtered_by_permission >= 1, "这几条查询必然命中无权限文档"
    assert len(hits) >= 1


def test_disabling_one_path_really_removes_it(index: HybridIndex) -> None:
    settings = RetrievalSettings(vector_top_k=0, keyword_top_k=5, final_top_k=5)
    hits, _ = index.search("住宿费", allowed_depts={"finance"}, settings=settings)
    assert hits
    assert all("vector" not in h.sources for h in hits)


def test_empty_index_raises(index: HybridIndex) -> None:
    from kbqa.errors import IndexNotReadyError

    empty = HybridIndex(HashingEmbedder(dim=64))
    with pytest.raises(IndexNotReadyError):
        empty.search("任意问题")


def test_results_are_stable_across_runs(index: HybridIndex) -> None:
    """同分时按 chunk_id 排序，否则评测数字会随字典顺序抖动。"""
    first = [h.chunk.chunk_id for h in index.search("年假", allowed_depts={"hr"})[0]]
    second = [h.chunk.chunk_id for h in index.search("年假", allowed_depts={"hr"})[0]]
    assert first == second


def test_vectors_are_normalised() -> None:
    """FAISS 用内积当余弦，前提是向量已归一化。"""
    emb = HashingEmbedder(dim=128)
    mat = emb.embed_documents(["测试文本一", "测试文本二"])
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_save_and_load_roundtrip(index: HybridIndex, tmp_path) -> None:
    index.save(tmp_path)
    loaded = HybridIndex(HashingEmbedder(dim=256))
    loaded.load(tmp_path)
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in index.chunks]
    hits, _ = loaded.search("住宿费", allowed_depts={"finance"})
    assert hits
