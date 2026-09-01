"""问答编排：事件流形状、缓存、空召回、权限。

这几条决定了「用户看到什么」：事件顺序错了前端会渲染出错，缓存 key 少了部门会泄密，
空召回还去调模型会得到一段编造的回答。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kbqa.cache import AnswerCache, CachedAnswer, make_key
from kbqa.config import Settings
from kbqa.pipeline import QAPipeline
from kbqa.store import Principal

DOCS = {
    "travel.md": """---
title: 差旅报销办法
dept: finance
---

# 差旅报销办法

## 住宿费标准

一线城市住宿费每晚上限 600 元，其他城市每晚上限 350 元。超出部分由个人承担。

## 报销时限

出差结束后 15 个自然日内提交报销单，超过 30 天需要部门负责人与财务双签。
""",
    "notice.md": """---
title: 考勤通知
---

# 考勤通知

## 打卡时间

标准工作时间为上午 9:00 至下午 18:00，弹性打卡范围为 8:30 至 10:00。
""",
}


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, content in DOCS.items():
        (docs / name).write_text(content, encoding="utf-8")
    return Settings(
        env="test",
        docs_dir=docs,
        index_dir=tmp_path / "index",
        store={"backend": "sqlite", "sqlite_path": tmp_path / "kb.db"},
        cache={"enabled": True, "redis_url": "", "ttl_s": 300, "max_entries": 16},
        retrieval={"vector_top_k": 8, "keyword_top_k": 8, "final_top_k": 3, "oversample": 4},
    )


@pytest.fixture()
def pipeline(settings: Settings) -> QAPipeline:
    p = QAPipeline(settings)
    p.ingest()
    return p


def collect(pipeline: QAPipeline, question: str, principal: Principal, **kwargs) -> list:
    return list(pipeline.stream(question, principal, **kwargs))


# ------------------------------------------------------------------ 事件流


def test_stream_starts_with_meta_then_tokens(pipeline: QAPipeline) -> None:
    """先 meta 再 token：前端要先拿到出处才能画引用。"""
    events = collect(
        pipeline, "一线城市住宿费上限多少", Principal("u1", depts=frozenset({"finance"}))
    )
    assert events[0].event == "meta"
    assert events[-1].event == "done"
    assert any(e.event == "token" for e in events)
    assert events[0].data["citations"], "meta 里必须带引用条目"


def test_meta_contains_retrieval_diagnostics(pipeline: QAPipeline) -> None:
    events = collect(pipeline, "住宿费上限", Principal("u1", depts=frozenset({"finance"})))
    diag = events[0].data["diagnostics"]
    assert {"vector_candidates", "keyword_candidates", "filtered_by_permission"} <= set(diag)


def test_answer_mentions_source_content(pipeline: QAPipeline) -> None:
    events = collect(
        pipeline, "一线城市住宿费上限多少", Principal("u1", depts=frozenset({"finance"}))
    )
    answer = "".join(e.data.get("text", "") for e in events if e.event == "token")
    assert "600" in answer


def test_empty_question_yields_error_event(pipeline: QAPipeline) -> None:
    events = collect(pipeline, "   ", Principal("u1"))
    assert events[0].event == "error"
    assert events[0].data["code"] == "empty_question"


def test_question_too_long_is_rejected(pipeline: QAPipeline) -> None:
    events = collect(pipeline, "问" * 600, Principal("u1"))
    assert events[0].event == "error"


# ------------------------------------------------------------------ 权限


def test_user_without_dept_cannot_see_finance_doc(pipeline: QAPipeline) -> None:
    hits, _ = pipeline.search("住宿费上限", Principal("u2", depts=frozenset()))
    assert all(h.chunk.dept != "finance" for h in hits)


def test_no_hits_does_not_call_the_generator(settings: Settings, pipeline: QAPipeline) -> None:
    """召回为空时直接返回「未找到依据」，不给模型编造的机会。

    注意这里显式把检索结果替换成空集，而不是指望「问一句无关的话就搜不到」。
    向量检索**永远**会返回 top-k，哪怕全是噪声——这正是它与关键词检索最大的区别，
    也是 RAG 最常见的失败模式：召回了一堆不相关的块，模型照样就着它们编。
    真正的兜底在生成侧（抽取式要求句子与问题有词重叠，否则拒答），不在检索侧。
    """
    calls: list[str] = []
    original = pipeline.generator.stream

    def spy(question, hits, **kwargs):  # noqa: ANN001, ANN003, ANN202
        calls.append(question)
        return original(question, hits, **kwargs)

    pipeline.generator.stream = spy  # type: ignore[method-assign]

    from kbqa.index import RetrievalDiagnostics

    def empty_search(question: str, principal: Principal):  # noqa: ANN202
        del question, principal
        return [], RetrievalDiagnostics()

    pipeline.search = empty_search  # type: ignore[method-assign]
    events = collect(pipeline, "量子计算机的退相干温度是多少", Principal("u3", depts=frozenset()))
    assert calls == []
    answer = "".join(e.data.get("text", "") for e in events if e.event == "token")
    assert "未找到" in answer


def test_extractive_generator_refuses_when_no_lexical_overlap(pipeline: QAPipeline) -> None:
    """没有词重叠的块即使被召回，抽取式生成也必须拒答而不是硬凑。"""
    from kbqa.index import Hit

    hit = Hit(chunk=pipeline.index.chunks[0], score=0.02)
    text = "".join(pipeline.generator.stream("量子退相干温度", [hit], titles={}))
    assert "未找到" in text


def test_extractive_citation_numbers_match_hit_positions(pipeline: QAPipeline) -> None:
    """正文里的 [3] 必须指第 3 条召回结果，且引用列表里能查到编号 3。

    编号若改成「第几个用到的块」，正文会写 [1][2] 而 citations 的 index 是 [1][4]，
    前端按编号高亮就会错位——而且不会报错，只会指错段落。
    """
    from kbqa.index import Hit

    hits = [Hit(chunk=c, score=0.02) for c in pipeline.index.chunks[:5]]
    text = "".join(pipeline.generator.stream("住宿费 报销单", hits, titles={}))
    body, _, refs = text.partition("\n\n引用来源：\n")
    assert refs, "引用段没输出"
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", body)}
    listed = {int(n) for n in re.findall(r"^\[(\d+)\]", refs, flags=re.M)}
    assert cited == listed, f"正文引用 {cited} 与列表 {listed} 不一致"


def test_extractive_puts_each_chunk_on_its_own_line() -> None:
    """不同块的句子不能直接粘在一起，否则读起来是半句话接半句话。"""
    from kbqa.chunking import Chunk
    from kbqa.generate import ExtractiveGenerator
    from kbqa.index import Hit

    def make(cid: str, text: str) -> Hit:
        doc_id, ordinal = cid.split("::")
        chunk = Chunk(
            chunk_id=cid,
            doc_id=doc_id,
            text=text,
            section_path=(doc_id,),
            ordinal=int(ordinal),
            char_start=0,
            char_end=len(text),
            dept="",
        )
        return Hit(chunk=chunk, score=0.02)

    hits = [
        make("a::0000", "住宿费每晚上限 600 元。超出部分由个人承担。"),
        make("b::0000", "住宿费发票须与行程单一起提交。"),
    ]
    text = "".join(ExtractiveGenerator().stream("住宿费", hits, titles={}))
    body = text.split("\n\n引用来源：")[0]
    # 两个块的句子之间必须断开，同块内的句子保持连排
    assert "\n" in body
    assert body.count("\n") == 1


# ------------------------------------------------------------------ 缓存


def test_second_ask_hits_cache(pipeline: QAPipeline) -> None:
    principal = Principal("u1", depts=frozenset({"finance"}))
    first = collect(pipeline, "住宿费上限多少", principal)
    second = collect(pipeline, "住宿费上限多少", principal)
    assert first[0].data["cached"] is False
    assert second[0].data["cached"] is True
    assert second[-1].data["cached"] is True


def test_cache_key_differs_by_department() -> None:
    """同一句问题，不同部门必须是不同的 key——否则会串答案。"""
    a = make_key("年终奖怎么算", {"hr"}, "digest", "kbqa")
    b = make_key("年终奖怎么算", {"finance"}, "digest", "kbqa")
    assert a != b


def test_cache_key_differs_by_index_digest() -> None:
    assert make_key("q", {"hr"}, "d1", "kbqa") != make_key("q", {"hr"}, "d2", "kbqa")


def test_cache_key_is_stable_for_same_inputs() -> None:
    assert make_key(" 住宿费 ", {"hr"}, "d", "kbqa") == make_key("住宿费", {"hr"}, "d", "kbqa")


def test_memory_cache_expires(monkeypatch) -> None:
    import time as _time

    cache = AnswerCache(
        Settings(
            env="test", cache={"enabled": True, "redis_url": "", "ttl_s": 1, "max_entries": 4}
        ).cache
    )
    base = _time.monotonic()
    cache.set("k", CachedAnswer(text="a", citations=[], chunk_ids=[], created_at=0.0))
    assert cache.get("k") is not None
    # monotonic 是相对时钟，必须基于当前时刻偏移，直接返回 10.0 会得到负的存活时间
    monkeypatch.setattr("kbqa.cache.time.monotonic", lambda: base + 10.0)
    assert cache.get("k") is None


def test_cache_disabled_returns_none() -> None:
    cache = AnswerCache(Settings(env="test", cache={"enabled": False, "redis_url": ""}).cache)
    cache.set("k", CachedAnswer(text="a", citations=[], chunk_ids=[], created_at=0.0))
    assert cache.get("k") is None


# ------------------------------------------------------------------ 入库


def test_ingest_registers_documents(pipeline: QAPipeline) -> None:
    docs = pipeline.store.list_documents()
    assert {d.doc_id for d in docs} == {"travel", "notice"}
    assert next(d for d in docs if d.doc_id == "travel").title == "差旅报销办法"


def test_index_is_persisted_and_reloadable(settings: Settings, pipeline: QAPipeline) -> None:
    assert (settings.index_dir / "chunks.jsonl").is_file()
    reloaded = QAPipeline(settings)
    reloaded.load()
    assert len(reloaded.index) == len(pipeline.index)
