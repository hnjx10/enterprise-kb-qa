"""问答编排：装载索引 → 权限 → 缓存 → 双路召回 → 流式生成。

这条链上每一步都能单独失败，所以每一步都有明确的降级动作：

| 环节 | 失败时 |
|---|---|
| Redis | 降级进程内 LRU，问答继续 |
| FAISS 文件缺失 | 索引退化为只用 BM25，记警告 |
| 模型不可用 | 降级抽取式生成，答案仍有出处 |
| 检索为空 | 直接回「未找到依据」，**不调模型** |

最后一条最重要：没有召回到任何内容还去调模型，等于给模型一个空白舞台，它一定会
用「一般来说」「通常情况下」把答案补完。这类答案在演示里最像样，在生产里最危险。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kbqa.cache import AnswerCache, CachedAnswer, make_key
from kbqa.chunking import Chunk, content_digest, split_document
from kbqa.config import Settings
from kbqa.embeddings import build_embedder
from kbqa.errors import EmptyQuestionError, IndexNotReadyError
from kbqa.generate import NO_ANSWER, build_citations, build_generator
from kbqa.index import Hit, HybridIndex, RetrievalDiagnostics
from kbqa.logging import get_logger
from kbqa.store import DocumentRecord, MetadataStore, Principal

logger = get_logger(__name__)


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "skipped": self.skipped,
            "digest": self.digest,
        }


@dataclass
class Event:
    """SSE 事件。``event`` 直接作为 SSE 的 event 字段。"""

    event: str
    data: dict[str, Any]

    def to_sse(self) -> dict[str, str]:
        return {"event": self.event, "data": json.dumps(self.data, ensure_ascii=False)}


class QAPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder = build_embedder(settings.embedding)
        self.index = HybridIndex(self.embedder)
        self.store = MetadataStore(settings.store)
        self.cache = AnswerCache(settings.cache)
        self.generator = build_generator(settings.llm)
        self._digest = ""
        self._titles: dict[str, str] = {}

    # ------------------------------------------------------------------ 装载

    def load(self) -> None:
        if not self.index.ready:
            self.index.load(self.settings.index_dir)
        self._titles = dict(self.index._titles)  # noqa: SLF001 - 同一模块族的内部约定
        self._digest = content_digest(self.index.chunks)

    @property
    def ready(self) -> bool:
        return self.index.ready

    # ------------------------------------------------------------------ 建库

    def ingest(self, docs_dir: Path | None = None, *, persist: bool = True) -> IngestReport:
        docs_dir = docs_dir or self.settings.docs_dir
        if not docs_dir.is_dir():
            raise IndexNotReadyError(f"文档目录不存在：{docs_dir}", details={"path": str(docs_dir)})

        chunks: list[Chunk] = []
        records: list[DocumentRecord] = []
        skipped: list[str] = []

        for path in sorted(docs_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".markdown"}:
                continue
            raw = path.read_text(encoding="utf-8")
            meta, body = _front_matter(raw)
            doc_id = path.stem
            title = str(meta.get("title") or _first_heading(body) or doc_id)
            dept = str(meta.get("dept") or "")
            doc_chunks = split_document(
                body, doc_id=doc_id, dept=dept, settings=self.settings.chunk
            )
            if not doc_chunks:
                skipped.append(f"{path.name}: 空文档")
                continue
            chunks.extend(doc_chunks)
            records.append(
                DocumentRecord(
                    doc_id=doc_id,
                    title=title,
                    dept=dept,
                    source_path=str(path.relative_to(docs_dir)),
                    chunk_count=len(doc_chunks),
                    digest=content_digest(doc_chunks),
                    updated_at="",
                )
            )

        if not chunks:
            raise IndexNotReadyError("没有可入库的文档", details={"dir": str(docs_dir)})

        titles = {r.doc_id: r.title for r in records}
        self.index.build(chunks, titles=titles)
        self.store.upsert_documents(records)
        self._titles = titles
        self._digest = content_digest(chunks)

        if persist:
            self.index.save(self.settings.index_dir)
            logger.info("index_saved", path=str(self.settings.index_dir), chunks=len(chunks))

        return IngestReport(
            documents=len(records), chunks=len(chunks), skipped=skipped, digest=self._digest
        )

    # ------------------------------------------------------------------ 检索

    def search(self, question: str, principal: Principal) -> tuple[list[Hit], RetrievalDiagnostics]:
        question = self._validate(question)
        return self.index.search(
            question,
            allowed_depts=principal.allowed_depts,
            settings=self.settings.retrieval,
        )

    # ------------------------------------------------------------------ 问答

    def stream(
        self,
        question: str,
        principal: Principal,
        *,
        history: Sequence[dict[str, str]] = (),
        use_cache: bool = True,
    ) -> Iterator[Event]:
        started = time.perf_counter()
        try:
            question = self._validate(question)
        except EmptyQuestionError as exc:
            yield Event("error", exc.to_dict()["error"])
            return

        key = make_key(
            question, principal.allowed_depts, self._digest, self.settings.cache.key_prefix
        )
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                yield Event(
                    "meta",
                    {
                        "cached": True,
                        "citations": cached.citations,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
                yield Event("token", {"text": cached.text})
                yield Event(
                    "done",
                    {
                        "cached": True,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    },
                )
                return

        try:
            hits, diag = self.search(question, principal)
        except IndexNotReadyError as exc:
            yield Event("error", exc.to_dict()["error"])
            return

        citations = build_citations(hits, self._titles)
        yield Event(
            "meta",
            {
                "cached": False,
                "citations": [c.to_dict() for c in citations],
                "diagnostics": diag.to_dict(),
                "retrieval_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )

        if not hits:
            yield Event("token", {"text": NO_ANSWER})
            yield Event("done", {"cached": False, "empty": True})
            return

        parts: list[str] = []
        try:
            for piece in self.generator.stream(
                question, hits, titles=self._titles, history=history
            ):
                parts.append(piece)
                yield Event("token", {"text": piece})
        except Exception as exc:
            logger.error("generation_failed", error=str(exc))
            yield Event("error", {"code": "generation_error", "message": str(exc)})
            return

        answer = "".join(parts)
        if use_cache and answer and answer != NO_ANSWER:
            self.cache.set(
                key,
                CachedAnswer(
                    text=answer,
                    citations=[c.to_dict() for c in citations],
                    chunk_ids=[h.chunk.chunk_id for h in hits],
                    created_at=time.time(),
                ),
            )

        yield Event(
            "done",
            {
                "cached": False,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "chunks": len(hits),
            },
        )

    def answer(self, question: str, principal: Principal) -> tuple[str, list[dict[str, Any]]]:
        """非流式入口，主要给 CLI 与评测用。"""
        text: list[str] = []
        citations: list[dict[str, Any]] = []
        for event in self.stream(question, principal):
            if event.event == "meta":
                citations = event.data.get("citations", [])
            elif event.event == "token":
                text.append(event.data.get("text", ""))
        return "".join(text), citations

    # ------------------------------------------------------------------ 内部

    def _validate(self, question: str) -> str:
        text = (question or "").strip()
        if not text:
            raise EmptyQuestionError("问题不能为空")
        limit = self.settings.api.max_question_chars
        if len(text) > limit:
            raise EmptyQuestionError(
                f"问题过长（上限 {limit} 字符）", details={"length": len(text)}
            )
        return text


def _front_matter(raw: str) -> tuple[dict[str, str], str]:
    """极简 front matter：``---`` 之间的 ``key: value``。

    不引入 PyYAML：这里只需要三个字段，为一个三行解析装一个 YAML 依赖不划算，
    而且 YAML 的类型推断（``dept: no`` 变成布尔）反而会咬人。
    """
    raw = raw.replace("\r\n", "\n")
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("\n---", 1)
    if len(parts) != 2:
        return {}, raw
    head, body = parts[0], parts[1]
    meta: dict[str, str] = {}
    for line in head.splitlines()[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip("'\"")
    return meta, body.lstrip("\n")


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return ""
