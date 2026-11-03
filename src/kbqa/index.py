"""双路召回与 RRF 融合。

## 为什么是两路，而不是「向量搜一下就行」

向量检索擅长语义相近：问「出差住宿能报多少」，能召回写「住宿费标准」的段落。
它对**精确串**反而很弱：型号 ``A-2024``、条款编号「第 3.2 条」、人名，向量把这些
拆成稠密数字后差异被抹平。BM25 正相反：它几乎不看语义，但「差旅报销」四个字出现在
文档里就一定排前面。

两条路的强项互补、弱项也不同，所以融合而不是二选一。RRF（Reciprocal Rank Fusion）
取的是**名次**而不是分数：``score = Σ 1/(k + rank)``。这点很关键——向量相似度是
0~1 的余弦，BM25 是无上界的实数，直接加权求和需要先归一化，而归一化系数又随查询
分布漂移。只取名次就绕开了这个标定问题，也是 RRF 在工程上几乎不需要调参的原因。

## 权限过滤为什么放在召回阶段

放到生成之后等于没做：模型已经看到了不该看的段落，你只能拦住它的输出，拦不住它
「知道了什么」。放到召回之后、排序之前同样有问题：无权限文档会先占掉 top-k 名额，
过滤完剩下的条目数随用户权限变化，权限越小召回越差——这在数据上是看不出来的，
只有低权限用户反馈「搜不到东西」时才会暴露。

所以流程是：**多取样 → 过滤 → 再截断**。多取样的倍数（``oversample``）就是为过滤
预留的余量，过滤掉多少条会作为诊断信息返回。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kbqa.chunking import Chunk
from kbqa.config import RetrievalSettings
from kbqa.embeddings import Embedder
from kbqa.errors import IndexNotReadyError
from kbqa.text import tokenize

PUBLIC = ""  # 空部门 = 公共文档，所有人可见


@dataclass(frozen=True)
class Hit:
    """一条召回结果。``sources`` 记录它被哪几路命中，融合效果靠这个排查。"""

    chunk: Chunk
    score: float
    sources: tuple[str, ...] = ()
    vector_rank: int | None = None
    keyword_rank: int | None = None

    def to_dict(self, doc_title: str = "") -> dict[str, object]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "text": self.chunk.text,
            "section": self.chunk.section_label,
            "citation": self.chunk.citation(doc_title),
            "score": round(self.score, 6),
            "sources": list(self.sources),
        }


@dataclass
class RetrievalDiagnostics:
    """一次检索的可观测信息。没有这些，调参就只能靠感觉。"""

    vector_candidates: int = 0
    keyword_candidates: int = 0
    filtered_by_permission: int = 0
    fused: int = 0
    returned: int = 0
    detail: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "vector_candidates": self.vector_candidates,
            "keyword_candidates": self.keyword_candidates,
            "filtered_by_permission": self.filtered_by_permission,
            "fused": self.fused,
            "returned": self.returned,
            **self.detail,
        }


class HybridIndex:
    """向量 + 关键词双路索引。构建一次，之后只读（可安全并发查询）。"""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        # faiss.Index 与 rank_bm25.BM25Okapi 都没有可用的类型标注，
        # 先声明成 Any 再在用到的地方靠 _faiss is None 判断
        self._faiss: Any = None
        self._bm25: Any = None
        self._titles: dict[str, str] = {}

    # ------------------------------------------------------------------ 构建

    def build(self, chunks: Sequence[Chunk], *, titles: dict[str, str] | None = None) -> None:
        self.chunks = list(chunks)
        self._titles = dict(titles or {})
        if not self.chunks:
            self._vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self._faiss = None
            self._bm25 = None
            return

        self._vectors = self.embedder.embed_documents([c.text for c in self.chunks])
        self._faiss = _build_faiss(self._vectors)
        self._build_bm25()

    def _build_bm25(self) -> None:
        from rank_bm25 import BM25Okapi

        corpus = [list(tokenize(c.text)) for c in self.chunks]
        # 空语料或全空 token 时 BM25Okapi 会除零，这里直接置空，查询时返回空结果
        self._bm25 = BM25Okapi(corpus) if any(corpus) else None

    @property
    def ready(self) -> bool:
        return bool(self.chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------ 检索

    def search(
        self,
        query: str,
        *,
        allowed_depts: Iterable[str] = (PUBLIC,),
        settings: RetrievalSettings | None = None,
    ) -> tuple[list[Hit], RetrievalDiagnostics]:
        """双路召回 → 权限过滤 → RRF 融合 → 截断。"""
        cfg = settings or RetrievalSettings()
        if not self.ready:
            raise IndexNotReadyError("索引为空，请先执行 kbqa ingest")

        allowed = set(allowed_depts) | {PUBLIC}
        diag = RetrievalDiagnostics()

        vector_hits, keyword_hits, filtered = self._recall(query, cfg, allowed, diag)
        diag.filtered_by_permission = filtered

        fused = rrf_fuse(vector_hits, keyword_hits, k=cfg.rrf_k)
        diag.fused = len(fused)
        hits = [h for h in fused if h.score >= cfg.min_score][: cfg.final_top_k]
        diag.returned = len(hits)
        return hits, diag

    def _recall(
        self,
        query: str,
        cfg: RetrievalSettings,
        allowed: set[str],
        diag: RetrievalDiagnostics,
    ) -> tuple[list[Hit], list[Hit], int]:
        # 权限过滤在截断之前，所以每次都要多取 oversample 倍
        v_k = min(cfg.vector_top_k * cfg.oversample, len(self.chunks))
        k_k = min(cfg.keyword_top_k * cfg.oversample, len(self.chunks))

        vector_hits = self._vector_search(query, v_k, allowed)
        keyword_hits = self._keyword_search(query, k_k, allowed)

        diag.vector_candidates = len(vector_hits)
        diag.keyword_candidates = len(keyword_hits)
        filtered = (v_k - len(vector_hits)) + (k_k - len(keyword_hits))
        return vector_hits, keyword_hits, max(0, filtered)

    def _vector_search(self, query: str, k: int, allowed: set[str]) -> list[Hit]:
        if self._faiss is None or k <= 0:
            return []
        qvec = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        scores, indices = self._faiss.search(qvec, k)
        hits: list[Hit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0], strict=False), start=1):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if chunk.dept not in allowed:
                continue
            hits.append(
                Hit(
                    chunk=chunk,
                    score=float(score),
                    sources=("vector",),
                    vector_rank=rank,
                )
            )
        return hits

    def _keyword_search(self, query: str, k: int, allowed: set[str]) -> list[Hit]:
        if self._bm25 is None or k <= 0:
            return []
        tokens = list(tokenize(query))
        if not tokens:
            return []
        scores = np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)
        order = np.argsort(-scores)[: k * 4]  # 先按分数取一批，过滤后再截断
        hits: list[Hit] = []
        for rank, idx in enumerate(order, start=1):
            if scores[idx] <= 0:
                break
            chunk = self.chunks[int(idx)]
            if chunk.dept not in allowed:
                continue
            hits.append(
                Hit(
                    chunk=chunk,
                    score=float(scores[idx]),
                    sources=("keyword",),
                    keyword_rank=rank,
                )
            )
            if len(hits) >= k:
                break
        return hits

    # ------------------------------------------------------------------ 持久化

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._faiss is not None:
            import faiss

            # 用 serialize 而不是 write_index：faiss 的 C++ 层按窄字符打开文件，
            # 路径里有中文（本机用户名）时直接失败。序列化成字节再交给 Python 写盘，
            # 绕开 path 编码问题，也顺带兼容 Windows 的路径长度限制。
            (directory / "vectors.faiss").write_bytes(faiss.serialize_index(self._faiss).tobytes())
        meta = directory / "chunks.jsonl"
        with meta.open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "text": chunk.text,
                            "section_path": list(chunk.section_path),
                            "ordinal": chunk.ordinal,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end,
                            "dept": chunk.dept,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        (directory / "titles.json").write_text(
            json.dumps(self._titles, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, directory: Path) -> None:
        meta = directory / "chunks.jsonl"
        if not meta.is_file():
            raise IndexNotReadyError(f"未找到索引：{meta}", details={"path": str(meta)})
        chunks: list[Chunk] = []
        with meta.open("r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=row["chunk_id"],
                        doc_id=row["doc_id"],
                        text=row["text"],
                        section_path=tuple(row["section_path"]),
                        ordinal=row["ordinal"],
                        char_start=row["char_start"],
                        char_end=row["char_end"],
                        dept=row.get("dept", ""),
                    )
                )
        titles_path = directory / "titles.json"
        titles = (
            json.loads(titles_path.read_text(encoding="utf-8")) if titles_path.is_file() else {}
        )

        self.chunks = chunks
        self._titles = titles
        vector_path = directory / "vectors.faiss"
        if vector_path.is_file() and chunks:
            import faiss

            raw = np.frombuffer(vector_path.read_bytes(), dtype=np.uint8)
            self._faiss = faiss.deserialize_index(raw)
            self._vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        else:
            # 向量文件缺失时退化为只用关键词路——降级要能用，而不是直接 503
            self._faiss = None
        self._build_bm25()

    def title_of(self, doc_id: str) -> str:
        return self._titles.get(doc_id, doc_id)


def _build_faiss(vectors: np.ndarray):  # noqa: ANN202 - faiss 对象没有可用类型标注
    import faiss

    index = faiss.IndexFlatIP(vectors.shape[1])  # 向量已 L2 归一化，内积即余弦
    index.add(vectors.astype(np.float32))
    return index


@dataclass
class _Fused:
    """融合过程中的累计状态。用 dataclass 而不是 ``dict[str, object]``：

    后者在 mypy 下每一步都要 ``type: ignore``，而这里恰恰是最容易写错的地方——
    把 rank 和 score 记混不会有任何报错，只会让排序悄悄错掉。
    """

    chunk: Chunk
    score: float = 0.0
    sources: list[str] = field(default_factory=list)
    vector_rank: int | None = None
    keyword_rank: int | None = None


def rrf_fuse(
    vector_hits: Sequence[Hit],
    keyword_hits: Sequence[Hit],
    *,
    k: int = 60,
) -> list[Hit]:
    """倒数排名融合：``Σ 1/(k + rank)``。

    只取名次，不取分数——两路的分数尺度不同，按分数加权需要标定，而标定系数
    会随查询分布漂移。
    """
    merged: dict[str, _Fused] = {}

    for hits, source in ((vector_hits, "vector"), (keyword_hits, "keyword")):
        for rank, hit in enumerate(hits, start=1):
            entry = merged.setdefault(hit.chunk.chunk_id, _Fused(chunk=hit.chunk))
            entry.score += 1.0 / (k + rank)
            if source not in entry.sources:
                entry.sources.append(source)
            if source == "vector":
                entry.vector_rank = rank
            else:
                entry.keyword_rank = rank

    fused = [
        Hit(
            chunk=entry.chunk,
            score=entry.score,
            sources=tuple(entry.sources),
            vector_rank=entry.vector_rank,
            keyword_rank=entry.keyword_rank,
        )
        for entry in merged.values()
    ]
    # 同分时按 chunk_id 排序，保证结果稳定（否则评测结果会随字典顺序抖动）
    fused.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
    return fused
