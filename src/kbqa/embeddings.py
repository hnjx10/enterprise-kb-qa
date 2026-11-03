"""向量化：离线哈希嵌入 + 真实模型嵌入，接口完全一致。

为什么默认走离线：评测必须能在任何一台机器上重跑。如果默认依赖外部 embedding API，
「召回率提升 18%」这句话就变成了「在某次网络状况下、某个模型版本上、某天的结果」——
别人复现不出来，你自己三个月后也复现不出来。

哈希嵌入的做法是把分词结果与字符 bigram 用带符号的哈希映射到固定维度，再 L2 归一化。
它不是语义模型，但有两个可用性质：同一批词得到的向量稳定、点积近似词重叠度。
接真实模型只需把 ``provider`` 改成 ``qwen``，上层（FAISS、检索、评测）一行不用改。

一个必须说清的边界：哈希嵌入**没有真正的语义泛化**——「年假」与「带薪休假」在它眼里
是两个不相干的词。语料里的同义表述要靠 BM25 与 RRF 兜，这也是这个项目坚持双路召回
的现实原因之一，而不是为了凑架构。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from kbqa.config import EmbeddingSettings
from kbqa.errors import ConfigError
from kbqa.text import char_ngrams, tokenize


class Embedder(Protocol):
    """最小接口：需要一个「文本 → 定长向量」的映射。"""

    dim: int
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(mat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return mat / norm


class HashingEmbedder:
    """带符号的特征哈希（hashing trick）。

    用 ``blake2b`` 而不是内置 ``hash``：后者在每个进程加盐，重启后同一段文字得到的
    向量完全不同，落盘的索引会立刻失效。
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        if dim < 64:
            raise ConfigError("embedding.dim 至少 64", details={"dim": dim})
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib_digest(token)
        return digest % self.dim, 1.0 if (digest >> 63) & 1 else -1.0

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = list(tokenize(text)) + list(char_ngrams(text, 2))
        if not tokens:
            return vec
        weight = 1.0 / np.sqrt(len(tokens))
        for token in tokens:
            idx, sign = self._bucket(token)
            vec[idx] += sign * weight
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.vstack([self._vector(t) for t in texts]).astype(np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vector(text).reshape(1, -1))[0]


def hashlib_digest(token: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "big")


class QwenEmbedder:
    """DashScope / 任意 OpenAI 兼容 embedding 接口。"""

    name = "qwen"

    def __init__(self, settings: EmbeddingSettings) -> None:
        if settings.api_key is None:
            raise ConfigError("embedding.provider=qwen 时必须配置 api_key")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失时给出可执行提示
            raise ConfigError("未安装 openai 包：pip install 'enterprise-kb-qa[llm]'") from exc

        self._client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            max_retries=3,
        )
        self._model = settings.model
        self._batch = settings.batch_size
        self.dim = settings.dim  # 首次调用后用真实维度覆盖

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend([d.embedding for d in resp.data])
        if not vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = np.asarray(vectors, dtype=np.float32)
        self.dim = mat.shape[1]
        return _l2(mat)

    def embed_query(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=self._model, input=[text])
        vec = np.asarray([resp.data[0].embedding], dtype=np.float32)
        self.dim = vec.shape[1]
        return _l2(vec)[0]


def build_embedder(settings: EmbeddingSettings) -> Embedder:
    if settings.provider == "qwen":
        return QwenEmbedder(settings)
    return HashingEmbedder(dim=settings.dim)
