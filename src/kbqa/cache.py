"""答案缓存：Redis 优先，进程内 LRU 兜底。

## 缓存 key 里必须有什么

只按问题文本做 key 是错误的，而且是**危险的错误**：同一句「年终奖怎么算」，
HR 部门的人问和普通员工问，答案里的数字不一样。key 必须含：

* 规范化后的问题；
* 提问者的部门集合（权限不同，可见文档不同）；
* 索引指纹（文档更新后旧答案必须失效）。

少任何一项，都会出现「A 看到了 B 的答案」这类事故——它不会报错，只会安静地泄密。

## 为什么答案缓存而不是检索缓存

检索是毫秒级，缓存它省不了多少；生成是秒级，那才是瓶颈。缓存检索结果还要处理
「命中但权限变了」的失效问题，收益小、风险大。
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from kbqa.config import CacheSettings
from kbqa.errors import KBQAError
from kbqa.logging import get_logger
from kbqa.text import normalize

logger = get_logger(__name__)


@dataclass
class CachedAnswer:
    text: str
    citations: list[dict[str, Any]]
    chunk_ids: list[str]
    created_at: float


def make_key(question: str, depts: Iterable[str], index_digest: str, prefix: str) -> str:
    payload = json.dumps(
        {
            "q": normalize(question),
            "d": sorted(set(depts)),
            "i": index_digest,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return f"{prefix}:{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]}"


class AnswerCache:
    """TTL + LRU。Redis 不可用时自动降级到进程内，且只在启动时决定一次。"""

    def __init__(self, settings: CacheSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._local: OrderedDict[str, tuple[float, CachedAnswer]] = OrderedDict()
        self._redis: Any | None = None
        self.backend = "memory"
        if settings.enabled and settings.redis_url:
            self._redis = self._connect(settings.redis_url)

    def _connect(self, url: str) -> Any | None:
        try:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.2)
            client.ping()
            self.backend = "redis"
            logger.info("cache_redis_connected", url=_redact(url))
            return client
        except Exception as exc:  # 连不上不是错误，降级即可
            logger.warning("cache_redis_unavailable_fallback_memory", error=str(exc))
            return None

    @property
    def enabled(self) -> bool:
        return self.settings.enabled and self.settings.ttl_s > 0

    def get(self, key: str) -> CachedAnswer | None:
        if not self.enabled:
            return None
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return _decode(raw) if raw else None
            except Exception as exc:
                # Redis 抖动时不能让整个问答失败：记一次日志，走本地兜底
                logger.warning("cache_redis_get_failed", error=str(exc))
        return self._local_get(key)

    def set(self, key: str, answer: CachedAnswer) -> None:
        if not self.enabled:
            return
        if self._redis is not None:
            try:
                self._redis.setex(key, self.settings.ttl_s, _encode(answer))
                return
            except Exception as exc:
                logger.warning("cache_redis_set_failed", error=str(exc))
        self._local_set(key, answer)

    def _local_get(self, key: str) -> CachedAnswer | None:
        now = time.monotonic()
        with self._lock:
            entry = self._local.get(key)
            if entry is None:
                return None
            stored, answer = entry
            if now - stored > self.settings.ttl_s:
                del self._local[key]
                return None
            self._local.move_to_end(key)
            return answer

    def _local_set(self, key: str, answer: CachedAnswer) -> None:
        if self.settings.max_entries <= 0:
            return
        with self._lock:
            self._local[key] = (time.monotonic(), answer)
            self._local.move_to_end(key)
            while len(self._local) > self.settings.max_entries:
                self._local.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._local.clear()


def _encode(answer: CachedAnswer) -> str:
    return json.dumps(
        {
            "text": answer.text,
            "citations": answer.citations,
            "chunk_ids": answer.chunk_ids,
            "created_at": answer.created_at,
        },
        ensure_ascii=False,
    )


def _decode(raw: str | bytes) -> CachedAnswer:
    data = json.loads(raw)
    return CachedAnswer(
        text=data["text"],
        citations=data["citations"],
        chunk_ids=data["chunk_ids"],
        created_at=float(data.get("created_at", 0.0)),
    )


def _redact(url: str) -> str:
    """日志里不能出现含密码的 URL。"""
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(url)
        if parts.password:
            netloc = f"{parts.username}:***@{parts.hostname}"
            if parts.port:
                netloc += f":{parts.port}"
            return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except (ValueError, KBQAError):
        pass
    return url
