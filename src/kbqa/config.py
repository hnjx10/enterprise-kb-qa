"""分层配置。

优先级：**构造参数 > 环境变量 > `.env` > 代码默认值**。环境变量前缀 ``KBQA_``，
嵌套字段用双下划线（``KBQA_RETRIEVAL__VECTOR_TOP_K=20``）。

三个「默认值即立场」的选择：

* ``embedding.provider="hashing"`` —— 离线可跑，不依赖任何 API key。接真实向量
  模型只需改一个配置项，接口不变。评测里的对比必须能在同一台机器上重跑，
  否则数字只是别人的数字。
* ``llm.provider="extractive"`` —— 抽取式生成：从命中的分块里摘句并附引用。
  它不是「假装有模型」，而是一条明确的降级路径：没有模型时系统仍然给出**有出处、
  可核对**的答案，而不是沉默或胡编。
* ``store.backend="sqlite"`` —— MySQL 是生产配置，SQLite 让 `pip install` 之后
  立刻能跑。两者共用同一份 SQL 语义（见 ``store.py``）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "KBQA_"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
# 索引是生成物（二进制 + 可重建），放在 var/ 而不是 data/：
# data/ 是「源材料」，会进版本库；var/ 是可丢弃的运行时产物。
DEFAULT_VAR_DIR = PROJECT_ROOT / "var"


class ChunkSettings(BaseModel):
    """分块参数。改这里等于改「检索的原子单位」，是效果最敏感的一组开关。"""

    target_chars: int = Field(default=300, ge=80, le=2000, description="单块目标字符数")
    max_chars: int = Field(default=700, ge=100, le=4000, description="硬上限，超了必须再切")
    min_chars: int = Field(default=80, ge=0, le=1000, description="过短的块会与相邻块合并")
    overlap_sentences: int = Field(default=1, ge=0, le=3, description="相邻块共享的句子数")
    # 中文句末标点；分号也算切点，因为制度类文本一句常常很长
    sentence_delimiters: str = "。！？；"


class EmbeddingSettings(BaseModel):
    provider: Literal["hashing", "qwen"] = "hashing"
    dim: int = Field(default=512, ge=64, le=4096, description="hashing 模式的哈希桶数量")
    model: str = "text-embedding-v3"
    api_key: SecretStr | None = None
    base_url: str | None = None
    batch_size: int = Field(default=25, ge=1, le=128)
    timeout_s: float = Field(default=30.0, gt=0)

    @property
    def configured(self) -> bool:
        return self.provider == "qwen" and self.api_key is not None


class RetrievalSettings(BaseModel):
    """双路召回与融合参数。"""

    # 置 0 表示关闭该路——评测里的单路基线就是这么来的，不需要另写一套检索器
    vector_top_k: int = Field(default=12, ge=0, le=200)
    keyword_top_k: int = Field(default=12, ge=0, le=200)
    rrf_k: int = Field(default=60, ge=1, description="RRF 平滑常数，论文默认 60")
    final_top_k: int = Field(default=5, ge=1, le=50, description="送给生成器的块数")
    #: 有权限过滤时先多取一些候选：过滤发生在召回之后，不多取样就会被无权限文档挤占名额
    oversample: int = Field(default=4, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, description="低于该 RRF 分数的块丢弃")


class CacheSettings(BaseModel):
    enabled: bool = True
    redis_url: str = "redis://127.0.0.1:6379/0"
    ttl_s: int = Field(default=300, ge=0)
    max_entries: int = Field(default=1024, ge=0, description="Redis 不可用时的进程内 LRU 上限")
    key_prefix: str = "kbqa:answer"


class LLMSettings(BaseModel):
    provider: Literal["extractive", "qwen"] = "extractive"
    model: str = "qwen-plus"
    api_key: SecretStr | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=800, ge=64)
    timeout_s: float = Field(default=30.0, gt=0)

    @property
    def configured(self) -> bool:
        return self.provider == "qwen" and self.api_key is not None


class StoreSettings(BaseModel):
    """元数据存储。生产是 MySQL，本地默认 SQLite，SQL 方言差异收敛在 store.py 里。"""

    backend: Literal["sqlite", "mysql"] = "sqlite"
    sqlite_path: Path = Path("var/kbqa.db")
    mysql_host: str = "127.0.0.1"
    mysql_port: int = Field(default=3306, ge=1, le=65535)
    mysql_user: str = "root"
    mysql_password: SecretStr | None = None
    mysql_database: str = "kbqa"
    pool_size: int = Field(default=5, ge=1, le=50)


class APISettings(BaseModel):
    max_question_chars: int = Field(default=500, ge=1, le=4000)
    history_turns: int = Field(default=3, ge=0, le=10)
    sse_heartbeat_s: float = Field(default=15.0, gt=0)
    cors_origins: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        validate_default=True,
    )

    env: Literal["dev", "staging", "prod", "test"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    data_dir: Path = DEFAULT_DATA_DIR
    index_dir: Path = DEFAULT_VAR_DIR / "index"
    docs_dir: Path = DEFAULT_DATA_DIR / "docs"

    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    api: APISettings = Field(default_factory=APISettings)

    @field_validator("data_dir", "index_dir", "docs_dir", mode="before")
    @classmethod
    def _expand(cls, value: Any) -> Any:
        if isinstance(value, str):
            return Path(value).expanduser()
        return value

    @property
    def vector_path(self) -> Path:
        return self.index_dir / "vectors.faiss"

    @property
    def meta_path(self) -> Path:
        return self.index_dir / "chunks.jsonl"


@lru_cache(maxsize=1)
def load_settings(**overrides: Any) -> Settings:
    """进程级配置单例。测试传 overrides 时绕开缓存。"""
    if overrides:
        return Settings(**overrides)
    return Settings()
