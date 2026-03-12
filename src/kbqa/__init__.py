"""企业智能知识问答平台。

面向企业内部知识库的检索增强问答：文档按标题 / 段落 / 句子三级切分后，
向量与关键词双路召回、RRF 融合、按文档权限过滤，再注入模型流式生成带引用的答案。

分层::

    api / cli        ← 交付层（HTTP + SSE、命令行）
    pipeline         ← 编排（检索 → 融合 → 生成）
    index / store    ← 能力层（FAISS 向量索引、BM25 关键词索引、文档与权限元数据）
    chunking / embeddings / cache / generate ← 领域与基础设施

设计取向和这个领域的常见 demo 不同：**先保证「检索不到就说检索不到」，
再谈答案好不好看**。RAG 最大的失败模式不是模型胡说，而是检索召回了一堆
看似相关的内容，模型于是照着编了一段有出处的假话。
"""

from kbqa.config import Settings, load_settings

__version__ = "1.0.0"
__all__ = ["Settings", "__version__", "load_settings"]
