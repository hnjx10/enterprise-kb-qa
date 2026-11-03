"""层级分块：标题 → 段落 → 句子。

分块是 RAG 里最不显眼、但最决定上限的一步。切太碎，一块里没有完整语义，模型拿到
半句话只能猜；切太粗，一块里塞三四个主题，向量被平均成一个谁都不像的方向，检索
自然不准。固定窗口切（每 500 字一刀）两个问题都占。

这里的做法是**按文档自身的结构切**，长度只作为兜底：

1. 先用标题（``#`` / ``##`` / ``###``）把文档切成 section，记下 ``section_path``。
   这一条同时是引用的来源——用户可以核对「第三章 > 报销标准」而不是只看到一个文件名。
2. section 内按空行切段落。段落是作者已经划好的语义边界，比任何启发式都准。
3. 单个段落超过 ``max_chars`` 时才退到句子级（按 ``。！？；``），避免一个长条款
   独占一块。
4. 相邻块共享 ``overlap_sentences`` 个句子。重叠不是为了「多召回一点」，而是为了
   防止答案正好落在切口上——一句话被劈成两半，两边都答不完整。

两个约束值得写下来：

* **块长度有目标值，也必须有硬顶**：``target_chars`` 是打包目标，``max_chars`` 是
  硬上限。只设目标不设硬顶，遇到一个 3000 字的段落照样会切出一块 3000 字。
* **chunk_id 必须确定**：``doc_id::0007`` 由序号生成，重建索引后同一条引用仍然指向
  同一段文字。用 uuid 会让评测集里的标注全部失效。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from kbqa.config import ChunkSettings

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Chunk:
    """检索的最小单位，也是引用的最小单位。"""

    chunk_id: str
    doc_id: str
    text: str
    section_path: tuple[str, ...]
    ordinal: int
    char_start: int
    char_end: int
    #: 文档所属部门；空串表示公共文档。这里**冗余存一份**：召回阶段要按权限过滤，
    #: 如果每个候选都回表查一次部门，一次检索就变成 N 次数据库往返。
    dept: str = ""

    @property
    def section_label(self) -> str:
        return " > ".join(self.section_path) if self.section_path else "（正文）"

    def citation(self, doc_title: str = "") -> str:
        return f"{doc_title or self.doc_id} · {self.section_label}"


@dataclass
class _Block:
    """一段原始文本及其在文档中的起始偏移。"""

    path: tuple[str, ...]
    text: str
    start: int


@dataclass
class _Unit:
    """打包的最小单元：一个句子或一个短段落。"""

    text: str
    start: int
    end: int


@dataclass
class _Buffer:
    units: list[_Unit] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(len(u.text) for u in self.units)

    def reset(self, keep_last: int) -> None:
        self.units = self.units[-keep_last:] if keep_last else []


def split_document(
    text: str,
    *,
    doc_id: str,
    dept: str = "",
    settings: ChunkSettings | None = None,
) -> list[Chunk]:
    """把一篇文档切成带层级路径的块。"""
    cfg = settings or ChunkSettings()
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    for block in _sectionize(text):
        chunks.extend(_pack_block(block, doc_id=doc_id, dept=dept, cfg=cfg))
    return _renumber(chunks)


# ------------------------------------------------------------------ 结构解析


def _sectionize(text: str) -> list[_Block]:
    """按标题切成 section，标题路径进 ``path``。无标题的文档退化为单 section。"""
    lines = text.split("\n")
    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1  # +1 是换行符

    blocks: list[_Block] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    buf_start = 0
    buf_path: tuple[str, ...] = ()

    def flush() -> None:
        nonlocal buf
        raw = "\n".join(buf)
        body = raw.strip()
        if body:
            lead = len(raw) - len(raw.lstrip())
            blocks.append(_Block(path=buf_path, text=body, start=buf_start + lead))
        buf = []

    for idx, line in enumerate(lines):
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buf_path = tuple(t for _, t in stack)
            continue
        if not buf:
            buf_start = offsets[idx]
        buf.append(line)
    flush()
    return blocks


# ------------------------------------------------------------------ 打包


def _pack_block(block: _Block, *, doc_id: str, dept: str, cfg: ChunkSettings) -> list[Chunk]:
    units = _units(block, cfg)
    if not units:
        return []

    chunks: list[Chunk] = []
    buf = _Buffer()
    for unit in units:
        # 单个句子就超硬顶（极端长句）：独立成块，不和别人合并，否则整块都超限
        if len(unit.text) > cfg.max_chars:
            if buf.units:
                chunks.append(_build(buf, block, doc_id, dept))
                buf = _Buffer()
            chunks.append(_build(_Buffer([unit]), block, doc_id, dept))
            continue

        buf.units.append(unit)
        if buf.size >= cfg.target_chars:
            chunks.append(_build(buf, block, doc_id, dept))
            buf.reset(cfg.overlap_sentences)

    if buf.units:
        tail = _build(buf, block, doc_id, dept)
        # 尾巴太短就并回上一块，避免切出「一、总则」这种只有四个字的块
        if chunks and len(tail.text) < cfg.min_chars:
            prev = chunks[-1]
            merged = f"{prev.text}\n{tail.text}"
            if len(merged) <= cfg.max_chars:
                chunks[-1] = _replace_span(prev, merged, tail.char_end)
                return _renumber(chunks)
        chunks.append(tail)

    return _renumber(chunks)


def _units(block: _Block, cfg: ChunkSettings) -> list[_Unit]:
    """段落内再判一次：短段落整段算一个单元，超长段落拆成句子。"""
    units: list[_Unit] = []
    offset = block.start
    for para in block.text.split("\n"):
        stripped = para.strip()
        if stripped:
            lead = len(para) - len(para.lstrip())
            start = offset + lead
            if len(stripped) <= cfg.max_chars:
                units.append(_Unit(stripped, start, start + len(stripped)))
            else:
                units.extend(_split_sentences(stripped, start, cfg.sentence_delimiters))
        offset += len(para) + 1
    return units


def _split_sentences(text: str, base: int, delimiters: str) -> list[_Unit]:
    units: list[_Unit] = []
    start = 0
    for idx, char in enumerate(text):
        if char in delimiters:
            piece = text[start : idx + 1]
            stripped = piece.strip()
            if stripped:
                lead = len(piece) - len(piece.lstrip())
                units.append(_Unit(stripped, base + start + lead, base + idx + 1))
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        lead = len(text[start:]) - len(text[start:].lstrip())
        units.append(_Unit(tail, base + start + lead, base + len(text)))
    return units


def _build(buf: _Buffer, block: _Block, doc_id: str, dept: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}::0000",
        doc_id=doc_id,
        text="\n".join(u.text for u in buf.units).strip(),
        section_path=block.path,
        ordinal=0,
        char_start=buf.units[0].start,
        char_end=buf.units[-1].end,
        dept=dept,
    )


def _replace_span(chunk: Chunk, text: str, end: int) -> Chunk:
    return Chunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=text,
        section_path=chunk.section_path,
        ordinal=chunk.ordinal,
        char_start=chunk.char_start,
        char_end=end,
        dept=chunk.dept,
    )


def _renumber(chunks: list[Chunk]) -> list[Chunk]:
    """序号在整篇文档内连续，保证 chunk_id 确定可复现。"""
    return [
        Chunk(
            chunk_id=f"{c.doc_id}::{i:04d}",
            doc_id=c.doc_id,
            text=c.text,
            section_path=c.section_path,
            ordinal=i,
            char_start=c.char_start,
            char_end=c.char_end,
            dept=c.dept,
        )
        for i, c in enumerate(chunks)
    ]


def content_digest(chunks: list[Chunk]) -> str:
    """索引指纹：文档内容变了，缓存里基于旧分块的答案就必须失效。"""
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.chunk_id.encode("utf-8"))
        h.update(chunk.text.encode("utf-8"))
    return h.hexdigest()[:16]
