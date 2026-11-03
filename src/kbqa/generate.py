"""答案生成：抽取式（默认）与模型式两条路径，对外都是「流式吐 token」。

## 为什么默认是抽取式

RAG 的目标是「基于资料回答」，不是「写得像人」。没有模型时最诚实的行为是**把资料里
最相关的那几句原样摘出来，并标清楚出处**——用户能核对，错了也知道错在哪一段。
反过来，如果没有模型却返回一句「抱歉我无法回答」，系统就退化成一个搜索框；
如果让它「自由发挥」，那就是在编。

抽取式不是玩具：它就是很多制度问答产品的真实形态（HR 政策、合规条款、SOP）。
模型在这里的价值是**改写与归纳**，而不是「有没有答案」。

## 引用编号怎么定

引用编号用的是**本次回答实际用到的块**，不是召回的全部块。把召回但没引用的块也列上，
引用列表会变成「看起来很详实」的装饰，用户核对时反而找不到对应文字。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from kbqa.config import LLMSettings
from kbqa.errors import ConfigError, GenerationError
from kbqa.index import Hit
from kbqa.text import tokenize

NO_ANSWER = "根据现有知识库未找到相关依据。"

SYSTEM_PROMPT = """你是企业知识问答助手。只能依据【检索资料】回答，不得补充资料之外的信息。
资料中没有依据时，直接回答「根据现有知识库未找到相关依据」。
回答末尾用 [1][2] 标注引用编号，不要编造编号。
回答控制在 200 字以内，先给结论，再给依据。"""


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    doc_id: str
    title: str
    section: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
        }

    def label(self) -> str:
        return f"[{self.index}] 《{self.title}》 {self.section}"


class Generator(Protocol):
    name: str

    def stream(
        self,
        question: str,
        hits: Sequence[Hit],
        *,
        titles: dict[str, str],
        history: Sequence[dict[str, str]] = (),
    ) -> Iterator[str]: ...


def build_citations(hits: Sequence[Hit], titles: dict[str, str]) -> list[Citation]:
    return [
        Citation(
            index=i,
            chunk_id=hit.chunk.chunk_id,
            doc_id=hit.chunk.doc_id,
            title=titles.get(hit.chunk.doc_id, hit.chunk.doc_id),
            section=hit.chunk.section_label,
        )
        for i, hit in enumerate(hits, start=1)
    ]


# ------------------------------------------------------------------ 抽取式


class ExtractiveGenerator:
    """按词重叠度从命中块里挑句子，原样输出并附引用。

    挑句规则：句子里命中查询词的个数 / 句子长度，取前 N 句。这比「取每块第一句」
    好得多——制度类文档的第一句常常是「本章适用于……」这种没有信息量的引导句。
    """

    name = "extractive"

    def __init__(self, *, max_sentences: int = 4, max_chars: int = 320) -> None:
        self.max_sentences = max_sentences
        self.max_chars = max_chars

    def stream(
        self,
        question: str,
        hits: Sequence[Hit],
        *,
        titles: dict[str, str],
        history: Sequence[dict[str, str]] = (),
    ) -> Iterator[str]:
        del history
        if not hits:
            yield NO_ANSWER
            return

        q_tokens = set(tokenize(question))
        scored = _score_sentences(hits, q_tokens)
        if not scored:
            yield NO_ANSWER
            return

        picked = scored[: self.max_sentences]
        # 选句按得分，输出按「块的相关性 + 块内原文顺序」：
        # 先让最相关的块排前面，块内保持语序。纯按得分排会把同一段打乱成
        # 「半句接半句」；纯按召回顺序排，第一句又可能来自一个只沾边的块。
        best: dict[int, float] = {}
        for score, hit_index, _seq, _sentence in picked:
            best[hit_index] = max(best.get(hit_index, 0.0), score)
        picked.sort(key=lambda item: (-best[item[1]], item[1], item[2]))
        body_parts: list[str] = []
        used: list[int] = []
        total = 0
        last_hit: int | None = None
        for score, hit_index, seq, sentence in picked:
            _ = score, seq
            if total + len(sentence) > self.max_chars and body_parts:
                break
            if last_hit is not None and hit_index != last_hit:
                # 换块就换行：不同段落的句子原样粘在一起读不成句
                body_parts.append("\n")
                total += 1
            body_parts.append(sentence)
            total += len(sentence)
            last_hit = hit_index
            if hit_index not in used:
                used.append(hit_index)

        body = "".join(body_parts)
        # 编号直接用「召回序号」而不是「第几个用到的块」：正文里出现 [3] 就一定是
        # 第 3 条召回结果，和 meta 里 citations[].index 一一对应。跳号是信息，不是 bug。
        refs = "".join(f"[{i + 1}]" for i in used)
        answer = f"{body}{refs}"

        # 逐段吐，保证 SSE 真的有「流」的形态，而不是一次性给完
        yield from _slice(answer, 24)

        if used:
            yield "\n\n引用来源：\n" + "\n".join(
                f"[{i + 1}] 《{titles.get(hits[i].chunk.doc_id, hits[i].chunk.doc_id)}》"
                f" · {hits[i].chunk.section_label}"
                for i in sorted(used)
            )


def _score_sentences(hits: Sequence[Hit], q_tokens: set[str]) -> list[tuple[float, int, int, str]]:
    """返回 ``(得分, 命中块序号, 块内句序, 句子)``，按得分降序。

    句序要一起带出来：挑句看分数，输出必须回到原文顺序（见 ``stream``）。
    """
    scored: list[tuple[float, int, int, str]] = []
    for hit_index, hit in enumerate(hits):
        for seq, sentence in enumerate(_sentences(hit.chunk.text)):
            tokens = set(tokenize(sentence))
            if not tokens:
                continue
            overlap = len(tokens & q_tokens)
            if overlap == 0:
                continue
            # 用重叠数除以句子长度的平方根：既奖励命中多，也压制「越长越容易命中」
            score = overlap / (len(sentence) ** 0.5)
            scored.append((score, hit_index, seq, sentence))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return scored


def _sentences(text: str) -> list[str]:
    """按中文句末标点断句。

    换行先删掉：制度文档里的换行是排版（一行 ~40 字就折行），不是句末。
    把 ``\\n`` 当句末切，会把「按日工资的 200%\\n补偿」切成两半，前半句以百分号
    结尾、后半句没有主语，两句都没法用。
    """
    text = "".join(line.strip() for line in text.splitlines())
    out: list[str] = []
    buf: list[str] = []
    for char in text:
        buf.append(char)
        if char in "。！？；":
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def _slice(text: str, size: int) -> Iterator[str]:
    for i in range(0, len(text), size):
        yield text[i : i + size]


# ------------------------------------------------------------------ 模型式


class QwenGenerator:
    """OpenAI 兼容接口的流式生成。"""

    name = "qwen"

    def __init__(self, settings: LLMSettings) -> None:
        if settings.api_key is None:
            raise ConfigError("llm.provider=qwen 时必须配置 api_key")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("未安装 openai 包") from exc
        self._client = OpenAI(
            api_key=settings.api_key.get_secret_value(),
            base_url=settings.base_url,
            timeout=settings.timeout_s,
            max_retries=2,
        )
        self._settings = settings

    def stream(
        self,
        question: str,
        hits: Sequence[Hit],
        *,
        titles: dict[str, str],
        history: Sequence[dict[str, str]] = (),
    ) -> Iterator[str]:
        if not hits:
            yield NO_ANSWER
            return

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": _build_prompt(question, hits, titles)})

        try:
            stream = self._client.chat.completions.create(
                model=self._settings.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=self._settings.temperature,
                max_tokens=self._settings.max_tokens,
                stream=True,
            )
            for event in stream:
                # openai 的 stream 是 Stream[ChatCompletionChunk] 与「错误时返回的
                # tuple」的联合类型，stub 里两种都列了出来
                delta = event.choices[0].delta.content if event.choices else None  # type: ignore[union-attr]
                if delta:
                    yield delta
        except Exception as exc:
            raise GenerationError(f"模型调用失败: {exc}") from exc


def _build_prompt(question: str, hits: Sequence[Hit], titles: dict[str, str]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        title = titles.get(hit.chunk.doc_id, hit.chunk.doc_id)
        blocks.append(f"[{i}] 《{title}》 {hit.chunk.section_label}\n{hit.chunk.text}")
    return "【检索资料】\n" + "\n\n".join(blocks) + f"\n\n【问题】\n{question}"


def build_generator(settings: LLMSettings) -> Generator:
    if settings.provider == "qwen":
        return QwenGenerator(settings)
    return ExtractiveGenerator()


def format_citations(citations: Iterable[Citation]) -> str:
    lines = [c.label() for c in citations]
    return "\n".join(lines)
