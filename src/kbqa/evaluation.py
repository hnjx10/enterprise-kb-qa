"""检索评测：把「效果好不好」变成一条能重跑的命令。

三个指标，各自回答一个不同的问题：

* **Recall@K（查全率）**——该被召回的段落，召回了多少。它衡量「检索层有没有漏」，
  与生成质量无关。
* **Hit@3（Top-3 命中率）**——用户最可能看的前三条里，有没有正确答案。它比
  Recall 更贴近体感：第 8 条命中了但用户不会翻到第 8 条。
* **MRR**——第一个正确结果排在第几位。用来发现「能召回但排序差」的情况，这类问题
  在 Recall 上看不出来。

## 单路 vs 双路必须在同一份标注上比

把向量路的数字和关键词路的数字分别在不同数据集上跑，然后说「提升了 18%」，
是这类项目里最常见的无效对比。这里三个模式（vector / keyword / hybrid）跑的是
**同一批问题、同一份标注、同一个索引**，唯一变化的只有召回路径。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal

from kbqa.config import RetrievalSettings
from kbqa.index import Hit
from kbqa.pipeline import QAPipeline
from kbqa.store import Principal

Mode = Literal["vector", "keyword", "hybrid"]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    doc_id: str
    section: str = ""
    depts: tuple[str, ...] = ()
    note: str = ""


@dataclass
class CaseResult:
    case_id: str
    question: str
    recall: float
    hit_at_3: bool
    reciprocal_rank: float
    retrieved: list[str]
    relevant: list[str]


@dataclass
class EvalReport:
    mode: str
    cases: list[CaseResult]
    k: int

    @property
    def recall(self) -> float:
        return mean([c.recall for c in self.cases]) if self.cases else 0.0

    @property
    def hit_at_3(self) -> float:
        return mean([1.0 if c.hit_at_3 else 0.0 for c in self.cases]) if self.cases else 0.0

    @property
    def mrr(self) -> float:
        return mean([c.reciprocal_rank for c in self.cases]) if self.cases else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cases": len(self.cases),
            "k": self.k,
            "recall": round(self.recall, 4),
            "hit_at_3": round(self.hit_at_3, 4),
            "mrr": round(self.mrr, 4),
            "failures": [
                {"case_id": c.case_id, "question": c.question, "retrieved": c.retrieved}
                for c in self.cases
                if not c.hit_at_3
            ],
        }


def load_cases(path: Path) -> list[EvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvalCase(
            case_id=row["id"],
            question=row["question"],
            doc_id=row["doc"],
            section=row.get("section", ""),
            depts=tuple(row.get("depts", ())),
            note=row.get("note", ""),
        )
        for row in payload["questions"]
    ]


def relevant_chunks(pipeline: QAPipeline, case: EvalCase) -> list[str]:
    """标注是「文档 + 章节标题」而不是 chunk_id。

    chunk_id 里带序号，改一次 ``target_chars`` 全部失效，标注就得重做。
    按「文档 + 章节」标注，只要内容没改，分块参数怎么调都还能用。
    """
    return [
        c.chunk_id
        for c in pipeline.index.chunks
        if c.doc_id == case.doc_id and (not case.section or case.section in c.section_label)
    ]


def evaluate(
    pipeline: QAPipeline,
    cases: Sequence[EvalCase],
    *,
    mode: Mode = "hybrid",
    k: int = 10,
) -> EvalReport:
    base = pipeline.settings.retrieval
    settings = RetrievalSettings(
        vector_top_k=base.vector_top_k if mode != "keyword" else 0,
        keyword_top_k=base.keyword_top_k if mode != "vector" else 0,
        rrf_k=base.rrf_k,
        final_top_k=k,
        oversample=base.oversample,
    )

    results: list[CaseResult] = []
    for case in cases:
        principal = Principal(user_id="eval", depts=frozenset(case.depts))
        hits, _ = pipeline.index.search(
            case.question, allowed_depts=principal.allowed_depts, settings=settings
        )
        results.append(_score(case, hits, relevant_chunks(pipeline, case)))
    return EvalReport(mode=mode, cases=results, k=k)


def _score(case: EvalCase, hits: Sequence[Hit], relevant: Sequence[str]) -> CaseResult:
    retrieved = [h.chunk.chunk_id for h in hits]
    relevant_set = set(relevant)
    relevant_list = list(relevant)

    if not relevant_set:
        # 标注本身有问题（章节名写错）时不能算 0 分：那会把评测数字拖成一个假象
        return CaseResult(
            case_id=case.case_id,
            question=case.question,
            recall=0.0,
            hit_at_3=False,
            reciprocal_rank=0.0,
            retrieved=retrieved,
            relevant=relevant_list,
        )

    found = [cid for cid in retrieved if cid in relevant_set]
    recall = len(set(found)) / len(relevant_set)
    hit3 = any(cid in relevant_set for cid in retrieved[:3])
    rr = next((1.0 / (i + 1) for i, cid in enumerate(retrieved) if cid in relevant_set), 0.0)
    # 七个同类型参数用关键字传：位置传参时把 recall 和 rr 写反，mypy 看不出来
    return CaseResult(
        case_id=case.case_id,
        question=case.question,
        recall=recall,
        hit_at_3=hit3,
        reciprocal_rank=rr,
        retrieved=retrieved,
        relevant=relevant_list,
    )


def compare(pipeline: QAPipeline, cases: Sequence[EvalCase], *, k: int = 10) -> dict[str, Any]:
    reports = {
        mode: evaluate(pipeline, cases, mode=mode, k=k)  # type: ignore[arg-type]
        for mode in ("vector", "keyword", "hybrid")
    }
    vector = reports["vector"]
    hybrid = reports["hybrid"]
    lift = (hybrid.recall - vector.recall) / vector.recall if vector.recall else 0.0
    return {
        "k": k,
        "cases": len(cases),
        "modes": {m: r.to_dict() for m, r in reports.items()},
        "recall_lift_vs_vector": round(lift, 4),
        "hit3_gain_vs_vector": round(hybrid.hit_at_3 - vector.hit_at_3, 4),
    }
