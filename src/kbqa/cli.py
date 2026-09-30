"""命令行入口：``kbqa ingest / ask / search / eval / doctor``。

CLI 的定位是「让每个数字都能被你自己重跑一遍」：``eval`` 输出单路与双路的对比，
``doctor`` 检查权限表和索引是否对得上，``search`` 打印每一路的名次与融合分数。
一个只在 README 里存在、跑不出来的指标，等于没有指标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kbqa.config import Settings, load_settings
from kbqa.evaluation import compare, load_cases
from kbqa.logging import configure_logging
from kbqa.pipeline import QAPipeline
from kbqa.store import Principal


def _principal(args: argparse.Namespace) -> Principal:
    depts = frozenset(d.strip() for d in (args.dept or "").split(",") if d.strip())
    return Principal(user_id=getattr(args, "user", "cli"), name="CLI", depts=depts)


def build_pipeline(settings: Settings, *, load: bool = True) -> QAPipeline:
    pipeline = QAPipeline(settings)
    if load:
        pipeline.load()
    return pipeline


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = QAPipeline(settings)
    report = pipeline.ingest(Path(args.docs) if args.docs else None)
    print(f"入库完成：{report.documents} 篇文档 / {report.chunks} 个分块  指纹 {report.digest}")
    for skip in report.skipped:
        print(f"  跳过 {skip}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = build_pipeline(settings)
    principal = _principal(args)

    tokens: list[str] = []
    citations: list[dict[str, Any]] = []
    meta: dict[str, Any] | None = None
    for event in pipeline.stream(args.question, principal, use_cache=not args.no_cache):
        if event.event == "meta":
            meta = event.data
            citations = event.data.get("citations", [])  # type: ignore[assignment]
        elif event.event == "token":
            tokens.append(str(event.data.get("text", "")))
        elif event.event == "error":
            print(f"错误: {event.data.get('message')}", file=sys.stderr)
            return 2

    answer = "".join(tokens)
    if args.json:
        print(
            json.dumps(
                {"answer": answer, "citations": citations, "meta": meta},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(answer)
    # 抽取式生成器自己会在末尾附「引用来源」，再打一遍就是重复。
    # 判据是「答案里有没有这段」而不是「结尾是不是『来源：』」——后者永远不成立，
    # 因为那段话后面还跟着若干行引用。
    if citations and "引用来源：" not in answer:
        print("\n引用来源：")
        for c in citations:
            print(f"  [{c['index']}] 《{c['title']}》 {c['section']}")
    if meta is not None:
        diag = meta.get("diagnostics", {})
        print(
            f"\n耗时 {meta.get('elapsed_ms', meta.get('retrieval_ms', 0))}ms"
            f"  缓存 {'命中' if meta.get('cached') else '未命中'}"
            f"  候选 向量{diag.get('vector_candidates', 0)}"
            f"/关键词{diag.get('keyword_candidates', 0)}"
            f"  权限过滤 {diag.get('filtered_by_permission', 0)}",
            file=sys.stderr,
        )
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = build_pipeline(settings)
    hits, diag = pipeline.search(args.question, _principal(args))
    payload = {
        "hits": [
            {
                **hit.to_dict(pipeline.index.title_of(hit.chunk.doc_id)),
                "vector_rank": hit.vector_rank,
                "keyword_rank": hit.keyword_rank,
            }
            for hit in hits
        ],
        "diagnostics": diag.to_dict(),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    for i, hit in enumerate(hits, start=1):
        title = pipeline.index.title_of(hit.chunk.doc_id)
        print(f"{i}. [{hit.score:.5f}] 《{title}》 {hit.chunk.section_label}")
        print(
            f"   命中路径 {','.join(hit.sources)}  "
            f"向量名次 {hit.vector_rank}  关键词名次 {hit.keyword_rank}"
        )
        print(f"   {hit.chunk.text[:80].replace(chr(10), ' ')}...")
    print(
        f"\n候选 向量 {diag.vector_candidates} / 关键词 {diag.keyword_candidates}"
        f"，权限过滤 {diag.filtered_by_permission} 条"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = build_pipeline(settings)
    cases = load_cases(
        Path(args.cases) if args.cases else settings.data_dir / "eval" / "questions.json"
    )
    result = compare(pipeline, cases, k=args.k)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    print(f"检索评测 · {len(cases)} 条标注 · K={args.k}\n")
    print(f"  {'模式':<10}{'Recall@K':>12}{'Hit@3':>10}{'MRR':>10}")
    for mode, row in result["modes"].items():
        print(f"  {mode:<10}{row['recall']:>12.4f}{row['hit_at_3']:>10.4f}{row['mrr']:>10.4f}")
    lift = result["recall_lift_vs_vector"]
    print(
        f"\n  双路 vs 单路向量：查全率 {lift:+.1%}，Top-3 命中率 "
        f"{result['hit3_gain_vs_vector']:+.4f}"
    )
    failures = result["modes"]["hybrid"].get("failures", [])
    if failures:
        print("\n  Top-3 未命中：")
        for row in failures[:8]:
            print(f"    {row['case_id']}  {row['question']}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    pipeline = QAPipeline(settings)
    problems: list[str] = []

    try:
        pipeline.load()
        print(f"索引       OK  {len(pipeline.index)} 个分块，指纹 {pipeline._digest}")  # noqa: SLF001
    except Exception as exc:
        problems.append(f"索引不可用: {exc}")
        print(f"索引       FAIL  {exc}")

    docs = pipeline.store.list_documents()
    print(f"元数据     OK  {len(docs)} 篇文档")
    index_docs = {c.doc_id for c in pipeline.index.chunks}
    orphan = index_docs - {d.doc_id for d in docs}
    missing = {d.doc_id for d in docs} - index_docs
    if orphan:
        problems.append(f"索引里有 {len(orphan)} 篇文档不在元数据表中: {sorted(orphan)}")
    if missing:
        problems.append(f"元数据表里有 {len(missing)} 篇文档没有分块: {sorted(missing)}")

    print(f"缓存       OK  后端 {pipeline.cache.backend}")
    print(f"生成器     OK  {pipeline.generator.name}")

    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\n结论：未发现问题")
    return 0


def _settings(args: argparse.Namespace) -> Settings:
    overrides: dict[str, Any] = {}
    if getattr(args, "docs", None):
        overrides["docs_dir"] = Path(args.docs)
    if getattr(args, "index_dir", None):
        overrides["index_dir"] = Path(args.index_dir)
    configure_logging(getattr(args, "log_level", "WARNING"), json_logs=False)
    if not overrides:
        return load_settings()
    # 键名只有上面这两个，取值由 argparse 保证是 str；类型检查在 pydantic 构造时做
    return Settings(**overrides)  # type: ignore[arg-type]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbqa", description="企业智能知识问答平台")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="解析文档并重建索引")
    p_ingest.add_argument("--docs", help="文档目录（默认 data/docs）")
    p_ingest.add_argument("--index-dir", help="索引输出目录")
    p_ingest.set_defaults(func=cmd_ingest)

    for name, func, helptext in (
        ("ask", cmd_ask, "提问（流式打印答案）"),
        ("search", cmd_search, "只看检索结果，不生成答案"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("question", help="问题文本")
        p.add_argument("--user", default="cli", help="用户标识")
        p.add_argument("--dept", default="", help="用户可访问部门，逗号分隔")
        p.add_argument("--json", action="store_true", help="输出 JSON")
        p.add_argument("--index-dir", help="索引目录")
        if name == "ask":
            p.add_argument("--no-cache", action="store_true", help="跳过缓存")
        p.set_defaults(func=func)

    p_eval = sub.add_parser("eval", help="检索评测：单路 vs 双路")
    p_eval.add_argument("--cases", help="标注集路径")
    p_eval.add_argument("--k", type=int, default=5, help="Recall@K 的 K，默认等于送给生成器的块数")
    p_eval.add_argument("--json", action="store_true")
    p_eval.add_argument("--index-dir", help="索引目录")
    p_eval.set_defaults(func=cmd_eval)

    p_doctor = sub.add_parser("doctor", help="自检")
    p_doctor.add_argument("--index-dir", help="索引目录")
    p_doctor.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
