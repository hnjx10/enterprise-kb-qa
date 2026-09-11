"""时延基准：检索、首字延迟（TTFT）、完整答案、缓存命中。

**为什么单独量 TTFT**

用户感知的不是「整个回答用了多久」，而是「多久开始出字」。一个 3 秒返回 800 字的
回答，比一个 200ms 出首字、再用 3 秒慢慢说完的回答，体感差一个数量级。所以流式
接口的核心指标是 TTFT，不是总时长。

口径（README 引用的就是这组数）：

* 检索：进程内 FAISS + BM25，不含网络；
* TTFT：从调用 ``stream()`` 到收到第一个 ``token`` 事件；
* 完整：从调用到 ``done`` 事件，生成器为抽取式（离线）；
* 缓存：同一问题第二次请求，走 Redis 或进程内 LRU。

运行：``python benchmarks/latency.py``
"""

from __future__ import annotations

import argparse
import statistics
import time

from kbqa.config import Settings
from kbqa.logging import configure_logging
from kbqa.pipeline import QAPipeline
from kbqa.store import Principal

QUESTIONS = [
    "一线城市出差住宿费上限是多少",
    "年假能不能跨年休",
    "忘记打卡一个月可以补几次",
    "采购五万元以下谁审批",
    "机器人电量低于多少会回充",
]

DEPTS = frozenset({"finance", "hr", "it", "sales", "robot"})


def _pct(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, int(len(ordered) * p))
    return ordered[idx]


def _row(name: str, samples: list[float]) -> str:
    return (
        f"  {name:<12}{len(samples):>5}"
        f"{_pct(samples, 0.5):>10.2f}{_pct(samples, 0.95):>10.2f}"
        f"{statistics.fmean(samples):>10.2f}{max(samples):>10.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="kbqa 时延基准")
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()

    configure_logging("CRITICAL", json_logs=False)
    settings = Settings()
    pipeline = QAPipeline(settings)
    pipeline.load()
    principal = Principal("bench", depts=DEPTS)

    retrieve: list[float] = []
    ttft: list[float] = []
    full: list[float] = []
    cached: list[float] = []

    for i in range(args.rounds):
        question = QUESTIONS[i % len(QUESTIONS)]

        start = time.perf_counter()
        pipeline.search(question, principal)
        retrieve.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        first: float | None = None
        for event in pipeline.stream(question, principal, use_cache=False):
            if event.event == "token" and first is None:
                first = (time.perf_counter() - start) * 1000
        total = (time.perf_counter() - start) * 1000
        if first is not None:
            ttft.append(first)
            full.append(total)

        start = time.perf_counter()
        for _event in pipeline.stream(question, principal, use_cache=True):
            pass
        cached.append((time.perf_counter() - start) * 1000)

    print(f"\n单条问答（{args.rounds} 次，离线抽取式生成）")
    print(f"  {'阶段':<10}{'n':>5}{'P50':>10}{'P95':>10}{'mean':>10}{'max':>10}   (ms)")
    print(_row("检索", retrieve))
    print(_row("首字 TTFT", ttft))
    print(_row("完整", full))
    print(_row("缓存命中", cached))
    print()


if __name__ == "__main__":
    main()
