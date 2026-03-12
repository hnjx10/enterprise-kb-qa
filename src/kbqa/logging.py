"""结构化日志：structlog 可用时输出 JSON，不可用时退回归标准 logging。

日志写 **stderr**，stdout 留给机器可读输出（``kbqa eval --json`` 要能被 ``jq`` 直接吃）。
两者混在一起，是这类 CLI 最常见的「本地好看、CI 解析失败」来源。
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    global _CONFIGURED
    numeric = getattr(logging, level.upper(), logging.INFO)
    if _CONFIGURED:
        logging.getLogger().setLevel(numeric)
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(numeric)

    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(ensure_ascii=False)
                if json_logs
                else structlog.dev.ConsoleRenderer(colors=False),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(numeric),
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )
        _CONFIGURED = True
        return
    except Exception:  # pragma: no cover - 没装 structlog 时用标准库，不影响功能
        _CONFIGURED = True


def get_logger(name: str = "kbqa"):  # noqa: ANN201
    try:
        import structlog

        return structlog.get_logger(name)
    except Exception:  # pragma: no cover
        return _StdLogger(name)


class _StdLogger:
    """structlog 缺失时的最小替代：保持 ``logger.info("event", key=value)`` 的调用形状。"""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(name)

    def _fmt(self, event: str, kv: dict[str, object]) -> str:
        extra = " ".join(f"{k}={v}" for k, v in kv.items())
        return f"{event} {extra}".strip()

    def info(self, event: str, **kv: object) -> None:
        self._log.info(self._fmt(event, kv))

    def warning(self, event: str, **kv: object) -> None:
        self._log.warning(self._fmt(event, kv))

    def error(self, event: str, **kv: object) -> None:
        self._log.error(self._fmt(event, kv))

    def exception(self, event: str, **kv: object) -> None:
        self._log.exception(self._fmt(event, kv))
