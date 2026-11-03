"""HTTP 接口：SSE 流式问答 + 检索调试 + 文档清单。

## 为什么先发 meta 再发 token

前端要的是「先看到出处，再看到答案」。如果先流式吐答案，用户读完之后才拿到引用，
引用就变成了事后补充，用户不会回去核对。``meta`` 事件在检索完成时立即下发，
里面带着引用条目和检索诊断，前端可以先把「依据：《报销管理办法》> 住宿费标准」画出来。

## 为什么 SSE 而不是 WebSocket

问答是单向流：客户端一问，服务端持续吐。WebSocket 的双向能力用不上，却要多维护
心跳、重连与代理兼容。SSE 走普通 HTTP，nginx 只要关掉缓冲就能用。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from kbqa import __version__
from kbqa.config import Settings
from kbqa.errors import KBQAError
from kbqa.logging import configure_logging, get_logger
from kbqa.pipeline import QAPipeline
from kbqa.store import Principal

logger = get_logger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    user_id: str = "anonymous"
    depts: list[str] = Field(default_factory=list)
    history: list[dict[str, str]] = Field(default_factory=list)
    use_cache: bool = True


class CitationOut(BaseModel):
    index: int
    chunk_id: str
    doc_id: str
    title: str
    section: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, json_logs=settings.log_json)

    app = FastAPI(
        title="企业智能知识问答平台",
        version=__version__,
        summary="文档层级分块 + 向量/关键词双路召回 + RRF 融合 + 文档级权限过滤",
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    pipeline = QAPipeline(settings)
    try:
        pipeline.load()
    except KBQAError as exc:
        # 启动时索引不可用不该让进程起不来：/readyz 会返回 503，流量自然不进来
        logger.warning("index_not_loaded", error=str(exc))
    app.state.pipeline = pipeline
    app.state.settings = settings

    async def _principal(
        x_user_id: str | None = Header(default=None),
        x_depts: str | None = Header(default=None),
    ) -> Principal:
        user_id = x_user_id or "anonymous"
        if x_depts is not None:
            depts = frozenset(d.strip() for d in x_depts.split(",") if d.strip())
            return Principal(user_id=user_id, depts=depts)
        stored = pipeline.store.get_principal(user_id)
        return stored or Principal(user_id=user_id)

    @app.exception_handler(KBQAError)
    async def _handle(request: Request, exc: KBQAError) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    @app.get("/healthz", summary="liveness")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", summary="readiness")
    async def readyz() -> JSONResponse:
        ready = pipeline.ready
        body = {
            "ready": ready,
            "chunks": len(pipeline.index),
            "generator": pipeline.generator.name,
            "cache": pipeline.cache.backend,
        }
        return JSONResponse(status_code=200 if ready else 503, content=body)

    @app.get("/version", summary="版本")
    async def version() -> dict[str, str]:
        return {
            "name": "enterprise-kb-qa",
            "product": "企业智能知识问答平台",
            "version": __version__,
        }

    @app.post("/v1/search", summary="只检索，不生成")
    async def search(req: AskRequest, principal: Principal = Depends(_principal)) -> dict[str, Any]:
        hits, diag = pipeline.search(req.question, principal)
        return {
            "hits": [h.to_dict(pipeline.index.title_of(h.chunk.doc_id)) for h in hits],
            "diagnostics": diag.to_dict(),
        }

    @app.get("/v1/documents", summary="文档清单（按权限过滤）")
    async def documents(principal: Principal = Depends(_principal)) -> dict[str, Any]:
        docs = pipeline.store.list_documents(depts=principal.allowed_depts)
        return {
            "documents": [
                {
                    "doc_id": d.doc_id,
                    "title": d.title,
                    "dept": d.dept,
                    "chunks": d.chunk_count,
                    "updated_at": d.updated_at,
                }
                for d in docs
            ]
        }

    @app.post("/v1/ask", summary="流式问答（SSE）")
    async def ask(
        req: AskRequest,
        request: Request,
        principal: Principal = Depends(_principal),
    ) -> EventSourceResponse:
        if len(req.question) > settings.api.max_question_chars:
            raise HTTPException(
                status_code=422,
                detail=f"问题过长（上限 {settings.api.max_question_chars} 字符）",
            )

        async def stream() -> AsyncIterator[dict[str, str]]:
            for event in pipeline.stream(
                req.question,
                principal,
                history=req.history[-settings.api.history_turns * 2 :],
                use_cache=req.use_cache,
            ):
                if await request.is_disconnected():
                    # 客户端断开后继续生成只是白烧 token——这里直接停
                    logger.info("sse_client_disconnected", user=principal.user_id)
                    return
                yield event.to_sse()

        return EventSourceResponse(
            stream(),
            ping=settings.api.sse_heartbeat_s,
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
