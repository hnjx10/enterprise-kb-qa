"""HTTP 接口：SSE 事件格式、权限头、错误映射。

SSE 的坑在于「看起来能用」：curl 里能看到数据，但前端解析不出事件名。所以这里直接
断言原始报文里的 ``event:`` / ``data:`` 两行，而不是只断言 HTTP 200。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kbqa.api import create_app
from kbqa.config import Settings

DOC = """---
title: 差旅报销办法
dept: finance
---

# 差旅报销办法

## 住宿费标准

一线城市住宿费每晚上限 600 元，其他城市 350 元。超出部分个人承担。
"""


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "travel.md").write_text(DOC, encoding="utf-8")
    settings = Settings(
        env="test",
        docs_dir=docs,
        index_dir=tmp_path / "index",
        store={"backend": "sqlite", "sqlite_path": tmp_path / "kb.db"},
        cache={"enabled": False, "redis_url": ""},
        log_level="CRITICAL",
    )
    app = create_app(settings)
    app.state.pipeline.ingest()
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"


def test_readyz_reports_chunk_count(client: TestClient) -> None:
    body = client.get("/readyz").json()
    assert body["ready"] is True
    assert body["chunks"] >= 1


def test_version_carries_product_name(client: TestClient) -> None:
    body = client.get("/version").json()
    assert body["product"] == "企业智能知识问答平台"


def test_sse_stream_has_named_events(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/ask",
        json={"question": "一线城市住宿费上限"},
        headers={"X-User-Id": "u1", "X-Depts": "finance"},
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = "".join(resp.iter_text())

    assert "event: meta" in raw
    assert "event: token" in raw
    assert "event: done" in raw
    assert "住宿费标准" in raw, "meta 里必须带章节信息，前端要展示出处"


def test_sse_respects_department_header(client: TestClient) -> None:
    """没带 finance 部门时，财务文档不可见，答案里不该出现 600 元。"""
    with client.stream(
        "POST", "/v1/ask", json={"question": "一线城市住宿费上限"}, headers={"X-Depts": "hr"}
    ) as resp:
        raw = "".join(resp.iter_text())
    assert "600" not in raw


def test_search_endpoint_returns_diagnostics(client: TestClient) -> None:
    body = client.post(
        "/v1/search",
        json={"question": "住宿费"},
        headers={"X-Depts": "finance"},
    ).json()
    assert body["hits"]
    assert "filtered_by_permission" in body["diagnostics"]


def test_question_too_long_returns_422(client: TestClient) -> None:
    resp = client.post("/v1/ask", json={"question": "问" * 600}, headers={"X-Depts": "finance"})
    assert resp.status_code == 422


def test_documents_are_permission_filtered(client: TestClient) -> None:
    visible = client.get("/v1/documents", headers={"X-Depts": "hr"}).json()["documents"]
    assert all(d["dept"] != "finance" for d in visible)
    visible = client.get("/v1/documents", headers={"X-Depts": "finance"}).json()["documents"]
    assert any(d["doc_id"] == "travel" for d in visible)
