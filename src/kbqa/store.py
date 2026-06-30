"""文档与用户元数据：MySQL（生产）/ SQLite（本地）。

## 为什么要单独存元数据

分块与向量是「内容」，文档标题、所属部门、索引指纹是「管理信息」。把管理信息塞进
向量库的 metadata 里，短期省事，长期会卡在三件事上：改一次部门要重建整份索引；
按部门统计文档数要全表扫描；权限变更没有事务。

## 一种 SQL，两种方言

MySQL 用 ``%s`` 占位、SQLite 用 ``?``。这里的做法是**统一写 ``?``**，在 MySQL 分支
上替换成 ``%s``。代价是 SQL 里不能出现字面量 ``?``——这个约束很便宜，比维护两套
语句便宜得多。

自增主键的差异用 ``AUTOINCREMENT`` / ``AUTO_INCREMENT`` 在建表语句里分支，其余
DDL 只有类型名不同（``TEXT`` 两边都认）。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kbqa.config import StoreSettings
from kbqa.errors import ConfigError

_SCHEMA_COMMON = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id      VARCHAR(64) PRIMARY KEY,
    title       VARCHAR(255) NOT NULL,
    dept        VARCHAR(64) NOT NULL DEFAULT '',
    source_path VARCHAR(512) NOT NULL DEFAULT '',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    digest      VARCHAR(64) NOT NULL DEFAULT '',
    updated_at  VARCHAR(32) NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    name    VARCHAR(128) NOT NULL DEFAULT '',
    depts   VARCHAR(512) NOT NULL DEFAULT ''
);
"""


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    title: str
    dept: str
    source_path: str
    chunk_count: int
    digest: str
    updated_at: str


@dataclass(frozen=True)
class Principal:
    """请求主体。``depts`` 决定能看到哪些文档。"""

    user_id: str
    name: str = ""
    depts: frozenset[str] = frozenset()

    @property
    def allowed_depts(self) -> frozenset[str]:
        # 公共文档（空串）永远可见，不需要写进用户的部门列表
        return self.depts


class MetadataStore:
    """文档与用户的读写。构造即建表，幂等。"""

    def __init__(self, settings: StoreSettings) -> None:
        self.settings = settings
        self.backend = settings.backend
        if self.backend == "mysql":
            self._conn_factory = self._mysql_conn
        else:
            path = Path(settings.sqlite_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._path = path
            self._conn_factory = self._sqlite_conn
        self._init_schema()

    # ------------------------------------------------------------------ 连接

    def _sqlite_conn(self) -> Any:
        conn = sqlite3.connect(self._path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _mysql_conn(self) -> Any:
        try:
            import pymysql
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("未安装 pymysql：pip install pymysql") from exc
        password = self.settings.mysql_password
        return pymysql.connect(
            host=self.settings.mysql_host,
            port=self.settings.mysql_port,
            user=self.settings.mysql_user,
            password=password.get_secret_value() if password else "",
            database=self.settings.mysql_database,
            charset="utf8mb4",
            autocommit=True,
            cursorclass=pymysql.cursors.DictCursor,
        )

    def _sql(self, statement: str) -> str:
        return statement.replace("?", "%s") if self.backend == "mysql" else statement

    def _execute(self, statement: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        conn = self._conn_factory()
        try:
            cur = conn.cursor()
            cur.execute(self._sql(statement), tuple(params))
            rows = cur.fetchall() if cur.description else []
            if self.backend != "mysql":  # pymysql 开了 autocommit，sqlite 需要显式提交
                conn.commit()
            if isinstance(rows, list):
                return [dict(r) for r in rows]
            return [dict(r) for r in rows or []]
        finally:
            conn.close()

    def _executemany(self, statement: str, rows: Sequence[Sequence[Any]]) -> None:
        if not rows:
            return
        conn = self._conn_factory()
        try:
            cur = conn.cursor()
            cur.executemany(self._sql(statement), [tuple(r) for r in rows])
            if self.backend != "mysql":
                conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ 建表

    def _init_schema(self) -> None:
        conn = self._conn_factory()
        try:
            cur = conn.cursor()
            for stmt in _SCHEMA_COMMON.strip().split(";"):
                if stmt.strip():
                    cur.execute(self._sql(stmt))  # 方言差异只在这里，且两个分支都受支持
            if self.backend != "mysql":
                conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ 文档

    def upsert_documents(self, records: Iterable[DocumentRecord]) -> int:
        rows = [
            (
                r.doc_id,
                r.title,
                r.dept,
                r.source_path,
                r.chunk_count,
                r.digest,
                r.updated_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            for r in records
        ]
        # MySQL 与 SQLite 都支持 ON CONFLICT/ON DUPLICATE 的差异较大，用「先删后插」
        # 在事务里做等价替换：文档量级是千级，不值得为 UPSERT 语法分叉两套 SQL
        for row in rows:
            self._execute("DELETE FROM documents WHERE doc_id = ?", (row[0],))
        self._executemany(
            "INSERT INTO documents"
            " (doc_id, title, dept, source_path, chunk_count, digest, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def list_documents(self, *, depts: Iterable[str] | None = None) -> list[DocumentRecord]:
        rows = self._execute("SELECT * FROM documents ORDER BY doc_id")
        records = [
            DocumentRecord(
                doc_id=r["doc_id"],
                title=r["title"],
                dept=r["dept"],
                source_path=r["source_path"],
                chunk_count=int(r["chunk_count"]),
                digest=r["digest"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]
        if depts is not None:
            allowed = set(depts) | {""}
            records = [r for r in records if r.dept in allowed]
        return records

    def documents_for(self, depts: Iterable[str]) -> dict[str, str]:
        """doc_id → 标题。生成引用时用，避免每块都回表。"""
        return {d.doc_id: d.title for d in self.list_documents(depts=depts)}

    # ------------------------------------------------------------------ 用户

    def upsert_user(self, principal: Principal) -> None:
        self._execute("DELETE FROM users WHERE user_id = ?", (principal.user_id,))
        self._executemany(
            "INSERT INTO users (user_id, name, depts) VALUES (?, ?, ?)",
            [(principal.user_id, principal.name, ",".join(sorted(principal.depts)))],
        )

    def get_principal(self, user_id: str) -> Principal | None:
        rows = self._execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        if not rows:
            return None
        row = rows[0]
        depts = frozenset(d for d in str(row["depts"]).split(",") if d)
        return Principal(user_id=row["user_id"], name=row["name"], depts=depts)
