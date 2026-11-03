PY ?= python
PORT ?= 8000

# 没执行 make install 时也能跑：源码目录直接进 sys.path
export PYTHONPATH := src

.PHONY: help install ingest run ask search eval bench test lint fmt type check clean

help:
	@echo "make install   安装（含 dev 依赖）"
	@echo "make ingest    解析 data/docs 并重建索引"
	@echo "make run       起 HTTP 服务（默认 :$(PORT)）"
	@echo "make ask       示例问答"
	@echo "make eval      检索评测（单路 vs 双路，K=5）"
	@echo "make bench     时延基准"
	@echo "make check     lint + type + test 一把过"

install:
	$(PY) -m pip install -e ".[dev]"

ingest:
	$(PY) -m kbqa.cli ingest

run:
	$(PY) -m uvicorn "kbqa.api:create_app" --factory --port $(PORT)

ask:
	$(PY) -m kbqa.cli ask "出差住宿费能报多少" --dept finance

search:
	$(PY) -m kbqa.cli search "年假有几天" --dept hr

eval:
	$(PY) -m kbqa.cli eval --k 5

bench:
	$(PY) benchmarks/latency.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

fmt:
	$(PY) -m ruff format .

type:
	$(PY) -m mypy src

check: lint test type
	@echo "全部通过"
