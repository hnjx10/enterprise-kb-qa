# 索引在构建期生成：制度文档是低频变更的，打进镜像比每次启动重建更可控。
# 文档更新 → 重建镜像 → 滚动发布，索引与文档版本天然一致。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY data ./data
RUN python -m kbqa.cli ingest

ENV KBQA_ENV=prod \
    KBQA_LOG_JSON=true

EXPOSE 8000

# readiness 探针打的是 /readyz：索引没装好时它返回 503，流量不会进来。
# 用 /healthz 做探针会让「进程活着但检索不可用」的实例继续接流量。
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/readyz')"

CMD ["python", "-m", "uvicorn", "kbqa.api:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
