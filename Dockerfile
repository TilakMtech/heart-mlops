# ---- Heart Disease Risk API ------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_META_PATH=/app/models/model_meta.json \
    PORT=8000

WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --upgrade pip && pip install -r requirements-serve.txt

COPY src/ src/
COPY api/ api/
COPY models/model.joblib models/model_meta.json models/

RUN useradd --create-home --uid 1001 appuser && chown -R appuser /app
USER appuser

# one worker per container: Prometheus counters stay consistent; scale with replicas
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --no-access-log"]
