# Build context must be the repository root so src/ and prompts/ are available.
# Railway: dockerfile path api/Dockerfile (see railway.toml). Do not set the
# service root to api/ — that hides the Grok client and prompt templates.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/api

WORKDIR /app

COPY api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY src /app/src
COPY prompts /app/prompts
COPY api/app /app/api/app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
