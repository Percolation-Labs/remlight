# REMLight Dockerfile
FROM python:3.12-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY remlight/ ./remlight/
COPY sql/ ./sql/
COPY schemas/ ./schemas/

# Install Python dependencies (including tracing for Phoenix)
RUN pip install --no-cache-dir -e ".[tracing]"

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Default: API server with reload for development
CMD ["uvicorn", "remlight.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
