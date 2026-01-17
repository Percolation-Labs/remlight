# ==============================================================================
# REMLight Dockerfile
# Minimal declarative agent framework with PostgreSQL memory
# Built with uv for fast, deterministic builds
# ==============================================================================
#
# NOTE: This Dockerfile is primarily for DEPLOYMENT and CI/CD.
# For LOCAL DEVELOPMENT, it's recommended to run the API directly:
#
#   # Start Postgres with Docker
#   docker-compose up postgres -d
#
#   # Run API locally with hot reload
#   uv venv && source .venv/bin/activate && uv sync
#   uvicorn remlight.api.main:app --host 0.0.0.0 --port 8080 --reload
#
# This gives you instant hot reload and better debugging capabilities.
# See app/README.md for full development setup instructions.
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Builder - Install dependencies with uv
# ------------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install build dependencies for packages with native extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Disable bytecode compilation for faster builds
ENV UV_COMPILE_BYTECODE=0

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Copy source code
COPY remlight/ ./remlight/

# Install dependencies and the package into .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ------------------------------------------------------------------------------
# Stage 2: Runtime - Minimal production image
# ------------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

WORKDIR /app

# Install runtime dependencies:
# - curl: health checks
# - ca-certificates: SSL/TLS connections
# - tesseract-ocr: OCR engine for PDF/image parsing
# - tesseract-ocr-eng: English language data for Tesseract
# - ffmpeg: Audio/video processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tesseract-ocr \
    tesseract-ocr-eng \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash remlight && \
    chown -R remlight:remlight /app

# Copy virtual environment from builder
COPY --from=builder --chown=remlight:remlight /app/.venv /app/.venv

# Copy source code from builder
COPY --from=builder --chown=remlight:remlight /app/remlight /app/remlight

# Copy additional required files
COPY --chown=remlight:remlight sql/ ./sql/
COPY --chown=remlight:remlight schemas/ ./schemas/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1

# Switch to non-root user
USER remlight

# Expose API port
EXPOSE 8000

# Default: API server with uvicorn
CMD ["uvicorn", "remlight.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
