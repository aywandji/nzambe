# Multi-stage Dockerfile for AWS ECS deployment
# Build with: docker build --build-arg APP_VERSION=$(git describe --tags --always) -t nzambe:latest .

# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM python:3.11-slim AS builder

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# 1. Install dependencies ONLY (Heavily cached)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

# 2. Now handle the application source (Changes frequently)
ARG APP_VERSION=dev
COPY src/ ./src/
COPY README.md ./

RUN echo "__version__ = \"${APP_VERSION}\"" > ./src/nzambe/_version.py

# 3. Install the project itself (Fast, as dependencies are already there)
RUN uv sync --no-editable --frozen --no-dev --no-cache

# ============================================
# Stage 2: Runtime - Final production image
# ============================================
FROM python:3.11-slim

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application config
COPY --chown=appuser:appuser config/ ./config/

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

# Expose port
EXPOSE 8000

# Health check for AWS ECS
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "nzambe.server.server:app", "--host", "0.0.0.0", "--port", "8000"]
