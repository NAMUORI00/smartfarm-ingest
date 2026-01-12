# Dataset Pipeline - Docker Image
# LLM-as-a-Judge 기반 QA 데이터셋 생성 파이프라인
#
# Build:
#   docker build -t dataset-pipeline .
#
# Usage:
#   docker run dataset-pipeline --help
#   docker run -e API_KEY=xxx dataset-pipeline config
#   docker run -v ./output:/app/output dataset-pipeline generate-qa ...

FROM python:3.11-slim

# Metadata
LABEL maintainer="SmartFarm Team"
LABEL description="LLM-as-a-Judge Dataset Generation Pipeline"
LABEL version="1.0"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements-docker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY pyproject.toml ./
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY src/ ./src/
COPY tests/ ./tests/

# Install package (editable mode with pyproject.toml)
RUN pip install --no-cache-dir -e .

# Create output directory
RUN mkdir -p /app/output

# Set working directory for CLI
WORKDIR /app/src

# Healthcheck (optional - for docker-compose)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "from dataset_pipeline.config import ConfigManager; print('OK')" || exit 1

# Default entrypoint: CLI
ENTRYPOINT ["python", "-m", "dataset_pipeline"]

# Default command: show help
CMD ["--help"]
