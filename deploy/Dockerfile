# One image for every Python service. It excludes the GNN extra, so it stays CPU-sized.
FROM python:3.12-slim AS base
# LightGBM and XGBoost need the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv==0.11.23
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev \
    --extra ml --extra streaming --extra agent --extra api
COPY src ./src
COPY apps ./apps
RUN uv sync --frozen --no-dev --extra ml --extra streaming --extra agent --extra api
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
USER nobody
