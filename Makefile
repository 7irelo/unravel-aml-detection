.PHONY: help install check test lint types data features trees gnn report demo demo-down clean-demo

NAME ?= HI-Small
COMPOSE = docker compose -f deploy/compose.yaml
KAGGLE = https://www.kaggle.com/api/v1/datasets/download/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

help:
	@echo "install   Install all dependencies"
	@echo "check     Lint, type-check and test (no data, GPU, broker or LLM key needed)"
	@echo "data      Download, ingest and audit NAME (default HI-Small)"
	@echo "features  Backfill features for NAME"
	@echo "trees     Train and study all tree arms on validation"
	@echo "gnn       Train the causal GNN (CPU budget; use a GPU for real runs)"
	@echo "demo      Run the full stack on synthetic data, no credentials needed"

install:
	uv sync --all-extras --python 3.12
	uv run pre-commit install

lint:
	uv run ruff check src tests apps
	uv run ruff format --check src tests apps

types:
	uv run mypy

test:
	uv run pytest -m "not integration and not live_llm and not real_data"

check: lint types test

data:
	mkdir -p data/raw
	curl -fSL -o data/raw/$(NAME)_Patterns.txt $(KAGGLE)/$(NAME)_Patterns.txt
	curl -fSL -o data/raw/$(NAME)_Trans.csv $(KAGGLE)/$(NAME)_Trans.csv
	uv run unravel ingest --name $(NAME)
	uv run unravel audit --name $(NAME)

features:
	uv run unravel features --name $(NAME)

trees:
	uv run unravel baselines --name $(NAME)
	uv run unravel study tune --name $(NAME) --trials 30
	uv run unravel study imbalance --name $(NAME)
	for arm in T1 T2 T3 T3x T3-nofmt; do uv run unravel train-tree $$arm --name $(NAME); done
	uv run unravel study calibration --name $(NAME)
	uv run unravel study shap --name $(NAME)
	uv run unravel study drift --name $(NAME)
	uv run unravel study agent-rules --name $(NAME)

gnn:
	uv run unravel train-gnn --arm G3 --protocol P2 --name $(NAME)

demo:
	$(COMPOSE) --profile demo up --build -d
	@echo "Review UI: http://localhost:8501  API: http://localhost:8000/docs  Grafana: http://localhost:3000"

demo-down:
	$(COMPOSE) --profile demo down

clean-demo:
	$(COMPOSE) --profile demo down -v
