# ── RAG System Makefile ──────────────────────────────────────────────────────
# Usage: make <target>
# Run `make help` to see all available targets.

.PHONY: help install install-dev lint format type-check test test-fast test-cov \
        test-integration api docker docker-push clean clean-all setup sample-data

PYTHON      ?= python
PIP         ?= pip
PYTEST      ?= pytest
PORT        ?= 8080
WORKERS     ?= 4
IMAGE_NAME  ?= rag-system
IMAGE_TAG   ?= latest

# ── Help ────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  RAG System — Available Make Targets"
	@echo "  ════════════════════════════════════"
	@echo ""
	@echo "  Setup"
	@echo "  ─────"
	@echo "  make install         Install production dependencies"
	@echo "  make install-dev     Install all dependencies (incl. dev tools)"
	@echo "  make setup           Full first-time setup (venv + deps + .env)"
	@echo "  make sample-data     Create sample documents for testing"
	@echo ""
	@echo "  Code Quality"
	@echo "  ────────────"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Auto-format code with black + ruff"
	@echo "  make type-check      Run mypy type checker"
	@echo "  make check           Run lint + type-check together"
	@echo ""
	@echo "  Testing"
	@echo "  ───────"
	@echo "  make test            Run all tests with coverage"
	@echo "  make test-fast       Run only unit tests (no API calls)"
	@echo "  make test-cov        Generate HTML coverage report"
	@echo "  make test-integration Run integration tests (requires API key or Ollama)"
	@echo ""
	@echo "  Running"
	@echo "  ───────"
	@echo "  make api             Run FastAPI development server"
	@echo "  make api-prod        Run FastAPI production server (gunicorn)"
	@echo ""
	@echo "  Docker"
	@echo "  ──────"
	@echo "  make docker          Build Docker image"
	@echo "  make docker-up       Start full stack with docker-compose"
	@echo "  make docker-down     Stop all containers"
	@echo "  make docker-logs     Follow API container logs"
	@echo ""
	@echo "  Cleanup"
	@echo "  ───────"
	@echo "  make clean           Remove build artifacts"
	@echo "  make clean-all       Remove all generated data (indexes, cache)"
	@echo ""

# ── Setup ───────────────────────────────────────────────────────────────────
setup:
	@echo "→ Creating virtual environment..."
	$(PYTHON) -m venv venv
	@echo "→ Installing dependencies..."
	venv/bin/pip install --upgrade pip
	venv/bin/pip install -r requirements.txt
	venv/bin/pip install pytest pytest-asyncio pytest-cov pytest-mock ruff black mypy
	@echo "→ Setting up .env..."
	@if [ ! -f .env ]; then cp .env.example .env && echo "  Created .env from template — fill in your API keys"; fi
	@echo "→ Creating required directories..."
	mkdir -p vector_store_data chroma_data .embed_cache sample_data logs
	@echo ""
	@echo "✓ Setup complete! Activate with: source venv/bin/activate"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-mock ruff black mypy

sample-data:
	$(PYTHON) -c "
import os, json
os.makedirs('sample_data', exist_ok=True)
with open('sample_data/policy.txt', 'w') as f:
    f.write('Refund Policy\n\nCustomers are eligible for a full refund within 30 days of purchase.\nRefunds are processed within 5-7 business days.\nDigital downloads are non-refundable once accessed.\n')
with open('sample_data/faq.md', 'w') as f:
    f.write('# FAQ\n\n## How do I track my order?\nYou will receive a tracking email within 24 hours.\n\n## What payment methods do you accept?\nVisa, Mastercard, American Express, PayPal, and Apple Pay.\n')
products = [
    {'text': 'XR-500 Laptop: 15-inch display, 32GB RAM, 2TB SSD. Price: \$$1,299.', 'id': 1},
    {'text': 'SmartWatch Pro: tracks heart rate, sleep, steps. Water-resistant. Price: \$$299.', 'id': 2},
]
with open('sample_data/products.json', 'w') as f:
    json.dump(products, f)
print('✓ Sample data created in ./sample_data/')
"

# ── Code Quality ─────────────────────────────────────────────────────────────
lint:
	ruff check . --fix

format:
	black . --line-length 99
	ruff check . --fix --select I   # sort imports

type-check:
	mypy rag_pipeline.py config.py \
	    data_pipeline/ embeddings/ vector_store/ retrieval/ generation/ evaluation/ \
	    --ignore-missing-imports \
	    --no-strict-optional \
	    --warn-return-any

check: lint type-check

# ── Testing ──────────────────────────────────────────────────────────────────
test:
	$(PYTEST) tests/ \
	    -v \
	    --tb=short \
	    --cov=. \
	    --cov-report=term-missing \
	    --cov-report=xml \
	    --cov-omit="tests/*,notebooks/*,venv/*" \
	    -p no:warnings

test-fast:
	$(PYTEST) tests/unit/ \
	    -v \
	    --tb=short \
	    -x \
	    -p no:warnings \
	    -m "not slow"

test-cov:
	$(PYTEST) tests/ \
	    --cov=. \
	    --cov-report=html:htmlcov \
	    --cov-omit="tests/*,notebooks/*,venv/*"
	@echo "✓ Coverage report generated: htmlcov/index.html"

test-integration:
	$(PYTEST) tests/integration/ -v --tb=short -s

# ── Running ──────────────────────────────────────────────────────────────────
api:
	$(PYTHON) -m uvicorn api.main:app \
	    --host 0.0.0.0 \
	    --port $(PORT) \
	    --reload \
	    --log-level info

api-prod:
	$(PYTHON) -m uvicorn api.main:app \
	    --host 0.0.0.0 \
	    --port $(PORT) \
	    --workers $(WORKERS) \
	    --log-level warning

# ── Docker ───────────────────────────────────────────────────────────────────
docker:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-up:
	docker-compose up -d --build rag-api redis
	@echo "✓ RAG API running at http://localhost:$(PORT)"
	@echo "  Health check: curl http://localhost:$(PORT)/health"

docker-up-full:
	docker-compose --profile monitoring up -d --build
	@echo "✓ Full stack running:"
	@echo "  API:        http://localhost:$(PORT)"
	@echo "  Grafana:    http://localhost:3001"
	@echo "  Prometheus: http://localhost:9090"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f rag-api

docker-push:
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) ghcr.io/$(GITHUB_REPOSITORY)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push ghcr.io/$(GITHUB_REPOSITORY)/$(IMAGE_NAME):$(IMAGE_TAG)

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml dist/ build/
	@echo "✓ Build artifacts cleaned"

clean-all: clean
	rm -rf vector_store_data/ chroma_data/ .embed_cache/ sample_data/ \
	       pipeline_state/ saved_index*/ bm25_data/ logs/ \
	       notebooks/*.png evaluation_report.json
	@echo "✓ All generated data cleaned"
