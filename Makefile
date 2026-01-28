.PHONY: help install dev test lint format clean build push run stop logs shell docker-up docker-down docker-logs buildx-setup build-multiarch dist publish publish-test

# Variables
VERSION := $(shell grep '^version' pyproject.toml | head -1 | cut -d'"' -f2)
IMAGE_NAME := likanwen/router-maestro
DOCKER_TAGS := -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest
PLATFORMS := linux/amd64,linux/arm64

# Default target
help:
	@echo "Router-Maestro Development Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install     Install dependencies"
	@echo "  make dev         Install with dev dependencies"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linter (ruff check)"
	@echo "  make format      Format code (ruff format)"
	@echo "  make clean       Clean build artifacts"
	@echo ""
	@echo "Build & Publish:"
	@echo "  make dist        Build Python package (sdist + wheel)"
	@echo "  make publish     Publish to PyPI (requires PYPI_TOKEN)"
	@echo ""
	@echo "Local Server:"
	@echo "  make run         Start local server (port 8080)"
	@echo "  make run-debug   Start with DEBUG logging"
	@echo "  make stop        Stop local server"
	@echo ""
	@echo "Docker:"
	@echo "  make build            Build Docker image ($(IMAGE_NAME):$(VERSION))"
	@echo "  make push             Push image to Docker Hub"
	@echo "  make buildx-setup     Setup Docker buildx for multi-arch builds"
	@echo "  make build-multiarch  Build and push multi-arch image (amd64, arm64)"
	@echo "  make shell            Open shell in container"
	@echo ""
	@echo "Docker Compose (Production):"
	@echo "  make docker-up   Start services (Traefik + Router-Maestro)"
	@echo "  make docker-down Stop services"
	@echo "  make docker-logs View logs"
	@echo ""
	@echo "Current version: $(VERSION)"

# ============== Development ==============

install:
	uv pip install .

dev:
	uv pip install -e ".[dev]"

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache src/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============== Build & Publish ==============

dist: clean
	uv build

publish: dist
	@if [ -z "$$PYPI_TOKEN" ]; then \
		echo "Error: PYPI_TOKEN environment variable is not set"; \
		echo "Usage: PYPI_TOKEN=your_token make publish"; \
		exit 1; \
	fi
	uv publish --token $$PYPI_TOKEN

publish-test: dist
	@if [ -z "$$PYPI_TEST_TOKEN" ]; then \
		echo "Error: PYPI_TEST_TOKEN environment variable is not set"; \
		echo "Usage: PYPI_TEST_TOKEN=your_token make publish-test"; \
		exit 1; \
	fi
	uv publish --token $$PYPI_TEST_TOKEN --publish-url https://test.pypi.org/legacy/

# ============== Local Server ==============

run:
	uv run router-maestro server start --port 8080

run-debug:
	uv run router-maestro server start --port 8080 --log-level DEBUG

stop:
	@echo "Use Ctrl+C to stop the server"

# ============== Docker ==============

build:
	docker build $(DOCKER_TAGS) .

push: build
	docker push $(IMAGE_NAME):$(VERSION)
	docker push $(IMAGE_NAME):latest

buildx-setup:
	@docker buildx inspect multiarch > /dev/null 2>&1 || docker buildx create --name multiarch --use
	docker buildx inspect --bootstrap

build-multiarch: buildx-setup
	docker buildx build --platform $(PLATFORMS) $(DOCKER_TAGS) --push .

shell:
	docker run --rm -it $(IMAGE_NAME):latest /bin/sh

# ============== Docker Compose ==============

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart router-maestro

docker-pull:
	docker compose pull

# ============== Release ==============

release: lint test build push
	@echo "Released $(IMAGE_NAME):$(VERSION)"
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	@echo "Don't forget to push tags: git push origin v$(VERSION)"
