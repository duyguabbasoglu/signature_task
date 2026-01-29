.PHONY: install install-dev test serve analyze clean help

# Default target
help:
	@echo "Bounding Box Detector - Makefile Commands"
	@echo ""
	@echo "  make install      - Install core dependencies"
	@echo "  make install-dev  - Install with dev & API dependencies"
	@echo "  make test         - Run pytest tests"
	@echo "  make serve        - Start FastAPI server (port 8000)"
	@echo "  make analyze      - Analyze all images in data/"
	@echo "  make clean        - Clean cache files"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[all]"

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src/bbox_detector --cov-report=term-missing

# API Server
serve:
	uvicorn bbox_detector.api.server:app --reload --host 0.0.0.0 --port 8000

# Analysis
analyze:
	bbox-detect data/

analyze-one:
	@echo "Usage: bbox-detect <image_path>"

# LLM Test
test-llm:
	./scripts/test_llm.sh

# Cleanup
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
