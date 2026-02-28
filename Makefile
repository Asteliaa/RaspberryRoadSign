.PHONY: help install install-dev train infer test lint format clean

help:
	@echo "RaspberryRoadSign - Traffic Sign Detection"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        Install package dependencies"
	@echo "  make install-dev    Install with development tools"
	@echo "  make train          Train YOLO model on RTSD dataset"
	@echo "  make infer          Run inference on test video"
	@echo "  make test           Run pytest test suite"
	@echo "  make lint           Run code quality checks (pylint, mypy)"
	@echo "  make format         Format code with black"
	@echo "  make clean          Remove generated files and caches"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

train:
	python scripts/train.py --config configs/training/rtsd.yaml

infer:
	python scripts/infer.py --model models/best.pt --video test_video/sample.mp4 --output test_results/output.mp4

test:
	pytest tests/ -v --cov=src/RaspberryRoadSign

lint:
	pylint src/RaspberryRoadSign
	mypy src/RaspberryRoadSign

format:
	black src/ scripts/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .coverage -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info

.DEFAULT_GOAL := help
