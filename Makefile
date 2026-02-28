.PHONY: install test download train evaluate report all clean lint format

install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=src

test-fast:
	pytest tests/ -v -m "not slow and not integration" --cov=src

download:
	python scripts/download_data.py --config config/config.yaml

train:
	python scripts/run_pipeline.py --stages train --config config/config.yaml

evaluate:
	python scripts/run_pipeline.py --stages evaluate --config config/config.yaml

report:
	python scripts/generate_report.py --config config/config.yaml

all: download train evaluate report

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/
