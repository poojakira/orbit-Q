.PHONY: help install run test lint clean

help:
	@echo "Orbit-Q Engineering Targets:"
	@echo "  dev        - Brings up ingestion, models, orchestrator, and UI via docker-compose"
	@echo "  install    - Install dependencies in current environment (.venv recommended)"
	@echo "  run        - Launch the Streamlit command center"
	@echo "  test       - Run the pytest suite"
	@echo "  lint       - Run static code analysis (flake8) and type checks (mypy)"
	@echo "  clean      - Remove pycache, build artifacts, and test outputs"

dev:
	docker-compose up --build -d

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .[dev]

run:
	orbit-q dashboard

test:
	pytest tests/ -v --cov=.

lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --max-complexity=10 --max-line-length=127 --statistics

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
