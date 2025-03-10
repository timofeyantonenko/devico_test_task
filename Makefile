.PHONY: run apply_formatting check_formatting test clean install

# Python settings
PYTHON := python3.11
VENV := venv
VENV_BIN := $(VENV)/bin

# Project settings
MAIN_FILE := src.main
TEST_DIR := tests

# Installation
install:
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install -U pip
	$(VENV_BIN)/pip install -r requirements.txt
	test -f .env.example || (echo ".env.example file is missing" && exit 1)
	test -f .env || cp .env.example .env

# Run the application
run:
	PYTHONPATH=$(PYTHONPATH):. $(VENV_BIN)/python -m $(MAIN_FILE)

# Format code using black
apply_formatting:
	$(VENV_BIN)/black .
	$(VENV_BIN)/isort .

# Check code formatting and linting
check_formatting:
	$(VENV_BIN)/black --check .
	$(VENV_BIN)/isort --check-only .
	$(VENV_BIN)/mypy src/ tests/

# Run tests with pytest
test:
	PYTHONPATH=$(PYTHONPATH):. $(VENV_BIN)/pytest $(TEST_DIR)/ -v -W ignore::pytest.PytestCollectionWarning --cov=. --cov-report=term-missing

# Clean up python cache and virtual environment
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +

# Help command to show available commands
help:
	@echo "Available commands:"
	@echo "  make install            - Install dependencies in virtual environment"
	@echo "  make run               - Run the application"
	@echo "  make apply_formatting  - Apply black and isort formatting"
	@echo "  make check_formatting  - Check code formatting with black, isort, flake8, and mypy"
	@echo "  make test              - Run tests with pytest and coverage"
	@echo "  make clean             - Clean up python cache and virtual environment"
	@echo "  make help              - Show this help message"