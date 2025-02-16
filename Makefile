.PHONY: run apply_formatting check_formatting test clean install

# Python settings
PYTHON := python3
VENV := venv
VENV_BIN := $(VENV)/bin

# Project settings
MAIN_FILE := main.py
TEST_DIR := tests

# Installation
install:
	test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(VENV_BIN)/pip install -U pip
	$(VENV_BIN)/pip install -r requirements.txt

# Run the application
run:
	$(VENV_BIN)/python $(MAIN_FILE)

# Format code using black
apply_formatting:
	$(VENV_BIN)/black .
	$(VENV_BIN)/isort .

# Check code formatting and linting
check_formatting:
	$(VENV_BIN)/black --check .
	$(VENV_BIN)/isort --check-only .
	$(VENV_BIN)/mypy .

# Run tests with pytest
test:
	$(VENV_BIN)/pytest $(TEST_DIR) -v --cov=. --cov-report=term-missing

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