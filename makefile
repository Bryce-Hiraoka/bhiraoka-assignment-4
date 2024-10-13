# Makefile for Flask Document Retrieval System

# Python interpreter
PYTHON = python3

# Virtual environment
VENV = venv
VENV_BIN = $(VENV)/bin

# Application file
APP = app.py

# Default target
.PHONY: all
all: install

# Create virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV)

# Install requirements
.PHONY: install
install: venv
	$(VENV_BIN)/pip install -r requirements.txt

# Run the application
.PHONY: run
run: venv
	$(VENV_BIN)/python $(APP)

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
