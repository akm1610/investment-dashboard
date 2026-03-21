# =============================================================================
# Makefile – Investment Dashboard unified task runner
#
# Targets
# -------
#   make all        – install dependencies and start all services (default)
#   make start      – alias for `make all`
#   make stop       – stop all services
#   make restart    – stop then start all services
#   make status     – show which services are running and their ports
#   make logs       – tail live logs from all services
#   make install    – install Python dependencies only
#   make test       – run the test suite with pytest
#   make check      – verify ports are free without starting services
#   make help       – print this help message
#
# Configuration (override on the command line or via .env)
# --------------------------------------------------------
#   API_PORT=9000        Flask API port  (default: 9000)
#   STREAMLIT_PORT=8501  Streamlit port  (default: 8501)
#
# Examples
# --------
#   make all
#   make all API_PORT=9001 STREAMLIT_PORT=8502
#   make restart --no-install
#   make stop
# =============================================================================

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := all

# Export variables so run_all.sh can read them
export API_PORT        ?= 9000
export STREAMLIT_PORT  ?= 8501

RUN_ALL := ./run_all.sh

# Make sure the script is executable before use
$(RUN_ALL):
	chmod +x $@

.PHONY: all start stop restart status logs install test check help

## Start all services (install deps + launch Flask API + Streamlit dashboard)
all: $(RUN_ALL)
	$(RUN_ALL)

## Alias for `make all`
start: all

## Stop all running services
stop: $(RUN_ALL)
	$(RUN_ALL) stop

## Stop then restart all services (skips dependency reinstall)
restart: $(RUN_ALL)
	$(RUN_ALL) stop
	$(RUN_ALL) --no-install

## Show status of each service
status: $(RUN_ALL)
	$(RUN_ALL) status

## Tail live logs from all services
logs: $(RUN_ALL)
	$(RUN_ALL) logs

## Install Python dependencies into the virtual environment
install:
	@if [ ! -d ".venv" ]; then python3 -m venv .venv; fi
	@.venv/bin/pip install --quiet -r requirements.txt
	@echo "[OK] Python dependencies installed."

## Run the test suite
test:
	@if [ -d ".venv" ]; then .venv/bin/python -m pytest tests/ -v; \
	else python3 -m pytest tests/ -v; fi

## Check that required ports are free (does not start services)
check:
	@echo "Checking port $$API_PORT (Flask API)…"
	@if lsof -ti tcp:$$API_PORT > /dev/null 2>&1; then \
	  echo "[WARN]  Port $$API_PORT is in use (PID $$(lsof -ti tcp:$$API_PORT))"; \
	else \
	  echo "[OK]    Port $$API_PORT is free."; \
	fi
	@echo "Checking port $$STREAMLIT_PORT (Streamlit)…"
	@if lsof -ti tcp:$$STREAMLIT_PORT > /dev/null 2>&1; then \
	  echo "[WARN]  Port $$STREAMLIT_PORT is in use (PID $$(lsof -ti tcp:$$STREAMLIT_PORT))"; \
	else \
	  echo "[OK]    Port $$STREAMLIT_PORT is free."; \
	fi

## Print this help message
help:
	@echo ""
	@echo "Investment Dashboard – Makefile targets"
	@echo "========================================"
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /'
	@echo ""
	@echo "Configuration variables (override on command line):"
	@echo "  API_PORT=9000        Flask API port  (current: $$API_PORT)"
	@echo "  STREAMLIT_PORT=8501  Streamlit port  (current: $$STREAMLIT_PORT)"
	@echo ""
	@echo "Examples:"
	@echo "  make all"
	@echo "  make all API_PORT=9001 STREAMLIT_PORT=8502"
	@echo "  make restart"
	@echo "  make stop"
	@echo ""
