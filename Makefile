.PHONY: install db-up db-down init ingest build backtest report test lint all

install:
	pip install -e ".[dev]"

db-up:
	docker compose up -d && sleep 4

db-down:
	docker compose down

init:
	volrv init-db

ingest:
	volrv ingest --source all

build:
	volrv build-features

backtest:
	volrv backtest --strategy all

report:
	volrv report

test:
	pytest -q

lint:
	ruff check src tests

all: init ingest build backtest report
