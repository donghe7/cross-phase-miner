PYTHON ?= python3

.PHONY: install check test test-postgres test-browser test-mqtt format build demo benchmark

install:
	$(PYTHON) -m pip install -e '.[server,experiment,dev]'

check:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	bash -n run.sh server/run_demo.sh server/postgres.sh
	node --check server/static/console.js
	npm run check
	$(PYTHON) -m unittest discover -s tests -t .

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

test-postgres:
	$(PYTHON) -m scripts.test_postgres

test-browser:
	$(PYTHON) -m tests.e2e.check_console

test-mqtt:
	$(PYTHON) -m tests.e2e.check_mqtt

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .
	npm run format

build:
	$(PYTHON) -m build

demo:
	./server/run_demo.sh --postgres --mqtt

benchmark:
	$(PYTHON) -m scripts.bench_scale
