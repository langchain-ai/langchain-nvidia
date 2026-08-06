.PHONY: all format lint test tests integration_tests e2e help

all: help

TEST_FILE ?= tests/unit_tests/

test tests:
	poetry run pytest $(PYTEST_ARGS) $(TEST_FILE)

integration_tests:
	poetry run pytest tests/integration_tests $(PYTEST_ARGS)

e2e:
	SWITCHYARD_LANGCHAIN_E2E=1 poetry run pytest \
		tests/unit_tests/test_e2e.py -v -m e2e -o addopts=

check_imports: $(shell find langchain_nvidia_switchyard -name '*.py')
	poetry run python ./scripts/check_imports.py $^

PYTHON_FILES=.
MYPY_CACHE=.mypy_cache
lint format: PYTHON_FILES=.
lint_diff format_diff: PYTHON_FILES=$(shell git diff --name-only --diff-filter=d master | grep -E '\.py$$|\.ipynb$$')
lint_package: PYTHON_FILES=langchain_nvidia_switchyard
lint_tests: PYTHON_FILES=tests
lint_tests: MYPY_CACHE=.mypy_cache_test

lint lint_diff lint_package lint_tests:
	./scripts/lint_imports.sh
	poetry run ruff check .
	[ "$(PYTHON_FILES)" = "" ] || poetry run ruff format --check $(PYTHON_FILES)
	[ "$(PYTHON_FILES)" = "" ] || poetry run mypy --cache-dir $(MYPY_CACHE) $(PYTHON_FILES)

format format_diff:
	poetry run ruff format $(PYTHON_FILES)
	poetry run ruff check --select I --fix $(PYTHON_FILES)

spell_check:
	poetry run codespell --toml pyproject.toml

spell_fix:
	poetry run codespell --toml pyproject.toml -w

help:
	@echo 'test                         - run unit tests'
	@echo 'integration_tests            - run compile-only integration tests'
	@echo 'e2e                          - run the paid OpenRouter test'
	@echo 'lint_package                  - lint and type-check the package'
	@echo 'lint_tests                    - lint and type-check tests'
	@echo 'format                        - format package and tests'
