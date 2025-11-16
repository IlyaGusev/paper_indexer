.PHONY: black validate install test

install:
	uv pip install -e .

black:
	uv run black paper_indexer tests --line-length 100

test:
	uv run pytest tests -vs

validate:
	uv run black paper_indexer tests --line-length 100
	uv run flake8 paper_indexer tests
	uv run mypy paper_indexer tests --strict --explicit-package-bases