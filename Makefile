.PHONY: black validate install

install:
	uv pip install -e .

black:
	uv run black paper_indexer --line-length 100

validate:
	uv run black paper_indexer --line-length 100
	uv run flake8 paper_indexer
	uv run mypy paper_indexer --strict --explicit-package-bases