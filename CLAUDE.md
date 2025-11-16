# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-source academic paper indexer that fetches papers from ArXiv, bioRxiv, medRxiv, PubMed, and ChemRxiv. It generates embeddings using Gemma models and stores them in a Qdrant vector database for semantic search across all sources.

## Architecture

The system consists of four main components:

1. **Fetchers** (`paper_indexer/fetchers.py`): Async API clients for each source using `httpx.AsyncClient`:
   - `BiorxivMedrxivFetcher`: Fetches from bioRxiv/medRxiv REST API (api.biorxiv.org)
   - `PubMedFetcher`: Fetches from NCBI E-utilities API with XML parsing
   - `ChemRxivFetcher`: Fetches from ChemRxiv Public API
   - ArXiv uses Kaggle dataset download (no fetcher needed)
   - All fetchers return `AsyncIterator[Dict[str, Any]]` for streaming results

2. **Embedder** (`paper_indexer/embedder.py`): Wrapper around SentenceTransformer using `unsloth/embeddinggemma-300m` model. Creates embeddings from paper titles and abstracts using the template `"title: {} | text: {}"`.

3. **Indexer** (`paper_indexer/index.py`): Source-specific processors that parse API responses, generate embeddings in chunks of 100, and upsert to Qdrant. Supports filtering by date and category/query per source.

4. **Query** (`paper_indexer/query.py`): Performs semantic search against the unified Qdrant collection with support for filtering by paper_id, source, authors, title, abstract, update_date range, and categories.

### Data Flow

- Papers are fetched from source APIs or Kaggle dataset
- Each record is parsed into a generic `PaperMetadata` Pydantic model with source-agnostic fields
- Papers are processed in batches (CHUNK_SIZE=100) for efficient embedding generation
- Point IDs in Qdrant are derived from paper_id using MD5 hashing to ensure deterministic IDs
- The Qdrant collection has indexes on: paper_id (keyword), source (keyword), authors (text), title (text), abstract (text), update_date (datetime), categories (keyword)

### Supported Sources

- **arxiv**: ArXiv preprints via Kaggle dataset
- **biorxiv**: Biology preprints via bioRxiv API
- **medrxiv**: Medical preprints via medRxiv API
- **pubmed**: PubMed articles via NCBI E-utilities
- **chemrxiv**: Chemistry preprints via ChemRxiv API

## Development Commands

### Setup
```bash
# Install package in development mode
make install
# or
uv pip install -e .
```

### Code Quality
```bash
# Format code with black (line length: 100)
make black

# Run tests
make test

# Run all validation (black, flake8, mypy)
make validate
```

The validation pipeline runs:
- `black` with --line-length 100
- `flake8` (configured in .flake8: ignores E203, F403, E501, SIM115, PIE786, W503, E226; max-line-length 120)
- `mypy` with --strict --explicit-package-bases (ignores missing imports via pyproject.toml config)

Tests are located in the `tests/` directory and use `pytest` with `pytest-asyncio` for async test support.

### Infrastructure
```bash
# Start Qdrant vector database
docker-compose up -d

# Qdrant will be available at:
# - HTTP: localhost:6333
# - gRPC: localhost:6334
# - Data stored in: /data/qdrant
```

### Running the Application

#### Indexing Papers

```bash
# ArXiv (downloads entire dataset from Kaggle)
uv run python -m paper_indexer.index --source=arxiv

# ArXiv with filters
uv run python -m paper_indexer.index \
  --source=arxiv \
  --start_date="2024-01-01T00:00:00" \
  --category="cs.AI"

# bioRxiv (requires date range)
uv run python -m paper_indexer.index \
  --source=biorxiv \
  --start_date="2024-01-01" \
  --end_date="2024-12-31"

# bioRxiv with category filter
uv run python -m paper_indexer.index \
  --source=biorxiv \
  --start_date="2024-01-01" \
  --end_date="2024-12-31" \
  --category="neuroscience"

# medRxiv (requires date range)
uv run python -m paper_indexer.index \
  --source=medrxiv \
  --start_date="2024-01-01" \
  --end_date="2024-12-31"

# PubMed (requires search query)
uv run python -m paper_indexer.index \
  --source=pubmed \
  --query="machine learning"

# PubMed with date filters and API key
uv run python -m paper_indexer.index \
  --source=pubmed \
  --query="CRISPR gene editing" \
  --start_date="2024-01-01" \
  --end_date="2024-12-31" \
  --max_results=5000 \
  --pubmed_api_key="YOUR_API_KEY"

# ChemRxiv
uv run python -m paper_indexer.index \
  --source=chemrxiv \
  --query="catalysis" \
  --start_date="2024-01-01" \
  --max_results=1000
```

#### Querying Papers

```bash
# Basic semantic search
uv run python -m paper_indexer.query --query="transformer architecture" --limit=10

# Filter by source
uv run python -m paper_indexer.query \
  --query="neural networks" \
  --source="pubmed" \
  --limit=20

# Search across multiple filters
uv run python -m paper_indexer.query \
  --query="machine learning" \
  --authors="Hinton" \
  --min_update_date="2023-01-01T00:00:00" \
  --categories='["cs.LG", "cs.AI"]' \
  --limit=20

# Search specific source with category
uv run python -m paper_indexer.query \
  --query="COVID-19 treatment" \
  --source="medrxiv" \
  --categories='["infectious diseases"]' \
  --limit=50
```

## Code Style

- **No Obvious Comments**: Do not write comments that simply restate what the code does
  - ❌ Bad: `# Parse XML response`, `# Close the HTTP client`, `# Filter by date`
  - ✅ Good: `# Exponential backoff: 1s, 2s, 4s`, `# API uses 'preprint_' prefix for fields`
- **No Verbose Docstrings**: Avoid docstrings that just repeat function signatures
  - ❌ Bad: Full docstrings with Args/Returns for `__init__`, `__del__`, obvious utility functions
  - ✅ Good: Short docstrings explaining non-obvious behavior, API format requirements
- **Keep Comments That Add Value**: Comments explaining why (not what), non-obvious behavior, API limits
- **Type Hints Over Comments**: Use type hints instead of comments to document types

## Important Implementation Details

- **Python Version**: Requires Python >=3.12, <=3.13
- **Package Manager**: Uses `uv` for dependency management
- **HTTP Library**: Uses `httpx` for all API requests with built-in timeouts and retry logic
- **Point ID Generation**: Uses `get_point_id()` in index.py which hashes paper_id to generate deterministic 63-bit integer IDs for Qdrant
- **Date Handling**:
  - ArXiv: ISO format with time (e.g., "2024-01-01T00:00:00")
  - bioRxiv/medRxiv/ChemRxiv: YYYY-MM-DD format
  - PubMed: YYYY/MM/DD format (auto-converted from YYYY-MM-DD)
- **Embedding Model**: Using unsloth/embeddinggemma-300m via SentenceTransformer
- **Vector Distance**: Qdrant collection uses COSINE distance metric
- **Collection Name**: "papers" (defined in QDRANT_COLLECTION_NAME constant)

### Source-Specific Details

**ArXiv**:
- Uses Kaggle dataset (`Cornell-University/arxiv`)
- Downloads entire dataset locally before processing
- Supports category filtering (e.g., "cs.AI", "cs.LG")
- paper_id format: ArXiv ID (e.g., "2103.12345")

**bioRxiv/medRxiv**:
- Uses official API at api.biorxiv.org and api.medrxiv.org
- Requires start_date and end_date
- Returns 100 results per API call with cursor-based pagination
- Supports category filtering (e.g., "neuroscience", "infectious diseases")
- paper_id format: DOI

**PubMed**:
- Uses NCBI E-utilities API
- Requires search query (supports PubMed query syntax)
- Rate limits: 3 requests/sec (10 with API key)
- Returns XML that is parsed into structured data
- paper_id format: "PMID:12345678"
- Set PUBMED_API_KEY environment variable or pass --pubmed_api_key for higher rate limits

**ChemRxiv**:
- Uses ChemRxiv Public API
- Supports optional search term
- Date filtering applied client-side
- paper_id format: DOI

### API Rate Limiting and Reliability

All fetchers use async `httpx.AsyncClient` for HTTP requests with:
- **Async/Await**: All fetchers are async and use `AsyncIterator` for streaming results
- **Context Managers**: Automatic cleanup using `async with` for HTTP client lifecycle
- **Timeout**: 30 second default timeout for all requests
- **Retry Logic**: Automatic retry with exponential backoff (1s, 2s, 4s) for up to 3 attempts via `retry_request()` helper
- **Rate Limiting**: Respectful delays between requests using `asyncio.sleep()`:
  - bioRxiv/medRxiv: 0.1s delay between requests
  - PubMed: 0.34s delay (3 req/s) or 0.1s with API key (10 req/s)
  - ChemRxiv: 0.1s delay between pagination requests

The retry mechanism handles transient network errors and API timeouts automatically. The indexer uses `asyncio.run()` to execute async fetchers from the synchronous CLI interface.
