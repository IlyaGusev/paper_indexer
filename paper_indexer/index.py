from typing import List, Dict, Optional, Any
import hashlib

import fire  # type: ignore
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from paper_indexer.embedder import GemmaEmbedder
from paper_indexer.fetchers.arxiv import ArxivFetcher
from paper_indexer.fetchers.biorxiv import BiorxivFetcher
from paper_indexer.fetchers.chemrxiv import ChemrxivFetcher


CHUNK_SIZE = 100
QDRANT_COLLECTION_NAME = "papers"


def get_point_id(paper_id: str) -> int:
    md5 = hashlib.md5(paper_id.encode()).hexdigest()
    return int(md5[:16], 16) % (2**63 - 1)


def _upsert_chunk(
    client: QdrantClient, embedder: GemmaEmbedder, chunk: List[Dict[str, Any]]
) -> None:
    if not chunk:
        return

    embeddings = embedder.encode_documents(chunk)
    client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=get_point_id(paper["paper_id"]),
                vector=embedding.tolist(),
                payload=paper,
            )
            for paper, embedding in zip(chunk, embeddings)
        ],
    )


def _ensure_collection(client: QdrantClient, embedder: GemmaEmbedder) -> None:
    if client.collection_exists(QDRANT_COLLECTION_NAME):
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="paper_id",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="source",
        field_schema="keyword",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="authors",
        field_schema="text",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="title",
        field_schema="text",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="abstract",
        field_schema="text",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="update_date",
        field_schema="datetime",
    )
    client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="categories",
        field_schema="keyword",
    )


def _process_arxiv(
    client: QdrantClient,
    embedder: GemmaEmbedder,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    category: Optional[str] = None,
) -> None:
    fetcher = ArxivFetcher()
    chunk = []
    for paper in fetcher.fetch_papers(start_date, end_date, category):
        chunk.append(paper.model_dump())
        if len(chunk) >= CHUNK_SIZE:
            _upsert_chunk(client, embedder, chunk)
            chunk = []
    if chunk:
        _upsert_chunk(client, embedder, chunk)
        chunk = []


async def _process_biorxiv_medrxiv(
    client: QdrantClient,
    embedder: GemmaEmbedder,
    server: str,
    start_date: str,
    end_date: str,
    category: Optional[str] = None,
) -> None:
    fetcher = BiorxivFetcher(server=server)
    chunk = []

    async for paper in fetcher.fetch_papers(start_date, end_date, category):
        chunk.append(paper.model_dump())
        if len(chunk) >= CHUNK_SIZE:
            _upsert_chunk(client, embedder, chunk)
            chunk = []
    if chunk:
        _upsert_chunk(client, embedder, chunk)
        chunk = []


async def _process_chemrxiv(
    client: QdrantClient,
    embedder: GemmaEmbedder,
    search_term: Optional[str] = None,
    limit: int = 10000,
) -> None:
    fetcher = ChemrxivFetcher()
    chunk = []

    async for paper in fetcher.fetch_papers(limit=limit, search_term=search_term):
        chunk.append(paper.model_dump())
        if len(chunk) >= CHUNK_SIZE:
            _upsert_chunk(client, embedder, chunk)
            chunk = []
    if chunk:
        _upsert_chunk(client, embedder, chunk)
        chunk = []


async def index(
    source: str = "arxiv",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    query: Optional[str] = None,
    category: Optional[str] = None,
    max_results: int = 10000,
) -> None:
    embedder = GemmaEmbedder()
    client = QdrantClient(host="localhost", port=6333)
    _ensure_collection(client, embedder)

    if source == "arxiv":
        _process_arxiv(
            client, embedder, start_date=start_date, end_date=end_date, category=category
        )
    elif source == "biorxiv":
        if not start_date or not end_date:
            raise ValueError("biorxiv requires start_date and end_date (YYYY-MM-DD)")
        await _process_biorxiv_medrxiv(client, embedder, "biorxiv", start_date, end_date, category)
    elif source == "medrxiv":
        if not start_date or not end_date:
            raise ValueError("medrxiv requires start_date and end_date (YYYY-MM-DD)")
        await _process_biorxiv_medrxiv(client, embedder, "medrxiv", start_date, end_date, category)
    elif source == "chemrxiv":
        await _process_chemrxiv(client, embedder, search_term=query, limit=max_results)
    else:
        raise ValueError(f"Unknown source: {source}")


if __name__ == "__main__":
    fire.Fire(index)
