from typing import Optional, List

import fire  # type: ignore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchText,
    MatchValue,
    DatetimeRange,
)

from paper_indexer.embedder import GemmaEmbedder


def query(
    query: str,
    arxiv_id: Optional[str] = None,
    authors: Optional[str] = None,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
    min_update_date: Optional[str] = None,
    max_update_date: Optional[str] = None,
    arxiv_categories: Optional[List[str]] = None,
    limit: int = 20,
) -> None:
    embedder = GemmaEmbedder()
    client = QdrantClient(host="localhost", port=6333)

    must_conditions = []

    if arxiv_id:
        must_conditions.append(FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id)))

    if authors:
        must_conditions.append(FieldCondition(key="authors", match=MatchText(text=authors)))

    if title:
        must_conditions.append(FieldCondition(key="title", match=MatchText(text=title)))

    if abstract:
        must_conditions.append(FieldCondition(key="abstract", match=MatchText(text=abstract)))

    if min_update_date or max_update_date:
        range_params = {}
        if min_update_date:
            range_params["gte"] = min_update_date
        if max_update_date:
            range_params["lte"] = max_update_date
        must_conditions.append(
            FieldCondition(key="update_date", range=DatetimeRange(**range_params))
        )

    if arxiv_categories:
        must_conditions.append(
            FieldCondition(key="arxiv_categories", match=MatchAny(any=arxiv_categories))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name="papers",
        query=embedder.encode_query(query),
        query_filter=query_filter,
        limit=limit,
    )
    print(results)


if __name__ == "__main__":
    fire.Fire(query)
