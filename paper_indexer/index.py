import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import hashlib

import fire  # type: ignore
import kagglehub  # type: ignore
from tqdm import tqdm
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from paper_indexer.embedder import GemmaEmbedder

CHUNK_SIZE = 100
DATASET_NAME = "Cornell-University/arxiv"
KAGGLE_FILE_NAME = "arxiv-metadata-oai-snapshot.json"
QDRANT_COLLECTION_NAME = "papers"


def get_point_id(arxiv_id: str) -> int:
    md5 = hashlib.md5(arxiv_id.encode()).hexdigest()
    return int(md5[:16], 16) % (2**63 - 1)


class PaperVersion(BaseModel):  # type: ignore
    version: str
    created: str


class PaperMetadata(BaseModel):  # type: ignore
    arxiv_id: str
    authors: str
    title: str
    abstract: str
    update_date: str
    arxiv_categories: List[str]
    arxiv_versions: List[PaperVersion]
    arxiv_submitter: Optional[str] = None
    arxiv_comments: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None
    license: Optional[str] = None


def parse_record(record: Dict[str, Any]) -> PaperMetadata:
    record["arxiv_id"] = record.pop("id")
    record["arxiv_submitter"] = record.pop("submitter")
    record["arxiv_comments"] = record.pop("comments")
    record["journal_ref"] = record.pop("journal-ref")
    record["abstract"] = " ".join(record["abstract"].split())
    record["categories"] = record["categories"].split()
    record["arxiv_categories"] = record.pop("categories")
    record["arxiv_versions"] = record.pop("versions")
    record.pop("report-no")
    paper: PaperMetadata = PaperMetadata.model_validate(record)
    return paper


def index(min_update_date: Optional[str] = None, category: Optional[str] = None) -> None:
    embedder = GemmaEmbedder()
    client = QdrantClient(host="localhost", port=6333)

    if not client.collection_exists(QDRANT_COLLECTION_NAME):
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION_NAME,
            field_name="arxiv_id",
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
            field_name="arxiv_categories",
            field_schema="keyword",
        )

    path = Path(kagglehub.dataset_download(DATASET_NAME))
    full_path = path / KAGGLE_FILE_NAME
    chunk = []
    parsed_min_update_date = datetime.fromisoformat(min_update_date) if min_update_date else None
    with open(full_path) as r:
        for line in tqdm(r, desc="Processing papers"):
            record = json.loads(line)
            paper = parse_record(record)
            paper_update_date = datetime.fromisoformat(paper.update_date)
            if parsed_min_update_date and paper_update_date < parsed_min_update_date:
                print(
                    "Skipping paper",
                    paper.arxiv_id,
                    "because it is older than",
                    parsed_min_update_date,
                )
                continue
            if category and category not in paper.arxiv_categories:
                continue
            chunk.append(paper.model_dump())
            if len(chunk) != CHUNK_SIZE:
                continue
            embeddings = embedder.encode_documents(chunk)
            client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=get_point_id(paper["arxiv_id"]),
                        vector=embedding.tolist(),
                        payload=paper,
                    )
                    for paper, embedding in zip(chunk, embeddings)
                ],
            )
            print("Processed chunk")
            chunk = []


if __name__ == "__main__":
    fire.Fire(index)
