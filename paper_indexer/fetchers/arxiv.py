import json
from pathlib import Path
from typing import Optional, Dict, Any, Generator
from datetime import datetime

import kagglehub  # type: ignore
from tqdm import tqdm

from paper_indexer.models import PaperMetadata, PaperVersion

DATASET_NAME = "Cornell-University/arxiv"
KAGGLE_FILE_NAME = "arxiv-metadata-oai-snapshot.json"


class ArxivFetcher:
    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        kaggle_file_name: str = KAGGLE_FILE_NAME,
    ) -> None:
        self.dataset_name = dataset_name
        self.kaggle_file_name = kaggle_file_name

    def _parse_record(self, record: Dict[str, Any]) -> PaperMetadata:
        return PaperMetadata(
            paper_id=record["id"],
            arxiv_id=record["id"],
            source="arxiv",
            title=record["title"],
            abstract=" ".join(record["abstract"].split()),
            authors=record["authors"],
            update_date=record["update_date"],
            categories=record["categories"].split(),
            doi=record.get("doi"),
            journal_ref=record.get("journal-ref"),
            license=record.get("license"),
            arxiv_versions=[PaperVersion(**v) for v in record.get("versions", [])],
            arxiv_submitter=record.get("submitter"),
            arxiv_comments=record.get("comments"),
        )

    def fetch_papers(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 10000,
    ) -> Generator[PaperMetadata, None, None]:
        path = Path(kagglehub.dataset_download(self.dataset_name))
        full_path = path / self.kaggle_file_name
        parsed_start_date = datetime.fromisoformat(start_date) if start_date else None
        parsed_end_date = datetime.fromisoformat(end_date) if end_date else None
        count = 0

        with open(full_path) as r:
            for line in tqdm(r, desc="Processing ArXiv papers"):
                record = json.loads(line)
                paper = self._parse_record(record)

                if parsed_start_date:
                    paper_update_date = datetime.fromisoformat(paper.update_date)
                    if paper_update_date < parsed_start_date:
                        continue
                if parsed_end_date:
                    paper_update_date = datetime.fromisoformat(paper.update_date)
                    if paper_update_date > parsed_end_date:
                        continue

                if category and category not in paper.categories:
                    continue

                yield paper
                count += 1
                if count >= limit:
                    break
