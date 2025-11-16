import asyncio
import httpx
from typing import Optional, Dict, Any, AsyncIterator
from tqdm import tqdm

from paper_indexer.utils import retry_request
from paper_indexer.models import PaperMetadata


BASE_URL = "https://api.biorxiv.org"


class BiorxivFetcher:
    def __init__(
        self,
        server: str = "biorxiv",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        rate_limit_delay: float = 0.1,
        base_url: str = BASE_URL,
    ) -> None:
        assert server in ["biorxiv", "medrxiv"], f"Invalid server: {server}"
        self.server = server
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.rate_limit_delay = rate_limit_delay
        self.base_url = base_url

    def _parse_record(self, record: Dict[str, Any], source: str) -> PaperMetadata:
        doi = record.get("preprint_doi", record.get("doi", ""))
        title = record.get("preprint_title", record.get("title", ""))
        abstract = record.get("preprint_abstract", record.get("abstract", ""))
        authors = record.get("preprint_authors", record.get("authors", ""))
        date = record.get("preprint_date", record.get("date", ""))
        category = record.get("preprint_category", record.get("category", ""))

        return PaperMetadata(
            paper_id=doi,
            source=source,
            title=title,
            abstract=" ".join(str(abstract).split()),
            authors=authors,
            update_date=date,
            categories=[category] if category else [],
            doi=doi,
            license=record.get("license"),
        )

    async def fetch_papers(
        self,
        start_date: str,
        end_date: str,
        category: Optional[str] = None,
    ) -> AsyncIterator[PaperMetadata]:
        cursor = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with tqdm(desc=f"Fetching {self.server} papers") as pbar:
                while True:
                    url = f"{self.base_url}/pubs/{self.server}/{start_date}/{end_date}/{cursor}"
                    params = {}
                    if category:
                        params["category"] = category

                    try:
                        response = await retry_request(
                            client=client,
                            method="GET",
                            url=url,
                            params=params,
                            max_retries=self.max_retries,
                            backoff_factor=self.backoff_factor,
                        )
                        data = response.json()
                    except Exception as e:
                        print(f"Error fetching {self.server} data: {e}")
                        break

                    collection = data.get("collection", [])
                    if not collection:
                        break

                    for paper in collection:
                        yield self._parse_record(paper, self.server)
                        pbar.update(1)

                    messages = data.get("messages", [])
                    cursor_message = next(
                        (m for m in messages if m.get("type") == "cursor_value"), None
                    )
                    if not cursor_message:
                        break

                    cursor = int(cursor_message.get("cursor", 0))
                    await asyncio.sleep(self.rate_limit_delay)
