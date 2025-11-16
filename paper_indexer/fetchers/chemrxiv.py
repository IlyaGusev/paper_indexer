import asyncio
from typing import Dict, Any, AsyncIterator, Optional, Union

import httpx
from tqdm import tqdm

from paper_indexer.utils import retry_request
from paper_indexer.models import PaperMetadata

BASE_URL = "https://chemrxiv.org/engage/chemrxiv/public-api/v1"


class ChemrxivFetcher:
    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        rate_limit_delay: float = 0.1,
        base_url: str = BASE_URL,
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.rate_limit_delay = rate_limit_delay
        self.base_url = base_url

    def _parse_record(self, record: Dict[str, Any]) -> PaperMetadata:
        item = record.get("item", record)
        doi = item.get("doi", "")
        title = item.get("title", "")
        abstract = item.get("description", item.get("abstract", ""))
        authors_list = item.get("authors", [])
        authors = ", ".join([a.get("name", "") for a in authors_list]) if authors_list else ""
        pub_date = item.get("publishedDate", "")

        categories = []
        for category in item.get("categories", []):
            if category.get("name"):
                categories.append(category.get("name"))
        return PaperMetadata(
            paper_id=doi,
            source="chemrxiv",
            title=title,
            abstract=" ".join(str(abstract).split()),
            authors=authors,
            update_date=pub_date,
            categories=categories,
            doi=doi,
            license=(
                item.get("license", {}).get("name")
                if isinstance(item.get("license"), dict)
                else None
            ),
        )

    async def fetch_papers(
        self,
        limit: int = 10000,
        page_size: int = 100,
        search_term: Optional[str] = None,
    ) -> AsyncIterator[PaperMetadata]:
        cursor = 0

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with tqdm(desc="Fetching ChemRxiv papers") as pbar:
                while cursor < limit:
                    url = f"{self.base_url}/items"
                    params: Dict[str, Union[int, str]] = {
                        "limit": min(page_size, limit - cursor),
                        "skip": cursor,
                    }

                    if search_term:
                        params["term"] = search_term

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
                        print(f"Error fetching ChemRxiv data: {e}")
                        break

                    items = data.get("itemHits", [])
                    if not items:
                        break

                    for item in items:
                        yield self._parse_record(item)
                        pbar.update(1)

                    cursor += len(items)

                    if len(items) < page_size:
                        break

                    await asyncio.sleep(self.rate_limit_delay)
