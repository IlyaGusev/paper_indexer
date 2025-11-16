import pytest
from paper_indexer.fetchers.chemrxiv import ChemrxivFetcher


@pytest.mark.asyncio
async def test_chemrxiv_fetcher() -> None:
    fetcher = ChemrxivFetcher()
    papers = []
    async for paper in fetcher.fetch_papers(limit=10):
        papers.append(paper)

    assert len(papers) > 0
