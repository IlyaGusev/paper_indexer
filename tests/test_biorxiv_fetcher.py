import pytest

from paper_indexer.fetchers.biorxiv import BiorxivFetcher


@pytest.mark.asyncio
async def test_biorxiv_fetcher() -> None:
    fetcher = BiorxivFetcher(server="biorxiv")
    papers = []
    async for paper in fetcher.fetch_papers("2024-01-01", "2024-01-02"):
        papers.append(paper)
        if len(papers) >= 5:
            break

    assert len(papers) > 0
    assert all(p.doi is not None for p in papers)


@pytest.mark.asyncio
async def test_medrxiv_fetcher() -> None:
    fetcher = BiorxivFetcher(server="medrxiv")
    papers = []
    async for paper in fetcher.fetch_papers("2024-01-01", "2024-01-02"):
        papers.append(paper)
        if len(papers) >= 5:
            break

    assert len(papers) > 0
    assert all(p.doi is not None for p in papers)
