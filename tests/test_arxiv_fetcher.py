from paper_indexer.fetchers.arxiv import ArxivFetcher


def test_arxiv_fetcher() -> None:
    fetcher = ArxivFetcher()
    papers = []
    for paper in fetcher.fetch_papers(limit=10):
        papers.append(paper)
    assert len(papers) > 0
    assert all(p.paper_id is not None for p in papers)
    assert all(p.categories is not None for p in papers)
    assert all(p.arxiv_versions is not None for p in papers)
    assert all(p.arxiv_submitter is not None for p in papers)
