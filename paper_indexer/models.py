from typing import List, Optional
from pydantic import BaseModel


class PaperVersion(BaseModel):  # type: ignore
    version: str
    created: str


class PaperMetadata(BaseModel):  # type: ignore
    paper_id: str
    source: str
    title: str
    abstract: str
    authors: str
    update_date: str
    categories: List[str] = []
    doi: Optional[str] = None
    journal_ref: Optional[str] = None
    license: Optional[str] = None
    arxiv_id: Optional[str] = None
    arxiv_versions: Optional[List[PaperVersion]] = None
    arxiv_submitter: Optional[str] = None
    arxiv_comments: Optional[str] = None
