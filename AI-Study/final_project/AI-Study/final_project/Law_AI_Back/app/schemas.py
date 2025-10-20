from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    top_k: int = 6

class SearchHit(BaseModel):
    title: str
    snippet: str
    source: str | None = None

class SearchResponse(BaseModel):
    results: list[SearchHit]
