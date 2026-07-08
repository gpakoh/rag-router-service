from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    tenant_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    strategy: str = "lightrag"
    top_k: int = Field(default=5, ge=1, le=100)


class SourceItem(BaseModel):
    content_chunk_id: str | None = None
    source_file: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    status: str
    tenant_id: str
    strategy: str
    answer: str
    sources: list[SourceItem] = []
