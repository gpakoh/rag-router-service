import logging

from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.models.query import QueryRequest, QueryResponse, SourceItem
from app.services.faiss_client import FaissServiceError
from app.services.lightrag_client import LightRAGServiceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["query"])

SUPPORTED_STRATEGIES = {"lightrag", "faiss"}


def _get_clients(request: Request):
    return (
        request.app.state.lightrag,
        request.app.state.faiss,
    )


@router.post("/api/v1/query")
async def query(request: Request, body: QueryRequest):
    if body.strategy not in SUPPORTED_STRATEGIES:
        return JSONResponse(
            status_code=501,
            content={
                "status": "error",
                "message": f"Strategy '{body.strategy}' not implemented. Supported: {SUPPORTED_STRATEGIES}",
            },
        )

    lightrag, faiss = _get_clients(request)

    if body.strategy == "lightrag":
        try:
            result = await lightrag.query(
                workspace=body.tenant_id,
                query_text=body.query,
                top_k=body.top_k,
            )
        except LightRAGServiceError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"status": "error", "message": e.message},
            )

        return QueryResponse(
            status="ok",
            tenant_id=body.tenant_id,
            strategy="lightrag",
            answer=result.get("response", ""),
        )

    if body.strategy == "faiss":
        try:
            result = await faiss.search(
                bot_id=body.tenant_id,
                query_text=body.query,
                top_k=body.top_k,
            )
        except FaissServiceError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"status": "error", "message": e.message},
            )

        sources = _faiss_sources(result)

        return QueryResponse(
            status="ok",
            tenant_id=body.tenant_id,
            strategy="faiss",
            answer="",
            sources=sources,
        )


def _faiss_sources(result: dict) -> list[SourceItem]:
    sources: list[SourceItem] = []
    for item in result.get("results", []):
        chunk = item.get("chunk", {})
        sources.append(
            SourceItem(
                content_chunk_id=chunk.get("content_chunk_id"),
                source_file=chunk.get("source_file"),
                score=item.get("score"),
            )
        )
    return sources
