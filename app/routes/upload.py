import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.exceptions import (
    FileBlockedError,
    FileSecurityProtocolError,
    FileSecurityTimeoutError,
    FileSecurityUnavailableError,
    ParserError,
)
from app.models.upload import UploadResponse
from app.services.faiss_client import FaissServiceError
from app.services.lightrag_client import LightRAGServiceError
from app.services.parser import parse_single_file_sync

logger = logging.getLogger(__name__)

router = APIRouter(tags=["upload"])

SUPPORTED_STRATEGIES = {"lightrag", "faiss"}


def _get_clients(request: Request):
    return (
        request.app.state.file_security,
        request.app.state.lightrag,
        request.app.state.faiss,
    )


@router.post("/api/v1/upload/file")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    strategy: str = Form("lightrag"),
    metadata: str | None = Form(None),
):
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="File is required")

    if not tenant_id or not tenant_id.strip():
        raise HTTPException(status_code=400, detail="tenant_id is required")

    if strategy not in SUPPORTED_STRATEGIES:
        return JSONResponse(
            status_code=501,
            content={
                "status": "error",
                "message": f"Strategy '{strategy}' not implemented. Supported: {SUPPORTED_STRATEGIES}",
            },
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    fsec, lightrag, faiss = _get_clients(request)

    # step 1: security scan
    try:
        safe_bytes = await fsec.scan_and_get_safe_artifact(
            content, file.filename, tenant_id
        )
    except FileBlockedError as e:
        return JSONResponse(
            status_code=422,
            content={
                "status": "blocked",
                "reason": e.reason or "Blocked by security policy",
            },
        )
    except FileSecurityTimeoutError:
        return JSONResponse(
            status_code=504,
            content={"status": "error", "message": "File scan timed out"},
        )
    except FileSecurityUnavailableError:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "File security service unavailable"},
        )
    except FileSecurityProtocolError as e:
        status = e.status_code or 502
        return JSONResponse(
            status_code=status,
            content={"status": "error", "message": e.message},
        )

    # step 2: parse
    try:
        text = parse_single_file_sync(file.filename, safe_bytes)
    except ParserError as e:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "message": str(e)},
        )

    # step 3: insert into backend
    if strategy == "lightrag":
        try:
            doc_id = await lightrag.insert_text(
                workspace=tenant_id,
                text=text,
                source_file=file.filename,
            )
        except LightRAGServiceError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"status": "error", "message": e.message},
            )
    elif strategy == "faiss":
        try:
            result = await faiss.rebuild(
                bot_id=tenant_id,
            )
            doc_id = _faiss_document_id(result)
        except FaissServiceError as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"status": "error", "message": e.message},
            )

    return UploadResponse(
        status="ok",
        tenant_id=tenant_id,
        strategy=strategy,
        document_id=doc_id,
    )


def _faiss_document_id(result: dict) -> str:
    return result.get(
        "index_path",
        f"{result.get('bot_id', '?')}/{result.get('chunk_profile_id', '?')}",
    )
