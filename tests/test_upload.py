from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def client():
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.fixture
def mock_services():
    with (
        patch("app.main.FileSecurityClient") as mock_fsec_cls,
        patch("app.main.LightRAGClient") as mock_lr_cls,
        patch("app.main.FaissClient") as mock_faiss_cls,
    ):
        mock_fsec = AsyncMock()
        mock_lr = AsyncMock()
        mock_faiss = AsyncMock()
        mock_fsec_cls.return_value = mock_fsec
        mock_lr_cls.return_value = mock_lr
        mock_faiss_cls.return_value = mock_faiss
        app.state.file_security = mock_fsec
        app.state.lightrag = mock_lr
        app.state.faiss = mock_faiss
        yield mock_fsec, mock_lr, mock_faiss


@pytest.mark.asyncio
async def test_upload_safe_file_returns_ok(client, mock_services):
    mock_fsec, mock_lr, _mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"safe content"
    mock_lr.insert_text.return_value = "doc-42"

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo"},
        files={"file": ("test.txt", b"hello world")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["tenant_id"] == "kojo"
    assert body["strategy"] == "lightrag"
    assert body["document_id"] == "doc-42"


@pytest.mark.asyncio
async def test_upload_invokes_security_parse_insert(client, mock_services):
    mock_fsec, mock_lr, _mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"safe content"
    mock_lr.insert_text.return_value = "doc-1"

    await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "t1"},
        files={"file": ("a.txt", b"hello")},
    )
    mock_fsec.scan_and_get_safe_artifact.assert_awaited_once()
    mock_lr.insert_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_blocked_file_returns_422(client, mock_services):
    mock_fsec, mock_lr, _mock_faiss = mock_services
    from app.exceptions import FileBlockedError

    mock_fsec.scan_and_get_safe_artifact.side_effect = FileBlockedError(
        "EICAR blocked", reason="EICAR-Test-File"
    )

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo"},
        files={"file": ("eicar.txt", b"X5O!P%@AP")},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "blocked"
    mock_lr.insert_text.assert_not_called()


@pytest.mark.asyncio
async def test_security_timeout_returns_504(client, mock_services):
    mock_fsec, mock_lr, _mock_faiss = mock_services
    from app.exceptions import FileSecurityTimeoutError

    mock_fsec.scan_and_get_safe_artifact.side_effect = FileSecurityTimeoutError()

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo"},
        files={"file": ("slow.txt", b"data")},
    )
    assert resp.status_code == 504
    mock_lr.insert_text.assert_not_called()


@pytest.mark.asyncio
async def test_security_unavailable_returns_503(client, mock_services):
    mock_fsec, mock_lr, _mock_faiss = mock_services
    from app.exceptions import FileSecurityUnavailableError

    mock_fsec.scan_and_get_safe_artifact.side_effect = FileSecurityUnavailableError()

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo"},
        files={"file": ("f.txt", b"data")},
    )
    assert resp.status_code == 503
    mock_lr.insert_text.assert_not_called()


@pytest.mark.asyncio
async def test_parser_error_returns_422(client, mock_services):
    mock_fsec, mock_lr, _mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"\x00\xff"

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo"},
        files={"file": ("corrupt.pdf", b"\x00\xff")},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "error"
    mock_lr.insert_text.assert_not_called()


@pytest.mark.asyncio
async def test_lightrag_unavailable_returns_503(client, mock_services):
    mock_fsec, mock_lr, _mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"safe"
    from app.services.lightrag_client import LightRAGServiceError

    mock_lr.insert_text.side_effect = LightRAGServiceError(
        "LightRAG unreachable", status_code=503
    )

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo"},
        files={"file": ("f.txt", b"hello")},
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "error"


@pytest.mark.asyncio
async def test_hybrid_strategy_returns_501(client, mock_services):
    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo", "strategy": "hybrid"},
        files={"file": ("f.txt", b"data")},
    )
    assert resp.status_code == 501


@pytest.mark.asyncio
async def test_missing_file_returns_422(client, mock_services):
    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_missing_tenant_id_returns_422(client, mock_services):
    resp = await client.post(
        "/api/v1/upload/file",
        files={"file": ("f.txt", b"data")},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_empty_file_returns_400(client, mock_services):
    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo"},
        files={"file": ("empty.txt", b"")},
    )
    assert resp.status_code == 400


# --- FAISS strategy ---


@pytest.mark.asyncio
async def test_faiss_calls_security_parse_rebuild(client, mock_services):
    mock_fsec, _mock_lr, mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"safe"
    mock_faiss.rebuild.return_value = {
        "status": "ok",
        "bot_id": "kojo",
        "chunk_profile_id": "default",
        "total_chunks": 3,
        "index_path": "/data/faiss_index/kojo/default",
    }

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo", "strategy": "faiss"},
        files={"file": ("doc.txt", b"hello world")},
    )
    assert resp.status_code == 200
    mock_fsec.scan_and_get_safe_artifact.assert_awaited_once()
    mock_faiss.rebuild.assert_awaited_once()
    body = resp.json()
    assert body["status"] == "ok"
    assert body["strategy"] == "faiss"
    assert body["document_id"]
    _mock_lr.insert_text.assert_not_called()


@pytest.mark.asyncio
async def test_faiss_unavailable_returns_503(client, mock_services):
    mock_fsec, _mock_lr, mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"safe"
    from app.services.faiss_client import FaissServiceError

    mock_faiss.rebuild.side_effect = FaissServiceError(
        "FAISS unreachable", status_code=503
    )

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo", "strategy": "faiss"},
        files={"file": ("f.txt", b"hello")},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_faiss_timeout_returns_504(client, mock_services):
    mock_fsec, _mock_lr, mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"safe"
    from app.services.faiss_client import FaissServiceError

    mock_faiss.rebuild.side_effect = FaissServiceError("FAISS timeout", status_code=504)

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo", "strategy": "faiss"},
        files={"file": ("f.txt", b"hello")},
    )
    assert resp.status_code == 504


@pytest.mark.asyncio
async def test_faiss_protocol_error_returns_502(client, mock_services):
    mock_fsec, _mock_lr, mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"safe"
    from app.services.faiss_client import FaissServiceError

    mock_faiss.rebuild.side_effect = FaissServiceError(
        "FAISS bad response", status_code=502
    )

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo", "strategy": "faiss"},
        files={"file": ("f.txt", b"hello")},
    )
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_faiss_blocked_file_stops_before_rebuild(client, mock_services):
    mock_fsec, _mock_lr, mock_faiss = mock_services
    from app.exceptions import FileBlockedError

    mock_fsec.scan_and_get_safe_artifact.side_effect = FileBlockedError(
        "blocked", reason="virus"
    )

    resp = await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "kojo", "strategy": "faiss"},
        files={"file": ("bad.txt", b"virus")},
    )
    assert resp.status_code == 422
    mock_faiss.rebuild.assert_not_called()


@pytest.mark.asyncio
async def test_lightrag_not_called_for_faiss(client, mock_services):
    mock_fsec, mock_lr, mock_faiss = mock_services
    mock_fsec.scan_and_get_safe_artifact.return_value = b"safe"
    mock_faiss.rebuild.return_value = {
        "status": "ok",
        "bot_id": "t1",
        "chunk_profile_id": "default",
        "index_path": "/data/faiss_index/t1/default",
    }

    await client.post(
        "/api/v1/upload/file",
        data={"tenant_id": "t1", "strategy": "faiss"},
        files={"file": ("doc.txt", b"data")},
    )
    mock_lr.insert_text.assert_not_called()
    mock_faiss.rebuild.assert_awaited_once()
