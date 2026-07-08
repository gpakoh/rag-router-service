import pytest
import httpx

from app.exceptions import (
    FileBlockedError,
    FileSecurityProtocolError,
    FileSecurityTimeoutError,
    FileSecurityUnavailableError,
)
from app.services.file_security_client import FileSecurityClient


@pytest.fixture
def client() -> FileSecurityClient:
    return FileSecurityClient()


@pytest.fixture
def mock_respx():
    import respx

    with respx.mock(
        base_url="http://file-security-service:8000",
        assert_all_called=False,
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_submit_sends_multipart_with_tenant_id(mock_respx, client):
    route = mock_respx.post("/api/v1/files/submit").respond(
        201, json={"id": "uuid-123", "status": "quarantined"}
    )
    result = await client.submit_file(b"content", "test.txt", "tenant-1")
    assert result["id"] == "uuid-123"
    assert route.called
    request = route.calls[0].request
    assert "tenant_id" in str(request.content)


@pytest.mark.asyncio
async def test_safe_flow_submit_poll_artifact(mock_respx, client):
    file_id = "550e8400-e29b-41d4-a716-446655440000"

    mock_respx.post("/api/v1/files/submit").respond(
        201, json={"id": file_id, "status": "quarantined"}
    )
    mock_respx.get(f"/api/v1/files/{file_id}").respond(
        200,
        json={
            "id": file_id,
            "security_status": "safe_for_rag",
            "scans": [],
        },
    )
    mock_respx.get(f"/api/v1/files/{file_id}/safe-artifact").respond(
        200, content=b"safe-bytes"
    )

    result = await client.scan_and_get_safe_artifact(b"content", "test.txt", "tenant-1")
    assert result == b"safe-bytes"


@pytest.mark.asyncio
async def test_blocked_submit_raises_file_blocked(mock_respx, client):
    mock_respx.post("/api/v1/files/submit").respond(
        422,
        json={"detail": "Extension .exe is blocked"},
    )
    with pytest.raises(FileBlockedError) as exc:
        await client.submit_file(b"exe", "malware.exe", "tenant-1")
    assert "blocked" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_blocked_poll_raises_file_blocked(mock_respx, client):
    file_id = "uuid-blocked"
    mock_respx.post("/api/v1/files/submit").respond(
        201, json={"id": file_id, "status": "quarantined"}
    )
    mock_respx.get(f"/api/v1/files/{file_id}").respond(
        200,
        json={
            "id": file_id,
            "security_status": "blocked",
            "scans": [{"scanner": "clamav", "verdict": "EICAR-Test-File"}],
        },
    )
    with pytest.raises(FileBlockedError) as exc:
        await client.scan_and_get_safe_artifact(
            b"X5O!P%@AP[4\\PZX54(P^)7CC)7}", "eicar.txt", "tenant-1"
        )
    assert "EICAR" in str(exc.value.reason)


@pytest.mark.asyncio
async def test_timeout_raises_timeout_error(mock_respx, client):
    file_id = "uuid-slow"
    mock_respx.post("/api/v1/files/submit").respond(
        201, json={"id": file_id, "status": "quarantined"}
    )
    mock_respx.get(f"/api/v1/files/{file_id}").respond(
        200,
        json={
            "id": file_id,
            "security_status": "quarantined",
            "scans": [],
        },
    )

    with pytest.raises(FileSecurityTimeoutError):
        await client.wait_for_safe(file_id, max_wait=0.1, poll_interval=0.05)


@pytest.mark.asyncio
async def test_unavailable_submit_503(mock_respx, client):
    mock_respx.post("/api/v1/files/submit").respond(503)
    with pytest.raises(FileSecurityUnavailableError):
        await client.submit_file(b"data", "f.txt", "t")


@pytest.mark.asyncio
async def test_unavailable_network_error(mock_respx, client):
    mock_respx.post("/api/v1/files/submit").side_effect = httpx.RequestError(
        "connection refused"
    )
    with pytest.raises(FileSecurityUnavailableError):
        await client.submit_file(b"data", "f.txt", "t")


@pytest.mark.asyncio
async def test_malformed_response_falls_back_to_detail(mock_respx, client):
    mock_respx.post("/api/v1/files/submit").respond(200, text="not-json")
    result = await client.submit_file(b"data", "f.txt", "t")
    assert result["detail"] == "not-json"


@pytest.mark.asyncio
async def test_get_safe_artifact_returns_bytes(mock_respx, client):
    file_id = "uuid-artifact"
    mock_respx.get(f"/api/v1/files/{file_id}/safe-artifact").respond(
        200, content=b"artifact-data"
    )
    result = await client.get_safe_artifact(file_id)
    assert result == b"artifact-data"


@pytest.mark.asyncio
async def test_get_safe_artifact_404(mock_respx, client):
    mock_respx.get("/api/v1/files/unknown/safe-artifact").respond(404)
    with pytest.raises(FileSecurityProtocolError):
        await client.get_safe_artifact("unknown")


@pytest.mark.asyncio
async def test_get_safe_artifact_409(mock_respx, client):
    mock_respx.get("/api/v1/files/pending/safe-artifact").respond(409)
    with pytest.raises(FileSecurityProtocolError):
        await client.get_safe_artifact("pending")


@pytest.mark.asyncio
async def test_submit_413_raises_protocol_error(mock_respx, client):
    mock_respx.post("/api/v1/files/submit").respond(413)
    with pytest.raises(FileSecurityProtocolError) as exc:
        await client.submit_file(b"x" * 999_999_999, "big.txt", "t")
    assert exc.value.status_code == 413
