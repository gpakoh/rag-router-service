import pytest
import httpx

from app.services.lightrag_client import LightRAGClient, LightRAGServiceError


@pytest.fixture
def client() -> LightRAGClient:
    return LightRAGClient()


@pytest.fixture
def mock_respx():
    import respx

    with respx.mock(
        base_url="http://lightrag-service:8787",
        assert_all_called=False,
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_insert_text_returns_document_id(mock_respx, client):
    mock_respx.post("/api/v1/workspaces/kojo/insert/text").respond(
        200, json={"status": "ok", "document_id": "doc-123", "workspace": "kojo"}
    )
    doc_id = await client.insert_text("kojo", "hello world")
    assert doc_id == "doc-123"


@pytest.mark.asyncio
async def test_insert_text_returns_none_when_missing(mock_respx, client):
    mock_respx.post("/api/v1/workspaces/test/insert/text").respond(
        200, json={"status": "ok"}
    )
    doc_id = await client.insert_text("test", "data")
    assert doc_id is None


@pytest.mark.asyncio
async def test_insert_text_sends_source_file(mock_respx, client):
    route = mock_respx.post("/api/v1/workspaces/ws/insert/text").respond(
        200, json={"status": "ok", "document_id": "d1"}
    )
    await client.insert_text("ws", "text", source_file="doc.pdf")
    import json

    sent = json.loads(route.calls[0].request.content)
    assert sent["source_file"] == "doc.pdf"


@pytest.mark.asyncio
async def test_unavailable_raises_503(mock_respx, client):
    mock_respx.post(
        "/api/v1/workspaces/kojo/insert/text"
    ).side_effect = httpx.RequestError("connection refused")
    with pytest.raises(LightRAGServiceError) as exc:
        await client.insert_text("kojo", "text")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_timeout_raises_504(mock_respx, client):
    mock_respx.post(
        "/api/v1/workspaces/kojo/insert/text"
    ).side_effect = httpx.TimeoutException("timed out")
    with pytest.raises(LightRAGServiceError) as exc:
        await client.insert_text("kojo", "text")
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_upstream_400_maps_to_upstream_code(mock_respx, client):
    mock_respx.post("/api/v1/workspaces/kojo/insert/text").respond(
        400, json={"status": "error", "error": "text_empty", "message": "Text is empty"}
    )
    with pytest.raises(LightRAGServiceError) as exc:
        await client.insert_text("kojo", "")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upstream_500_maps_to_502(mock_respx, client):
    mock_respx.post("/api/v1/workspaces/kojo/insert/text").respond(500)
    with pytest.raises(LightRAGServiceError) as exc:
        await client.insert_text("kojo", "text")
    assert exc.value.status_code == 502
