import pytest
import httpx

from app.services.faiss_client import FaissClient, FaissServiceError


@pytest.fixture
def client() -> FaissClient:
    return FaissClient()


@pytest.fixture
def mock_respx():
    import respx

    with respx.mock(
        base_url="http://rag-faiss-service:8020",
        assert_all_called=False,
    ) as mock:
        yield mock


@pytest.mark.asyncio
async def test_rebuild_returns_result_dict(mock_respx, client):
    mock_respx.post("/api/v1/index/rebuild").respond(
        200,
        json={
            "status": "ok",
            "bot_id": "kojo",
            "chunk_profile_id": "default",
            "total_chunks": 5,
            "total_vectors": 5,
            "duration_seconds": 0.5,
            "index_path": "/data/faiss_index/kojo/default",
            "is_gpu": False,
        },
    )
    result = await client.rebuild("kojo")
    assert result["status"] == "ok"
    assert result["total_chunks"] == 5


@pytest.mark.asyncio
async def test_rebuild_passes_chunk_profile(mock_respx, client):
    route = mock_respx.post("/api/v1/index/rebuild").respond(
        200, json={"status": "ok", "bot_id": "ws", "chunk_profile_id": "custom"}
    )
    await client.rebuild("ws", chunk_profile_id="custom")
    import json

    sent = json.loads(route.calls[0].request.content)
    assert sent["bot_id"] == "ws"
    assert sent["chunk_profile_id"] == "custom"


@pytest.mark.asyncio
async def test_unavailable_raises_503(mock_respx, client):
    mock_respx.post("/api/v1/index/rebuild").side_effect = httpx.RequestError(
        "connection refused"
    )
    with pytest.raises(FaissServiceError) as exc:
        await client.rebuild("kojo")
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_timeout_raises_504(mock_respx, client):
    mock_respx.post("/api/v1/index/rebuild").side_effect = httpx.TimeoutException(
        "timed out"
    )
    with pytest.raises(FaissServiceError) as exc:
        await client.rebuild("kojo")
    assert exc.value.status_code == 504


@pytest.mark.asyncio
async def test_upstream_503_maps_to_503(mock_respx, client):
    mock_respx.post("/api/v1/index/rebuild").respond(
        503, json={"detail": "FAISS not initialized"}
    )
    with pytest.raises(FaissServiceError) as exc:
        await client.rebuild("kojo")
    assert exc.value.status_code == 503
    assert "FAISS not initialized" in exc.value.message


@pytest.mark.asyncio
async def test_upstream_400_maps_to_upstream_code(mock_respx, client):
    mock_respx.post("/api/v1/index/rebuild").respond(
        400, json={"detail": "bad request"}
    )
    with pytest.raises(FaissServiceError) as exc:
        await client.rebuild("kojo")
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upstream_500_maps_to_502(mock_respx, client):
    mock_respx.post("/api/v1/index/rebuild").respond(500)
    with pytest.raises(FaissServiceError) as exc:
        await client.rebuild("kojo")
    assert exc.value.status_code == 502
