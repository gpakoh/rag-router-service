import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_returns_200(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_returns_expected_shape(client: AsyncClient) -> None:
    resp = await client.get("/health")
    body = resp.json()
    assert body["status"] == "ok"
    assert "backends" in body
    for bk in ("file_security", "lightrag", "faiss"):
        assert bk in body["backends"]
