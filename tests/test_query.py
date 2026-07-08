import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
def client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
def patch_state():
    from unittest.mock import AsyncMock

    app.state.lightrag = AsyncMock()
    app.state.faiss = AsyncMock()
    yield


@pytest.mark.asyncio
async def test_lightrag_strategy_calls_lightrag(client):
    mock_lr = app.state.lightrag
    mock_lr.query.return_value = {
        "status": "ok",
        "response": "RAG is a technique...",
        "mode": "hybrid",
        "workspace": "kojo",
    }

    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "What is RAG?", "strategy": "lightrag"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["tenant_id"] == "kojo"
    assert body["strategy"] == "lightrag"
    assert body["answer"] == "RAG is a technique..."
    assert body["sources"] == []
    mock_lr.query.assert_awaited_once_with(
        workspace="kojo", query_text="What is RAG?", top_k=5
    )


@pytest.mark.asyncio
async def test_faiss_strategy_calls_faiss(client):
    mock_faiss = app.state.faiss
    mock_faiss.search.return_value = {
        "query": "What is RAG?",
        "top_k": 5,
        "results": [
            {
                "chunk": {
                    "content_chunk_id": "chunk-1",
                    "source_file": "doc.pdf",
                    "text": "RAG stands for...",
                },
                "score": 0.95,
            },
            {
                "chunk": {
                    "content_chunk_id": "chunk-2",
                    "source_file": "guide.pdf",
                },
                "score": 0.82,
            },
        ],
        "total_time_ms": 5.2,
    }

    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "What is RAG?", "strategy": "faiss"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["tenant_id"] == "kojo"
    assert body["strategy"] == "faiss"
    assert body["answer"] == ""
    assert len(body["sources"]) == 2
    assert body["sources"][0]["content_chunk_id"] == "chunk-1"
    assert body["sources"][0]["source_file"] == "doc.pdf"
    assert body["sources"][0]["score"] == 0.95
    assert body["sources"][1]["content_chunk_id"] == "chunk-2"
    assert body["sources"][1]["score"] == 0.82
    mock_faiss.search.assert_awaited_once_with(
        bot_id="kojo", query_text="What is RAG?", top_k=5
    )


@pytest.mark.asyncio
async def test_hybrid_strategy_returns_501(client):
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "test", "strategy": "hybrid"},
    )
    assert resp.status_code == 501
    assert resp.json()["status"] == "error"


@pytest.mark.asyncio
async def test_missing_query_rejected(client):
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "", "strategy": "lightrag"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_missing_tenant_id_rejected(client):
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "", "query": "hello", "strategy": "lightrag"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_lightrag_timeout_returns_504(client):
    from app.services.lightrag_client import LightRAGServiceError

    app.state.lightrag.query.side_effect = LightRAGServiceError(
        "timeout", status_code=504
    )
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "hello", "strategy": "lightrag"},
    )
    assert resp.status_code == 504


@pytest.mark.asyncio
async def test_lightrag_unavailable_returns_503(client):
    from app.services.lightrag_client import LightRAGServiceError

    app.state.lightrag.query.side_effect = LightRAGServiceError(
        "unavailable", status_code=503
    )
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "hello", "strategy": "lightrag"},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_lightrag_bad_response_returns_502(client):
    from app.services.lightrag_client import LightRAGServiceError

    app.state.lightrag.query.side_effect = LightRAGServiceError(
        "bad response", status_code=502
    )
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "hello", "strategy": "lightrag"},
    )
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_faiss_timeout_returns_504(client):
    from app.services.faiss_client import FaissServiceError

    app.state.faiss.search.side_effect = FaissServiceError("timeout", status_code=504)
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "hello", "strategy": "faiss"},
    )
    assert resp.status_code == 504


@pytest.mark.asyncio
async def test_faiss_unavailable_returns_503(client):
    from app.services.faiss_client import FaissServiceError

    app.state.faiss.search.side_effect = FaissServiceError(
        "unavailable", status_code=503
    )
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "hello", "strategy": "faiss"},
    )
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_faiss_bad_response_returns_502(client):
    from app.services.faiss_client import FaissServiceError

    app.state.faiss.search.side_effect = FaissServiceError(
        "bad response", status_code=502
    )
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "hello", "strategy": "faiss"},
    )
    assert resp.status_code == 502


@pytest.mark.asyncio
async def test_top_k_passed_through(client):
    mock_lr = app.state.lightrag
    mock_lr.query.return_value = {
        "status": "ok",
        "response": "answer",
        "mode": "hybrid",
        "workspace": "t1",
    }
    await client.post(
        "/api/v1/query",
        json={"tenant_id": "t1", "query": "q", "top_k": 3},
    )
    mock_lr.query.assert_awaited_once_with(workspace="t1", query_text="q", top_k=3)


@pytest.mark.asyncio
async def test_faiss_empty_results_returns_empty_sources(client):
    mock_faiss = app.state.faiss
    mock_faiss.search.return_value = {
        "query": "test",
        "top_k": 5,
        "results": [],
        "total_time_ms": 0.1,
    }
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "test", "strategy": "faiss"},
    )
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


@pytest.mark.asyncio
async def test_top_k_invalid_returns_422(client):
    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "hello", "top_k": 0},
    )
    assert resp.status_code == 422

    resp = await client.post(
        "/api/v1/query",
        json={"tenant_id": "kojo", "query": "hello", "top_k": 101},
    )
    assert resp.status_code == 422
