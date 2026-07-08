import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class FaissServiceError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class FaissClient:
    def __init__(self) -> None:
        self.base_url = settings.faiss_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=settings.faiss_timeout
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def search(
        self,
        bot_id: str,
        query_text: str,
        chunk_profile_id: str = "default",
        top_k: int = 5,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "bot_id": bot_id,
            "chunk_profile_id": chunk_profile_id,
            "query": query_text,
            "top_k": top_k,
        }

        try:
            resp = await self._client.post("/api/v1/search", json=payload)
        except httpx.TimeoutException as e:
            raise FaissServiceError(
                f"FAISS search timeout: {e}", status_code=504
            ) from e
        except httpx.RequestError as e:
            raise FaissServiceError(
                f"FAISS search unreachable: {e}", status_code=503
            ) from e

        if resp.status_code == 503:
            body = _parse_body(resp)
            raise FaissServiceError(
                body.get("detail", "FAISS not initialized"), status_code=503
            )

        if resp.status_code == 404:
            body = _parse_body(resp)
            raise FaissServiceError(
                body.get("detail", "FAISS index not found"), status_code=404
            )

        if resp.status_code >= 400:
            body = _parse_body(resp)
            mapped = 502 if resp.status_code >= 500 else resp.status_code
            raise FaissServiceError(
                body.get("detail", f"FAISS search failed: {resp.status_code}"),
                status_code=mapped,
            )

        return _parse_body(resp)

    async def rebuild(
        self,
        bot_id: str,
        chunk_profile_id: str = "default",
    ) -> dict[str, Any]:
        payload: dict[str, str] = {
            "bot_id": bot_id,
            "chunk_profile_id": chunk_profile_id,
        }

        try:
            resp = await self._client.post("/api/v1/index/rebuild", json=payload)
        except httpx.TimeoutException as e:
            raise FaissServiceError(
                f"FAISS service timeout: {e}", status_code=504
            ) from e
        except httpx.RequestError as e:
            raise FaissServiceError(
                f"FAISS service unreachable: {e}", status_code=503
            ) from e

        if resp.status_code == 503:
            body = _parse_body(resp)
            raise FaissServiceError(
                body.get("detail", "FAISS not initialized"), status_code=503
            )

        if resp.status_code >= 400:
            body = _parse_body(resp)
            mapped = 502 if resp.status_code >= 500 else resp.status_code
            raise FaissServiceError(
                body.get("detail", f"FAISS rebuild failed: {resp.status_code}"),
                status_code=mapped,
            )

        return _parse_body(resp)


def _parse_body(resp: httpx.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"detail": resp.text}
