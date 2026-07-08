import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LightRAGServiceError(Exception):
    def __init__(self, message: str, status_code: int = 502) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class LightRAGClient:
    def __init__(self) -> None:
        self.base_url = settings.lightrag_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=settings.lightrag_timeout
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def insert_text(
        self,
        workspace: str,
        text: str,
        source_file: str | None = None,
    ) -> str | None:
        payload: dict[str, Any] = {"text": text}
        if source_file:
            payload["source_file"] = source_file

        try:
            resp = await self._client.post(
                f"/api/v1/workspaces/{workspace}/insert/text",
                json=payload,
            )
        except httpx.TimeoutException as e:
            raise LightRAGServiceError(
                f"LightRAG service timeout: {e}", status_code=504
            ) from e
        except httpx.RequestError as e:
            raise LightRAGServiceError(
                f"LightRAG service unreachable: {e}", status_code=503
            ) from e

        if resp.status_code >= 400:
            body = _parse_body(resp)
            upstream_code = resp.status_code
            mapped = 502 if upstream_code >= 500 else upstream_code
            raise LightRAGServiceError(
                body.get("message", f"LightRAG insert failed: {upstream_code}"),
                status_code=mapped,
            )

        body = _parse_body(resp)
        return body.get("document_id")


def _parse_body(resp: httpx.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"detail": resp.text}
