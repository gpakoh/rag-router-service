import asyncio
import logging
from typing import Any

import httpx

from app.config import settings
from app.exceptions import (
    FileBlockedError,
    FileSecurityProtocolError,
    FileSecurityTimeoutError,
    FileSecurityUnavailableError,
)

logger = logging.getLogger(__name__)

POLL_MAX_WAIT = 60.0
POLL_INTERVAL = 1.0


class FileSecurityClient:
    def __init__(self) -> None:
        self.base_url = settings.file_security_url.rstrip("/")
        self._timeout = settings.file_security_timeout
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def submit_file(
        self,
        content: bytes,
        filename: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        files = {"file": (filename, content)}
        data = {"tenant_id": tenant_id}
        try:
            resp = await self._client.post(
                "/api/v1/files/submit", files=files, data=data
            )
        except httpx.RequestError as e:
            raise FileSecurityUnavailableError(
                f"File security service unreachable: {e}"
            ) from e

        if resp.status_code == 503:
            raise FileSecurityUnavailableError(
                "File security service unavailable (fail-closed)"
            )
        if resp.status_code == 422:
            body = _parse_body(resp)
            raise FileBlockedError(
                message=body.get("detail", "File blocked by security policy"),
                reason=str(body),
            )
        if resp.status_code == 413:
            raise FileSecurityProtocolError("File too large", status_code=413)
        if resp.status_code >= 400:
            body = _parse_body(resp)
            raise FileSecurityProtocolError(
                body.get("detail", f"Submit failed with status {resp.status_code}"),
                status_code=resp.status_code,
            )

        body = _parse_body(resp)
        return body

    async def get_file_status(self, file_asset_id: str) -> dict[str, Any]:
        try:
            resp = await self._client.get(f"/api/v1/files/{file_asset_id}")
        except httpx.RequestError as e:
            raise FileSecurityUnavailableError(
                f"File security service unreachable: {e}"
            ) from e

        if resp.status_code == 404:
            raise FileSecurityProtocolError(
                f"File asset not found: {file_asset_id}", status_code=404
            )
        if resp.status_code >= 400:
            body = _parse_body(resp)
            raise FileSecurityProtocolError(
                body.get("detail", f"Status check failed: {resp.status_code}"),
                status_code=resp.status_code,
            )

        return _parse_body(resp)

    async def wait_for_safe(
        self,
        file_asset_id: str,
        max_wait: float = POLL_MAX_WAIT,
        poll_interval: float = POLL_INTERVAL,
    ) -> str:
        deadline = asyncio.get_event_loop().time() + max_wait
        while True:
            status_data = await self.get_file_status(file_asset_id)
            sec_status = status_data.get("security_status", "")
            if sec_status in ("safe_for_rag", "safe_for_download"):
                return sec_status
            if sec_status in ("blocked", "failed"):
                scans = status_data.get("scans", [])
                verdict = scans[0].get("verdict", "blocked") if scans else "blocked"
                raise FileBlockedError(
                    message=f"File blocked: {verdict}",
                    reason=verdict,
                )
            if asyncio.get_event_loop().time() >= deadline:
                raise FileSecurityTimeoutError(f"File scan timeout after {max_wait}s")
            await asyncio.sleep(poll_interval)

    async def get_safe_artifact(self, file_asset_id: str) -> bytes:
        try:
            resp = await self._client.get(
                f"/api/v1/files/{file_asset_id}/safe-artifact"
            )
        except httpx.RequestError as e:
            raise FileSecurityUnavailableError(
                f"File security service unreachable: {e}"
            ) from e

        if resp.status_code == 404:
            raise FileSecurityProtocolError(
                f"Safe artifact not found for {file_asset_id}", status_code=404
            )
        if resp.status_code == 409:
            raise FileSecurityProtocolError(
                "File not yet safe for access", status_code=409
            )
        if resp.status_code >= 400:
            body = _parse_body(resp)
            raise FileSecurityProtocolError(
                body.get(
                    "detail", f"Safe artifact download failed: {resp.status_code}"
                ),
                status_code=resp.status_code,
            )

        return resp.content

    async def scan_and_get_safe_artifact(
        self,
        content: bytes,
        filename: str,
        tenant_id: str,
    ) -> bytes:
        submit_resp = await self.submit_file(content, filename, tenant_id)
        file_id = submit_resp.get("id")
        if not file_id:
            raise FileSecurityProtocolError(
                f"Submit response missing 'id': {submit_resp}"
            )
        await self.wait_for_safe(str(file_id))
        return await self.get_safe_artifact(str(file_id))


def _parse_body(resp: httpx.Response) -> dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"detail": resp.text}
