import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LightragClient:
    def __init__(self) -> None:
        self.base_url = settings.lightrag_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=settings.lightrag_timeout
        )

    async def close(self) -> None:
        await self._client.aclose()
