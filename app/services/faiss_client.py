import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class FaissClient:
    def __init__(self) -> None:
        self.base_url = settings.faiss_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url, timeout=settings.faiss_timeout
        )

    async def close(self) -> None:
        await self._client.aclose()
