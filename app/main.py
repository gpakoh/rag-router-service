from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.upload import router as upload_router
from app.services.file_security_client import FileSecurityClient
from app.services.lightrag_client import LightRAGClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.file_security = FileSecurityClient()
    app.state.lightrag = LightRAGClient()
    yield
    await app.state.file_security.close()
    await app.state.lightrag.close()


app = FastAPI(title="RAG Router Service", version="0.1.0", lifespan=lifespan)
app.include_router(health_router)
app.include_router(upload_router)
