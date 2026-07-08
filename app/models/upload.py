from pydantic import BaseModel


class UploadResponse(BaseModel):
    status: str
    tenant_id: str
    strategy: str
    document_id: str | None = None
