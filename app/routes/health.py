from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "backends": {
            "file_security": "unknown",
            "lightrag": "unknown",
            "faiss": "unknown",
        },
    }
