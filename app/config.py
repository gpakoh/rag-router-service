from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "RAG_ROUTER__"}

    host: str = "0.0.0.0"
    port: int = 8030

    file_security_url: str = "http://file-security-service:8000"
    file_security_timeout: int = 60

    lightrag_url: str = "http://lightrag-service:8787"
    lightrag_timeout: int = 120

    faiss_url: str = "http://rag-faiss-service:8020"
    faiss_timeout: int = 120


settings = Settings()
