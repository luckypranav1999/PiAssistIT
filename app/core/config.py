
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Enterprise IT Support Agentic RAG Copilot"
    app_env: str = "development"

    # Azure AI Foundry / OpenAI-compatible configuration
    openai_api_key: str = ""
    openai_base_url: str = ""
    openai_model: str = "gpt-4.1-mini"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"

    # External services
    tavily_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "fde-it-support-rag"
    pinecone_namespace: str = "company-it-kb"

    # RAG configuration
    top_k: int = 4
    max_retries: int = 1

    # Application settings
    admin_api_key: str = "change-me"
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
