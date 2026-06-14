from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM / embeddings
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o"
    rerank_backend: str = "cross-encoder"  # cross-encoder | llm
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # OpenSearch
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "rag-chunks"
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = False
    opensearch_user: str = "admin"
    opensearch_password: str = "admin"
    opensearch_aws_auth: bool = False
    aws_region: str = "us-east-1"

    # Retrieval tuning
    dense_k: int = 10
    sparse_k: int = 10
    rrf_k: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rerank_top_n: int = 20
    final_top_k: int = 5
    confidence_threshold: float = 0.35

    embedding_dim: int = 1536  # text-embedding-3-small


@lru_cache
def get_settings() -> Settings:
    return Settings()
