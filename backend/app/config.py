"""
Central configuration for the RAG backend.
All values are read from environment variables (see .env.example).
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Which LLM provider to use for answer generation: "anthropic" or "groq"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic").lower()

    # Anthropic Claude settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    # Groq settings (free tier, OpenAI-compatible API, open-source models)
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1024"))

    # Retry / rate-limit settings for LLM calls
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "4"))
    LLM_BASE_BACKOFF_SECONDS: float = float(os.getenv("LLM_BASE_BACKOFF_SECONDS", "1.0"))
    LLM_MIN_SECONDS_BETWEEN_CALLS: float = float(os.getenv("LLM_MIN_SECONDS_BETWEEN_CALLS", "0.5"))

    # Embedding model (runs locally, open-source, no API key needed)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    # Chunking
    CHUNK_SIZE_TOKENS: int = int(os.getenv("CHUNK_SIZE_TOKENS", "400"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "50"))

    # Retrieval
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "4"))

    # Storage
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "documents")

    # Server
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")


settings = Settings()
