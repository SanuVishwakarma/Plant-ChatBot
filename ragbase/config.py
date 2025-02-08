import os
from pathlib import Path
from typing import Optional


class Config:
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    CONVERSATION_MESSAGES_LIMIT = int(os.getenv("CONVERSATION_MESSAGES_LIMIT", "50"))
    
    # API Configuration
    API_VERSION = "1.0.0"
    API_TITLE = "Plant Shop Assistant API"
    API_DESCRIPTION = "API for plant shop assistant with RAG capabilities"
    
    # Security Configuration
    API_KEY_NAME = "Plant-API-Key"
    DEFAULT_API_KEY = os.getenv("API_KEY")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
    RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "3600"))  # in seconds

    class Path:
        BASE_DIR = Path(__file__).parent.parent
        DATABASE_DIR = BASE_DIR / "database"
        DATA_DIR = BASE_DIR / "data"

    class Database:
        DOCUMENTS_COLLECTION = "documents"
        COLLECTION_PATH: Optional[str] = os.getenv("QDRANT_COLLECTION_PATH")

    class Model:
        USE_LOCAL = os.getenv("USE_LOCAL_MODEL", "False").lower() == "true"
        LOCAL_LLM = os.getenv("LOCAL_LLM", "llama2")
        REMOTE_LLM = os.getenv("REMOTE_LLM", "llama-3.1-70b-versatile")
        TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
        MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))
        EMBEDDINGS = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        RERANKER = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

    class Retriever:
        USE_RERANKER = os.getenv("USE_RERANKER", "True").lower() == "true"
        USE_CHAIN_FILTER = os.getenv("USE_CHAIN_FILTER", "False").lower() == "true"
        TOP_K = int(os.getenv("RETRIEVER_TOP_K", "5"))