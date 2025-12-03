# Конфигурация для MCP сервера 1С RAG

import os
from dotenv import load_dotenv

load_dotenv()

# Qdrant settings
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "1cservice_rag")

# Embedding service settings
EMBEDDING_SERVICE_URL = os.getenv(
    "EMBEDDING_SERVICE_URL", "http://localhost:5000")

# Server settings
SERVER_NAME = os.getenv("SERVER_NAME", "MCP 1C RAG Server")
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8001"))

# RAG settings
DEFAULT_SEARCH_LIMIT = int(os.getenv("DEFAULT_SEARCH_LIMIT", "5"))
MAX_SEARCH_LIMIT = int(os.getenv("MAX_SEARCH_LIMIT", "10"))
MIN_SEARCH_LIMIT = int(os.getenv("MIN_SEARCH_LIMIT", "1"))

# Request timeout settings
EMBEDDING_REQUEST_TIMEOUT = int(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "10"))
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))

# Transport settings  
TRANSPORT_TYPE = os.getenv("TRANSPORT_TYPE", "streamable-http")  # stdio, sse или streamable-http

# Multivector search settings
# Имена векторов в коллекции Qdrant (не настраиваются через переменные окружения)
OBJECT_NAME_VECTOR = "object_name"
FRIENDLY_NAME_VECTOR = "friendly_name"
# Множитель для prefetch лимита в мультивекторном поиске
PREFETCH_LIMIT_MULTIPLIER = int(os.getenv("PREFETCH_LIMIT_MULTIPLIER", "3"))
