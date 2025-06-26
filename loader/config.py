# Конфигурация для loader
import os
from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

EMBEDDING_SERVICE_URL = os.getenv(
    "EMBEDDING_SERVICE_URL", "http://localhost:5000")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "1c_rag")

# Параметры батчинга
# Обрабатывать строки CSV пачками по 250
ROW_BATCH_SIZE = int(os.getenv("ROW_BATCH_SIZE", "250"))
# Размер батча для эмбеддингов
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "50"))
