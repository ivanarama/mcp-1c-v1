# MCP 1C Vector Database Project

Проект состоит из трех сервисов для работы с векторной базой данных на основе документации 1C:

1. **Embedding Service** - сервис генерации векторных представлений
2. **Loader** - веб-интерфейс для загрузки markdown файлов в векторную БД Qdrant
3. **MCP Server** - сервер для ответов на вопросы с использованием RAG

## Быстрый запуск

### Запуск всех сервисов

```bash
chmod +x start.sh && chmod +x stop.sh
# Запуск всех сервисов
./start.sh

# Или вручную
docker-compose up --build
```

### Остановка сервисов

```bash
# Остановка всех сервисов
./stop.sh

# Или вручную
docker-compose down
```

## Доступные сервисы

После запуска будут доступны:

- **Loader (Streamlit)**: http://localhost:8501 - веб-интерфейс для загрузки файлов
- **Embedding Service**: http://localhost:5000 - API для генерации эмбеддингов
- **Qdrant**: http://localhost:6333/dashboard - векторная база данных
- **MCP Server**: http://localhost:8000/mcp - сервер для ответов на вопросы
- **MCP Inspector**: http://localhost:6274 - веб-интерфейс проверки/отладки MCP серверов

## Структура проекта

```
mcp-1c-v1/
├── embeddings/           # Сервис генерации эмбеддингов
│   ├── Dockerfile
│   ├── embedding_service.py
│   ├── config.json
│   └── requirements.txt
├── loader/              # Веб-интерфейс для загрузки данных
│   ├── Dockerfile
│   ├── loader.py
│   ├── config.py
│   └── requirements.txt
├── mcp/                 # MCP 1С RAG сервер
│   ├── Dockerfile
│   ├── mcp_server.py
│   ├── config.py
│   └── requirements.txt
├── inspector/           # MCP Inspector
│   ├── Dockerfile
│   ├── mcp_server.py
│   ├── config.py
│   └── requirements.txt
├── docker-compose.yml   # Конфигурация всех сервисов
├── start.sh            # Скрипт запуска
└── stop.sh             # Скрипт остановки
```

## Использование

1. Запустите все сервисы: `./start.sh`
2. Откройте веб-интерфейс загрузчика: http://localhost:8501
3. Загрузите ZIP-архив с markdown файлами и файлом objects.csv
4. Нажмите "Начать обработку"
5. После загрузки данные будут доступны в Qdrant для поиска через MCP Server

Для агентов (Cursor, RooCode) поддерживающих современный протокол Streamable HTTP, указываем: http://youaddress:8000/mcp
Для VSCode Copilot, хоть и заявлена поддержка Streamable HTTP, но у меня работает только как SSE, поэтому указывал: http://youraddress:8000/mcp/sse

**Для Copilot .../YourProject/.vscode/mcp.json**
```json
{
    "servers": {
        "my-1c-mcp-server": {
            "url": "http://youraddress:8000/mcp/sse"
        }
    }
}
```

**Для Cursor .../YourProject/.cursor/mcp.json**
```json
{
    "servers": {
        "my-1c-mcp-server": {
            "url": "http://youraddress:8000/mcp"
        }
    }
}
```


## Переменные окружения

Можно настроить через переменные окружения:

- `EMBEDDING_SERVICE_URL` - URL сервиса эмбеддингов (по умолчанию: http://localhost:5000)
- `QDRANT_HOST` - хост Qdrant (по умолчанию: localhost)
- `QDRANT_PORT` - порт Qdrant (по умолчанию: 6333)
- `COLLECTION_NAME` - имя коллекции в Qdrant (по умолчанию: 1c_rag)
- `ROW_BATCH_SIZE` - размер батча строк (по умолчанию: 250)
- `EMBEDDING_BATCH_SIZE` - размер батча эмбеддингов (по умолчанию: 50)

## Отладка

Для просмотра логов отдельного сервиса:

```bash
# Логи конкретного сервиса
docker-compose logs -f loader
docker-compose logs -f embedding-service
docker-compose logs -f qdrant
docker-compose logs -f mcp-server

# Логи всех сервисов
docker-compose logs -f
```
