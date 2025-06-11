#!/bin/bash

# Скрипт для запуска всех сервисов

echo "🚀 Запуск всех сервисов..."

# Создание Docker сети если она не существует
docker network create mcp-network 2>/dev/null || true

# Запуск всех сервисов
docker-compose up --build -d

echo "✅ Все сервисы запущены!"
echo ""
echo "📊 Доступные сервисы:"
echo "- Loader (Streamlit): http://localhost:8501"
echo "- Embedding Service: http://localhost:5000"
echo "- Qdrant: http://localhost:6333/dashboard"
echo "- MCP Server: http://localhost:8000/mcp"
echo "- MCP Inspector: http://localhost:6274"
