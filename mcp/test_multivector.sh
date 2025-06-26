#!/bin/bash

# Тестирование мультивекторного поиска MCP 1C RAG Server
# Убедитесь, что сервер запущен на localhost:9000

echo "=== Тестирование мультивекторного поиска ==="
echo

echo "1. Мультивекторный поиск (RRF):"
curl -X POST http://localhost:9000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "справочник номенклатура",
    "object_type": "Справочник",
    "limit": 3,
    "use_multivector": true
  }' | jq .

echo
echo "2. Обычный поиск (для сравнения):"
curl -X POST http://localhost:9000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "справочник номенклатура", 
    "object_type": "Справочник",
    "limit": 3,
    "use_multivector": false
  }' | jq .

echo
echo "3. Мультивекторный поиск без фильтра по типу:"
curl -X POST http://localhost:9000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "документ продажи",
    "limit": 5,
    "use_multivector": true
  }' | jq .

echo
echo "4. Поиск с кастомной коллекцией:"
curl -X POST http://localhost:9000/search \
  -H "Content-Type: application/json" \
  -H "x-collection-name: custom_collection" \
  -d '{
    "query": "регистр сведений",
    "limit": 3,
    "use_multivector": true
  }' | jq .

echo
echo "5. Проверка health check:"
curl -X GET http://localhost:9000/health | jq .

echo
echo "=== Тестирование завершено ==="
