# Мультивекторный поиск в MCP 1C RAG Server

## Описание

MCP сервер теперь поддерживает мультивекторный поиск, который использует два векторных представления для каждого объекта документации:
- `object_name` - эмбеддинг имени объекта
- `friendly_name` - эмбеддинг дружественного имени объекта

## Как работает мультивекторный поиск

1. **RRF (Reciprocal Rank Fusion)** - объединяет результаты поиска по двум векторам
2. **Prefetch** - выполняет предварительный поиск по каждому вектору с увеличенным лимитом
3. **Fusion** - объединяет и ранжирует результаты для получения финального списка

## Параметры поиска

### Конфигурационные параметры

В `config.py` добавлены новые параметры:

```python
# Multivector search settings
OBJECT_NAME_VECTOR = "object_name"
FRIENDLY_NAME_VECTOR = "friendly_name"
PREFETCH_LIMIT_MULTIPLIER = int(os.getenv("PREFETCH_LIMIT_MULTIPLIER", "3"))
```

## Примеры использования

### MCP инструмент

```json
{
  "query": "справочник номенклатура",
  "object_type": "Справочник",
  "limit": 5
}
```

### REST API

```bash
curl -X POST http://localhost:9000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "справочник номенклатура",
    "object_type": "Справочник", 
    "limit": 5,
    "use_multivector": true
  }'
```
Если нужно использовать обычный поиск (например, для отладки), то используйте `use_multivector: false`. Для MCP сделал, чтобы всегда был мультивекторный поиск.


## Архитектура поиска

### Мультивекторный поиск (use_multivector=true)

```python
qdrant_client.query_points(
    collection_name=collection_name,
    prefetch=[
        Prefetch(
            query=query_embedding,
            using="object_name",
            filter=query_filter,
            limit=limit * 3  # PREFETCH_LIMIT_MULTIPLIER
        ),
        Prefetch(
            query=query_embedding,
            using="friendly_name", 
            filter=query_filter,
            limit=limit * 3
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=limit
)
```

### Обычный поиск (use_multivector=false)

```python
qdrant_client.query_points(
    collection_name=collection_name,
    query=query_embedding,
    using="friendly_name",  # Используем friendly_name как основной
    query_filter=query_filter,
    limit=limit
)
```

## Преимущества мультивекторного поиска

1. **Более точное ранжирование** - учитываются разные аспекты объектов
2. **Лучшее покрытие** - поиск по разным представлениям может найти больше релевантных результатов
3. **RRF** - надежный алгоритм объединения результатов, который не требует калибровки весов

Подробнее об RRF можно почитать в [статье](https://medium.com/@Iraj/how-qdrant-combines-query-results-explaining-rrf-and-dbsf-cd08cd272a80).

## Настройка производительности

- **PREFETCH_LIMIT_MULTIPLIER** - контролирует сколько кандидатов берется на этапе prefetch
- Увеличение значения улучшает качество, но замедляет поиск
- Рекомендуемые значения: 2-5
