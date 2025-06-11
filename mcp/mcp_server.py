from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse
import json
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

from config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_SERVICE_URL,
    SERVER_HOST, SERVER_PORT, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
    MIN_SEARCH_LIMIT, SERVER_NAME,
    EMBEDDING_REQUEST_TIMEOUT, HEALTH_CHECK_TIMEOUT
)

mcp = FastMCP(name=SERVER_NAME)

# Подключение к Qdrant
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


class SearchRequest(BaseModel):
    """Модель запроса для поиска в документации 1С"""
    query: str = Field(
        description="Наименование объекта конфигурации или часть его имени для поиска в документации 1С",
        min_length=1,
        max_length=500
    )
    object_type: Literal["Справочник", "Документ", "РегистрСведений", "РегистрНакопления",
                         "Константа", "Перечисление", "ПланВидовХарактеристик"] | None = Field(
        default=None,
        description="Фильтр по типу объекта конфигурации 1С. Если не указан, поиск выполняется по всем типам объектов"
    )
    limit: int = Field(
        default=DEFAULT_SEARCH_LIMIT,
        description=f"Максимальное количество результатов поиска (по умолчанию {DEFAULT_SEARCH_LIMIT})",
        ge=MIN_SEARCH_LIMIT,
        le=MAX_SEARCH_LIMIT
    )


def get_query_embedding(query: str) -> List[float]:
    """Получение эмбеддинга для запроса"""
    payload = json.dumps({
        "texts": [query],
        "task": "retrieval.query"
    })
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.request(
            "POST", f"{EMBEDDING_SERVICE_URL}/embed", headers=headers, data=payload, timeout=EMBEDDING_REQUEST_TIMEOUT)

        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]
    except requests.RequestException as e:
        raise Exception(f"Ошибка получения эмбеддинга: {str(e)}")


def rag_search(query: str, object_type: str = None, limit: int = DEFAULT_SEARCH_LIMIT) -> List[Dict[str, Any]]:
    """Выполнение RAG-поиска в документации 1С"""
    try:
        # Получение эмбеддинга для запроса
        query_embedding = get_query_embedding(query)

        # Подготовка фильтра по типу объекта
        query_filter = None
        if object_type:
            query_filter = Filter(
                must=[
                    {
                        "key": "object_type",
                        "match": {
                            "value": object_type
                        }
                    }
                ]
            )

        # Поиск в Qdrant
        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit
        )

        # Форматирование результатов
        results = []
        for result in search_results.points:
            results.append({
                "score": result.score,
                "object_name": result.payload.get("object_name", ""),
                "object_type": result.payload.get("object_type", ""),
                "description": result.payload.get("doc", "")
            })

        return results
    except Exception as e:
        raise Exception(f"Ошибка поиска в документации: {str(e)}")


@mcp.tool
def search_1c_documentation(search_params: SearchRequest) -> str:
    """Поиск описания объектов конфигурации 1С Предприятие 8 в документации.

    Args:
        search_params: Параметры поиска включающие запрос, тип объекта и лимит результатов
    """
    try:
        results = rag_search(search_params.query,
                             search_params.object_type, search_params.limit)

        if not results:
            filter_text = f" по типу '{search_params.object_type}'" if search_params.object_type else ""
            return f"По запросу '{search_params.query}'{filter_text} ничего не найдено в документации 1С."

        formatted_results = []
        filter_text = f" (фильтр по типу: {search_params.object_type})" if search_params.object_type else ""
        formatted_results.append(
            f"Результаты поиска по запросу: '{search_params.query}'{filter_text}\n")

        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"\nРезультат {i} (релевантность: {result['score']:.3f})")
            formatted_results.append(f"Объект: {result['object_name']}")
            formatted_results.append(f"Тип: {result['object_type']}")
            formatted_results.append(f"Описание:")
            formatted_results.append(f"{result['description']}")
            formatted_results.append("---")

        return "\n".join(formatted_results)

    except Exception as e:
        return f"Ошибка при поиске в документации 1С: {str(e)}"


@mcp.custom_route("/", methods=["GET"])
async def root(request: Request) -> JSONResponse:
    return JSONResponse({"message": "MCP 1C RAG Server запущен"})


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Проверка работоспособности сервера и подключений"""
    try:
        # Проверяем подключение к Qdrant
        collections = qdrant_client.get_collections()
        qdrant_status = "OK"

        # Проверяем сервис эмбеддингов
        embedding_status = "OK"
        try:
            response = requests.get(
                f"{EMBEDDING_SERVICE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
            if response.status_code != 200:
                embedding_status = "UNAVAILABLE"
        except:
            embedding_status = "UNAVAILABLE"

        return JSONResponse({
            "status": "healthy",
            "qdrant": qdrant_status,
            "embedding_service": embedding_status,
            "collection": COLLECTION_NAME
        })
    except Exception as e:
        return JSONResponse({
            "status": "unhealthy",
            "error": str(e)
        })


@mcp.custom_route("/search", methods=["POST"])
async def manual_search(request: Request) -> JSONResponse:
    """REST endpoint для ручного тестирования поиска"""
    try:
        req_data = await request.json()

        # Валидация данных через Pydantic модель
        search_request = SearchRequest(**req_data)

        # Выполнение поиска
        results = rag_search(
            query=search_request.query,
            object_type=search_request.object_type,
            limit=search_request.limit
        )

        return JSONResponse({
            "query": search_request.query,
            "object_type": search_request.object_type,
            "limit": search_request.limit,
            "results_count": len(results),
            "results": results
        })

    except ValueError as e:
        # Ошибки валидации Pydantic
        return JSONResponse({
            "error": f"Ошибка валидации данных: {str(e)}"
        }, status_code=400)
    except Exception as e:
        return JSONResponse({
            "error": f"Ошибка поиска: {str(e)}"
        }, status_code=500)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=SERVER_HOST,
            port=SERVER_PORT, log_level="info")
