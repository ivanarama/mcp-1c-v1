from fastmcp import FastMCP
from fastmcp.server.dependencies import get_http_headers
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
import json
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, Prefetch, FusionQuery, Fusion
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
import uuid

from config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_SERVICE_URL,
    SERVER_HOST, SERVER_PORT, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
    MIN_SEARCH_LIMIT, SERVER_NAME,
    EMBEDDING_REQUEST_TIMEOUT, HEALTH_CHECK_TIMEOUT,
    OBJECT_NAME_VECTOR, FRIENDLY_NAME_VECTOR, PREFETCH_LIMIT_MULTIPLIER,
    TRANSPORT_TYPE
)


class SessionIdMiddleware(BaseHTTPMiddleware):
    """Middleware для автоматического добавления mcp-session-id если его нет.
    Это позволяет MCP клиентам, которые не поддерживают session management,
    работать с streamable-http транспортом FastMCP.
    """
    async def dispatch(self, request: Request, call_next):
        # Добавляем session-id только для /mcp endpoint если его нет
        if request.url.path == "/mcp" and "mcp-session-id" not in request.headers:
            # Генерируем уникальный session-id
            session_id = str(uuid.uuid4())
            # Создаём новый запрос с добавленным заголовком
            request.headers.__dict__["_list"].append(
                (b"mcp-session-id", session_id.encode())
            )
        response = await call_next(request)
        return response


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
    use_multivector: bool = Field(
        default=True,
        description="Использовать мультивекторный поиск с RRF для более точного ранжирования результатов"
    )


class SearchRequestMCP(BaseModel):
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


def rag_search(query: str, collection_name: str, object_type: str = None, limit: int = DEFAULT_SEARCH_LIMIT, use_multivector: bool = True) -> List[Dict[str, Any]]:
    """Выполнение RAG-поиска в документации 1С с поддержкой мультивекторного поиска"""
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

        if use_multivector:
            # Мультивекторный поиск с RRF
            search_results = qdrant_client.query_points(
                collection_name=collection_name,
                prefetch=[
                    Prefetch(
                        query=query_embedding,
                        using=OBJECT_NAME_VECTOR,
                        filter=query_filter,
                        limit=limit * PREFETCH_LIMIT_MULTIPLIER
                    ),
                    Prefetch(
                        query=query_embedding,
                        using=FRIENDLY_NAME_VECTOR,
                        filter=query_filter,
                        limit=limit * PREFETCH_LIMIT_MULTIPLIER
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit
            )
        else:
            # Обычный поиск по одному вектору
            search_results = qdrant_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                using=FRIENDLY_NAME_VECTOR,  # Используем friendly_name как основной вектор
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
def search_1c_documentation(query: str, object_type: str = None, limit: int = DEFAULT_SEARCH_LIMIT) -> str:
    """Поиск описания объектов конфигурации 1С Предприятие 8 в документации.

    Args:
        query: Наименование объекта конфигурации или часть его имени для поиска в документации 1С
        object_type: Фильтр по типу объекта конфигурации 1С (Справочник, Документ, РегистрСведений, РегистрНакопления, Константа, Перечисление, ПланВидовХарактерик)
        limit: Максимальное количество результатов поиска
    """
    try:

        headers = get_http_headers()
        # Определяем имя коллекции по приоритету:
        # 1. Из HTTP-заголовка x-collection-name
        # 2. Значение по умолчанию из конфигурации
        collection_name = (
            headers.get("x-collection-name") or
            COLLECTION_NAME
        )

        # Проверяем, что коллекция существует
        if not qdrant_client.collection_exists(collection_name):
            return f"Ошибка: коллекция '{collection_name}' не существует в Qdrant."

        use_multivector = True
        results = rag_search(
            query,
            collection_name,
            object_type,
            limit,
            use_multivector
        )

        if not results:
            filter_text = f" по типу '{object_type}'" if object_type else ""
            search_type = "мультивекторный" if use_multivector else "обычный"
            return f"По запросу '{query}'{filter_text} ничего не найдено в документации 1С (коллекция: {collection_name}, поиск: {search_type})."

        formatted_results = []
        filter_text = f" (фильтр по типу: {object_type})" if object_type else ""
        search_type = "мультивекторный (RRF)" if use_multivector else "обычный"
        formatted_results.append(
            f"Результаты поиска по запросу: '{query}'{filter_text} (коллекция: {collection_name}, поиск: {search_type})\n")

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


@mcp.custom_route("/mcp", methods=["POST"])
async def mcp_proxy(request: Request) -> JSONResponse:
    """Proxy endpoint для streamable-http, который автоматически добавляет session-id.
    Это позволяет MCP клиентам, которые не поддерживают session management, работать.
    """
    # Получаем или создаём session-id
    session_id = request.headers.get("mcp-session-id") or str(uuid.uuid4())

    # Читаем тело запроса
    body = await request.body()

    # Делаем внутренний запрос к streamable-http endpoint с session-id
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"http://{SERVER_HOST}:{SERVER_PORT}/mcp",
            content=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "mcp-session-id": session_id,
            },
            timeout=30.0
        )

        # Возвращаем ответ
        return JSONResponse(
            content=response.json(),
            status_code=response.status_code,
            headers={"mcp-session-id": session_id}
        )


def create_sse_response(data: Dict) -> Response:
    """Создаёт SSE ответ из данных JSON-RPC"""
    sse_data = json.dumps(data, ensure_ascii=False)
    return Response(
        content=f"event: message\ndata: {sse_data}\n\n",
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@mcp.custom_route("/jsonrpc", methods=["GET", "POST"])
async def simple_jsonrpc(request: Request) -> Response:
    """Простой HTTP JSON-RPC endpoint для MCP SuperAssistant и других клиентов.

    Этот endpoint принимает стандартные JSON-RPC запросы и возвращает ответы в SSE формате,
    который ожидает MCP SuperAssistant Proxy при использовании streamable-http типа.
    """
    # Обрабатываем GET запросы (handshake)
    if request.method == "GET":
        return Response(
            content="event: endpoint\ndata: /jsonrpc\n\n",
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )

    # Обрабатываем POST запросы
    try:
        req_data = await request.json()

        # Проверяем, что это JSON-RPC запрос
        if not isinstance(req_data, dict):
            return create_sse_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"}
            })

        method = req_data.get("method")
        req_id = req_data.get("id")

        # Обрабатываем tools/list
        if method == "tools/list":
            # Получаем список инструментов из MCP
            tools = []
            tools_dict = await mcp.get_tools()  # get_tools возвращает словарь {name: Tool}
            for tool_name, tool in tools_dict.items():
                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": tool.inputSchema if hasattr(tool, 'inputSchema') else {
                        "type": "object",
                        "properties": {}
                    }
                })

            return create_sse_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": tools}
            })

        # Обрабатываем initialize
        elif method == "initialize":
            return create_sse_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": "1.0.0"
                    },
                    "capabilities": {
                        "tools": {}
                    }
                }
            })

        # Обрабатываем tools/call
        elif method == "tools/call":
            params = req_data.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})

            # Вызываем инструмент через FastMCP
            try:
                # Получаем все инструменты и ищем нужный
                tools_dict = await mcp.get_tools()
                if name in tools_dict:
                    tool = tools_dict[name]
                    # У FunctionTool есть атрибут fn - это реальная функция
                    if hasattr(tool, 'fn') and callable(tool.fn):
                        result = tool.fn(**arguments)
                        return create_sse_response({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"content": [{"type": "text", "text": str(result)}]}
                        })
                    else:
                        return create_sse_response({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": {"code": -32603, "message": f"Tool {name} is not callable"}
                        })
                else:
                    return create_sse_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Tool not found: {name}"}
                    })
            except Exception as e:
                return create_sse_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": f"Tool execution error: {str(e)}"}
                })

        # Метод ping для health check
        elif method == "ping":
            return create_sse_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"status": "ok"}
            })

        # Обрабатываем уведомления (notifications) - это запросы без id или с method начинающимся с notifications/
        elif method and method.startswith("notifications/"):
            # Уведомления не требуют ответа с result, возвращаем пустой успешный ответ
            if req_id is not None:
                return create_sse_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {}
                })
            else:
                # Для уведомлений без id возвращаем пустой SSE ответ
                return Response(
                    content=": \n\n",
                    media_type="text/event-stream"
                )

        # Обрабатываем resources/list (пустой список, ресурсов нет)
        elif method == "resources/list":
            return create_sse_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"resources": []}
            })

        # Обрабатываем prompts/list (пустой список, промптов нет)
        elif method == "prompts/list":
            return create_sse_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"prompts": []}
            })

        else:
            return create_sse_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            })

    except json.JSONDecodeError:
        return create_sse_response({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error"}
        })
    except Exception as e:
        return create_sse_response({
            "jsonrpc": "2.0",
            "id": req_data.get("id") if 'req_data' in locals() else None,
            "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
        })


@mcp.custom_route("/search", methods=["POST"])
async def manual_search(request: Request) -> JSONResponse:
    """REST endpoint для ручного тестирования поиска"""
    try:
        req_data = await request.json()
        print(req_data)

        # Валидация данных через Pydantic модель
        search_request = SearchRequest(**req_data)

        # Определяем имя коллекции по приоритету:
        # 1. Из HTTP-заголовка x-collection-name
        # 2. Значение по умолчанию из конфигурации
        collection_name = (
            request.headers.get("x-collection-name") or
            COLLECTION_NAME
        )

        # Проверяем, что коллекция существует
        if not qdrant_client.collection_exists(collection_name):
            return JSONResponse({
                "error": f"Коллекция '{collection_name}' не существует в Qdrant."
            }, status_code=400)

        # Выполнение поиска
        results = rag_search(
            query=search_request.query,
            collection_name=collection_name,
            object_type=search_request.object_type,
            limit=search_request.limit,
            use_multivector=search_request.use_multivector
        )

        return JSONResponse({
            "query": search_request.query,
            "object_type": search_request.object_type,
            "collection_name": collection_name,
            "limit": search_request.limit,
            "use_multivector": search_request.use_multivector,
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
    mcp.run(transport=TRANSPORT_TYPE, host=SERVER_HOST,
            port=SERVER_PORT, log_level="info")
