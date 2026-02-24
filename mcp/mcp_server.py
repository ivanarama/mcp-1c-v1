from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.middleware.base import BaseHTTPMiddleware
import json
import asyncio
import time
from contextvars import ContextVar

# Context variable для хранения текущего request
_current_request_ctx: ContextVar[Request] = ContextVar("_current_request_ctx", default=None)
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, Prefetch, FusionQuery, Fusion
from typing import Dict, Any, List, Literal, Optional, Mapping, AsyncIterator
from pydantic import BaseModel, Field, ValidationError, ConfigDict
import uuid
from datetime import datetime, timezone

from config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_SERVICE_URL,
    SERVER_HOST, SERVER_PORT, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT,
    MIN_SEARCH_LIMIT, SERVER_NAME,
    EMBEDDING_REQUEST_TIMEOUT, HEALTH_CHECK_TIMEOUT,
    OBJECT_NAME_VECTOR, FRIENDLY_NAME_VECTOR, PREFETCH_LIMIT_MULTIPLIER,
    TRANSPORT_TYPE
)


# Middleware для автоматического добавления mcp-session-id и сохранения request
class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware для сохранения request и автоматического добавления mcp-session-id."""
    async def dispatch(self, request: Request, call_next):
        # Сохраняем request в context variable
        _current_request_ctx.set(request)

        # Автоматически добавляем mcp-session-id для /mcp endpoint
        if request.url.path in ("/mcp", "/mcp/") and "mcp-session-id" not in request.headers:
            from starlette.requests import Request as NewRequest

            session_id = str(uuid.uuid4())

            # Создаём новый scope с добавленным заголовком
            scope = dict(request.scope)
            headers = list(scope.get("headers", []))
            headers.append((b"mcp-session-id", session_id.encode()))
            scope["headers"] = headers

            # Создаём новый request с обновленными заголовками
            new_request = NewRequest(scope)

            try:
                response = await call_next(new_request)
                # Добавляем session-id в ответ для клиента
                if hasattr(response, 'headers'):
                    response.headers["mcp-session-id"] = session_id
                return response
            finally:
                _current_request_ctx.set(None)
        else:
            try:
                response = await call_next(request)
                return response
            finally:
                _current_request_ctx.set(None)


def get_collection_from_header() -> str | None:
    """Получает x-collection-name из заголовков текущего запроса."""
    request = _current_request_ctx.get()
    if request:
        return request.headers.get("x-collection-name")
    return None


mcp = FastMCP(name=SERVER_NAME)

# Monkey-patch для добавления middleware в Starlette приложение FastMCP
# Это нужно для автоматического добавления mcp-session-id
def add_middleware_to_fastmcp():
    """Добавляет middleware в Starlette приложение FastMCP."""
    try:
        # Получаем внутреннее приложение Starlette из FastMCP
        if hasattr(mcp, '_app'):
            app = mcp._app
        elif hasattr(mcp, 'app'):
            app = mcp.app
        else:
            # Попробуем получить приложение через mcp.__dict__
            for attr_name, attr_value in mcp.__dict__.items():
                if hasattr(attr_value, 'add_middleware') and hasattr(attr_value, 'routes'):
                    app = attr_value
                    break
            else:
                print("Warning: Could not find Starlette app in FastMCP")
                return False

        # Добавляем middleware через Middleware класс
        from starlette.middleware import Middleware
        middleware = Middleware(RequestContextMiddleware)
        app.user_middleware.insert(0, middleware)
        return True
    except Exception as e:
        print(f"Warning: Could not add middleware to FastMCP: {e}")
        return False

# Пытаемся добавить middleware
add_middleware_to_fastmcp()

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

        # Определяем имя коллекции по приоритету:
        # 1. Из HTTP-заголовка x-collection-name (через middleware)
        # 2. Значение по умолчанию из конфигурации
        collection_name = get_collection_from_header() or COLLECTION_NAME

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


def _oauth_authorization_server_metadata(request: Request) -> Dict[str, Any]:
    """Минимальные discovery-метаданные для клиентов, которые проверяют OAuth endpoint."""
    base = str(request.base_url).rstrip("/")
    return {
        "issuer": base,
        "resource": f"{base}/mcp-auto",
        "authorization_endpoint": f"{base}/auth/disabled",
        "token_endpoint": f"{base}/auth/disabled",
        "response_types_supported": [],
        "grant_types_supported": [],
        "token_endpoint_auth_methods_supported": [],
        "scopes_supported": [],
    }


async def oauth_authorization_server_root(request: Request) -> JSONResponse:
    return JSONResponse(_oauth_authorization_server_metadata(request))


async def oauth_authorization_server_mcp_auto(request: Request) -> JSONResponse:
    return JSONResponse(_oauth_authorization_server_metadata(request))


async def oauth_authorization_server_nested(request: Request) -> JSONResponse:
    return JSONResponse(_oauth_authorization_server_metadata(request))


def _register_oauth_well_known_routes() -> None:
    """FastMCP custom_route не принимает .well-known, регистрируем вручную."""
    route_map = {
        "/.well-known/oauth-authorization-server": oauth_authorization_server_root,
        "/.well-known/oauth-authorization-server/mcp-auto": oauth_authorization_server_mcp_auto,
        "/mcp-auto/.well-known/oauth-authorization-server": oauth_authorization_server_nested,
    }
    existing = {getattr(route, "path", "") for route in mcp._additional_http_routes}
    for path, endpoint in route_map.items():
        if path in existing:
            continue
        mcp._additional_http_routes.append(Route(path, endpoint=endpoint, methods=["GET"]))


# Не публикуем OAuth discovery для /mcp-auto: Codex в этом случае
# переходит в auth-flow (Not logged in) вместо обычного streamable-http.
# _register_oauth_well_known_routes()


def create_sse_response(data: Dict, session_id: str = None) -> Response:
    """
    Создаёт SSE ответ из данных JSON-RPC с правильными заголовками для stream.

    Использует StreamingResponse для настоящего SSE с chunked encoding.
    """
    sse_data = json.dumps(data, ensure_ascii=False)

    # Генератор для SSE контента (для StreamingResponse)
    async def sse_generator():
        yield f"event: message\ndata: {sse_data}\n\n"

    # Заголовки для SSE stream
    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
    }
    if session_id:
        headers["MCP-Session-Id"] = session_id

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers=headers
    )


# ==================== SSE Session Manager ====================
import logging
from asyncio import sleep
from dataclasses import dataclass, field

# Настройка логирования
mcp_logger = logging.getLogger("mcp.handshake")
if not mcp_logger.handlers:
    mcp_handler = logging.StreamHandler()
    mcp_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    mcp_logger.addHandler(mcp_handler)
mcp_logger.setLevel(logging.INFO)


@dataclass
class MCPSession:
    """Состояние MCP сессии для SSE streaming."""
    session_id: str
    initialized: bool = False
    capabilities_sent: bool = False
    collection_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)


class MCPSessionManager:
    """Управляет MCP сессиями для SSE streaming."""

    def __init__(self, session_timeout: float = 300.0):
        self.sessions: Dict[str, MCPSession] = {}
        self.session_timeout = session_timeout

    def get_or_create_session(self, session_id: str) -> MCPSession:
        """Получает или создаёт сессию."""
        if session_id not in self.sessions:
            mcp_logger.info(f"[session_manager] Creating new session: {session_id}")
            self.sessions[session_id] = MCPSession(session_id=session_id)
        else:
            mcp_logger.info(f"[session_manager] Reusing existing session: {session_id}")
        session = self.sessions[session_id]
        session.last_activity = time.time()
        return session

    def remove_session(self, session_id: str) -> None:
        """Удаляет сессию."""
        if session_id in self.sessions:
            mcp_logger.info(f"[session_manager] Removing session: {session_id}")
            del self.sessions[session_id]


# Глобальный менеджер сессий
session_manager = MCPSessionManager()


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request/notification."""

    model_config = ConfigDict(extra="allow")

    jsonrpc: Literal["2.0"]
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)
    id: str | int | None = None


def _extract_mcp_session_id(headers: Mapping[str, str]) -> Optional[str]:
    """Нормализует получение session-id из заголовков (case-insensitive)."""
    return headers.get("mcp-session-id") or headers.get("MCP-Session-Id")


def _log_mcp_auto_event(
    *,
    request_id: str,
    session_id: str,
    method: Optional[str],
    phase: str,
    status_code: int,
    latency_ms: float,
    error: Optional[str] = None,
) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "session_id": session_id,
        "method": method,
        "phase": phase,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "error": error,
    }
    mcp_logger.info(json.dumps(payload, ensure_ascii=False))


def _mcp_sse_response(
    *,
    payload: Dict[str, Any],
    session_id: str,
    status_code: int = 200,
) -> StreamingResponse:
    async def event_generator() -> AsyncIterator[str]:
        yield f"event: message\ndata: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"

    return StreamingResponse(
        event_generator(),
        status_code=status_code,
        media_type=None,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "mcp-session-id": session_id,
        },
    )


def _jsonrpc_error(code: int, message: str, req_id: str | int | None = None) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


class MCPAutoTransportAdapter:
    """Transport adapter только для /mcp-auto."""

    def __init__(self, session: MCPSession, request: Request, request_id: str):
        self.session = session
        self.request = request
        self.request_id = request_id

    async def dispatch(self, rpc: JsonRpcRequest) -> Dict[str, Any]:
        method = rpc.method
        req_id = rpc.id
        if method == "initialize":
            collection = self.request.headers.get("x-collection-name")
            if collection and not self.session.collection_name:
                self.session.collection_name = collection
            self.session.capabilities_sent = True
            self.session.initialized = False
            init_opts = mcp._mcp_server.create_initialization_options()
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": init_opts.capabilities.model_dump(exclude_none=True),
                    "serverInfo": {
                        "name": init_opts.server_name,
                        "version": init_opts.server_version,
                    },
                },
            }

        if method == "notifications/initialized":
            self.session.initialized = True
            response = {"jsonrpc": "2.0", "result": {}}
            if req_id is not None:
                response["id"] = req_id
            return response

        if method == "resources/list":
            if not self.session.initialized:
                return _jsonrpc_error(-32002, "Session not initialized. Send notifications/initialized first.", req_id)
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": []}}

        if method == "tools/list":
            tools_list = await mcp.list_tools()
            tools = []
            tool_schemas = {
                "search_1c_documentation": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Наименование объекта конфигурации или часть его имени для поиска в документации 1С",
                        },
                        "object_type": {
                            "type": "string",
                            "description": "Фильтр по типу объекта конфигурации 1С",
                            "enum": [
                                "Справочник",
                                "Документ",
                                "РегистрСведений",
                                "РегистрНакопления",
                                "Константа",
                                "Перечисление",
                                "ПланВидовХарактеристик",
                            ],
                        },
                        "limit": {
                            "type": "integer",
                            "description": f"Максимальное количество результатов поиска (по умолчанию {DEFAULT_SEARCH_LIMIT})",
                        },
                    },
                    "required": ["query"],
                }
            }
            for tool in tools_list:
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "inputSchema": tool_schemas.get(
                            tool.name,
                            {"type": "object", "properties": {}},
                        ),
                    }
                )
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

        if method in ("tools/call", "tool/call"):
            if not self.session.initialized:
                return _jsonrpc_error(-32002, "Session not initialized. Send notifications/initialized first.", req_id)

            params = rpc.params if isinstance(rpc.params, dict) else {}
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not tool_name:
                return _jsonrpc_error(-32602, "Invalid params: missing tool name", req_id)
            if arguments is None:
                arguments = {}
            if not isinstance(arguments, dict):
                return _jsonrpc_error(-32602, "Invalid params: arguments must be an object", req_id)

            try:
                tool_result = await mcp.call_tool(tool_name, arguments)
            except Exception as e:
                return _jsonrpc_error(-32603, f"Tool execution error: {str(e)}", req_id)

            content_items = []
            for item in getattr(tool_result, "content", []) or []:
                if hasattr(item, "model_dump"):
                    content_items.append(item.model_dump(exclude_none=True))
                elif isinstance(item, dict):
                    content_items.append(item)
                else:
                    content_items.append({"type": "text", "text": str(item)})

            result_payload: Dict[str, Any] = {"content": content_items}
            if hasattr(tool_result, "isError"):
                result_payload["isError"] = bool(getattr(tool_result, "isError"))

            return {"jsonrpc": "2.0", "id": req_id, "result": result_payload}

        if method == "resources/templates/list":
            if not self.session.initialized:
                return _jsonrpc_error(-32002, "Session not initialized. Send notifications/initialized first.", req_id)
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resourceTemplates": []}}

        if method and method.startswith("notifications/"):
            response = {"jsonrpc": "2.0", "result": {}}
            if req_id is not None:
                response["id"] = req_id
            return response

        return _jsonrpc_error(-32601, f"Method not found: {method}", req_id)


# ==================== SSE Streaming Transport ====================

class MCPSSEStream:
    """Реализует SSE streaming для MCP JSON-RPC сообщений."""

    def __init__(self, session: MCPSession, request: Request):
        self.session = session
        self.request = request
        self._closed = False

    async def _process_request(self, req_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Обрабатывает один JSON-RPC запрос и возвращает ответ."""
        method = req_data.get("method")
        req_id = req_data.get("id")
        is_notification = req_id is None

        mcp_logger.info(f"[{self.session.session_id}] processing method={method} id={req_id} notification={is_notification}")

        start_time = time.time()

        # === initialize ===
        if method == "initialize":
            mcp_logger.info(f"[{self.session.session_id}] initialize_received")
            self.session.initialized = True

            # Сохраняем collection_name из заголовка для будущих запросов
            collection = self.request.headers.get("x-collection-name")
            if collection:
                self.session.collection_name = collection
                mcp_logger.info(f"[{self.session.session_id}] collection_set={collection}")

            latency = (time.time() - start_time) * 1000
            mcp_logger.info(f"[{self.session.session_id}] initialize_sent latency={latency:.2f}ms")

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": "1.0.0"
                    },
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    }
                }
            }

        # === notifications/initialized ===
        elif method == "notifications/initialized":
            mcp_logger.info(f"[{self.session.session_id}] initialized_received notification=True")
            latency = (time.time() - start_time) * 1000
            mcp_logger.info(f"[{self.session.session_id}] initialized_ack_sent latency={latency:.2f}ms stream_kept_open=True")

            # Возвращаем пустой результат для совместимости с MCP transport layer
            # JSON-RPC 2.0 говорит что notifications не должны получать ответ,
            # но MCP streamable-http транспорт ожидает ответ для подтверждения обработки
            return {"jsonrpc": "2.0", "result": {}}

        # === Общие notifications ===
        elif method and method.startswith("notifications/"):
            mcp_logger.info(f"[{self.session.session_id}] notification_received method={method}")
            return {"jsonrpc": "2.0", "result": {}}

        # === tools/list ===
        elif method == "tools/list":
            tools_dict = await mcp.get_tools()
            tools = []
            tool_schemas = {
                "search_1c_documentation": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Наименование объекта для поиска"},
                        "object_type": {"type": "string", "description": "Тип объекта"},
                        "limit": {"type": "integer", "description": f"Макс. результатов (по умолчанию {DEFAULT_SEARCH_LIMIT})"}
                    },
                    "required": ["query"]
                }
            }
            for tool_name, tool in tools_dict.items():
                input_schema = tool_schemas.get(tool_name, {"type": "object", "properties": {}})
                tools.append({"name": tool.name, "description": tool.description or "", "inputSchema": input_schema})

            latency = (time.time() - start_time) * 1000
            mcp_logger.info(f"[{self.session.session_id}] tools/list_sent latency={latency:.2f}ms")

            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}

        # === tools/call ===
        elif method == "tools/call":
            params = req_data.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})

            # Используем сохранённое collection_name из сессии
            original_collection = get_collection_from_header()
            if self.session.collection_name:
                _current_request_ctx.set(self.request)

            try:
                tools_dict = await mcp.get_tools()
                if name in tools_dict:
                    tool = tools_dict[name]
                    if hasattr(tool, 'fn') and callable(tool.fn):
                        result = tool.fn(**arguments)
                        latency = (time.time() - start_time) * 1000
                        mcp_logger.info(f"[{self.session.session_id}] tools/call_sent tool={name} latency={latency:.2f}ms")
                        return {
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {"content": [{"type": "text", "text": str(result)}]}
                        }

                latency = (time.time() - start_time) * 1000
                mcp_logger.warning(f"[{self.session.session_id}] tools/call_failed tool={name} latency={latency:.2f}ms")
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Tool not found: {name}"}
                }
            finally:
                if original_collection is None:
                    _current_request_ctx.set(None)

        # === resources/list ===
        elif method == "resources/list":
            mcp_logger.info(f"[{self.session.session_id}] resources/list_received")
            latency = (time.time() - start_time) * 1000
            mcp_logger.info(f"[{self.session.session_id}] resources/list_sent latency={latency:.2f}ms")
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": []}}

        # === resources/templates/list ===
        elif method == "resources/templates/list":
            mcp_logger.info(f"[{self.session.session_id}] resources/templates/list_received")
            latency = (time.time() - start_time) * 1000
            mcp_logger.info(f"[{self.session.session_id}] resources/templates/list_sent latency={latency:.2f}ms")
            return {"jsonrpc": "2.0", "id": req_id, "result": {"resourceTemplates": []}}

        # === resources/read ===
        elif method == "resources/read":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": "No resources available"}
            }

        # === prompts/list ===
        elif method == "prompts/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"prompts": []}}

        # === ping ===
        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"status": "ok"}}

        # === Неизвестный метод ===
        else:
            mcp_logger.warning(f"[{self.session.session_id}] unknown_method method={method}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"}
            }

    async def _sse_event(self, data: Dict[str, Any]) -> str:
        """Создаёт SSE событие из данных."""
        return f"event: message\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

    async def stream(self) -> AsyncIterator[str]:
        """
        Основной метод SSE streaming.

        Читает запрос, обрабатывает его, отправляет ответ.
        """
        mcp_logger.info(f"[{self.session.session_id}] stream_opened")

        try:
            # Читаем запрос из тела POST
            req_data = await self._read_single_request()

            if req_data:
                # Обрабатываем запрос
                response = await self._process_request(req_data)

                if response:
                    # Отправляем SSE событие
                    sse_event = await self._sse_event(response)
                    yield sse_event

                # Логируем завершение обработки
                if req_data.get("id") is None:
                    mcp_logger.info(f"[{self.session.session_id}] notification_processed stream_complete=True")
                else:
                    mcp_logger.info(f"[{self.session.session_id}] request_processed stream_complete=True")

        except Exception as e:
            mcp_logger.error(f"[{self.session.session_id}] stream_error: {e}")
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
            yield await self._sse_event(error_response)
        finally:
            mcp_logger.info(f"[{self.session.session_id}] stream_closed")

    async def _read_single_request(self) -> Optional[Dict[str, Any]]:
        """Читает один JSON-RPC запрос из тела POST запроса."""
        try:
            if self.request.method == "POST":
                body = await self.request.body()
                if body:
                    req_data = json.loads(body.decode('utf-8'))
                    mcp_logger.info(f"[{self.session.session_id}] request_read method={req_data.get('method')}")
                    self.session.last_activity = time.time()
                    return req_data
        except Exception as e:
            mcp_logger.error(f"[{self.session.session_id}] read_request_error: {e}")
        return None


@mcp.custom_route("/mcp-auto", methods=["GET", "POST"])
async def mcp_auto_session(request: Request) -> Response:
    """MCP streamable-http адаптер на одном URL (/mcp-auto)."""
    started = time.time()
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    session_id = _extract_mcp_session_id(request.headers) or str(uuid.uuid4())

    _log_mcp_auto_event(
        request_id=request_id,
        session_id=session_id,
        method=None,
        phase="stream_opened",
        status_code=200,
        latency_ms=(time.time() - started) * 1000,
    )

    if request.method == "GET":
        incoming_session_id = _extract_mcp_session_id(request.headers)
        if not incoming_session_id:
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=None,
                phase="stream_closed",
                status_code=400,
                latency_ms=(time.time() - started) * 1000,
                error="missing_session_id",
            )
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": "server-error",
                    "error": {"code": -32600, "message": "Bad Request: Missing session ID"},
                },
                status_code=400,
                headers={"mcp-session-id": session_id},
            )
        session_id = incoming_session_id
        if session_id not in session_manager.sessions:
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=None,
                phase="stream_closed",
                status_code=404,
                latency_ms=(time.time() - started) * 1000,
                error="session_not_found",
            )
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": "server-error",
                    "error": {"code": -32600, "message": "Session not found"},
                },
                status_code=404,
                headers={"mcp-session-id": session_id},
            )

        async def keepalive_stream() -> AsyncIterator[str]:
            try:
                while True:
                    await sleep(15)
                    yield ": keepalive\n\n"
            except asyncio.CancelledError:
                # Клиент закрыл соединение - нормальный сценарий для long-lived SSE.
                pass
            finally:
                _log_mcp_auto_event(
                    request_id=request_id,
                    session_id=session_id,
                    method=None,
                    phase="stream_closed",
                    status_code=200,
                    latency_ms=(time.time() - started) * 1000,
                )

        return StreamingResponse(
            keepalive_stream(),
            media_type=None,
            headers={
                "Content-Type": "text/event-stream; charset=utf-8",
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
                "mcp-session-id": session_id,
            },
        )

    try:
        raw_body = await request.body()
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            response = _jsonrpc_error(-32700, "Parse error", None)
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=None,
                phase="stream_closed",
                status_code=200,
                latency_ms=(time.time() - started) * 1000,
                error="parse_error",
            )
            return _mcp_sse_response(payload=response, session_id=session_id, status_code=200)

        try:
            rpc = JsonRpcRequest.model_validate(payload)
        except ValidationError:
            response = _jsonrpc_error(-32600, "Invalid Request", payload.get("id") if isinstance(payload, dict) else None)
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=None,
                phase="stream_closed",
                status_code=200,
                latency_ms=(time.time() - started) * 1000,
                error="validation_error",
            )
            return _mcp_sse_response(payload=response, session_id=session_id, status_code=200)

        session = session_manager.get_or_create_session(session_id)
        session.last_activity = time.time()
        adapter = MCPAutoTransportAdapter(session=session, request=request, request_id=request_id)

        if rpc.method == "initialize":
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=rpc.method,
                phase="initialize_received",
                status_code=200,
                latency_ms=(time.time() - started) * 1000,
            )
        elif rpc.method == "notifications/initialized":
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=rpc.method,
                phase="initialized_received",
                status_code=200,
                latency_ms=(time.time() - started) * 1000,
            )

        response_payload = await adapter.dispatch(rpc)

        response_status = 200
        if rpc.method == "initialize":
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=rpc.method,
                phase="initialize_sent",
                status_code=response_status,
                latency_ms=(time.time() - started) * 1000,
            )
        elif rpc.method == "notifications/initialized":
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=rpc.method,
                phase="initialized_ack_sent",
                status_code=response_status,
                latency_ms=(time.time() - started) * 1000,
            )
            _log_mcp_auto_event(
                request_id=request_id,
                session_id=session_id,
                method=rpc.method,
                phase="stream_kept_open=true",
                status_code=response_status,
                latency_ms=(time.time() - started) * 1000,
            )

        _log_mcp_auto_event(
            request_id=request_id,
            session_id=session_id,
            method=rpc.method,
            phase="stream_closed",
            status_code=response_status,
            latency_ms=(time.time() - started) * 1000,
        )
        return _mcp_sse_response(payload=response_payload, session_id=session_id, status_code=response_status)
    except asyncio.CancelledError:
        _log_mcp_auto_event(
            request_id=request_id,
            session_id=session_id,
            method=None,
            phase="stream_closed",
            status_code=499,
            latency_ms=(time.time() - started) * 1000,
            error="request_cancelled",
        )
        raise
    except Exception as e:
        _log_mcp_auto_event(
            request_id=request_id,
            session_id=session_id,
            method=None,
            phase="stream_closed",
            status_code=500,
            latency_ms=(time.time() - started) * 1000,
            error=str(e),
        )
        return _mcp_sse_response(
            payload=_jsonrpc_error(-32603, f"Internal error: {str(e)}", None),
            session_id=session_id,
            status_code=200,
        )


@mcp.custom_route("/mcp-auto", methods=["DELETE"])
async def mcp_auto_session_delete(request: Request) -> Response:
    """DELETE ack для /mcp-auto без мгновенного удаления сессии (избегаем race у клиентов)."""
    session_id = _extract_mcp_session_id(request.headers)
    if session_id and session_id in session_manager.sessions:
        session_manager.sessions[session_id].last_activity = time.time()
    return Response(
        status_code=200,
        headers={
            "mcp-session-id": session_id,
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
        } if session_id else {},
    )


@mcp.custom_route("/mcp-custom", methods=["POST"])
async def mcp_custom_proxy(request: Request) -> Response:
    """Кастомный proxy для /mcp endpoint с поддержкой x-collection-name.

    FastMCP не предоставляет доступ к middleware для streamable-http транспорта,
    поэтому мы создаём свой endpoint который сохраняет request и проксирует запрос.
    """
    _current_request_ctx.set(request)
    try:
        # Читаем тело запроса
        body = await request.body()

        # FastMCP обрабатывает /mcp endpoint автоматически через internal router
        # Мы не можем легко проксировать, поэтому используем SSE endpoint
        # Возвращаем ошибку с предложением использовать /jsonrpc
        return JSONResponse({
            "error": "Please use /jsonrpc endpoint for custom headers support",
            "jsonrpc": "2.0",
            "id": None,
            "message": "For x-collection-name header support, use /jsonrpc endpoint instead"
        }, status_code=400)
    finally:
        _current_request = None


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

            # Определяем схему для каждого tool вручную
            tool_schemas = {
                "search_1c_documentation": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Наименование объекта конфигурации или часть его имени для поиска в документации 1С"
                        },
                        "object_type": {
                            "type": "string",
                            "description": "Фильтр по типу объекта конфигурации 1С",
                            "enum": ["Справочник", "Документ", "РегистрСведений", "РегистрНакопления", "Константа", "Перечисление", "ПланВидовХарактеристик"]
                        },
                        "limit": {
                            "type": "integer",
                            "description": f"Максимальное количество результатов поиска (по умолчанию {DEFAULT_SEARCH_LIMIT})"
                        }
                    },
                    "required": ["query"]
                }
            }

            for tool_name, tool in tools_dict.items():
                input_schema = tool_schemas.get(tool_name, {
                    "type": "object",
                    "properties": {}
                })

                tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": input_schema
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

            # Сохраняем request в глобальной переменной для доступа в tool
            _current_request_ctx.set(request)
            _current_request = request

            try:
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
            finally:
                _current_request_ctx.set(None)  # Очищаем в любом случае

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
