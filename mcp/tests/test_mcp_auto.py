import json
import sys
from pathlib import Path

import httpx
import pytest


CURRENT_DIR = Path(__file__).resolve().parent
MCP_DIR = CURRENT_DIR.parent
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

import mcp_server as server


def _resolve_asgi_app():
    if hasattr(server.mcp, "http_app"):
        return server.mcp.http_app(path="/mcp", transport="streamable-http")

    for attr_name in ("_app", "app"):
        app_obj = getattr(server.mcp, attr_name, None)
        if app_obj is None:
            continue
        if callable(app_obj):
            try:
                app_obj = app_obj()
            except TypeError:
                pass
        if app_obj is not None:
            return app_obj
    raise RuntimeError("Unable to resolve ASGI app from FastMCP instance")


def _parse_sse_message(response: httpx.Response) -> dict:
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "content-length" not in response.headers
    body = response.text
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise AssertionError(f"SSE data frame not found. Body: {body}")


async def _post_json(client: httpx.AsyncClient, payload: dict, session_id: str | None = None) -> httpx.Response:
    headers = {"content-type": "application/json", "accept": "text/event-stream"}
    if session_id:
        headers["mcp-session-id"] = session_id
    return await client.post("/mcp-auto", json=payload, headers=headers)


async def _perform_handshake(client: httpx.AsyncClient) -> str:
    initialize_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "1.0"},
        },
    }
    init_resp = await _post_json(client, initialize_payload)
    init_data = _parse_sse_message(init_resp)
    assert init_data["result"]["protocolVersion"] == "2024-11-05"
    session_id = init_resp.headers.get("mcp-session-id")
    assert session_id

    initialized_notification = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    initialized_resp = await _post_json(client, initialized_notification, session_id=session_id)
    assert initialized_resp.status_code in (200, 202)
    if initialized_resp.status_code == 200:
        initialized_data = _parse_sse_message(initialized_resp)
        assert "error" not in initialized_data
    else:
        assert initialized_resp.text == ""
    assert initialized_resp.headers.get("mcp-session-id") == session_id
    return session_id


@pytest.fixture(autouse=True)
def clear_sessions():
    server.session_manager.sessions.clear()
    yield
    server.session_manager.sessions.clear()


@pytest.fixture
async def client():
    app = _resolve_asgi_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


@pytest.mark.asyncio
async def test_mcp_auto_full_handshake(client: httpx.AsyncClient):
    session_id = await _perform_handshake(client)

    resources_resp = await _post_json(
        client,
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
        session_id=session_id,
    )
    resources_data = _parse_sse_message(resources_resp)
    assert resources_data["id"] == 2
    assert resources_data["result"] == {"resources": []}
    assert resources_resp.headers.get("mcp-session-id") == session_id
    assert server.session_manager.sessions[session_id].initialized is True


@pytest.mark.asyncio
async def test_mcp_auto_templates_list(client: httpx.AsyncClient):
    session_id = await _perform_handshake(client)

    templates_resp = await _post_json(
        client,
        {"jsonrpc": "2.0", "id": 3, "method": "resources/templates/list"},
        session_id=session_id,
    )
    templates_data = _parse_sse_message(templates_resp)
    assert templates_data["id"] == 3
    assert templates_data["result"] == {"resourceTemplates": []}


@pytest.mark.asyncio
async def test_mcp_auto_session_reuse(client: httpx.AsyncClient):
    session_id = await _perform_handshake(client)

    resources_resp = await _post_json(
        client,
        {"jsonrpc": "2.0", "id": 4, "method": "resources/list"},
        session_id=session_id,
    )
    resources_data = _parse_sse_message(resources_resp)
    assert resources_data["result"] == {"resources": []}
    assert resources_resp.headers.get("mcp-session-id") == session_id
    assert list(server.session_manager.sessions.keys()) == [session_id]


@pytest.mark.asyncio
async def test_mcp_auto_notification_without_id(client: httpx.AsyncClient):
    initialize_resp = await _post_json(
        client,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "1.0"}},
        },
    )
    session_id = initialize_resp.headers.get("mcp-session-id")
    assert session_id

    notification_resp = await _post_json(
        client,
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        session_id=session_id,
    )
    assert notification_resp.status_code in (200, 202)
    if notification_resp.status_code == 200:
        notification_data = _parse_sse_message(notification_resp)
        assert notification_data == {"jsonrpc": "2.0", "result": {}}
    else:
        assert notification_resp.text == ""
    assert server.session_manager.sessions[session_id].initialized is True


@pytest.mark.asyncio
async def test_mcp_auto_10x_stability(client: httpx.AsyncClient):
    for run in range(10):
        session_id = await _perform_handshake(client)
        resources_resp = await _post_json(
            client,
            {"jsonrpc": "2.0", "id": run + 100, "method": "resources/list"},
            session_id=session_id,
        )
        resources_data = _parse_sse_message(resources_resp)
        assert resources_data["result"] == {"resources": []}
        assert "error" not in resources_data
