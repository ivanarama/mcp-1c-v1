#!/usr/bin/env python3
"""
Интеграционные тесты MCP handshake для endpoint /mcp-auto.

Проверяет:
1. Полный handshake: initialize -> notifications/initialized -> resources/list
2. Канал не закрывается после notifications/initialized
3. Session lifecycle: MCP-Session-Id стабилен
4. SSE режим: chunked encoding, без content-length
5. 10-кратный стресс-тест handshake
"""

import requests
import json
import time
import sys

MCP_URL = "http://192.168.1.13:8001/mcp-auto"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def log(msg, color=""):
    print(f"{color}{msg}{Colors.ENDC}")

def check_response(resp, expected_contains=None, expected_status=200):
    """Проверяет HTTP ответ."""
    if resp.status_code != expected_status:
        log(f"  ✗ Status code: {resp.status_code} (expected {expected_status})", Colors.RED)
        return False
    log(f"  ✓ Status code: {resp.status_code}", Colors.GREEN)

    # Проверяем SSE заголовки
    ct = resp.headers.get('content-type', '')
    if 'text/event-stream' not in ct:
        log(f"  ✗ Content-Type: {ct} (expected text/event-stream)", Colors.RED)
        return False
    log(f"  ✓ Content-Type: {ct}", Colors.GREEN)

    # Проверяем chunked encoding (нет content-length)
    if 'content-length' in resp.headers:
        cl = resp.headers['content-length']
        log(f"  ✗ Has Content-Length: {cl} (should use chunked)", Colors.RED)
        return False
    log(f"  ✓ Using chunked encoding (no Content-Length)", Colors.GREEN)

    # Проверяем keep-alive (для HTTP/1.1 keep-alive по умолчанию, заголовок может отсутствовать)
    conn = resp.headers.get('connection', '').lower()
    # Пустой connection или keep-alive - оба норма для HTTP/1.1
    if conn and conn != 'keep-alive':
        log(f"  ! Connection: {conn} (not keep-alive, but may be OK for HTTP/1.1)", Colors.YELLOW)
    else:
        log(f"  ✓ Connection OK (keep-alive default for HTTP/1.1)", Colors.GREEN)

    if expected_contains:
        if expected_contains in resp.text:
            log(f"  ✓ Response contains: {expected_contains[:50]}...", Colors.GREEN)
        else:
            log(f"  ✗ Response missing: {expected_contains[:50] if len(expected_contains) > 50 else expected_contains}", Colors.RED)
            log(f"    Got: {resp.text[:200]}", Colors.RED)
            return False

    return True

def test_get_handshake():
    """Тест GET handshake."""
    log("\n=== TEST 1: GET Handshake ===", Colors.BLUE)
    resp = requests.get(MCP_URL)

    if not check_response(resp, expected_status=200):
        return False

    if 'event: endpoint' not in resp.text or '/mcp-auto' not in resp.text:
        log(f"  ✗ Invalid handshake response", Colors.RED)
        log(f"    Got: {resp.text}", Colors.RED)
        return False

    log(f"  ✓ Valid SSE handshake", Colors.GREEN)
    return True

def test_initialize():
    """Тест initialize."""
    log("\n=== TEST 2: Initialize ===", Colors.BLUE)

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"}
        }
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }

    resp = requests.post(MCP_URL, json=payload, headers=headers)

    if not check_response(resp, expected_contains='"protocolVersion"'):
        return False

    # Проверяем MCP-Session-Id
    session_id = resp.headers.get('mcp-session-id') or resp.headers.get('MCP-Session-Id')
    if not session_id:
        log(f"  ✗ No MCP-Session-Id header", Colors.RED)
        return False
    log(f"  ✓ MCP-Session-Id: {session_id}", Colors.GREEN)

    # Проверяем результат
    try:
        # Парсим SSE ответ
        lines = resp.text.strip().split('\n')
        data_line = None
        for line in lines:
            if line.startswith('data: '):
                data_line = line[6:]
                break

        if not data_line:
            log(f"  ✗ No data in SSE response", Colors.RED)
            return False

        result = json.loads(data_line)
        if result.get('jsonrpc') != '2.0':
            log(f"  ✗ Invalid JSON-RPC version", Colors.RED)
            return False
        if 'result' not in result:
            log(f"  ✗ No result in response", Colors.RED)
            return False

        log(f"  ✓ Valid JSON-RPC response", Colors.GREEN)
        return session_id  # Возвращаем session_id для следующих тестов

    except json.JSONDecodeError as e:
        log(f"  ✗ JSON parse error: {e}", Colors.RED)
        return False

def test_initialized_notification(session_id=None):
    """Тест notifications/initialized."""
    log("\n=== TEST 3: notifications/initialized ===", Colors.BLUE)

    payload = {
        "jsonrpc": "2.0",
        "method": "notifications/initialized"
        # Обратите внимание: БЕЗ id - это notification
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    if session_id:
        headers['mcp-session-id'] = session_id

    resp = requests.post(MCP_URL, json=payload, headers=headers)

    # Для notifications согласно JSON-RPC 2.0 spec ожидаем HTTP 204 No Content
    if resp.status_code == 204:
        log(f"  ✓ Notification accepted (HTTP 204 No Content)", Colors.GREEN)

        # Проверяем что session_id сохранился
        returned_session = resp.headers.get('mcp-session-id') or resp.headers.get('MCP-Session-Id')
        if session_id and returned_session != session_id:
            log(f"  ✗ Session changed! Was: {session_id}, now: {returned_session}", Colors.RED)
            return False

        if returned_session:
            log(f"  ✓ Session stable: {returned_session}", Colors.GREEN)

        return True

    # Для обратной совместимости также проверяем HTTP 200 с JSON
    if not check_response(resp, expected_status=200):
        return False

    # Проверяем что вернулся корректный ответ (не ошибка)
    lines = resp.text.strip().split('\n')
    data_line = None
    for line in lines:
        if line.startswith('data: '):
            data_line = line[6:]
            break

    if not data_line:
        log(f"  ✗ No data in SSE response after initialized", Colors.RED)
        return False

    try:
        result = json.loads(data_line)

        # Для notification может быть result или error
        if 'error' in result:
            log(f"  ✗ Server returned error: {result['error']}", Colors.RED)
            return False

        log(f"  ✓ Valid response to notification", Colors.GREEN)

        # Проверяем что session_id сохранился
        returned_session = resp.headers.get('mcp-session-id') or resp.headers.get('MCP-Session-Id')
        if session_id and returned_session != session_id:
            log(f"  ✗ Session changed! Was: {session_id}, now: {returned_session}", Colors.RED)
            return False

        if returned_session:
            log(f"  ✓ Session stable: {returned_session}", Colors.GREEN)

        return True

    except json.JSONDecodeError as e:
        log(f"  ✗ JSON parse error: {e}", Colors.RED)
        log(f"    Got: {data_line}", Colors.RED)
        return False

def test_resources_list_after_initialized(session_id=None):
    """
    КРИТИЧЕСКИЙ ТЕСТ: Канал НЕ закрыт после initialized.

    Этот тест выполняется СРАЗУ после initialized и проверяет,
    что соединение всё ещё работает и сессия сохранилась.
    """
    log("\n=== TEST 4: resources/list AFTER initialized (channel check) ===", Colors.BLUE)

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "resources/list"
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    if session_id:
        headers['mcp-session-id'] = session_id

    try:
        resp = requests.post(MCP_URL, json=payload, headers=headers, timeout=5)
    except requests.exceptions.ConnectionError as e:
        log(f"  ✗ Connection closed! Error: {e}", Colors.RED)
        log(f"    This means the server closed the channel after initialized!", Colors.RED)
        return False
    except requests.exceptions.Timeout:
        log(f"  ✗ Request timeout - server may have closed connection", Colors.RED)
        return False

    if not check_response(resp, expected_contains='"resources"'):
        return False

    # Проверяем session
    returned_session = resp.headers.get('mcp-session-id') or resp.headers.get('MCP-Session-Id')
    if session_id and returned_session != session_id:
        log(f"  ✗ Session changed after initialized!", Colors.RED)
        return False

    log(f"  ✓ Channel still OPEN after initialized", Colors.GREEN)
    log(f"  ✓ Session maintained: {returned_session}", Colors.GREEN)
    return True

def test_resources_templates_list(session_id=None):
    """Тест resources/templates/list."""
    log("\n=== TEST 5: resources/templates/list ===", Colors.BLUE)

    payload = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "resources/templates/list"
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream"
    }

    if session_id:
        headers['mcp-session-id'] = session_id

    resp = requests.post(MCP_URL, json=payload, headers=headers)

    if not check_response(resp, expected_contains='"resourceTemplates"'):
        return False

    log(f"  ✓ resources/templates/list works", Colors.GREEN)
    return True

def test_handshake_sequence():
    """
    ПОЛНЫЙ ТЕСТ HANDSHAKE последовательности как у rmcp/Codex.

    Порядок: initialize -> notifications/initialized -> resources/list
    """
    log("\n" + "="*60, Colors.BLUE)
    log("FULL HANDSHAKE SEQUENCE TEST (like rmcp/Codex)", Colors.BLUE)
    log("="*60, Colors.BLUE)

    # 1. GET handshake (опционально, но некоторые клиенты делают)
    if not test_get_handshake():
        return False

    # 2. initialize
    session_id = test_initialize()
    if not session_id:
        return False

    # 3. notifications/initialized
    if not test_initialized_notification(session_id):
        return False

    # 4. СРАЗУ после initialized - resources/list (критический тест канала)
    if not test_resources_list_after_initialized(session_id):
        return False

    # 5. resources/templates/list
    if not test_resources_templates_list(session_id):
        return False

    log("\n" + "="*60, Colors.GREEN)
    log("✓ FULL HANDSHAKE SEQUENCE PASSED", Colors.GREEN)
    log("="*60 + "\n", Colors.GREEN)

    return True

def test_stress_handshake(runs=10):
    """
    СТРЕСС-ТЕСТ: 10 последовательных прогонов handshake.

    Критерий: 0 ошибок transport closed/EOF/reset.
    """
    log("\n" + "="*60, Colors.BLUE)
    log(f"STRESS TEST: {runs} sequential handshake runs", Colors.BLUE)
    log("="*60, Colors.BLUE)

    passed = 0
    failed = 0
    errors = []

    for i in range(1, runs + 1):
        log(f"\n--- Run {i}/{runs} ---", Colors.BLUE)

        try:
            # Выполняем полный handshake
            if test_handshake_sequence():
                passed += 1
            else:
                failed += 1
                errors.append(f"Run {i}: Handshake sequence failed")

        except requests.exceptions.ConnectionError as e:
            failed += 1
            errors.append(f"Run {i}: ConnectionError - {e}")
            log(f"  ✗ Connection error: {e}", Colors.RED)

        except requests.exceptions.Timeout as e:
            failed += 1
            errors.append(f"Run {i}: Timeout - {e}")
            log(f"  ✗ Timeout: {e}", Colors.RED)

        except Exception as e:
            failed += 1
            errors.append(f"Run {i}: Unexpected error - {e}")
            log(f"  ✗ Unexpected error: {e}", Colors.RED)

        # Небольшая пауза между запусками
        time.sleep(0.1)

    log("\n" + "="*60, Colors.BLUE)
    log(f"STRESS TEST RESULTS: {runs} runs, {passed} passed, {failed} failed", Colors.BLUE)
    log("="*60, Colors.BLUE)

    if errors:
        log("\nErrors:", Colors.RED)
        for error in errors:
            log(f"  - {error}", Colors.RED)

    if failed == 0:
        log(f"\n✓ ALL {runs} RUNS PASSED - 0 transport errors!", Colors.GREEN)
        return True
    else:
        log(f"\n✗ {failed} RUN(S) FAILED", Colors.RED)
        return False

def main():
    """Запуск всех тестов."""
    log("\n" + "="*60, Colors.BLUE)
    log("MCP HANDSHAKE INTEGRATION TESTS", Colors.BLUE)
    log(f"Testing endpoint: {MCP_URL}")
    log("="*60 + "\n", Colors.BLUE)

    # Проверяем что сервер доступен
    try:
        resp = requests.get(MCP_URL.replace('/mcp-auto', '/health'), timeout=5)
        if resp.status_code != 200:
            log(f"Server health check failed: {resp.status_code}", Colors.RED)
            sys.exit(1)
    except Exception as e:
        log(f"Cannot connect to server: {e}", Colors.RED)
        sys.exit(1)

    log("Server is running\n", Colors.GREEN)

    # Запускаем стресс-тест
    success = test_stress_handshake(runs=10)

    if success:
        log("\n" + "="*60, Colors.GREEN)
        log("ALL TESTS PASSED ✓", Colors.GREEN)
        log("="*60 + "\n", Colors.GREEN)
        sys.exit(0)
    else:
        log("\n" + "="*60, Colors.RED)
        log("SOME TESTS FAILED ✗", Colors.RED)
        log("="*60 + "\n", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main()
