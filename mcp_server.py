import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx


MCP_SERVER_NAME = "rag-semantic-cache-plugin"
MCP_SERVER_VERSION = "1.3.0"
API_BASE_URL = os.getenv("RAG_CAG_API_BASE_URL", "http://127.0.0.1:8000/api/v1")


def _write_message(payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _read_message() -> Optional[Dict[str, Any]]:
    content_length = None

    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None

        if line in (b"\r\n", b"\n"):
            break

        lowered = line.decode("ascii", errors="ignore").strip().lower()
        if lowered.startswith("content-length:"):
            value = lowered.split(":", maxsplit=1)[1].strip()
            content_length = int(value)

    if content_length is None:
        return None

    body = sys.stdin.buffer.read(content_length)
    if not body:
        return None

    return json.loads(body.decode("utf-8"))


def _success(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tools_payload() -> List[Dict[str, Any]]:
    return [
        {
            "name": "query",
            "description": "Query exact cache -> semantic cache -> RAG -> LLM pipeline.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "use_cache": {"type": "boolean", "default": True},
                    "use_rag": {"type": "boolean", "default": True},
                },
                "required": ["query"],
            },
        },
        {
            "name": "ingest_document",
            "description": "Ingest a document into the local RAG store.",
            "inputSchema": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
        {
            "name": "ingest_file",
            "description": "Ingest a local text file into the RAG store (chunked).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mode": {"type": "string", "enum": ["append", "replace"], "default": "append"},
                },
                "required": ["path"],
            },
        },
        {
            "name": "ingest_repo",
            "description": "Ingest a local repository directory into the RAG store.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "mode": {"type": "string", "enum": ["append", "replace"], "default": "replace"},
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "cache_stats",
            "description": "Get semantic cache and document store counters.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "clear_cache",
            "description": "Delete all semantic cache entries.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "health_check",
            "description": "Check local API health.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "get_metrics",
            "description": "Get cache hit rate, latency averages, and estimated token savings.",
            "inputSchema": {"type": "object", "properties": {}},
        },
    ]


def _http_call(
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Dict[str, Any]:
    url = f"{API_BASE_URL}{path}"
    with httpx.Client(timeout=timeout) as client:
        response = client.request(method=method, url=url, json=payload)
        response.raise_for_status()
        return response.json()


def _tool_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "query":
        request_payload = {
            "query": arguments.get("query", ""),
            "use_cache": arguments.get("use_cache", True),
            "use_rag": arguments.get("use_rag", True),
        }
        if not request_payload["query"]:
            raise ValueError("query is required")
        return _http_call("POST", "/tools/query", request_payload)

    if tool_name == "ingest_document":
        content = arguments.get("content", "")
        if not content:
            raise ValueError("content is required")
        return _http_call("POST", "/tools/ingest", {"content": content})

    if tool_name == "ingest_file":
        path = arguments.get("path", "")
        if not path:
            raise ValueError("path is required")
        return _http_call(
            "POST",
            "/tools/ingest/file",
            {"path": path, "mode": arguments.get("mode", "append")},
            timeout=300.0,
        )

    if tool_name == "ingest_repo":
        path = arguments.get("path", "")
        if not path:
            raise ValueError("path is required")
        payload: Dict[str, Any] = {
            "path": path,
            "mode": arguments.get("mode", "replace"),
        }
        if arguments.get("extensions"):
            payload["extensions"] = arguments["extensions"]
        return _http_call("POST", "/tools/ingest/repo", payload, timeout=600.0)

    if tool_name == "cache_stats":
        return _http_call("GET", "/tools/cache/stats")

    if tool_name == "clear_cache":
        return _http_call("DELETE", "/tools/cache")

    if tool_name == "health_check":
        # Health route lives outside /api/v1 prefix.
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{API_BASE_URL.rsplit('/api/v1', maxsplit=1)[0]}/health")
            response.raise_for_status()
            return response.json()

    if tool_name == "get_metrics":
        return _http_call("GET", "/tools/metrics")

    raise ValueError(f"Unknown tool: {tool_name}")


def _to_tool_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2),
            }
        ]
    }


def _handle_request(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    method = message.get("method")
    request_id = message.get("id")

    if method == "initialize":
        return _success(
            request_id,
            {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
                "capabilities": {"tools": {}},
            },
        )

    if method == "notifications/initialized":
        return None

    if method == "ping":
        return _success(request_id, {})

    if method == "tools/list":
        return _success(request_id, {"tools": _tools_payload()})

    if method == "tools/call":
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not tool_name:
            return _error(request_id, -32602, "Missing tool name")

        try:
            result = _tool_call(tool_name, arguments)
            return _success(request_id, _to_tool_result(result))
        except httpx.HTTPError as exc:
            return _success(request_id, _to_tool_result({"status": "error", "detail": str(exc)}))
        except Exception as exc:
            return _error(request_id, -32000, str(exc))

    if request_id is not None:
        return _error(request_id, -32601, f"Method not found: {method}")
    return None


def main() -> None:
    while True:
        message = _read_message()
        if message is None:
            break

        response = _handle_request(message)
        if response is not None:
            _write_message(response)


if __name__ == "__main__":
    main()
