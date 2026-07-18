"""Request-level defenses for a localhost server.

A local HTTP server is reachable by every webpage the user's browser visits:
DNS rebinding hands a hostile page a hostname it controls that resolves to
127.0.0.1, and plain cross-origin fetches can fire state-changing POSTs (the
same-origin policy blocks reading the *response*, not sending the request).
Two checks close this down:

- Host allowlist: only loopback hosts (127.0.0.1, localhost, ::1 — any
  port). A rebound request arrives with the attacker's hostname in Host and
  gets a 403.
- Origin allowlist on state-changing methods: a request carrying an Origin
  header that is not a loopback (or Tauri shell) origin gets a 403.
  Requests with no Origin at all — curl, the Tauri shell, same-origin
  navigation — pass untouched.

WebSocket scopes get both checks too: browsers do not apply the same-origin
policy to WebSocket connections, so a hostile page could otherwise read job
and download event streams.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.websockets import WebSocketClose

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Receive, Scope, Send

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

# The Tauri shell's webview origins (tauri://localhost on Linux/macOS,
# http://tauri.localhost on Windows). Tauri usually sends no Origin on API
# calls — these cover the configurations that do.
DEFAULT_ALLOWED_ORIGINS = frozenset({"tauri://localhost", "http://tauri.localhost"})


def host_is_loopback(host_header: str) -> bool:
    """True if a Host header (optionally with port) names a loopback address."""
    host = host_header.strip().lower()
    if host.startswith("["):  # [::1]:port
        host = host[1 : host.index("]")] if "]" in host else ""
    elif host.count(":") == 1:  # host:port
        host = host.rsplit(":", 1)[0]
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def origin_allowed(origin: str, extra: frozenset[str] = DEFAULT_ALLOWED_ORIGINS) -> bool:
    if origin in extra:
        return True
    parts = urlsplit(origin)
    return parts.scheme in ("http", "https") and host_is_loopback(parts.netloc)


class LocalRequestsOnly:
    """Pure ASGI middleware: reject rebound Hosts and cross-origin writes."""

    def __init__(
        self, app: ASGIApp, allowed_origins: frozenset[str] = DEFAULT_ALLOWED_ORIGINS
    ) -> None:
        self.app = app
        self.allowed_origins = allowed_origins

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if not host_is_loopback(headers.get("host", "")):
            await self._reject(
                scope,
                receive,
                send,
                "request Host is not a loopback address — LoRAForge only answers local callers",
            )
            return
        origin = headers.get("origin")
        guarded = scope["type"] == "websocket" or scope.get("method") not in _SAFE_METHODS
        if origin is not None and guarded and not origin_allowed(origin, self.allowed_origins):
            await self._reject(
                scope, receive, send, f"cross-origin request from '{origin}' refused"
            )
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send, message: str) -> None:
        if scope["type"] == "http":
            await PlainTextResponse(message, status_code=403)(scope, receive, send)
        else:
            await WebSocketClose(code=4403, reason=message)(scope, receive, send)
