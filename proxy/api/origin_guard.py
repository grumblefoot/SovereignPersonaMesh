"""Brief FX-B: cross-site request guard for the admin API (review finding 3).

With CORS `allow_origins=["*"]` removed, any web page open in the user's
browser could still POST to http://localhost:5050/admin/api/v1/... with a
simple request (no preflight): build a `full_logs` report, download the
private bundle, or trigger shutdown. SillyTavern calls `/v1` from its SERVER
(not the browser) and the admin dashboard is same-origin, so no browser
request legitimately crosses origins — this middleware refuses the ones that
do, on every path starting with `/admin`.

Rules (Origin-first, Referer as a fallback for unsafe methods):
- `Origin` present and its `scheme://host[:port]` is neither the request's own
  origin (from Host + scheme) nor `http://127.0.0.1:<port>` /
  `http://localhost:<port>` for the port in Host -> 403.
- Unsafe method (POST/PUT/PATCH/DELETE) without `Origin` but with a foreign
  `Referer` -> 403. No `Origin` and no `Referer` (curl, scripts, the CLI):
  allowed.
- `/v1` and `/health` are not affected.

Purely normative checks on the visible request headers; no DNS, no sockets.
"""
from urllib.parse import urlsplit

_ADMIN_PREFIX = "/admin"
# The world engine mounts the same guard on its whole API (/api/v1): it binds loopback
# by default and its endpoints (shutdown, reset, room/character mutations) take simple
# POSTs that any web page could send without a preflight (second review, item 1).
_ENGINE_PREFIX = "/api/v1"
_UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost")


def _origin_of(url: str):
    """'scheme://host[:port]' lowercased, or None if not an absolute URL."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return None
    scheme, netloc = parts.scheme.lower(), parts.netloc.lower()
    if not scheme or not netloc:
        return None
    return f"{scheme}://{netloc}"


def request_origin(scope) -> str:
    """The request's own origin: scheme (TLS or X-Forwarded-Proto) + Host."""
    headers = {k.decode("latin-1").lower(): v.decode("latin-1")
               for k, v in scope.get("headers", [])}
    scheme = "https" if scope.get("scheme") == "https" else \
        (headers.get("x-forwarded-proto", "").split(",")[0].strip().lower() or "http")
    host = headers.get("host", "").strip().lower()
    return f"{scheme}://{host}"


def allowed_origins(request_origin: str) -> set:
    """Same-origin plus the loopback aliases for the Host's port."""
    allowed = {request_origin}
    for h in _LOOPBACK_HOSTS:
        for scheme in ("http", "https"):
            allowed.add(f"{scheme}://{h}")
    try:
        port = urlsplit(request_origin).port
    except ValueError:
        port = None
    if port is not None:
        for h in _LOOPBACK_HOSTS:
            for scheme in ("http", "https"):
                allowed.add(f"{scheme}://{h}:{port}")
    return allowed


def is_allowed_origin(origin_header: str, own_origin: str) -> bool:
    """True when an `Origin` header value is same-origin or a loopback alias."""
    if origin_header in ("null", ""):
        # 'null': sandboxed/file: pages — never a legitimate admin caller.
        return False
    origin = _origin_of(origin_header)
    if origin is None:
        return False
    return origin in allowed_origins(own_origin)


def is_referer_same_site(referer_header: str, own_origin: str) -> bool:
    """True when a `Referer`'s origin is same-site (any of the allowed origins)."""
    referer = _origin_of(referer_header)
    if referer is None:
        # Unparsable, or same-origin file:// / about: refs with no origin part.
        return True
    return referer in allowed_origins(own_origin)


def _host_ok(host_header: str, bind_env: str = "SPM_HOST") -> bool:
    """Loopback Host names are always fine. Other names only when the service was
    deliberately opened to the network (`bind_env` — SPM_HOST for the proxy,
    SPM_ENGINE_HOST for the engine — not a loopback address); a test client's default
    'testserver' counts as local."""
    import os
    name = host_header.strip().lower()
    if name.startswith("["):                      # [::1]:5050
        name = name[1:].split("]", 1)[0]
    else:
        name = name.split(":", 1)[0]
    if name in ("127.0.0.1", "localhost", "::1", "testserver", ""):
        return True
    if name == "test" or name.endswith(".test"):
        # RFC 2606 reserves .test: it can never resolve on the public DNS, so no
        # rebinding page can carry it. The suite's ASGI clients use such hosts.
        return True
    if os.getenv("SPM_IN_CONTAINER") == "1" and os.getenv("SPM_LAN_ACCESS") != "1":
        # The Docker bundle binds 0.0.0.0 inside the container but publishes 127.0.0.1 only.
        return False
    bind = os.getenv(bind_env, "127.0.0.1").strip().lower()
    return bind not in ("127.0.0.1", "localhost", "::1")


class OriginGuardMiddleware:
    """Pure ASGI middleware guarding `prefixes` (default: /admin) against cross-site
    requests. The engine passes prefixes=("/api/v1",) and bind_env="SPM_ENGINE_HOST"."""

    def __init__(self, app, prefixes=(_ADMIN_PREFIX,), bind_env: str = "SPM_HOST"):
        self.app = app
        self.prefixes = tuple(prefixes)
        self.bind_env = bind_env

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or \
                not scope.get("path", "").startswith(self.prefixes):
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1").lower(): v.decode("latin-1")
                   for k, v in scope.get("headers", [])}
        own = request_origin(scope)
        origin = headers.get("origin")
        method = scope.get("method", "").upper()

        # DNS rebinding: a page on evil.example can make its own name resolve to 127.0.0.1;
        # its requests then carry Host: evil.example and look same-origin. While SPM listens
        # on this machine only (the default), an admin request must name a loopback host.
        if not _host_ok(headers.get("host", ""), self.bind_env):
            await self._deny(send)
            return

        if origin is not None:
            ok = is_allowed_origin(origin, own)
        elif method in _UNSAFE_METHODS:
            referer = headers.get("referer")
            ok = referer is None or is_referer_same_site(referer, own)
        else:
            ok = True

        if not ok:
            await self._deny(send)
            return
        await self.app(scope, receive, send)

    async def _deny(self, send):
        import json
        body = json.dumps({"error": "cross-site request refused"}).encode()
        await send({"type": "http.response.start", "status": 403,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def require_loopback(request) -> None:
    """403 unless the peer address is loopback (brief FX-B: /report full_logs
    and /shutdown are loopback-only, like the engine's shutdown endpoint)."""
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost"):
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="loopback clients only")
