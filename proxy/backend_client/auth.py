"""BACKEND_API_KEY on requests to the model server, and nowhere else.

Local servers (Lemonade, LM Studio, llama-server) ignore keys, so SPM never sent
one; a hosted OpenAI-compatible API answers 401 without it. Every client that
talks to the model server uses BackendKeyAuth: it adds "Authorization: Bearer
<key>" only when the request goes to the backend URL's own scheme, host and port,
so the key never reaches the world engine, an embedding server on another host,
or a redirect target.
"""

from typing import Callable, Optional, Union

import httpx

StrOrFn = Union[str, Callable[[], str], None]


def backend_api_key() -> str:
    """BACKEND_API_KEY from the effective settings, read at call time (hot reload)."""
    try:
        from config.manager import get_settings_manager
        return str(get_settings_manager().get_settings().get("BACKEND_API_KEY") or "").strip()
    except Exception:
        return ""


def same_origin(a: str, b: str) -> bool:
    """True when two URLs share scheme, host and port (default ports filled in)."""
    try:
        ua, ub = httpx.URL(a), httpx.URL(b)
    except Exception:
        return False
    if not ua.host or not ub.host:
        return False

    def port(u: httpx.URL) -> Optional[int]:
        return u.port or {"http": 80, "https": 443}.get(u.scheme)

    return (ua.scheme, ua.host.lower(), port(ua)) == (ub.scheme, ub.host.lower(), port(ub))


def _value(v: StrOrFn) -> str:
    return str((v() if callable(v) else v) or "").strip()


class BackendKeyAuth(httpx.Auth):
    """httpx auth: Bearer key for requests to `base_url`'s origin only.

    `base_url` and `key` may be strings or zero-argument callables (the live
    client follows the settings without a restart). An Authorization header the
    caller set explicitly is left alone.
    """

    def __init__(self, base_url: StrOrFn, key: StrOrFn = backend_api_key):
        self._base_url = base_url
        self._key = key

    def auth_flow(self, request: httpx.Request):
        key = _value(self._key)
        if key and "authorization" not in request.headers \
                and same_origin(str(request.url), _value(self._base_url)):
            request.headers["Authorization"] = f"Bearer {key}"
        yield request
