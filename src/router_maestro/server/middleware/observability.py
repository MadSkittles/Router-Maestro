"""HTTP observability middleware."""

import re
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from router_maestro.server.observability import (
    HttpMetrics,
    normalize_http_method,
    path_template_from_scope,
)
from router_maestro.utils import get_logger

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

logger = get_logger("server.middleware.observability")


def resolve_request_id(scope: Scope) -> str:
    """Return a safe request ID from headers or generate a new one."""
    headers = scope.get("headers", ())
    for name, value in headers:
        if name.lower() == b"x-request-id":
            try:
                request_id = value.decode("ascii")
            except UnicodeDecodeError:
                break
            if REQUEST_ID_PATTERN.fullmatch(request_id):
                return request_id
            break
    return uuid.uuid4().hex


def record_http_request(
    *,
    scope: Scope,
    method: str,
    path_template: str,
    status_code: str,
    duration_seconds: float,
    request_id: str,
) -> None:
    """Record HTTP request metrics and completion log fields."""
    app = scope.get("app")
    metrics = getattr(getattr(app, "state", None), "http_metrics", None)
    if isinstance(metrics, HttpMetrics):
        metrics.observe_request(
            method=method,
            path_template=path_template,
            status=status_code,
            duration_seconds=duration_seconds,
        )

    logger.info(
        "HTTP request completed: request_id=%s method=%s path_template=%s status=%s "
        "elapsed_ms=%.1f",
        request_id,
        method,
        path_template,
        status_code,
        duration_seconds * 1000,
    )


class ObservabilityMiddleware:
    """Record HTTP metrics at ASGI response completion and attach request IDs."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id

        method = normalize_http_method(scope.get("method", ""))
        path_template = path_template_from_scope(scope)
        logger.info(
            "HTTP request started: request_id=%s method=%s path_template=%s",
            request_id,
            method,
            path_template,
        )

        start_time = time.perf_counter()
        status_code = "500"
        recorded = False

        def record(status: str) -> None:
            nonlocal recorded
            if recorded:
                return
            recorded = True
            elapsed_seconds = time.perf_counter() - start_time
            record_http_request(
                scope=scope,
                method=method,
                path_template=path_template_from_scope(scope),
                status_code=status,
                duration_seconds=elapsed_seconds,
                request_id=request_id,
            )

        async def send_with_observability(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = str(message["status"])
                raw_headers = list(message.get("headers", ()))
                raw_headers.append((REQUEST_ID_HEADER.lower().encode("ascii"), request_id.encode()))
                message = {**message, "headers": raw_headers}
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                record(status_code)

            await send(message)

        try:
            await self.app(scope, receive, send_with_observability)
        except Exception:
            record("500")
            raise
