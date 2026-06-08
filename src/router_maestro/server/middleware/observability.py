"""HTTP observability middleware."""

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from router_maestro.server.observability import HttpMetrics, path_template_from_scope
from router_maestro.utils import get_logger

REQUEST_ID_HEADER = "X-Request-ID"

logger = get_logger("server.middleware.observability")


async def observability_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Record HTTP metrics and attach a request id to every response."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
    request.state.request_id = request_id

    method = request.method
    path_template = path_template_from_scope(request.scope)
    logger.info(
        "HTTP request started: request_id=%s method=%s path_template=%s",
        request_id,
        method,
        path_template,
    )

    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_seconds = time.perf_counter() - start_time

    # Route resolution happens inside call_next, so refresh the template before
    # metrics/logging to avoid recording raw request paths.
    path_template = path_template_from_scope(request.scope)
    status_code = str(response.status_code)

    response.headers[REQUEST_ID_HEADER] = request_id
    metrics = getattr(request.app.state, "http_metrics", None)
    if isinstance(metrics, HttpMetrics):
        metrics.observe_request(
            method=method,
            path_template=path_template,
            status=status_code,
            duration_seconds=elapsed_seconds,
        )

    logger.info(
        "HTTP request completed: request_id=%s method=%s path_template=%s status=%s "
        "elapsed_ms=%.1f",
        request_id,
        method,
        path_template,
        status_code,
        elapsed_seconds * 1000,
    )
    return response
