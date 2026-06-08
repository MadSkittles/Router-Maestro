"""Prometheus metrics helpers for Router-Maestro server observability."""

from dataclasses import dataclass

from prometheus_client import (
    CONTENT_TYPE_LATEST as PROMETHEUS_CONTENT_TYPE_LATEST,
)
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
)
from prometheus_client.exposition import generate_latest

METRIC_PREFIX = "router_maestro"
CONTENT_TYPE_LATEST = PROMETHEUS_CONTENT_TYPE_LATEST

HTTP_REQUESTS_TOTAL = f"{METRIC_PREFIX}_http_requests_total"
HTTP_REQUEST_DURATION_SECONDS = f"{METRIC_PREFIX}_http_request_duration_seconds"

HTTP_DURATION_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60, 120)
HTTP_LABELS = ("method", "path_template", "status")


@dataclass(frozen=True)
class HttpMetrics:
    """HTTP-level Prometheus metrics bound to a specific registry."""

    registry: CollectorRegistry
    requests_total: Counter
    request_duration: Histogram

    def observe_request(
        self,
        *,
        method: str,
        path_template: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record one completed HTTP request."""
        labels = {
            "method": method,
            "path_template": path_template,
            "status": status,
        }
        self.requests_total.labels(**labels).inc()
        self.request_duration.labels(**labels).observe(duration_seconds)


def create_registry() -> CollectorRegistry:
    """Create an isolated Prometheus registry.

    The module intentionally does not register collectors with the default
    global registry at import time, which keeps tests order-independent.
    """
    return CollectorRegistry(auto_describe=True)


def create_http_metrics(registry: CollectorRegistry | None = None) -> HttpMetrics:
    """Create HTTP metrics bound to an explicit or isolated registry."""
    target_registry = registry or create_registry()
    return HttpMetrics(
        registry=target_registry,
        requests_total=Counter(
            HTTP_REQUESTS_TOTAL,
            "Total HTTP requests handled by Router-Maestro.",
            HTTP_LABELS,
            registry=target_registry,
        ),
        request_duration=Histogram(
            HTTP_REQUEST_DURATION_SECONDS,
            "HTTP request duration in seconds.",
            HTTP_LABELS,
            buckets=HTTP_DURATION_BUCKETS,
            registry=target_registry,
        ),
    )


def render_metrics(registry: CollectorRegistry) -> bytes:
    """Render metrics from a registry in Prometheus text format."""
    return generate_latest(registry)
