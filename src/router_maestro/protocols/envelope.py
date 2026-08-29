"""Lazy request envelope that preserves the native fast path."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from router_maestro.protocols.models import RequestManifest, SemanticRequest, WireProtocol
from router_maestro.protocols.runtime import ProtocolRuntime


class RequestEnvelope:
    """An isolated raw request with loop-neutral, single-flight IR materialization."""

    def __init__(
        self,
        runtime: ProtocolRuntime,
        payload: Mapping[str, Any],
        *,
        path: str = "",
        query: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        take_ownership: bool = False,
    ) -> None:
        if not isinstance(path, str):
            raise TypeError("request path must be a string")
        self._runtime = runtime
        # HTTP routes hand over a request-scoped JSON object and avoid an eager
        # copy. Public/direct callers keep the defensive snapshot by default.
        # Shallow inspection and identity preparation treat nested values as
        # read-only; provider dialects own any copy they need for mutation.
        self._raw_payload = (
            payload if take_ownership and isinstance(payload, dict) else dict(payload)
        )
        if not take_ownership:
            self._raw_payload = deepcopy(self._raw_payload)
        self._path = path
        self._query = tuple(dict(query or {}).items())
        self._headers = tuple(dict(headers or {}).items())
        self._manifest = runtime.inspect_request(self._raw_payload)
        if self._manifest.protocol is not runtime.protocol:
            raise ValueError("request manifest protocol must match its runtime")

        self._state_lock = threading.Lock()
        self._semantic_request: SemanticRequest | None = None
        self._materialization_count = 0
        self._flight: concurrent.futures.Future[SemanticRequest] | None = None

    def replace_manifest(self, manifest: RequestManifest) -> None:
        """Replace only shallow routing facts before dispatch begins."""
        if manifest.protocol is not self._runtime.protocol:
            raise ValueError("replacement manifest protocol must match its runtime")
        with self._state_lock:
            if self._semantic_request is not None or self._flight is not None:
                raise RuntimeError("request manifest cannot change after IR materialization begins")
            self._manifest = manifest

    def require_context_tokens(self, tokens: int) -> None:
        """Attach a client-side context hint to this request's shallow manifest."""
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError("context token requirement must be a positive integer")
        from dataclasses import replace

        self.replace_manifest(replace(self._manifest, requested_context_tokens=tokens))

    def estimate_input_tokens(self) -> int:
        """Memoize a shallow wire estimate for Auto context filtering only."""
        if self._manifest.estimated_input_tokens is not None:
            return self._manifest.estimated_input_tokens
        import json
        from dataclasses import replace

        from router_maestro.utils.tokens import estimate_tokens

        serialized = json.dumps(
            self._raw_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        estimated = estimate_tokens(serialized)
        self.replace_manifest(replace(self._manifest, estimated_input_tokens=estimated))
        return estimated

    @property
    def protocol(self) -> WireProtocol:
        return self._runtime.protocol

    @property
    def runtime(self) -> ProtocolRuntime:
        """The request-scoped ingress codec used to build this envelope."""
        return self._runtime

    @property
    def manifest(self) -> RequestManifest:
        return self._manifest

    @property
    def model(self) -> str | None:
        """Model discovered by the runtime's shallow inspector."""
        return self._manifest.model

    @property
    def stream(self) -> bool:
        """Streaming mode discovered without semantic IR materialization."""
        return self._manifest.stream

    @property
    def path(self) -> str:
        return self._path

    @property
    def query(self) -> dict[str, str]:
        """Return an isolated snapshot of the ingress query context."""
        return dict(self._query)

    @property
    def headers(self) -> dict[str, str]:
        """Return an isolated snapshot of the ingress header context."""
        return dict(self._headers)

    @property
    def raw_payload(self) -> dict[str, Any]:
        """Return a fresh native-wire copy; callers can mutate it safely."""
        return deepcopy(self._raw_payload)

    def native_payload(self) -> dict[str, Any]:
        """Return a top-level copy for copy-on-write identity preparation.

        Nested wire values remain shared with the preserved snapshot and must
        be treated as read-only. A provider dialect that needs to rewrite a
        nested branch owns copying that branch before mutation.
        """
        return dict(self._raw_payload)

    @property
    def materialization_count(self) -> int:
        with self._state_lock:
            return self._materialization_count

    async def semantic_ir(self) -> SemanticRequest:
        """Materialize semantic IR once, sharing the result across loops and threads."""
        with self._state_lock:
            if self._semantic_request is not None:
                return self._semantic_request
            flight = self._flight
            owns_flight = flight is None
            if flight is None:
                flight = concurrent.futures.Future()
                self._flight = flight

        if not owns_flight:
            return await asyncio.shield(asyncio.wrap_future(flight))

        try:
            # Decoders are allowed to normalize their working copy. Keep that
            # mutation isolated from the snapshot used by identity fallbacks.
            materialized = await self._runtime.decode_request(self.raw_payload)
            if not isinstance(materialized, SemanticRequest):
                raise TypeError("protocol runtime must decode requests to SemanticRequest")
        except BaseException as error:
            with self._state_lock:
                if self._flight is flight:
                    self._flight = None
            if not flight.done():
                flight.set_exception(error)
            raise

        with self._state_lock:
            if self._semantic_request is None:
                self._semantic_request = materialized
                self._materialization_count += 1
            cached = self._semantic_request
            if self._flight is flight:
                self._flight = None

        if not flight.done():
            flight.set_result(cached)
        return cached
