"""Provider-qualified model identity used by routing internals."""

from dataclasses import dataclass


def validate_provider_id(provider: str) -> str:
    """Validate one provider component in the public ``provider/model`` domain."""
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("provider ID must be a non-empty string")
    if provider != provider.strip():
        raise ValueError("provider ID cannot contain leading or trailing whitespace")
    if "/" in provider:
        raise ValueError("provider ID cannot contain '/'")
    return provider


def validate_upstream_model_id(model_id: str) -> str:
    """Validate one non-empty upstream model suffix."""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("upstream model ID must be a non-empty string")
    if model_id != model_id.strip():
        raise ValueError("upstream model ID cannot contain leading or trailing whitespace")
    if any(not segment for segment in model_id.split("/")):
        raise ValueError("upstream model ID cannot contain empty path segments")
    return model_id


def qualify_model_id(provider: str, model_id: str) -> str:
    """Normalize a possibly public ID for legacy/UI callers.

    Core routing code should use ``ModelRef(...).qualified_id`` because only
    typed provenance can distinguish a public prefix from an upstream namespace.
    """
    validate_provider_id(provider)
    validate_upstream_model_id(model_id)
    prefix = f"{provider}/"
    if model_id.startswith(prefix):
        validate_upstream_model_id(model_id[len(prefix) :])
        return model_id
    return f"{prefix}{model_id}"


def catalog_model_public_id(
    provider: str,
    catalog_id: str,
    *,
    id_is_qualified: bool = False,
) -> str:
    """Encode a catalog model using its explicit raw/public provenance."""
    ref = (
        ModelRef.from_qualified_catalog_id(provider, catalog_id)
        if id_is_qualified
        else ModelRef.from_catalog_id(provider, catalog_id)
    )
    return ref.qualified_id


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Unique reference to one upstream model owned by one provider."""

    provider: str
    upstream_id: str

    def __post_init__(self) -> None:
        validate_provider_id(self.provider)
        validate_upstream_model_id(self.upstream_id)

    @classmethod
    def from_catalog_id(cls, provider: str, catalog_id: str) -> ModelRef:
        """Construct from a raw upstream catalog ID."""
        validate_provider_id(provider)
        validate_upstream_model_id(catalog_id)
        return cls(provider=provider, upstream_id=catalog_id)

    @classmethod
    def from_qualified_catalog_id(cls, provider: str, catalog_id: str) -> ModelRef:
        """Construct from a catalog source explicitly known to return public IDs."""
        validate_provider_id(provider)
        validate_upstream_model_id(catalog_id)
        prefix = f"{provider}/"
        if not catalog_id.startswith(prefix):
            raise ValueError("qualified catalog model prefix does not match its provider")
        return cls(provider=provider, upstream_id=catalog_id[len(prefix) :])

    @property
    def qualified_id(self) -> str:
        # Both components are already validated and provenance is explicit, so
        # always encode them. An upstream ID may legitimately begin with the
        # same text as its provider (``openrouter/openrouter/auto``).
        return f"{self.provider}/{self.upstream_id}"
