"""Fuzzy model ID matching utilities."""

from __future__ import annotations

from rapidfuzz import fuzz, process

from router_maestro.providers.base import ModelInfo
from router_maestro.utils.model_sort import parse_model_id, strip_date_suffix


def normalize_model_id(model_id: str) -> str:
    """Normalize a model ID for fuzzy comparison.

    Pipeline:
    1. Lowercase
    2. Replace spaces with hyphens
    3. Replace dots with hyphens (version segments)
    4. Strip date suffix

    Args:
        model_id: Raw model ID string (e.g. "Opus 4.6")

    Returns:
        Normalized model ID (e.g. "opus-4-6")
    """
    result = model_id.lower()
    result = result.replace(" ", "-")
    result = result.replace(".", "-")
    result = strip_date_suffix(result)
    return result


def fuzzy_match_model(
    query: str,
    models_cache: dict[str, tuple[str, ModelInfo]],
) -> str | None:
    """Find the best fuzzy match for a model query in the cache.

    Args:
        query: User-provided model ID (e.g. "Opus 4.6", "opus-4-6")
        models_cache: Router's model cache mapping cache_key -> (provider_name, ModelInfo)

    Returns:
        The original cache key of the best match, or None if no match found
    """
    if not models_cache:
        return None

    # Extract provider filter from query if present
    provider_filter: str | None = None
    match_query = query
    if "/" in query:
        provider_filter, match_query = query.split("/", 1)

    normalized_query = normalize_model_id(match_query)

    # Build candidates: normalized_id -> list of (cache_key, provider_name, ModelInfo)
    # When provider filter is set, scan prefixed keys to find that provider's models.
    # Otherwise, scan bare keys only (avoiding duplicates).
    candidates: dict[str, list[tuple[str, str, ModelInfo]]] = {}
    for cache_key, (provider_name, model_info) in models_cache.items():
        has_slash = "/" in cache_key
        if provider_filter is not None:
            # With a provider filter, use prefixed keys for the target provider
            if not has_slash:
                continue
            key_provider, bare_id = cache_key.split("/", 1)
            if key_provider != provider_filter:
                continue
            # Return the prefixed cache key so the router resolves the correct provider
            normalized = normalize_model_id(bare_id)
            if normalized not in candidates:
                candidates[normalized] = []
            candidates[normalized].append((cache_key, provider_name, model_info))
        else:
            # Without filter, use bare keys only to avoid duplicates
            if has_slash:
                continue
            normalized = normalize_model_id(cache_key)
            if normalized not in candidates:
                candidates[normalized] = []
            candidates[normalized].append((cache_key, provider_name, model_info))

    if not candidates:
        return None

    # Check for exact normalized match first (prevents false positives like gpt-4o -> gpt-4o-mini)
    if normalized_query in candidates:
        all_hits = candidates[normalized_query]
        return _select_best(all_hits)

    # Use rapidfuzz to find matches
    matches = process.extract(
        normalized_query,
        list(candidates.keys()),
        scorer=fuzz.WRatio,
        limit=5,
        score_cutoff=80.0,
    )

    if not matches:
        return None

    # Collect all original models behind the matched normalized keys
    all_hits = []
    for matched_normalized, _score, _idx in matches:
        all_hits.extend(candidates[matched_normalized])

    if not all_hits:
        return None

    return _select_best(all_hits)


def find_extended_context_variant(
    model_id: str,
    models_cache: dict[str, tuple[str, ModelInfo]],
) -> str | None:
    """Find the extended-context variant of a model in the cache.

    Given a base model ID like ``claude-opus-4-6`` or ``claude-opus-4-7``,
    search the cache for a matching ``-1m`` suffixed model (e.g.
    ``claude-opus-4.6-1m``) or a ``-1m-internal`` suffixed model (e.g.
    ``claude-opus-4.7-1m-internal``). The plain ``-1m`` variant is preferred
    when both are present.

    The lookup normalizes both query and cache keys so that dot/hyphen differences
    (``4.6`` vs ``4-6``) are handled transparently.

    Args:
        model_id: The base model ID from the request (may include provider prefix).
        models_cache: Router's model cache mapping cache_key -> (provider_name, ModelInfo).

    Returns:
        The original cache key of the variant, or None if not found.
    """
    if not models_cache:
        return None

    # Strip provider prefix for normalization, but remember it for filtering
    provider_filter: str | None = None
    bare_query = model_id
    if "/" in model_id:
        provider_filter, bare_query = model_id.split("/", 1)

    normalized_query = normalize_model_id(bare_query)
    # Target candidates, in preference order. Plain "-1m" wins when both exist
    # (keeps the 4.6 behavior); "-1m-internal" is the fallback for opus-4.7.
    targets = (f"{normalized_query}-1m", f"{normalized_query}-1m-internal")
    matches: dict[str, str] = {}

    for cache_key, (_provider_name, _model_info) in models_cache.items():
        has_slash = "/" in cache_key
        if provider_filter is not None:
            if not has_slash:
                continue
            key_provider, bare_key = cache_key.split("/", 1)
            if key_provider != provider_filter:
                continue
        else:
            if has_slash:
                continue
            bare_key = cache_key

        normalized_key = normalize_model_id(bare_key)
        if normalized_key in targets and normalized_key not in matches:
            matches[normalized_key] = cache_key

    for target in targets:
        if target in matches:
            return matches[target]
    return None


_REASONING_SUFFIXES: tuple[str, ...] = ("-low", "-medium", "-high", "-xhigh")


def find_reasoning_variant(
    model_id: str,
    effort: str,
    models_cache: dict[str, tuple[str, ModelInfo]],
) -> str | None:
    """Find a ``<model>-<effort>`` variant in the cache.

    For example, given ``claude-opus-4-7`` and ``effort="high"``, returns the
    cache key of ``claude-opus-4.7-high`` if present. Returns ``None`` when
    ``model_id`` already encodes a reasoning suffix or when no matching variant
    exists.

    Args:
        model_id: The base model ID from the request (may include provider prefix).
        effort: Reasoning effort tier (``"high"`` or ``"xhigh"``).
        models_cache: Router's model cache mapping cache_key -> (provider_name, ModelInfo).

    Returns:
        The original cache key of the variant, or None if not found.
    """
    if not models_cache:
        return None

    provider_filter: str | None = None
    bare_query = model_id
    if "/" in model_id:
        provider_filter, bare_query = model_id.split("/", 1)

    normalized_query = normalize_model_id(bare_query)
    # If the request already targets an effort-encoded variant, leave it alone.
    if any(normalized_query.endswith(suffix) for suffix in _REASONING_SUFFIXES):
        return None

    target = f"{normalized_query}-{effort}"

    for cache_key, (_provider_name, _model_info) in models_cache.items():
        has_slash = "/" in cache_key
        if provider_filter is not None:
            if not has_slash:
                continue
            key_provider, bare_key = cache_key.split("/", 1)
            if key_provider != provider_filter:
                continue
        else:
            if has_slash:
                continue
            bare_key = cache_key

        if normalize_model_id(bare_key) == target:
            return cache_key

    return None


def _select_best(hits: list[tuple[str, str, ModelInfo]]) -> str:
    """Select the best model from a list of candidates.

    Prefers newest version (by date), dated over undated, first-registered as tiebreaker.

    Args:
        hits: List of (cache_key, provider_name, ModelInfo)

    Returns:
        The cache key of the best match
    """

    def _sort_key(item: tuple[str, str, ModelInfo]) -> tuple[int, int]:
        _cache_key, _provider_name, model_info = item
        parsed = parse_model_id(model_info.id)
        # Dated models (version > 0) win over undated (version == 0)
        has_date = 1 if parsed.version > 0 else 0
        return (has_date, parsed.version)

    best = max(hits, key=_sort_key)
    return best[0]
