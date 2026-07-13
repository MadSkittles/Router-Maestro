"""Fuzzy model ID matching utilities."""

from __future__ import annotations

from rapidfuzz import fuzz, process

from router_maestro.providers.base import ModelInfo
from router_maestro.utils.model_sort import parse_model_id, strip_date_suffix

MIN_MATCH_SCORE = 80.0
CONFIDENT_MATCH_SCORE = 85.0
AMBIGUITY_SCORE_MARGIN = 1.0


class AmbiguousModelMatchError(ValueError):
    """Raised when a model alias cannot select one provider/model identity."""


def normalize_model_identity(model_id: str) -> str:
    """Normalize spelling while preserving a concrete date/version identity."""
    result = model_id.lower()
    result = result.replace(" ", "-")
    return result.replace(".", "-")


def normalize_model_id(model_id: str) -> str:
    """Normalize a model ID to its date-independent family for fuzzy comparison.

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
    return strip_date_suffix(normalize_model_identity(model_id))


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

    # An exact raw cache alias preserves an upstream namespace. Every other
    # slash is a provider boundary, including an unknown provider name.
    provider_filter: str | None = None
    match_query = query
    if "/" in query:
        exact_entry = models_cache.get(query)
        exact_is_raw_alias = exact_entry is not None and query != (
            f"{exact_entry[0]}/{exact_entry[1].id}"
        )
        if not exact_is_raw_alias:
            possible_provider, possible_query = query.split("/", 1)
            provider_filter = possible_provider.lower()
            match_query = possible_query

    identity_query = normalize_model_identity(match_query)

    # Build identities first so case/dot aliases keep a concrete dated version.
    # When provider filter is set, scan prefixed keys to find that provider's models.
    # Otherwise, scan bare keys only (avoiding duplicates).
    identity_candidates: dict[str, list[tuple[str, str, ModelInfo]]] = {}
    for cache_key, (provider_name, model_info) in models_cache.items():
        qualified_key = f"{provider_name}/{model_info.id}"
        is_qualified = cache_key == qualified_key
        if provider_filter is not None:
            # With a provider filter, use prefixed keys for the target provider
            if not is_qualified:
                continue
            if provider_name.lower() != provider_filter:
                continue
            # Return the prefixed cache key so the router resolves the correct provider
            identity = normalize_model_identity(model_info.id)
            identity_candidates.setdefault(identity, []).append(
                (cache_key, provider_name, model_info)
            )
        else:
            # Without filter, use bare keys only to avoid duplicates
            if is_qualified:
                continue
            identity = normalize_model_identity(model_info.id)
            identity_candidates.setdefault(identity, []).append(
                (cache_key, provider_name, model_info)
            )

    if not identity_candidates:
        return None

    if identity_query in identity_candidates:
        return identity_candidates[identity_query][0][0]

    normalized_query = strip_date_suffix(identity_query)
    candidates: dict[str, list[tuple[str, str, ModelInfo]]] = {}
    for identity, hits in identity_candidates.items():
        candidates.setdefault(strip_date_suffix(identity), []).extend(hits)

    # Family aliases intentionally select the newest concrete catalog version.
    if normalized_query in candidates:
        return _select_best(candidates[normalized_query])

    # Use rapidfuzz to find matches
    matches = process.extract(
        normalized_query,
        list(candidates.keys()),
        scorer=fuzz.WRatio,
        limit=5,
        score_cutoff=MIN_MATCH_SCORE,
    )

    if not matches:
        return None

    top_normalized, top_score, _index = matches[0]
    if top_score < CONFIDENT_MATCH_SCORE:
        raise AmbiguousModelMatchError(
            f"Model alias '{query}' is a low-confidence match; use provider/model"
        )
    tied_families = {
        matched_normalized
        for matched_normalized, score, _index in matches
        if top_score - score <= AMBIGUITY_SCORE_MARGIN
    }
    if len(tied_families) > 1:
        raise AmbiguousModelMatchError(
            f"Model alias '{query}' matches multiple model families; use provider/model"
        )

    # Score is authoritative across families. Date/version only breaks ties among
    # concrete catalog entries that normalize to the winning family.
    return _select_best(candidates[top_normalized])


def _select_best(hits: list[tuple[str, str, ModelInfo]]) -> str:
    """Select the best model from a list of candidates.

    All hits must belong to one normalized family. Prefer newest date/version,
    then retain catalog order as the stable provider tiebreaker.

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
