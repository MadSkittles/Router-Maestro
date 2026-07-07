"""Beta Header Auto-Strip: configurable removal of anthropic-beta tokens.

Strips specified beta tokens from the anthropic-beta header before forwarding
requests upstream. Supports trailing wildcard patterns (e.g. "output-128k-*").

This replaces ad-hoc per-issue hotfixes (like #113) with a configurable
strip list in priorities.json.
"""


def strip_beta_tokens(
    header_value: str | None,
    strip_patterns: list[str],
) -> str | None:
    """Remove matching tokens from an anthropic-beta header value.

    Args:
        header_value: Comma-separated beta token string (e.g. "token-a,token-b")
        strip_patterns: Patterns to strip. Each is either an exact match or
            supports trailing wildcard (*). E.g. "output-128k-*" strips any
            token starting with "output-128k-".

    Returns:
        The filtered header value with matching tokens removed, or None if
        all tokens were stripped (meaning the header should be omitted).
    """
    if not header_value or not strip_patterns:
        return header_value if header_value else None

    tokens = [t.strip() for t in header_value.split(",") if t.strip()]
    if not tokens:
        return None

    filtered = [t for t in tokens if not _matches_any(t, strip_patterns)]

    if not filtered:
        return None
    return ",".join(filtered)


def _matches_any(token: str, patterns: list[str]) -> bool:
    """Check if a token matches any of the strip patterns."""
    for pattern in patterns:
        if pattern.endswith("*"):
            if token.startswith(pattern[:-1]):
                return True
        elif token == pattern:
            return True
    return False
