"""Canonical model identity domain tests."""

import pytest

from router_maestro.routing.model_ref import ModelRef, qualify_model_id


@pytest.mark.parametrize(
    ("provider", "upstream_id"),
    [
        ("", "m"),
        ("   ", "m"),
        (" p", "m"),
        ("p ", "m"),
        ("team/p", "m"),
        ("p", ""),
        ("p", "   "),
        ("p", " m"),
        ("p", "m "),
    ],
)
def test_model_ref_rejects_invalid_identity_components(provider: str, upstream_id: str) -> None:
    with pytest.raises(ValueError):
        ModelRef(provider, upstream_id)

    with pytest.raises(ValueError):
        qualify_model_id(provider, upstream_id)


@pytest.mark.parametrize("catalog_id", ["", "   ", "p/", "p/   "])
def test_catalog_identity_rejects_empty_upstream_component(catalog_id: str) -> None:
    with pytest.raises(ValueError):
        ModelRef.from_catalog_id("p", catalog_id)


def test_catalog_identity_preserves_unambiguous_upstream_slash_suffix() -> None:
    ref = ModelRef.from_qualified_catalog_id("p", "p/team/model")

    assert ref == ModelRef("p", "team/model")
    assert ref.qualified_id == "p/team/model"


def test_catalog_identity_preserves_namespaced_upstream_id() -> None:
    ref = ModelRef.from_catalog_id("openrouter", "meta-llama/llama-3.1-8b-instruct")

    assert ref == ModelRef("openrouter", "meta-llama/llama-3.1-8b-instruct")
    assert ref.qualified_id == "openrouter/meta-llama/llama-3.1-8b-instruct"


def test_catalog_identity_preserves_upstream_id_starting_with_provider_name() -> None:
    ref = ModelRef.from_catalog_id("openrouter", "openrouter/auto")

    assert ref == ModelRef("openrouter", "openrouter/auto")
    assert ref.qualified_id == "openrouter/openrouter/auto"
