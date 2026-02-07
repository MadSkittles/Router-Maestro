"""Tests for model ID parsing and sorting."""

from router_maestro.providers.base import ModelInfo
from router_maestro.utils.model_sort import ParsedModelId, parse_model_id, sort_models


class TestParseModelId:
    """Tests for parse_model_id function."""

    def test_openai_base_model(self):
        result = parse_model_id("gpt-4o")
        assert result == ParsedModelId(family="gpt-4o", variant="", version=0, raw_id="gpt-4o")

    def test_openai_mini_variant(self):
        result = parse_model_id("gpt-4o-mini")
        assert result == ParsedModelId(
            family="gpt-4o", variant="mini", version=0, raw_id="gpt-4o-mini"
        )

    def test_openai_turbo_variant(self):
        result = parse_model_id("gpt-4-turbo")
        assert result == ParsedModelId(
            family="gpt-4", variant="turbo", version=0, raw_id="gpt-4-turbo"
        )

    def test_o_series_mini(self):
        result = parse_model_id("o1-mini")
        assert result == ParsedModelId(family="o1", variant="mini", version=0, raw_id="o1-mini")

    def test_o_series_preview(self):
        result = parse_model_id("o1-preview")
        assert result == ParsedModelId(
            family="o1", variant="preview", version=0, raw_id="o1-preview"
        )

    def test_o3_mini(self):
        result = parse_model_id("o3-mini")
        assert result == ParsedModelId(family="o3", variant="mini", version=0, raw_id="o3-mini")

    def test_claude_with_date(self):
        result = parse_model_id("claude-3-5-sonnet-20241022")
        assert result == ParsedModelId(
            family="claude-3-5-sonnet",
            variant="",
            version=20241022,
            raw_id="claude-3-5-sonnet-20241022",
        )

    def test_claude_new_naming(self):
        result = parse_model_id("claude-sonnet-4-20250514")
        assert result == ParsedModelId(
            family="claude-sonnet-4",
            variant="",
            version=20250514,
            raw_id="claude-sonnet-4-20250514",
        )

    def test_unknown_model(self):
        result = parse_model_id("my-custom-model")
        assert result == ParsedModelId(
            family="my-custom-model",
            variant="",
            version=0,
            raw_id="my-custom-model",
        )

    def test_variant_plus_dashed_date(self):
        result = parse_model_id("gpt-4o-mini-2024-07-18")
        assert result == ParsedModelId(
            family="gpt-4o",
            variant="mini",
            version=20240718,
            raw_id="gpt-4o-mini-2024-07-18",
        )

    def test_unknown_suffix_stays_in_family(self):
        result = parse_model_id("my-model-xyz")
        assert result == ParsedModelId(
            family="my-model-xyz",
            variant="",
            version=0,
            raw_id="my-model-xyz",
        )


class TestSortModels:
    """Tests for sort_models function."""

    @staticmethod
    def _model(model_id: str, provider: str, name: str = "") -> ModelInfo:
        return ModelInfo(id=model_id, name=name or model_id, provider=provider)

    def test_group_by_provider(self):
        models = [
            self._model("gpt-4o", "openai"),
            self._model("claude-3-5-sonnet-20241022", "anthropic"),
            self._model("gpt-4", "openai"),
        ]
        result = sort_models(models)
        assert [m.provider for m in result] == ["anthropic", "openai", "openai"]

    def test_group_by_family_within_provider(self):
        models = [
            self._model("gpt-4-turbo", "openai", "GPT-4 Turbo"),
            self._model("gpt-4o-mini", "openai", "GPT-4o Mini"),
            self._model("gpt-4", "openai", "GPT-4"),
            self._model("gpt-4o", "openai", "GPT-4o"),
        ]
        result = sort_models(models)
        # GPT-4 base first, then GPT-4 Turbo (" " < "o"), then GPT-4o, then GPT-4o Mini
        assert result[0].id == "gpt-4"
        assert result[1].id == "gpt-4-turbo"
        assert result[2].id == "gpt-4o"
        assert result[3].id == "gpt-4o-mini"

    def test_newer_family_version_first(self):
        """Models with higher version numbers in family should sort first."""
        models = [
            self._model("gpt-4.1", "openai", "GPT-4.1"),
            self._model("gpt-5.2", "openai", "GPT-5.2"),
            self._model("gpt-5", "openai", "GPT-5"),
            self._model("gpt-5.1", "openai", "GPT-5.1"),
        ]
        result = sort_models(models)
        assert [m.id for m in result] == ["gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1"]

    def test_newer_family_version_first_across_series(self):
        """Newer model series sort before older ones within same provider."""
        models = [
            self._model("claude-opus-41", "github-copilot", "Claude Opus 4.1"),
            self._model("claude-opus-4.6", "github-copilot", "Claude Opus 4.6"),
            self._model("claude-opus-4.5", "github-copilot", "Claude Opus 4.5"),
            self._model("gemini-2.5-pro", "github-copilot", "Gemini 2.5 Pro"),
            self._model("gemini-3-pro-preview", "github-copilot", "Gemini 3 Pro (Preview)"),
        ]
        result = sort_models(models)
        assert result[0].id == "claude-opus-4.6"
        assert result[1].id == "claude-opus-4.5"
        assert result[2].id == "claude-opus-41"
        assert result[3].id == "gemini-3-pro-preview"
        assert result[4].id == "gemini-2.5-pro"

    def test_inconsistent_id_uses_name_for_sorting(self):
        """Model IDs with inconsistent formatting sort correctly via display name."""
        models = [
            self._model("claude-opus-41", "github-copilot", "Claude Opus 4.1"),
            self._model("claude-opus-4.6", "github-copilot", "Claude Opus 4.6"),
            self._model("claude-opus-4.5", "github-copilot", "Claude Opus 4.5"),
        ]
        result = sort_models(models)
        # Sorted by display name: 4.6 > 4.5 > 4.1
        assert result[0].id == "claude-opus-4.6"
        assert result[1].id == "claude-opus-4.5"
        assert result[2].id == "claude-opus-41"

    def test_newer_version_first(self):
        models = [
            self._model("claude-3-5-sonnet-20240620", "anthropic"),
            self._model("claude-3-5-sonnet-20241022", "anthropic"),
        ]
        result = sort_models(models)
        assert result[0].id == "claude-3-5-sonnet-20241022"
        assert result[1].id == "claude-3-5-sonnet-20240620"

    def test_variant_after_base(self):
        models = [
            self._model("gpt-4o-mini", "openai"),
            self._model("gpt-4o", "openai"),
        ]
        result = sort_models(models)
        # base (variant="") sorts before variant="mini"
        assert result[0].id == "gpt-4o"
        assert result[1].id == "gpt-4o-mini"

    def test_immutability(self):
        original = [
            self._model("gpt-4o-mini", "openai"),
            self._model("gpt-4o", "openai"),
        ]
        original_copy = list(original)
        sort_models(original)
        assert original == original_copy

    def test_empty_list(self):
        assert sort_models([]) == []

    def test_comprehensive_real_models(self):
        models = [
            self._model("o1-preview", "github-copilot", "O1 Preview"),
            self._model("gpt-4o-mini-2024-07-18", "openai", "GPT-4o Mini 2024-07-18"),
            self._model("claude-3-5-sonnet-20241022", "anthropic", "Claude 3.5 Sonnet 20241022"),
            self._model("gpt-4o", "openai", "GPT-4o"),
            self._model("claude-3-5-sonnet-20240620", "anthropic", "Claude 3.5 Sonnet 20240620"),
            self._model("gpt-4o-mini", "openai", "GPT-4o Mini"),
            self._model("gpt-4-turbo", "openai", "GPT-4 Turbo"),
            self._model("o1-mini", "github-copilot", "O1 Mini"),
            self._model("claude-sonnet-4-20250514", "anthropic", "Claude Sonnet 4 20250514"),
        ]
        result = sort_models(models)

        # Anthropic first, then github-copilot, then openai
        providers = [m.provider for m in result]
        assert providers == [
            "anthropic",
            "anthropic",
            "anthropic",
            "github-copilot",
            "github-copilot",
            "openai",
            "openai",
            "openai",
            "openai",
        ]

        # Within anthropic: "Claude 3.5" sorts before "Claude Sonnet" (space < 'S'),
        # and within claude-3-5-sonnet, newer date first
        anthropic_models = [m for m in result if m.provider == "anthropic"]
        assert anthropic_models[0].id == "claude-3-5-sonnet-20241022"
        assert anthropic_models[1].id == "claude-3-5-sonnet-20240620"
        assert anthropic_models[2].id == "claude-sonnet-4-20250514"

        # Within github-copilot: O1 family, mini then preview (alphabetical)
        copilot_models = [m for m in result if m.provider == "github-copilot"]
        assert copilot_models[0].id == "o1-mini"
        assert copilot_models[1].id == "o1-preview"

        # Within openai: GPT-4 Turbo first (" " < "o"), then GPT-4o, GPT-4o Mini variants
        openai_models = [m for m in result if m.provider == "openai"]
        assert openai_models[0].id == "gpt-4-turbo"
        assert openai_models[1].id == "gpt-4o"
        assert openai_models[2].id == "gpt-4o-mini"
        assert openai_models[3].id == "gpt-4o-mini-2024-07-18"
