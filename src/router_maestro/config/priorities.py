"""Model priority configuration."""

from enum import Enum

from pydantic import BaseModel, Field


class FallbackStrategy(str, Enum):
    """Fallback strategy options."""

    PRIORITY = "priority"  # Fallback to next model in priorities list
    SAME_MODEL = "same-model"  # Only fallback to providers with the same model
    NONE = "none"  # Disable fallback, fail immediately


class FallbackConfig(BaseModel):
    """Fallback configuration."""

    strategy: FallbackStrategy = Field(
        default=FallbackStrategy.PRIORITY,
        description="Fallback strategy",
    )
    maxRetries: int = Field(  # noqa: N815
        default=2,
        ge=0,
        le=10,
        description="Maximum number of fallback retries",
    )


class ModelOverride(BaseModel):
    """Per-model token limit overrides."""

    max_prompt_tokens: int | None = None
    max_output_tokens: int | None = None
    max_context_window_tokens: int | None = None


class ThinkingBudgetConfig(BaseModel):
    """Server-side thinking budget defaults."""

    default_budget: int = Field(default=16000, ge=1024, le=128000)
    auto_enable: bool = Field(
        default=False,
        description="Auto-enable thinking for capable models when client doesn't request it",
    )
    model_budgets: dict[str, int] = Field(
        default_factory=dict,
        description="Per-model budget overrides keyed by model name",
    )


class PrioritiesConfig(BaseModel):
    """Configuration for model priorities and fallback."""

    priorities: list[str] = Field(
        default_factory=list,
        description="Model priorities in format 'provider/model', highest to lowest",
    )
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    model_overrides: dict[str, ModelOverride] = Field(
        default_factory=dict,
        description="Per-model token limit overrides keyed by 'provider/model' or 'model'",
    )
    thinking: ThinkingBudgetConfig = Field(default_factory=ThinkingBudgetConfig)

    @classmethod
    def get_default(cls) -> "PrioritiesConfig":
        """Get default empty priorities configuration."""
        return cls(priorities=[])

    def get_priority(self, provider: str, model: str) -> int:
        """Get priority for a model.

        Args:
            provider: Provider name
            model: Model ID

        Returns:
            Priority index (lower = higher priority), or 999999 if not in list
        """
        key = f"{provider}/{model}"
        try:
            return self.priorities.index(key)
        except ValueError:
            return 999999

    def add_priority(self, provider: str, model: str, position: int | None = None) -> None:
        """Add a model to priorities.

        Args:
            provider: Provider name
            model: Model ID
            position: Position to insert (None = append to end)
        """
        key = f"{provider}/{model}"
        # Remove if already exists
        if key in self.priorities:
            self.priorities.remove(key)
        # Insert at position
        if position is None:
            self.priorities.append(key)
        else:
            self.priorities.insert(position, key)

    def remove_priority(self, provider: str, model: str) -> bool:
        """Remove a model from priorities.

        Args:
            provider: Provider name
            model: Model ID

        Returns:
            True if removed, False if not found
        """
        key = f"{provider}/{model}"
        if key in self.priorities:
            self.priorities.remove(key)
            return True
        return False
