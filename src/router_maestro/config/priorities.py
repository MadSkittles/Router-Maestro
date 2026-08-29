"""Runtime routing, fallback, and automatic model configuration."""

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_AUTO_ROUTER_MODEL = "github-copilot/gpt-5.6-luna"
DEFAULT_AUTO_TASK_MODELS = {
    "fast": "github-copilot/gpt-5.6-luna",
    "general": "github-copilot/gpt-5.6-terra",
    "coding": "github-copilot/gpt-5.6-terra",
    "deep_reasoning": "github-copilot/gpt-5.6-sol",
}


def _validate_qualified_model_id(value: str) -> str:
    if value != value.strip() or "/" not in value:
        raise ValueError("model must use the qualified 'provider/model' form")
    provider, model = value.split("/", 1)
    if not provider or not model or any(not segment for segment in model.split("/")):
        raise ValueError("model must use the qualified 'provider/model' form")
    if value == "router-maestro" or model == "router-maestro":
        raise ValueError("an Auto target cannot recursively select router-maestro")
    return value


class AutoMode(StrEnum):
    """How the virtual ``router-maestro`` model selects an execution model."""

    TASK_ROUTER = "task-router"
    PRIORITY_CHAIN = "priority-chain"


class AutoCapabilityPolicy(StrEnum):
    """How Auto treats catalog capabilities that the provider did not declare."""

    STRICT = "strict"
    OPTIMISTIC = "optimistic"


class AutoTaskType(StrEnum):
    """Stable task classes returned by the configured router model."""

    FAST = "fast"
    GENERAL = "general"
    CODING = "coding"
    DEEP_REASONING = "deep_reasoning"


class TaskRouterConfig(BaseModel):
    """LLM classifier and the bounded task-to-model mapping it may select."""

    router_model: str = Field(
        default=DEFAULT_AUTO_ROUTER_MODEL,
        description="Qualified model used only to classify Auto requests",
    )
    task_models: dict[AutoTaskType, str] = Field(
        default_factory=lambda: {
            AutoTaskType(task): model for task, model in DEFAULT_AUTO_TASK_MODELS.items()
        },
        description="One qualified execution model for every supported task type",
    )

    @field_validator("router_model")
    @classmethod
    def validate_router_model(cls, value: str) -> str:
        return _validate_qualified_model_id(value)

    @field_validator("task_models")
    @classmethod
    def validate_task_models(cls, value: dict[AutoTaskType, str]) -> dict[AutoTaskType, str]:
        expected = set(AutoTaskType)
        actual = set(value)
        if actual != expected:
            missing = sorted(task.value for task in expected - actual)
            extra = sorted(str(task) for task in actual - expected)
            details = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if extra:
                details.append(f"unknown: {', '.join(extra)}")
            raise ValueError(f"task_models must configure every task type ({'; '.join(details)})")
        return {task: _validate_qualified_model_id(model) for task, model in value.items()}


class AutoConfig(BaseModel):
    """Configuration of the virtual ``router-maestro`` model."""

    mode: AutoMode = AutoMode.TASK_ROUTER
    capability_policy: AutoCapabilityPolicy = AutoCapabilityPolicy.STRICT
    priority_chain: list[str] = Field(
        default_factory=list,
        description="Strict ordered fallback chain used by priority-chain mode",
    )
    task_router: TaskRouterConfig = Field(default_factory=TaskRouterConfig)

    @field_validator("priority_chain")
    @classmethod
    def validate_priority_chain(cls, value: list[str]) -> list[str]:
        validated = [_validate_qualified_model_id(model) for model in value]
        if len(validated) != len(set(validated)):
            raise ValueError("priority_chain cannot contain duplicate models")
        return validated

    @model_validator(mode="after")
    def validate_active_profile(self) -> Self:
        if self.mode is AutoMode.PRIORITY_CHAIN and not self.priority_chain:
            raise ValueError("priority_chain must contain at least one model")
        return self


class FallbackStrategy(StrEnum):
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


class LeakGuardConfig(BaseModel):
    """Leak guard configuration."""

    enabled: bool = Field(default=True, description="Enable response leak detection")


class RunawayGuardConfig(BaseModel):
    """Runaway guard configuration."""

    enabled: bool = Field(default=True, description="Enable runaway generation detection")
    max_bytes: int = Field(
        default=10_000_000,
        ge=100_000,
        description="Abort if total streamed bytes exceed this",
    )
    max_deltas: int = Field(
        default=50_000,
        ge=1000,
        description="Delta count threshold for tiny-fragment detection",
    )


class GuardsConfig(BaseModel):
    """Stream guards configuration."""

    leak_guard: LeakGuardConfig = Field(default_factory=LeakGuardConfig)
    runaway_guard: RunawayGuardConfig = Field(default_factory=RunawayGuardConfig)


class AuditConfig(BaseModel):
    """Per-request audit tracing configuration."""

    enabled: bool = Field(default=False, description="Enable per-request audit tracing")
    trace_dir: str | None = Field(
        default=None,
        description="Directory for trace files (default: ~/.local/share/router-maestro/traces/)",
    )


class PrioritiesConfig(BaseModel):
    """Configuration for model priorities and fallback."""

    priorities: list[str] = Field(
        default_factory=list,
        description="Model priorities in format 'provider/model', highest to lowest",
    )
    auto: AutoConfig = Field(default_factory=AutoConfig)
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    model_overrides: dict[str, ModelOverride] = Field(
        default_factory=dict,
        description="Per-model token limit overrides keyed by 'provider/model' or 'model'",
    )
    thinking: ThinkingBudgetConfig = Field(default_factory=ThinkingBudgetConfig)
    guards: GuardsConfig = Field(default_factory=GuardsConfig)
    beta_strip: list[str] = Field(
        default_factory=list,
        description="anthropic-beta tokens to strip (supports trailing * wildcard)",
    )
    audit: AuditConfig = Field(default_factory=AuditConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_auto_config(cls, value: Any) -> Any:
        """Preserve a non-empty legacy Auto chain while defaulting new installs to Smart Auto."""
        if not isinstance(value, dict) or "auto" in value:
            return value
        priorities = value.get("priorities")
        if not isinstance(priorities, list) or not priorities:
            return value
        if any(not isinstance(model, str) or "/" not in model for model in priorities):
            # Very old/test-only configs allowed bare priority labels. Keep
            # validating those through the legacy fields, but do not promote
            # ambiguous identities into the new strict Auto profile.
            return value
        unique_priorities = list(dict.fromkeys(priorities))
        migrated = dict(value)
        migrated["auto"] = {
            "mode": AutoMode.PRIORITY_CHAIN,
            "priority_chain": unique_priorities,
        }
        return migrated

    @classmethod
    def get_default(cls) -> PrioritiesConfig:
        """Get the default Smart Auto configuration."""
        return cls()

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
