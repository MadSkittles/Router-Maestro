"""Shared model-catalog schemas."""

from pydantic import BaseModel, Field


class ContextWindowOption(BaseModel):
    """One context size a client may select for a model."""

    tier: str = Field(..., description="Provider-normalized context tier identifier")
    max_prompt_tokens: int = Field(
        ...,
        gt=0,
        description="Maximum prompt tokens available in this context tier",
    )
    is_default: bool = Field(
        default=False,
        description="Whether clients should select this tier by default",
    )
