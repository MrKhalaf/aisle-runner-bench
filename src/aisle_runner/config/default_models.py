"""Default model configurations for benchmarking.

NOTE: API keys must be set as environment variables:
- ANTHROPIC_API_KEY: For Claude models
- OPENAI_API_KEY: For GPT models
- GOOGLE_API_KEY: For Gemini models
"""

from ..models import ModelConfig

# Current production models with actual pricing (as of Jan 2025)
DEFAULT_MODELS: list[ModelConfig] = [
    # Anthropic Models
    ModelConfig(
        id="claude-opus-4-5",
        provider="anthropic",
        model_id="claude-opus-4-5-20251101",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="claude-sonnet-4",
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="claude-haiku-35",
        provider="anthropic",
        model_id="claude-haiku-3-5-20241022",
        cost_per_1k_input=0.001,
        cost_per_1k_output=0.005,
        supports_vision=True,
        max_tokens=8192,
    ),
    # OpenAI Models
    ModelConfig(
        id="gpt-4o",
        provider="openai",
        model_id="gpt-4o",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        supports_vision=True,
        max_tokens=4096,
    ),
    ModelConfig(
        id="gpt-4o-mini",
        provider="openai",
        model_id="gpt-4o-mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        supports_vision=True,
        max_tokens=4096,
    ),
    ModelConfig(
        id="o1",
        provider="openai",
        model_id="o1",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.060,
        supports_vision=True,
        max_tokens=32768,
    ),
    # Google Models
    ModelConfig(
        id="gemini-2-flash",
        provider="google",
        model_id="gemini-2.0-flash",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0004,
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="gemini-2-flash-thinking",
        provider="google",
        model_id="gemini-2.0-flash-thinking-exp",
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0004,
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="gemini-15-pro",
        provider="google",
        model_id="gemini-1.5-pro",
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
        supports_vision=True,
        max_tokens=8192,
    ),
]

# Quick reference sets
ANTHROPIC_MODELS = [m for m in DEFAULT_MODELS if m.provider == "anthropic"]
OPENAI_MODELS = [m for m in DEFAULT_MODELS if m.provider == "openai"]
GOOGLE_MODELS = [m for m in DEFAULT_MODELS if m.provider == "google"]

# Budget-friendly subset for quick testing
BUDGET_MODELS = [
    m for m in DEFAULT_MODELS
    if m.id in ("claude-haiku-35", "gpt-4o-mini", "gemini-2-flash")
]

# Premium models for best accuracy
PREMIUM_MODELS = [
    m for m in DEFAULT_MODELS
    if m.id in ("claude-opus-4-5", "o1", "gemini-15-pro")
]


def get_model_by_id(model_id: str) -> ModelConfig | None:
    """Get a model config by its ID."""
    for model in DEFAULT_MODELS:
        if model.id == model_id:
            return model
    return None


def get_models_by_provider(provider: str) -> list[ModelConfig]:
    """Get all models from a specific provider."""
    return [m for m in DEFAULT_MODELS if m.provider == provider]
