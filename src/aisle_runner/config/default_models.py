"""Default model configurations for benchmarking.

NOTE: API keys must be set as environment variables:
- ANTHROPIC_API_KEY: For Claude models
- OPENAI_API_KEY: For GPT models
- GOOGLE_API_KEY: For Gemini models
"""

from ..models import ModelConfig

DEFAULT_MODELS: list[ModelConfig] = [
    # Anthropic Models
    # https://docs.anthropic.com/en/docs/about-claude/models
    ModelConfig(
        id="claude-opus-4-5",
        provider="anthropic",
        model_id="claude-opus-4-5-20251101",
        cost_per_1k_input=0.005,    # $5/1M tokens
        cost_per_1k_output=0.025,   # $25/1M tokens
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="claude-sonnet-4-5",
        provider="anthropic",
        model_id="claude-sonnet-4-5-20250929",
        cost_per_1k_input=0.003,    # $3/1M tokens
        cost_per_1k_output=0.015,   # $15/1M tokens
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="claude-sonnet-4",
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        cost_per_1k_input=0.003,    # $3/1M tokens
        cost_per_1k_output=0.015,   # $15/1M tokens
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="claude-haiku-4-5",
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        cost_per_1k_input=0.001,    # $1/1M tokens
        cost_per_1k_output=0.005,   # $5/1M tokens
        supports_vision=True,
        max_tokens=8192,
    ),
    # OpenAI Models
    # https://platform.openai.com/docs/models
    ModelConfig(
        id="gpt-5.2",
        provider="openai",
        model_id="gpt-5.2",
        cost_per_1k_input=0.00125,  # $1.25/1M tokens
        cost_per_1k_output=0.010,   # $10/1M tokens
        supports_vision=True,
        max_tokens=128000,
    ),
    ModelConfig(
        id="gpt-5",
        provider="openai",
        model_id="gpt-5-2025-08-07",
        cost_per_1k_input=0.00125,  # $1.25/1M tokens
        cost_per_1k_output=0.010,   # $10/1M tokens
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="gpt-5-mini",
        provider="openai",
        model_id="gpt-5-mini-2025-08-07",
        cost_per_1k_input=0.00025,  # $0.25/1M tokens
        cost_per_1k_output=0.002,   # $2/1M tokens
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="gpt-4o",
        provider="openai",
        model_id="gpt-4o",
        cost_per_1k_input=0.005,    # $5/1M tokens
        cost_per_1k_output=0.015,   # $15/1M tokens
        supports_vision=True,
        max_tokens=4096,
    ),
    # Google Models
    # https://ai.google.dev/gemini-api/docs/models
    ModelConfig(
        id="gemini-3-flash",
        provider="google",
        model_id="gemini-3-flash-preview",
        cost_per_1k_input=0.0005,   # $0.50/1M tokens
        cost_per_1k_output=0.003,   # $3/1M tokens
        supports_vision=True,
        max_tokens=8192,
    ),
    ModelConfig(
        id="gemini-2-flash",
        provider="google",
        model_id="gemini-2.0-flash",
        cost_per_1k_input=0.0001,   # $0.10/1M tokens
        cost_per_1k_output=0.0004,  # $0.40/1M tokens
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
    if m.id in ("claude-haiku-4-5", "gpt-5-mini", "gemini-2-flash")
]

# Premium models for best accuracy
PREMIUM_MODELS = [
    m for m in DEFAULT_MODELS
    if m.id in ("claude-opus-4-5", "gpt-5.2", "gemini-3-flash")
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
