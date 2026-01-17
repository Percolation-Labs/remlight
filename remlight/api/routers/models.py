"""Models router - List available LLM models.

Provides endpoint to list available LLM models that can be used
with the chat completions endpoint.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/models", tags=["models"])


class ModelInfo(BaseModel):
    """Model information."""

    id: str
    name: str
    provider: str
    context_window: int | None = None
    supports_tools: bool = True
    supports_streaming: bool = True


class ModelListResponse(BaseModel):
    """List of models response."""

    models: list[ModelInfo]


# Available models - these should match what's configured in the provider
AVAILABLE_MODELS: list[ModelInfo] = [
    # OpenAI models
    ModelInfo(
        id="openai:gpt-4.1",
        name="GPT-4.1",
        provider="openai",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
    ),
    ModelInfo(
        id="openai:gpt-4.1-mini",
        name="GPT-4.1 Mini",
        provider="openai",
        context_window=128000,
        supports_tools=True,
        supports_streaming=True,
    ),
    # Anthropic models
    ModelInfo(
        id="anthropic:claude-opus-4-5-20251101",
        name="Claude Opus 4.5",
        provider="anthropic",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
    ),
    ModelInfo(
        id="anthropic:claude-sonnet-4-5-20250929",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
    ),
    ModelInfo(
        id="anthropic:claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider="anthropic",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
    ),
    ModelInfo(
        id="anthropic:claude-3-5-sonnet-20241022",
        name="Claude 3.5 Sonnet",
        provider="anthropic",
        context_window=200000,
        supports_tools=True,
        supports_streaming=True,
    ),
]


@router.get("", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List all available LLM models.

    Returns models that can be used with the chat completions endpoint.
    Use the model ID in the X-Model-Name header or model field.
    """
    return ModelListResponse(models=AVAILABLE_MODELS)


@router.get("/{model_id:path}", response_model=ModelInfo)
async def get_model(model_id: str) -> ModelInfo:
    """Get information about a specific model.

    Args:
        model_id: The model ID (e.g., "openai:gpt-4.1")
    """
    for model in AVAILABLE_MODELS:
        if model.id == model_id:
            return model

    from fastapi import HTTPException

    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
