/**
 * Models API - Fetch available LLM models
 */

import { apiRequest } from "@/lib/api-client"
import type { Model } from "@/types/chat"

interface ModelListResponse {
  models: Model[]
}

/**
 * Default models if API is unavailable.
 */
const DEFAULT_MODELS: Model[] = [
  { id: "openai:gpt-4.1", name: "GPT-4.1", provider: "openai" },
  { id: "anthropic:claude-sonnet-4-5-20250929", name: "Claude Sonnet 4.5", provider: "anthropic" },
]

/**
 * Fetch list of available models from the API.
 * Falls back to default models on error.
 */
export async function fetchModels(): Promise<Model[]> {
  try {
    const response = await apiRequest<ModelListResponse>("/v1/models")
    return response.models || DEFAULT_MODELS
  } catch (error) {
    console.warn("Failed to fetch models, using defaults:", error)
    return DEFAULT_MODELS
  }
}
