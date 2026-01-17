/**
 * Chat API - Streaming chat completions endpoint
 */

import { getApiHeaders } from "@/lib/api-client"

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api"

export interface ChatCompletionRequest {
  messages: Array<{ role: string; content: string }>
  stream?: boolean
  model?: string
}

export interface ChatCompletionOptions {
  /** Messages to send */
  messages: ChatCompletionRequest["messages"]
  /** Model override */
  model?: string
  /** Agent schema to use */
  agentSchema?: string
  /** Session ID for conversation continuity */
  sessionId?: string
  /** AbortSignal for cancellation */
  signal?: AbortSignal
}

/**
 * Send a chat completion request with streaming.
 * Returns a Response object for SSE parsing.
 */
export async function chatCompletions(
  options: ChatCompletionOptions
): Promise<Response> {
  const { messages, model, agentSchema, sessionId, signal } = options

  const headers: Record<string, string> = {}

  if (agentSchema) {
    headers["X-Agent-Schema"] = agentSchema
  }

  if (sessionId) {
    headers["X-Session-Id"] = sessionId
  }

  if (model) {
    headers["X-Model-Name"] = model
  }

  console.log("[Chat API] Sending request:", { agentSchema, sessionId, model, headers })

  const url = `${API_BASE_URL}/v1/chat/completions`

  const response = await fetch(url, {
    method: "POST",
    headers: getApiHeaders(headers),
    body: JSON.stringify({
      messages,
      stream: true,
      model,
    }),
    signal,
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `HTTP ${response.status}`)
  }

  return response
}
