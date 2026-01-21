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
  /** Context to include (e.g., current schema state) */
  context?: string
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
  const { messages, model, agentSchema, sessionId, context, signal } = options

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

  // If context is provided, prepend it to the last user message
  // Note: Backend ignores system messages from request, uses agent schema's description instead
  let messagesWithContext = messages
  if (context && messages.length > 0) {
    const lastIdx = messages.length - 1
    const lastMsg = messages[lastIdx]
    if (lastMsg.role === "user") {
      messagesWithContext = [
        ...messages.slice(0, lastIdx),
        { role: "user" as const, content: `${context}\n\n---\n\nUSER REQUEST:\n${lastMsg.content}` }
      ]
    }
  }

  console.log("[Chat API] Sending request:", { agentSchema, sessionId, model, headers, hasContext: !!context })

  const url = `${API_BASE_URL}/v1/chat/completions`

  const response = await fetch(url, {
    method: "POST",
    headers: getApiHeaders(headers),
    body: JSON.stringify({
      messages: messagesWithContext,
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
