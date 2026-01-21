/**
 * useSSE - Custom hook for Server-Sent Events streaming
 *
 * Implements fetch-based SSE parsing with proper buffer management.
 * Uses AbortController for cancellation support.
 */

import { useCallback, useRef } from "react"
import type { SSEEvent, RawSSEEvent } from "@/types/sse-events"

export interface UseSSEOptions {
  /** Callback for each parsed event */
  onEvent: (event: SSEEvent) => void
  /** Callback on error */
  onError?: (error: Error) => void
  /** Callback when stream completes */
  onComplete?: () => void
}

export interface UseSSEReturn {
  /** Start streaming from a Response */
  stream: (response: Response) => Promise<void>
  /** Abort the current stream */
  abort: () => void
  /** Whether currently streaming */
  isStreaming: boolean
}

/**
 * Parse a raw SSE event into a typed event.
 * Handles event type detection and fallback logic.
 */
function parseSSEEvent(data: RawSSEEvent, currentEventType: string): SSEEvent {
  // Start with explicit type from data
  let eventType = data.type

  // Detect text_delta events by OpenAI structure (BEFORE fallback to currentEventType)
  // This is important because OpenAI-format chunks don't have explicit event: lines
  if (!eventType && data.choices) {
    eventType = "text_delta"
  }

  // Detect metadata events by structure
  if (!eventType && (data.confidence !== undefined || data.sources !== undefined || data.message_id !== undefined)) {
    eventType = "metadata"
  }

  // Detect tool_call events by structure
  // API sends 'tool_id' but frontend expects 'tool_call_id', so check both
  if (!eventType && (data.tool_name !== undefined || data.tool_call_id !== undefined || data.tool_id !== undefined)) {
    eventType = "tool_call"
  }

  // Detect done events
  if (!eventType && data.reason !== undefined) {
    eventType = "done"
  }

  // Detect schema events from action tool (used by agent-builder)
  if (!eventType && data._action_event && data.action_type === "schema_update") {
    eventType = "schema_update"
  }
  if (!eventType && data._action_event && data.action_type === "schema_focus") {
    eventType = "schema_focus"
  }

  // Fall back to currentEventType only if we couldn't detect from structure
  if (!eventType) {
    eventType = currentEventType || "text_delta"
  }

  return { type: eventType, ...data } as SSEEvent
}

/**
 * Hook for consuming SSE streams from the chat API.
 */
export function useSSE(options: UseSSEOptions): UseSSEReturn {
  const { onEvent, onError, onComplete } = options
  const abortControllerRef = useRef<AbortController | null>(null)
  const isStreamingRef = useRef(false)

  /**
   * Stream and parse events from a Response object.
   */
  const stream = useCallback(
    async (response: Response) => {
      if (!response.body) {
        throw new Error("Response body is null")
      }

      isStreamingRef.current = true
      abortControllerRef.current = new AbortController()

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      let currentEventType = "text_delta"

      try {
        while (true) {
          const { value, done } = await reader.read()

          if (done) {
            break
          }

          // Decode chunk and append to buffer
          buffer += decoder.decode(value, { stream: true })

          // Split buffer by newlines
          const lines = buffer.split("\n")
          // Keep the last potentially incomplete line in buffer
          buffer = lines.pop() || ""

          for (const line of lines) {
            const trimmedLine = line.trim()

            if (!trimmedLine) {
              continue
            }

            // Parse event type line
            if (trimmedLine.startsWith("event:")) {
              currentEventType = trimmedLine.slice(6).trim()
              continue
            }

            // Parse data line
            if (trimmedLine.startsWith("data:")) {
              const jsonStr = trimmedLine.slice(5).trim()

              // Skip [DONE] marker
              if (jsonStr === "[DONE]") {
                onEvent({ type: "done" })
                continue
              }

              try {
                const data = JSON.parse(jsonStr) as RawSSEEvent
                const event = parseSSEEvent(data, currentEventType)
                onEvent(event)
              } catch {
                // Skip malformed JSON silently
                console.debug("Skipping malformed SSE data:", jsonStr)
              }
            }
          }
        }

        // Process any remaining buffer
        if (buffer.trim()) {
          const trimmedLine = buffer.trim()
          if (trimmedLine.startsWith("data:")) {
            const jsonStr = trimmedLine.slice(5).trim()
            if (jsonStr !== "[DONE]") {
              try {
                const data = JSON.parse(jsonStr) as RawSSEEvent
                const event = parseSSEEvent(data, currentEventType)
                onEvent(event)
              } catch {
                // Skip
              }
            }
          }
        }

        onComplete?.()
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          // Stream was aborted, not an error
          return
        }
        onError?.(error as Error)
      } finally {
        isStreamingRef.current = false
        abortControllerRef.current = null
      }
    },
    [onEvent, onError, onComplete]
  )

  /**
   * Abort the current stream.
   */
  const abort = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      isStreamingRef.current = false
    }
  }, [])

  return {
    stream,
    abort,
    get isStreaming() {
      return isStreamingRef.current
    },
  }
}
