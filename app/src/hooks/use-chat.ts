/**
 * useChat - Chat state management hook
 *
 * Manages messages, streaming state, and SSE event processing.
 */

import { useState, useCallback, useRef } from "react"
import { useSSE } from "./use-sse"
import { chatCompletions } from "@/api/chat"
import { generateId } from "@/lib/utils"
import type { Message, ToolCall } from "@/types/chat"
import type { SSEEvent, ToolCallEvent, MetadataEvent, TextDeltaEvent, ChildToolEvent } from "@/types/sse-events"

export interface UseChatOptions {
  /** Initial messages */
  initialMessages?: Message[]
  /** Agent schema to use */
  agentSchema?: string
  /** Model to use */
  model?: string
  /** Session ID */
  sessionId?: string
  /** Context to include with each message (e.g., current schema state) */
  context?: string
  /** Callback when session ID changes */
  onSessionIdChange?: (sessionId: string) => void
  /** Callback when an action event is received */
  onActionEvent?: (actionType: string, payload: Record<string, unknown>) => void
}

export interface UseChatReturn {
  /** Current messages */
  messages: Message[]
  /** Whether currently streaming */
  isLoading: boolean
  /** Send a message */
  sendMessage: (content: string) => Promise<void>
  /** Stop current stream */
  stop: () => void
  /** Clear messages */
  clear: () => void
  /** Set messages */
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>
}

/**
 * Map SSE tool status to our ToolCallState.
 */
function mapToolStatus(status: string): ToolCall["state"] {
  switch (status) {
    case "started":
    case "in_progress":
    case "executing":
      return "in_progress"
    case "completed":
      return "completed"
    case "failed":
      return "failed"
    default:
      return "in_progress"
  }
}

export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const { initialMessages = [], agentSchema, model, sessionId, context, onActionEvent } = options

  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [isLoading, setIsLoading] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  const currentMessageRef = useRef<Message | null>(null)
  // Track the current parent tool for nesting child agent calls
  const currentParentToolRef = useRef<string | null>(null)

  /**
   * Process an SSE event and update the current message.
   */
  const handleEvent = useCallback((event: SSEEvent) => {
    if (!currentMessageRef.current) {
      return
    }

    const msg = currentMessageRef.current

    switch (event.type) {
      case "text_delta": {
        const textEvent = event as TextDeltaEvent
        const chunk = textEvent.choices?.[0]?.delta?.content || ""
        if (chunk) {
          msg.content += chunk
          msg.status = "streaming"
          setMessages((prev) =>
            prev.map((m) => (m.id === msg.id ? { ...msg } : m))
          )
        }
        break
      }

      case "tool_call": {
        const toolEvent = event as ToolCallEvent
        // API sends 'tool_id', but we also support 'tool_call_id' for compatibility
        const toolId = toolEvent.tool_call_id || toolEvent.tool_id || `tool-${Date.now()}`

        // Check if this is a child agent tool (format: "agent_name:tool_name")
        const isChildTool = toolEvent.tool_name.includes(":")
        let agentName: string | undefined
        let actualToolName = toolEvent.tool_name

        if (isChildTool) {
          const parts = toolEvent.tool_name.split(":")
          agentName = parts[0]
          actualToolName = parts.slice(1).join(":")
        }

        const toolCall: ToolCall = {
          id: toolId,
          name: actualToolName,
          args: (toolEvent.arguments as Record<string, unknown>) || {},
          state: mapToolStatus(toolEvent.status),
          output: toolEvent.result,
          error: toolEvent.error,
          // If this is a child tool, link to current parent (ask_agent)
          parentId: isChildTool ? currentParentToolRef.current || undefined : undefined,
          agentName: agentName,
        }

        // Handle tool results that emit action events (action, save_agent, etc.)
        // Any tool can return _action_event: true with action_type to trigger frontend actions
        if (toolEvent.status === "completed" && toolEvent.result) {
          const result = toolEvent.result as Record<string, unknown>
          if (result._action_event && result.action_type && onActionEvent) {
            const actionType = result.action_type as string
            const payload = (result.payload || {}) as Record<string, unknown>
            onActionEvent(actionType, payload)
          }
        }

        if (toolEvent.status === "started") {
          // Track ask_agent as parent for subsequent child tools
          if (toolEvent.tool_name === "ask_agent") {
            currentParentToolRef.current = toolId
          }
          msg.toolCalls = [...(msg.toolCalls || []), toolCall]
        } else {
          // Update existing tool call - also update args if provided in completion
          msg.toolCalls = (msg.toolCalls || []).map((t) =>
            t.id === toolId || (t.name === actualToolName && t.state === "in_progress" && t.agentName === agentName)
              ? {
                  ...t,
                  state: toolCall.state,
                  output: toolCall.output,
                  error: toolCall.error,
                  // Update args if provided in completion (may have actual values now)
                  args: Object.keys(toolCall.args).length > 0 ? toolCall.args : t.args,
                }
              : t
          )
          // Clear parent tracking when ask_agent completes
          if (toolEvent.tool_name === "ask_agent" && toolEvent.status === "completed") {
            currentParentToolRef.current = null
          }
        }
        setMessages((prev) =>
          prev.map((m) => (m.id === msg.id ? { ...msg } : m))
        )
        break
      }

      case "schema_update": {
        const actionEvent = event as { payload?: Record<string, unknown>; section?: string; value?: unknown }
        const payload = actionEvent.payload || actionEvent
        if (onActionEvent && payload.section !== undefined) {
          onActionEvent("schema_update", payload as Record<string, unknown>)
        }
        break
      }

      case "schema_focus": {
        const actionEvent = event as { payload?: Record<string, unknown>; section?: string }
        const payload = actionEvent.payload || actionEvent
        if (onActionEvent && payload.section !== undefined) {
          onActionEvent("schema_focus", payload as Record<string, unknown>)
        }
        break
      }

      case "patch_schema": {
        const actionEvent = event as { payload?: { patches?: unknown[] } }
        const payload = actionEvent.payload
        if (onActionEvent && payload && Array.isArray(payload.patches)) {
          onActionEvent("patch_schema", payload as Record<string, unknown>)
        }
        break
      }

      case "trigger_save": {
        if (onActionEvent) {
          onActionEvent("trigger_save", {})
        }
        break
      }

      case "metadata": {
        const metaEvent = event as MetadataEvent
        // IMPORTANT: Use message_id from metadata for database ID
        if (metaEvent.message_id) {
          msg.id = metaEvent.message_id
        }
        msg.metadata = {
          confidence: metaEvent.confidence,
          sources: metaEvent.sources,
          agentName: metaEvent.agent_schema || metaEvent.responding_agent,
          modelVersion: metaEvent.model_version,
          latencyMs: metaEvent.latency_ms,
          tokenCount: metaEvent.token_count,
          sessionName: metaEvent.session_name,
        }
        setMessages((prev) =>
          prev.map((m) => (m.id === msg.id ? { ...msg } : m))
        )
        break
      }

      case "done": {
        msg.status = "completed"
        // Mark any remaining in-progress tool calls as completed
        // (they may not have received explicit completion events)
        msg.toolCalls = (msg.toolCalls || []).map((t) =>
          t.state === "in_progress" ? { ...t, state: "completed" as const } : t
        )
        setMessages((prev) =>
          prev.map((m) => (m.id === msg.id ? { ...msg } : m))
        )
        break
      }

      case "error": {
        msg.status = "failed"
        msg.content += `\n\nError: ${(event as { message: string }).message}`
        setMessages((prev) =>
          prev.map((m) => (m.id === msg.id ? { ...msg } : m))
        )
        break
      }

      case "child_tool_start":
      case "child_tool_result": {
        // Handle explicit child agent tool events
        const childEvent = event as ChildToolEvent
        const childToolId = childEvent.tool_call_id || `child-${Date.now()}`
        const isStart = event.type === "child_tool_start"

        if (isStart) {
          const childToolCall: ToolCall = {
            id: childToolId,
            name: childEvent.tool_name,
            args: childEvent.arguments || {},
            state: "in_progress",
            parentId: currentParentToolRef.current || undefined,
            agentName: childEvent.agent_name,
          }
          msg.toolCalls = [...(msg.toolCalls || []), childToolCall]
        } else {
          // Update the child tool call with result - match by tool_call_id or fallback to name+agent
          msg.toolCalls = (msg.toolCalls || []).map((t) => {
            const matchById = childEvent.tool_call_id && t.id === childEvent.tool_call_id
            const matchByName = t.name === childEvent.tool_name && t.state === "in_progress" && t.agentName === childEvent.agent_name
            if (matchById || matchByName) {
              return {
                ...t,
                args: childEvent.arguments || t.args,
                state: "completed" as const,
                output: childEvent.result,
              }
            }
            return t
          })
        }
        setMessages((prev) =>
          prev.map((m) => (m.id === msg.id ? { ...msg } : m))
        )
        break
      }
    }
  }, [onActionEvent])

  const handleError = useCallback((error: Error) => {
    if (currentMessageRef.current) {
      const msg = currentMessageRef.current
      msg.status = "failed"
      msg.content += `\n\nError: ${error.message}`
      // Mark any in-progress tool calls as failed
      msg.toolCalls = (msg.toolCalls || []).map((t) =>
        t.state === "in_progress" ? { ...t, state: "failed" as const } : t
      )
      setMessages((prev) =>
        prev.map((m) => (m.id === msg.id ? { ...msg } : m))
      )
    }
    setIsLoading(false)
  }, [])

  const handleComplete = useCallback(() => {
    if (currentMessageRef.current) {
      const msg = currentMessageRef.current
      if (msg.status === "streaming" || msg.status === "pending") {
        msg.status = "completed"
        // Mark any remaining in-progress tool calls as completed
        msg.toolCalls = (msg.toolCalls || []).map((t) =>
          t.state === "in_progress" ? { ...t, state: "completed" as const } : t
        )
        setMessages((prev) =>
          prev.map((m) => (m.id === msg.id ? { ...msg } : m))
        )
      }
    }
    currentMessageRef.current = null
    currentParentToolRef.current = null
    setIsLoading(false)
  }, [])

  const { stream } = useSSE({
    onEvent: handleEvent,
    onError: handleError,
    onComplete: handleComplete,
  })

  /**
   * Send a user message and stream the response.
   */
  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || isLoading) return

      // Create user message
      const userMessage: Message = {
        id: generateId(),
        role: "user",
        content: content.trim(),
        status: "completed",
        createdAt: new Date(),
        sessionId,
      }

      // Create placeholder assistant message
      const assistantMessage: Message = {
        id: generateId(),
        role: "assistant",
        content: "",
        status: "pending",
        toolCalls: [],
        createdAt: new Date(),
        sessionId,
      }

      setMessages((prev) => [...prev, userMessage, assistantMessage])
      currentMessageRef.current = assistantMessage
      setIsLoading(true)

      // Build messages array for API
      const apiMessages = [
        ...messages.map((m) => ({ role: m.role, content: m.content })),
        { role: "user" as const, content: content.trim() },
      ]

      abortControllerRef.current = new AbortController()

      try {
        const response = await chatCompletions({
          messages: apiMessages,
          agentSchema,
          model,
          sessionId,
          context,
          signal: abortControllerRef.current.signal,
        })

        await stream(response)
      } catch (error) {
        if ((error as Error).name === "AbortError") {
          return
        }
        handleError(error as Error)
      }
    },
    [messages, isLoading, agentSchema, model, sessionId, context, stream, handleError]
  )

  /**
   * Stop the current stream.
   */
  const stop = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      setIsLoading(false)
    }
  }, [])

  /**
   * Clear all messages.
   */
  const clear = useCallback(() => {
    setMessages([])
    currentMessageRef.current = null
    currentParentToolRef.current = null
  }, [])

  return {
    messages,
    isLoading,
    sendMessage,
    stop,
    clear,
    setMessages,
  }
}
