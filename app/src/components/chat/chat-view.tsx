/**
 * ChatView - Main chat container component
 *
 * Integrates message list, input, toolbar, and manages chat state.
 */

import { useState, useCallback, useEffect, useRef } from "react"
import { useChat } from "@/hooks/use-chat"
import { MessageList } from "./message-list"
import { ChatInput } from "./chat-input"
import { ChatToolbar } from "./chat-toolbar"
import { AddScenarioModal } from "./add-scenario-modal"
import { exportSessionAsYaml, fetchSessionMessages } from "@/api/sessions"
import { generateId } from "@/lib/utils"
import type { Message } from "@/types/chat"

interface ChatViewProps {
  /** Initial session ID */
  sessionId?: string
  /** Callback when session changes */
  onSessionChange?: (sessionId: string) => void
}

export function ChatView({
  sessionId: initialSessionId,
  onSessionChange,
}: ChatViewProps) {
  // Track previous sessionId to detect changes
  const prevSessionIdRef = useRef<string | undefined>(initialSessionId)
  const [sessionId, setSessionId] = useState(() => initialSessionId || generateId())
  const [selectedAgent, setSelectedAgent] = useState<string | undefined>()
  const [selectedModel, setSelectedModel] = useState<string>()
  const [showAddScenarioModal, setShowAddScenarioModal] = useState(false)

  // Sync sessionId when initialSessionId changes (e.g., selecting from sidebar or new chat)
  useEffect(() => {
    const prevSessionId = prevSessionIdRef.current
    prevSessionIdRef.current = initialSessionId

    if (initialSessionId !== prevSessionId) {
      if (initialSessionId) {
        // Switched to an existing session
        setSessionId(initialSessionId)
      } else {
        // New chat - generate a new session ID
        setSessionId(generateId())
      }
    }
  }, [initialSessionId])

  const {
    messages,
    isLoading,
    sendMessage,
    stop,
    setMessages,
    clear,
  } = useChat({
    agentSchema: selectedAgent,
    model: selectedModel,
    sessionId,
    onSessionIdChange: (newId) => {
      setSessionId(newId)
      onSessionChange?.(newId)
    },
  })

  // Load session messages when sessionId changes
  useEffect(() => {
    const loadSessionMessages = async () => {
      if (initialSessionId) {
        // Loading an existing session - fetch its messages
        try {
          const sessionMessages = await fetchSessionMessages(initialSessionId)
          if (sessionMessages.length > 0) {
            // Reconstruct messages with tool calls from DB rows
            // DB stores: user, tool (with tool_call_id/tool_name in metadata), assistant
            // Tool messages represent BOTH the tool invocation AND result in the same row
            // The content is the tool arguments (JSON), and we don't have separate result storage
            const formattedMessages: Message[] = []
            let currentAssistant: Message | null = null
            // Track tool calls by tool_call_id to handle nested agent calls
            const toolCallsById: Map<string, import("@/types/chat").ToolCall> = new Map()
            // Track the current parent tool (ask_agent) for nesting
            let currentParentToolId: string | null = null

            for (const msg of sessionMessages) {
              const createdAtValue = msg.createdAt || msg.created_at
              const metadata = msg.metadata as Record<string, unknown> | undefined

              if (msg.role === "user") {
                // Flush any pending assistant message
                if (currentAssistant) {
                  formattedMessages.push(currentAssistant)
                  currentAssistant = null
                }
                toolCallsById.clear()
                currentParentToolId = null

                formattedMessages.push({
                  id: msg.id,
                  role: "user",
                  content: msg.content,
                  status: "completed" as const,
                  toolCalls: [],
                  metadata: msg.metadata as Message["metadata"],
                  createdAt: createdAtValue ? new Date(createdAtValue as string) : new Date(),
                  sessionId: initialSessionId,
                })
              } else if (msg.role === "assistant") {
                // Flush any previous assistant message
                if (currentAssistant) {
                  formattedMessages.push(currentAssistant)
                }
                toolCallsById.clear()
                currentParentToolId = null

                currentAssistant = {
                  id: msg.id,
                  role: "assistant",
                  content: msg.content,
                  status: "completed" as const,
                  toolCalls: [],
                  metadata: msg.metadata as Message["metadata"],
                  createdAt: createdAtValue ? new Date(createdAtValue as string) : new Date(),
                  sessionId: initialSessionId,
                }
              } else if (msg.role === "tool") {
                // Tool message - contains tool_call_id, tool_name in metadata
                // Content is the tool arguments (JSON)
                const toolCallId = metadata?.tool_call_id as string | undefined
                const toolName = metadata?.tool_name as string | undefined
                const agentSchema = metadata?.agent_schema as string | undefined

                if (!toolCallId || !toolName) {
                  // Skip malformed tool messages
                  continue
                }

                // Parse tool arguments from content
                let toolArgs: Record<string, unknown> = {}
                try {
                  toolArgs = typeof msg.content === "string" ? JSON.parse(msg.content) : (msg.content as Record<string, unknown>)
                } catch {
                  // Content may not be valid JSON, use empty args
                }

                // Check if this is a child agent tool (format: "agent_name:tool_name")
                const isChildTool = toolName.includes(":")
                let agentName: string | undefined
                let actualToolName = toolName

                if (isChildTool) {
                  const parts = toolName.split(":")
                  agentName = parts[0]
                  actualToolName = parts.slice(1).join(":")
                }

                // Create tool call object
                const toolCall: import("@/types/chat").ToolCall = {
                  id: toolCallId,
                  name: actualToolName,
                  args: toolArgs,
                  state: "completed" as const,
                  // For child tools, set parentId to the current ask_agent call
                  parentId: isChildTool ? currentParentToolId || undefined : undefined,
                  agentName: agentName || agentSchema,
                }

                // Track ask_agent as parent for subsequent child tools
                if (toolName === "ask_agent") {
                  currentParentToolId = toolCallId
                }

                // Store for reference
                toolCallsById.set(toolCallId, toolCall)

                // Create assistant message if we don't have one yet
                // (tool calls come before the final assistant text response)
                if (!currentAssistant) {
                  currentAssistant = {
                    id: generateId(),
                    role: "assistant",
                    content: "",
                    status: "completed" as const,
                    toolCalls: [],
                    createdAt: createdAtValue ? new Date(createdAtValue as string) : new Date(),
                    sessionId: initialSessionId,
                  }
                }

                // Add tool call to current assistant message
                if (currentAssistant.toolCalls) {
                  currentAssistant.toolCalls.push(toolCall)
                }
              }
            }

            // Don't forget the last assistant message
            if (currentAssistant) {
              formattedMessages.push(currentAssistant)
            }

            setMessages(formattedMessages)
          } else {
            clear()
          }
        } catch (error) {
          console.error("Failed to load session messages:", error)
          clear()
        }
      } else {
        // New chat - clear messages
        clear()
      }
    }

    loadSessionMessages()
  }, [initialSessionId, setMessages, clear])

  /**
   * Export the current session as YAML.
   */
  const handleExport = useCallback(async () => {
    console.log("Export button clicked, sessionId:", sessionId)
    if (sessionId) {
      try {
        await exportSessionAsYaml(sessionId)
      } catch (error) {
        console.error("Export failed:", error)
      }
    }
  }, [sessionId])

  /**
   * Open the add scenario modal.
   */
  const handleAddScenario = useCallback(() => {
    console.log("Add to scenario button clicked, sessionId:", sessionId)
    setShowAddScenarioModal(true)
  }, [sessionId])

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Toolbar */}
      <ChatToolbar
        selectedAgent={selectedAgent}
        selectedModel={selectedModel}
        onAgentChange={setSelectedAgent}
        onModelChange={setSelectedModel}
      />

      {/* Messages */}
      <div className="flex-1 overflow-auto">
        <MessageList
          messages={messages}
          sessionId={sessionId}
          onExport={handleExport}
          onAddScenario={handleAddScenario}
        />
      </div>

      {/* Input */}
      <ChatInput
        onSend={sendMessage}
        onStop={stop}
        isLoading={isLoading}
      />

      {/* Add Scenario Modal */}
      {showAddScenarioModal && (
        <AddScenarioModal
          sessionId={sessionId}
          agentName={selectedAgent}
          onClose={() => setShowAddScenarioModal(false)}
          onSuccess={(scenarioId) => {
            console.log("Scenario created:", scenarioId)
          }}
        />
      )}
    </div>
  )
}

// Export loadSession for external use
export type { ChatViewProps }
