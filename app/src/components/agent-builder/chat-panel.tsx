/**
 * ChatPanel - Right panel with agent-builder chat
 *
 * Reuses existing chat components with agent-builder agent.
 */

import { useCallback, useEffect } from "react"
import { MessageList } from "@/components/chat/message-list"
import { ChatInput } from "@/components/chat/chat-input"
import { useChat } from "@/hooks/use-chat"
import type { AgentSchemaState, SchemaUpdatePayload, SchemaFocusPayload } from "@/types/agent-schema"

interface ChatPanelProps {
  schema: AgentSchemaState
  onSchemaUpdate: (payload: SchemaUpdatePayload) => void
  onSchemaFocus: (payload: SchemaFocusPayload) => void
}

export function ChatPanel({ schema, onSchemaUpdate, onSchemaFocus }: ChatPanelProps) {
  const { messages, isLoading, sendMessage, stop, setMessages } = useChat({
    agentSchema: "agent-builder",
  })

  // Handle schema-specific action events from SSE
  useEffect(() => {
    const lastMessage = messages[messages.length - 1]
    if (lastMessage?.role === "assistant") {
      // Check for embedded action events in the message
      // This is a simplified handler - real implementation would parse SSE events
    }
  }, [messages, onSchemaUpdate, onSchemaFocus])

  const handleSend = useCallback(
    (content: string) => {
      sendMessage(content)
    },
    [sendMessage]
  )

  // Welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([
        {
          id: "welcome",
          role: "assistant",
          content:
            "Hi! I'm the Agent Builder. I'll help you create a new agent schema step by step.\n\nTo get started, tell me: **What should your agent do?**\n\nFor example:\n- \"Help users analyze customer feedback\"\n- \"Search documentation and answer questions\"\n- \"Generate code based on requirements\"",
          status: "completed",
          createdAt: new Date(),
        },
      ])
    }
  }, [messages.length, setMessages])

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium text-zinc-800">Agent Builder</span>
        </div>
        <span className="text-xs text-zinc-400">Building: {schema.metadata.name || "New Agent"}</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-hidden">
        <MessageList messages={messages} />
        {isLoading && (
          <div className="px-4 py-2 text-xs text-zinc-400">Thinking...</div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-zinc-200 p-4">
        <ChatInput
          onSend={handleSend}
          onStop={stop}
          isLoading={isLoading}
          placeholder="Describe your agent or ask a question..."
        />
      </div>
    </div>
  )
}
