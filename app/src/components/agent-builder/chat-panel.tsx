/**
 * ChatPanel - Right panel with agent-builder chat
 *
 * Provides context-aware chat for building agent schemas.
 * The chat knows the current schema state and can modify it via action events.
 */

import { useCallback, useEffect, useMemo, useState } from "react"
import { stringify as stringifyYaml } from "yaml"
import type { Operation } from "fast-json-patch"
import { MessageList } from "@/components/chat/message-list"
import { ChatInput } from "@/components/chat/chat-input"
import { useChat } from "@/hooks/use-chat"
import { generateId } from "@/lib/utils"
import type { AgentSchemaState, SchemaUpdatePayload, SchemaFocusPayload } from "@/types/agent-schema"

interface ChatPanelProps {
  schema: AgentSchemaState
  onSchemaUpdate: (payload: SchemaUpdatePayload) => void
  onSchemaFocus: (payload: SchemaFocusPayload) => void
  onPatchSchema?: (patches: Operation[]) => void
  onTriggerSave?: () => void
}

export function ChatPanel({ schema, onSchemaUpdate, onSchemaFocus, onPatchSchema, onTriggerSave }: ChatPanelProps) {
  // Generate a stable session ID for this edit session (must be valid UUID format)
  const [sessionId] = useState(() => generateId())

  // Generate context string from current schema state
  const schemaContext = useMemo(() => {
    const yamlPreview = stringifyYaml({
      name: schema.metadata.name || "(not set)",
      version: schema.metadata.version,
      description: schema.description ? schema.description.slice(0, 500) + (schema.description.length > 500 ? "..." : "") : "(not set)",
      tools: schema.metadata.tools.map(t => t.name),
      properties: schema.properties,  // Full structure so agent can see types
      structured_output: schema.metadata.structured_output,
      tags: schema.metadata.tags,
    }, { lineWidth: 80 })

    return `CURRENT AGENT SCHEMA STATE:
\`\`\`yaml
${yamlPreview}
\`\`\`

You are helping the user build/edit this agent schema. Use JSON Patch (RFC 6902) to make changes:

action(type="patch_schema", patches=[...])

Schema paths:
- /description - system prompt (string)
- /properties/<name> - output property (object with type, description)
- /metadata/name - agent name
- /metadata/tools - array of {name, description?}
- /metadata/tags - array of strings

Examples:
- Set system prompt: {"op":"replace","path":"/description","value":"You are..."}
- Add property: {"op":"add","path":"/properties/summary","value":{"type":"string","description":"..."}}
- Add tool: {"op":"add","path":"/metadata/tools/-","value":{"name":"search"}}
- Remove property: {"op":"remove","path":"/properties/oldfield"}`
  }, [schema])

  // Handle action events from the agent
  const handleActionEvent = useCallback((actionType: string, payload: Record<string, unknown>) => {
    if (actionType === "patch_schema" && Array.isArray(payload.patches) && onPatchSchema) {
      onPatchSchema(payload.patches as Operation[])
    } else if (actionType === "schema_update" && payload.section && payload.value !== undefined) {
      onSchemaUpdate(payload as unknown as SchemaUpdatePayload)
    } else if (actionType === "schema_focus" && payload.section) {
      onSchemaFocus(payload as unknown as SchemaFocusPayload)
    } else if (actionType === "trigger_save" && onTriggerSave) {
      onTriggerSave()
    }
  }, [onSchemaUpdate, onSchemaFocus, onPatchSchema, onTriggerSave])

  const { messages, isLoading, sendMessage, stop, setMessages } = useChat({
    agentSchema: "agent-builder",
    sessionId,
    context: schemaContext,
    onActionEvent: handleActionEvent,
  })

  const handleSend = useCallback(
    (content: string) => {
      sendMessage(content)
    },
    [sendMessage]
  )

  // Welcome message with context about the current schema
  useEffect(() => {
    if (messages.length === 0) {
      const isExisting = schema.metadata.name && schema.metadata.name.length > 0
      const welcomeContent = isExisting
        ? `I see you're editing **${schema.metadata.name}**. I can help you:\n\n` +
          `- Modify the system prompt\n` +
          `- Add or remove tools\n` +
          `- Update the output schema\n` +
          `- Change metadata (name, version, tags)\n\n` +
          `What would you like to change?`
        : `Hi! I'm the Agent Builder. I'll help you create a new agent schema step by step.\n\n` +
          `To get started, tell me: **What should your agent do?**\n\n` +
          `For example:\n` +
          `- "Help users analyze customer feedback"\n` +
          `- "Search documentation and answer questions"\n` +
          `- "Generate code based on requirements"`

      setMessages([
        {
          id: "welcome",
          role: "assistant",
          content: welcomeContent,
          status: "completed",
          createdAt: new Date(),
        },
      ])
    }
  }, [messages.length, setMessages, schema.metadata.name])

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200">
        <div className="flex items-center gap-2">
          <div className="h-2 w-2 rounded-full bg-green-500" />
          <span className="text-sm font-medium text-zinc-800">Agent Builder</span>
        </div>
        <span className="text-xs text-zinc-400">
          {schema.metadata.name ? `Editing: ${schema.metadata.name}` : "New Agent"}
        </span>
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
          placeholder="Describe changes or ask questions about your agent..."
        />
      </div>
    </div>
  )
}
