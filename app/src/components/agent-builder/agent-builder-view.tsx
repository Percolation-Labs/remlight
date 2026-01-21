/**
 * AgentBuilderView - Main split layout for agent builder
 *
 * Left: Schema preview panel
 * Right: Chat panel with agent-builder agent
 */

import { useCallback, useEffect } from "react"
import { SchemaPreviewPanel } from "./schema-preview-panel"
import { ChatPanel } from "./chat-panel"
import { useAgentSchema } from "@/hooks/use-agent-schema"
import type { SchemaUpdatePayload, SchemaFocusPayload, UserSchemaEdit } from "@/types/agent-schema"

interface AgentBuilderViewProps {
  initialAgentName?: string
}

export function AgentBuilderView({ initialAgentName }: AgentBuilderViewProps) {
  const handleUserEdit = useCallback((edit: UserSchemaEdit) => {
    // This will be sent to the chat as context
    console.log("User edit:", edit)
  }, [])

  const {
    schema,
    focusState,
    setDescription,
    addTool,
    removeTool,
    removeProperty,
    applySchemaUpdate,
    applySchemaFocus,
    isValid,
    validationErrors,
  } = useAgentSchema({
    initialSchema: initialAgentName
      ? { metadata: { kind: "agent", name: initialAgentName, version: "1.0.0", tools: [], resources: [], structured_output: false, tags: [] } }
      : undefined,
    onUserEdit: handleUserEdit,
  })

  // Handle SSE events from chat
  const handleSchemaUpdate = useCallback(
    (payload: SchemaUpdatePayload) => {
      applySchemaUpdate(payload)
    },
    [applySchemaUpdate]
  )

  const handleSchemaFocus = useCallback(
    (payload: SchemaFocusPayload) => {
      applySchemaFocus(payload)
    },
    [applySchemaFocus]
  )

  const handleEditProperty = useCallback((path: string) => {
    // TODO: Open property editor modal
    console.log("Edit property:", path)
  }, [])

  const handleAddProperty = useCallback(() => {
    // TODO: Open add property modal
    console.log("Add property")
  }, [])

  return (
    <div className="flex h-full">
      {/* Schema Preview Panel (Left) */}
      <div className="w-[400px] border-r border-zinc-200 flex-shrink-0">
        <SchemaPreviewPanel
          schema={schema}
          focusState={focusState}
          onDescriptionChange={setDescription}
          onAddTool={addTool}
          onRemoveTool={removeTool}
          onEditProperty={handleEditProperty}
          onRemoveProperty={removeProperty}
          onAddProperty={handleAddProperty}
        />
      </div>

      {/* Chat Panel (Right) */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <ChatPanel
          schema={schema}
          onSchemaUpdate={handleSchemaUpdate}
          onSchemaFocus={handleSchemaFocus}
        />
      </div>
    </div>
  )
}
