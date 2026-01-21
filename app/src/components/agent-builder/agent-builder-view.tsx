/**
 * AgentBuilderView - Main split layout for agent builder
 *
 * Left: Schema preview panel
 * Right: Chat panel with agent-builder agent
 */

import { useCallback, useEffect, useState } from "react"
import { SchemaPreviewPanel } from "./schema-preview-panel"
import { ChatPanel } from "./chat-panel"
import { useAgentSchema } from "@/hooks/use-agent-schema"
import { fetchAgentContent } from "@/api/agents"
import type { SchemaUpdatePayload, SchemaFocusPayload, UserSchemaEdit, AgentSchemaState, ToolReference, PropertyDefinition } from "@/types/agent-schema"

interface AgentBuilderViewProps {
  initialAgentName?: string
}

/**
 * Parse YAML content to extract schema state.
 * Simple parser for agent YAML format.
 */
function parseAgentYaml(content: string): Partial<AgentSchemaState> | null {
  try {
    // Extract description (multiline string after "description: |" or "description:")
    const descMatch = content.match(/description:\s*\|?\s*\n([\s\S]*?)(?=\n\w|\nproperties:|\nrequired:|\njson_schema_extra:)/i)
    let description = ""
    if (descMatch) {
      // Remove leading indentation from description
      description = descMatch[1]
        .split("\n")
        .map(line => line.replace(/^  /, ""))
        .join("\n")
        .trim()
    }

    // Extract json_schema_extra section
    const metaMatch = content.match(/json_schema_extra:\s*\n([\s\S]*?)(?=\n[^\s]|$)/i)
    const metadata: AgentSchemaState["metadata"] = {
      kind: "agent",
      name: "",
      version: "1.0.0",
      tools: [],
      resources: [],
      structured_output: false,
      tags: [],
    }

    if (metaMatch) {
      const metaSection = metaMatch[1]

      // Extract name
      const nameMatch = metaSection.match(/name:\s*(.+)/i)
      if (nameMatch) metadata.name = nameMatch[1].trim()

      // Extract version
      const versionMatch = metaSection.match(/version:\s*["']?([^"'\n]+)["']?/i)
      if (versionMatch) metadata.version = versionMatch[1].trim()

      // Extract structured_output
      const structuredMatch = metaSection.match(/structured_output:\s*(true|false)/i)
      if (structuredMatch) metadata.structured_output = structuredMatch[1].toLowerCase() === "true"

      // Extract tools
      const toolsMatch = metaSection.match(/tools:\s*\n((?:\s+-[^\n]+\n?)+)/i)
      if (toolsMatch) {
        const toolLines = toolsMatch[1].match(/-\s*name:\s*(\S+)/gi) || []
        metadata.tools = toolLines.map(line => {
          const name = line.match(/-\s*name:\s*(\S+)/i)?.[1] || ""
          return { name, description: "" } as ToolReference
        })
      }

      // Extract tags
      const tagsMatch = metaSection.match(/tags:\s*\[([^\]]*)\]/i)
      if (tagsMatch) {
        metadata.tags = tagsMatch[1].split(",").map(t => t.trim().replace(/["']/g, "")).filter(Boolean)
      }
    }

    // Extract properties
    const properties: Record<string, PropertyDefinition> = {}
    const propsMatch = content.match(/properties:\s*\n([\s\S]*?)(?=\nrequired:|\njson_schema_extra:|$)/i)
    if (propsMatch) {
      // Simple property extraction (top-level only for now)
      const propSection = propsMatch[1]
      const propBlocks = propSection.match(/^  (\w+):\s*\n((?:    [^\n]+\n?)+)/gm) || []
      const validTypes = ["string", "number", "integer", "boolean", "array", "object"] as const
      for (const block of propBlocks) {
        const propNameMatch = block.match(/^  (\w+):/m)
        if (propNameMatch) {
          const propName = propNameMatch[1]
          const typeMatch = block.match(/type:\s*(\w+)/i)
          const descMatch = block.match(/description:\s*(.+)/i)
          const rawType = typeMatch?.[1] || "string"
          const propType = validTypes.includes(rawType as typeof validTypes[number])
            ? rawType as typeof validTypes[number]
            : "string"
          properties[propName] = {
            type: propType,
            description: descMatch?.[1] || "",
          }
        }
      }
    }

    // Extract required fields
    const reqMatch = content.match(/required:\s*\n((?:\s*-\s*\w+\n?)+)/i)
    const required: string[] = []
    if (reqMatch) {
      const reqLines = reqMatch[1].match(/-\s*(\w+)/g) || []
      for (const line of reqLines) {
        const field = line.match(/-\s*(\w+)/)?.[1]
        if (field) required.push(field)
      }
    }

    return {
      description,
      properties,
      required,
      metadata,
    }
  } catch (e) {
    console.error("Failed to parse agent YAML:", e)
    return null
  }
}

export function AgentBuilderView({ initialAgentName }: AgentBuilderViewProps) {
  const [isLoading, setIsLoading] = useState(!!initialAgentName)

  const handleUserEdit = useCallback((edit: UserSchemaEdit) => {
    // This will be sent to the chat as context
    console.log("User edit:", edit)
  }, [])

  const {
    schema,
    setSchema,
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

  // Load existing agent content when editing
  useEffect(() => {
    if (!initialAgentName) return

    const loadAgent = async () => {
      setIsLoading(true)
      try {
        const agentContent = await fetchAgentContent(initialAgentName)
        if (agentContent?.content) {
          const parsed = parseAgentYaml(agentContent.content)
          if (parsed) {
            setSchema(prev => ({
              description: parsed.description || prev.description,
              properties: parsed.properties || prev.properties,
              required: parsed.required || prev.required,
              metadata: {
                ...prev.metadata,
                ...parsed.metadata,
              },
            }))
          }
        }
      } catch (e) {
        console.error("Failed to load agent:", e)
      } finally {
        setIsLoading(false)
      }
    }

    loadAgent()
  }, [initialAgentName, setSchema])

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
