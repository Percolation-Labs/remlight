/**
 * AgentBuilderView - Main split layout for agent builder
 *
 * Left: Schema preview panel
 * Right: Chat panel with agent-builder agent
 */

import { useCallback, useEffect, useState } from "react"
import { parse as parseYaml } from "yaml"
import { SchemaPreviewPanel } from "./schema-preview-panel"
import { ChatPanel } from "./chat-panel"
import { useAgentSchema } from "@/hooks/use-agent-schema"
import { fetchAgentContent } from "@/api/agents"
import type { SchemaUpdatePayload, SchemaFocusPayload, UserSchemaEdit, AgentSchemaState, ToolReference, PropertyDefinition } from "@/types/agent-schema"

interface AgentBuilderViewProps {
  initialAgentName?: string
  onExportYaml?: (yaml: string) => void
}

// Valid property types
const VALID_TYPES = ["string", "number", "integer", "boolean", "array", "object"] as const
type ValidType = typeof VALID_TYPES[number]

/**
 * Recursively convert parsed YAML properties to PropertyDefinition.
 */
function convertProperties(props: Record<string, unknown> | undefined): Record<string, PropertyDefinition> {
  if (!props || typeof props !== "object") return {}

  const result: Record<string, PropertyDefinition> = {}

  for (const [name, value] of Object.entries(props)) {
    if (value && typeof value === "object") {
      const prop = value as Record<string, unknown>
      const rawType = String(prop.type || "string")
      const propType: ValidType = VALID_TYPES.includes(rawType as ValidType)
        ? (rawType as ValidType)
        : "string"

      result[name] = {
        type: propType,
        description: prop.description ? String(prop.description) : undefined,
        enum: Array.isArray(prop.enum) ? prop.enum.map(String) : undefined,
        minimum: typeof prop.minimum === "number" ? prop.minimum : undefined,
        maximum: typeof prop.maximum === "number" ? prop.maximum : undefined,
        default: prop.default,
        properties: prop.properties ? convertProperties(prop.properties as Record<string, unknown>) : undefined,
        items: prop.items ? convertProperties({ item: prop.items })["item"] : undefined,
        required: Array.isArray(prop.required) ? prop.required.map(String) : undefined,
      }
    }
  }

  return result
}

/**
 * Parse YAML content to extract schema state using proper YAML parser.
 */
function parseAgentYamlContent(content: string): Partial<AgentSchemaState> | null {
  try {
    const parsed = parseYaml(content) as Record<string, unknown>

    // Extract description (system prompt)
    const description = typeof parsed.description === "string" ? parsed.description : ""

    // Extract json_schema_extra (metadata)
    const extra = (parsed.json_schema_extra || {}) as Record<string, unknown>

    // Convert tools
    const rawTools = Array.isArray(extra.tools) ? extra.tools : []
    const tools: ToolReference[] = rawTools.map((t: unknown) => {
      if (typeof t === "object" && t !== null) {
        const tool = t as Record<string, unknown>
        return {
          name: String(tool.name || ""),
          description: tool.description ? String(tool.description) : undefined,
          server: tool.server ? String(tool.server) : undefined,
        }
      }
      return { name: String(t) }
    })

    // Convert tags
    const rawTags = Array.isArray(extra.tags) ? extra.tags : []
    const tags = rawTags.map((t: unknown) => String(t))

    const metadata: AgentSchemaState["metadata"] = {
      kind: "agent",
      name: extra.name ? String(extra.name) : "",
      version: extra.version ? String(extra.version) : "1.0.0",
      tools,
      resources: [],
      structured_output: extra.structured_output === true,
      tags,
    }

    // Extract properties
    const properties = convertProperties(parsed.properties as Record<string, unknown> | undefined)

    // Extract required fields
    const rawRequired = Array.isArray(parsed.required) ? parsed.required : []
    const required = rawRequired.map((r: unknown) => String(r))

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
  const [, setIsLoading] = useState(!!initialAgentName)

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
    toYaml,
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
          const parsed = parseAgentYamlContent(agentContent.content)
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
      <div className="w-[440px] min-w-[440px] border-r border-zinc-200 shrink-0">
        <SchemaPreviewPanel
          schema={schema}
          focusState={focusState}
          onDescriptionChange={setDescription}
          onAddTool={addTool}
          onRemoveTool={removeTool}
          onEditProperty={handleEditProperty}
          onRemoveProperty={removeProperty}
          onAddProperty={handleAddProperty}
          onExportYaml={toYaml}
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
