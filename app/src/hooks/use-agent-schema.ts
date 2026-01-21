/**
 * useAgentSchema - State management hook for the agent builder
 *
 * Manages schema state, focus state, and bi-directional sync with chat.
 */

import { useState, useCallback, useRef, useEffect } from "react"
import { stringify as stringifyYaml } from "yaml"
import type {
  AgentSchemaState,
  FocusState,
  SchemaUpdatePayload,
  SchemaFocusPayload,
  PropertyDefinition,
  ToolReference,
  UserSchemaEdit,
} from "@/types/agent-schema"

export interface UseAgentSchemaOptions {
  initialSchema?: Partial<AgentSchemaState>
  onUserEdit?: (edit: UserSchemaEdit) => void
}

export interface UseAgentSchemaReturn {
  // Schema state
  schema: AgentSchemaState
  setSchema: React.Dispatch<React.SetStateAction<AgentSchemaState>>

  // Section updaters
  setDescription: (description: string) => void
  addTool: (tool: ToolReference) => void
  removeTool: (toolName: string) => void
  setProperty: (path: string, definition: PropertyDefinition) => void
  removeProperty: (path: string) => void
  setRequired: (fields: string[]) => void
  setMetadata: (updates: Partial<AgentSchemaState["metadata"]>) => void

  // SSE event handlers
  applySchemaUpdate: (payload: SchemaUpdatePayload) => void
  applySchemaFocus: (payload: SchemaFocusPayload) => void

  // Focus state
  focusState: FocusState
  setFocusState: React.Dispatch<React.SetStateAction<FocusState>>
  clearFocus: () => void

  // Validation
  isValid: boolean
  validationErrors: string[]

  // Export
  toYaml: () => string
}

/**
 * Set a value in a nested object by dot-separated path.
 */
function setValueByPath(obj: Record<string, unknown>, path: string, value: unknown): void {
  const keys = path.split(".")
  let current = obj as Record<string, unknown>

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i]
    if (!(key in current) || typeof current[key] !== "object") {
      current[key] = {}
    }
    current = current[key] as Record<string, unknown>
  }

  current[keys[keys.length - 1]] = value
}

/**
 * Delete a value from a nested object by dot-separated path.
 */
function deleteValueByPath(obj: Record<string, unknown>, path: string): void {
  const keys = path.split(".")
  let current = obj as Record<string, unknown>

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i]
    if (!(key in current) || typeof current[key] !== "object") {
      return
    }
    current = current[key] as Record<string, unknown>
  }

  delete current[keys[keys.length - 1]]
}

/**
 * Creates an empty default schema state.
 */
function createDefaultSchema(): AgentSchemaState {
  return {
    description: "",
    properties: {},
    required: [],
    metadata: {
      kind: "agent",
      name: "",
      version: "1.0.0",
      tools: [],
      resources: [],
      structured_output: false,
      tags: [],
    },
  }
}

export function useAgentSchema(options: UseAgentSchemaOptions = {}): UseAgentSchemaReturn {
  const { initialSchema, onUserEdit } = options

  // Main schema state
  const [schema, setSchema] = useState<AgentSchemaState>(() => ({
    ...createDefaultSchema(),
    ...initialSchema,
    metadata: {
      ...createDefaultSchema().metadata,
      ...initialSchema?.metadata,
    },
  }))

  // Focus state with timeout
  const [focusState, setFocusState] = useState<FocusState>({
    section: null,
  })
  const focusTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Clear focus timeout on unmount
  useEffect(() => {
    return () => {
      if (focusTimeoutRef.current) {
        clearTimeout(focusTimeoutRef.current)
      }
    }
  }, [])

  // Section updaters with bi-directional sync
  const setDescription = useCallback(
    (description: string) => {
      setSchema((prev) => ({ ...prev, description }))
      onUserEdit?.({
        type: "user_schema_edit",
        section: "system_prompt",
        value: description,
        summary: "Updated system prompt",
      })
    },
    [onUserEdit]
  )

  const addTool = useCallback(
    (tool: ToolReference) => {
      setSchema((prev) => ({
        ...prev,
        metadata: {
          ...prev.metadata,
          tools: [...prev.metadata.tools.filter((t) => t.name !== tool.name), tool],
        },
      }))
      onUserEdit?.({
        type: "user_schema_edit",
        section: "tools",
        value: tool,
        summary: `Added tool: ${tool.name}`,
      })
    },
    [onUserEdit]
  )

  const removeTool = useCallback(
    (toolName: string) => {
      setSchema((prev) => ({
        ...prev,
        metadata: {
          ...prev.metadata,
          tools: prev.metadata.tools.filter((t) => t.name !== toolName),
        },
      }))
      onUserEdit?.({
        type: "user_schema_edit",
        section: "tools",
        value: toolName,
        summary: `Removed tool: ${toolName}`,
      })
    },
    [onUserEdit]
  )

  const setProperty = useCallback(
    (path: string, definition: PropertyDefinition) => {
      setSchema((prev) => {
        const newProperties = { ...prev.properties }
        setValueByPath(newProperties as Record<string, unknown>, path, definition)
        return { ...prev, properties: newProperties }
      })
      onUserEdit?.({
        type: "user_schema_edit",
        section: "properties",
        value: { path, definition },
        summary: `Updated property: ${path}`,
      })
    },
    [onUserEdit]
  )

  const removeProperty = useCallback(
    (path: string) => {
      setSchema((prev) => {
        const newProperties = { ...prev.properties }
        deleteValueByPath(newProperties as Record<string, unknown>, path)
        return { ...prev, properties: newProperties }
      })
      onUserEdit?.({
        type: "user_schema_edit",
        section: "properties",
        value: path,
        summary: `Removed property: ${path}`,
      })
    },
    [onUserEdit]
  )

  const setRequired = useCallback(
    (fields: string[]) => {
      setSchema((prev) => ({ ...prev, required: fields }))
    },
    []
  )

  const setMetadata = useCallback(
    (updates: Partial<AgentSchemaState["metadata"]>) => {
      setSchema((prev) => ({
        ...prev,
        metadata: { ...prev.metadata, ...updates },
      }))
      onUserEdit?.({
        type: "user_schema_edit",
        section: "metadata",
        value: updates,
        summary: `Updated metadata`,
      })
    },
    [onUserEdit]
  )

  // SSE event handlers
  const applySchemaUpdate = useCallback((payload: SchemaUpdatePayload) => {
    const { section, value, operation = "set", path } = payload

    setSchema((prev) => {
      switch (section) {
        case "system_prompt":
          return { ...prev, description: value as string }

        case "tools":
          if (operation === "append") {
            return {
              ...prev,
              metadata: {
                ...prev.metadata,
                tools: [...prev.metadata.tools, value as ToolReference],
              },
            }
          } else if (operation === "remove") {
            return {
              ...prev,
              metadata: {
                ...prev.metadata,
                tools: prev.metadata.tools.filter(
                  (t) => t.name !== (value as ToolReference).name
                ),
              },
            }
          } else {
            return {
              ...prev,
              metadata: {
                ...prev.metadata,
                tools: value as ToolReference[],
              },
            }
          }

        case "properties":
          if (path) {
            const newProperties = { ...prev.properties }
            if (operation === "remove") {
              deleteValueByPath(newProperties as Record<string, unknown>, path)
            } else {
              setValueByPath(newProperties as Record<string, unknown>, path, value)
            }
            return { ...prev, properties: newProperties }
          } else {
            return { ...prev, properties: value as Record<string, PropertyDefinition> }
          }

        case "metadata":
          return {
            ...prev,
            metadata: { ...prev.metadata, ...(value as Partial<AgentSchemaState["metadata"]>) },
          }

        default:
          return prev
      }
    })
  }, [])

  const applySchemaFocus = useCallback((payload: SchemaFocusPayload) => {
    const { section, property_path, message } = payload

    // Clear existing timeout
    if (focusTimeoutRef.current) {
      clearTimeout(focusTimeoutRef.current)
    }

    setFocusState({
      section,
      propertyPath: property_path,
      message,
      expiresAt: Date.now() + 3000, // 3 second focus
    })

    // Auto-clear focus after duration
    focusTimeoutRef.current = setTimeout(() => {
      setFocusState({ section: null })
    }, 3000)
  }, [])

  const clearFocus = useCallback(() => {
    if (focusTimeoutRef.current) {
      clearTimeout(focusTimeoutRef.current)
    }
    setFocusState({ section: null })
  }, [])

  // Validation
  const validationErrors: string[] = []
  if (!schema.metadata.name) {
    validationErrors.push("Agent name is required")
  }
  if (!schema.description) {
    validationErrors.push("System prompt is required")
  }
  const isValid = validationErrors.length === 0

  // Export to YAML string
  const toYaml = useCallback(() => {
    // Build the YAML object structure
    const yamlObj: Record<string, unknown> = {
      type: "object",
      description: schema.description,
    }

    // Only include properties if there are any
    if (Object.keys(schema.properties).length > 0) {
      yamlObj.properties = schema.properties
    }

    // Only include required if there are any
    if (schema.required.length > 0) {
      yamlObj.required = schema.required
    }

    // Build json_schema_extra with clean tool references
    const tools = schema.metadata.tools.map(tool => {
      const t: Record<string, string> = { name: tool.name }
      if (tool.description) t.description = tool.description
      if (tool.server) t.server = tool.server
      return t
    })

    yamlObj.json_schema_extra = {
      kind: schema.metadata.kind,
      name: schema.metadata.name,
      version: schema.metadata.version,
      structured_output: schema.metadata.structured_output,
      tools,
      tags: schema.metadata.tags,
    }

    return stringifyYaml(yamlObj, {
      lineWidth: 100,
      defaultStringType: "PLAIN",
      defaultKeyType: "PLAIN",
    })
  }, [schema])

  return {
    schema,
    setSchema,
    setDescription,
    addTool,
    removeTool,
    setProperty,
    removeProperty,
    setRequired,
    setMetadata,
    applySchemaUpdate,
    applySchemaFocus,
    focusState,
    setFocusState,
    clearFocus,
    isValid,
    validationErrors,
    toYaml,
  }
}
