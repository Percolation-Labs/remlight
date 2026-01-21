/**
 * useAgentSchema - State management hook for the agent builder
 *
 * Manages schema state, focus state, and bi-directional sync with chat.
 */

import { useState, useCallback, useRef, useEffect } from "react"
import { stringify as stringifyYaml, parse as parseYaml } from "yaml"
import { applyPatch } from "fast-json-patch"
import type { Operation } from "fast-json-patch"
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
  setSchemaFromYaml: (yaml: string) => void
  applyJsonPatch: (patches: Operation[]) => void

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

  // Keep a ref to the latest schema for synchronous access (e.g., in toYaml)
  const schemaRef = useRef<AgentSchemaState>(schema)
  useEffect(() => {
    schemaRef.current = schema
  }, [schema])

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
            const currentTools = prev.metadata?.tools || []
            return {
              ...prev,
              metadata: {
                ...prev.metadata,
                tools: [...currentTools, value as ToolReference],
              },
            }
          } else if (operation === "remove") {
            const currentTools = prev.metadata?.tools || []
            return {
              ...prev,
              metadata: {
                ...prev.metadata,
                tools: currentTools.filter(
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

        case "properties": {
          const newProperties = { ...(prev.properties || {}) }
          if (path) {
            // Path provided - set/remove at specific path
            if (operation === "remove") {
              deleteValueByPath(newProperties as Record<string, unknown>, path)
            } else {
              setValueByPath(newProperties as Record<string, unknown>, path, value)
            }
          } else if (value && typeof value === "object" && !Array.isArray(value)) {
            // No path - value should be the property name with definition, or full properties object
            // Check if value looks like a property definition (has 'type' key)
            if ("type" in value) {
              // Agent sent {name: "foo", type: "string", ...} - extract name and set
              const propValue = value as Record<string, unknown>
              const propName = propValue.name as string
              if (propName) {
                const { name: _, ...definition } = propValue
                newProperties[propName] = definition as PropertyDefinition
              }
            } else {
              // Value is a full properties object - merge it
              Object.assign(newProperties, value)
            }
          }
          // Ignore invalid values (strings, arrays, etc.)
          return { ...prev, properties: newProperties }
        }

        case "metadata": {
          const newMetadata = { ...prev.metadata }
          if (operation === "remove" && typeof value === "string") {
            // Remove a specific field by name
            delete (newMetadata as Record<string, unknown>)[value]
          } else if (value && typeof value === "object") {
            // Merge values, removing any that are explicitly null/undefined
            const updates = value as Record<string, unknown>
            for (const [key, val] of Object.entries(updates)) {
              if (val === null || val === undefined) {
                delete (newMetadata as Record<string, unknown>)[key]
              } else {
                (newMetadata as Record<string, unknown>)[key] = val
              }
            }
          }
          return { ...prev, metadata: newMetadata as AgentSchemaState["metadata"] }
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

  // Export to YAML string - reads from ref for immediate access after patches
  const toYaml = useCallback(() => {
    // Use schemaRef.current for synchronous access to latest state
    const currentSchema = schemaRef.current

    // Build the YAML object structure
    const yamlObj: Record<string, unknown> = {
      type: "object",
      description: currentSchema.description,
    }

    // Only include properties if there are any
    if (Object.keys(currentSchema.properties).length > 0) {
      yamlObj.properties = currentSchema.properties
    }

    // Only include required if there are any
    if (currentSchema.required.length > 0) {
      yamlObj.required = currentSchema.required
    }

    // Build json_schema_extra with clean tool references
    const tools = currentSchema.metadata.tools.map(tool => {
      const t: Record<string, string> = { name: tool.name }
      if (tool.description) t.description = tool.description
      if (tool.server) t.server = tool.server
      return t
    })

    yamlObj.json_schema_extra = {
      kind: currentSchema.metadata.kind,
      name: currentSchema.metadata.name,
      version: currentSchema.metadata.version,
      structured_output: currentSchema.metadata.structured_output,
      tools,
      tags: currentSchema.metadata.tags,
    }

    return stringifyYaml(yamlObj, {
      lineWidth: 100,
      defaultStringType: "PLAIN",
      defaultKeyType: "PLAIN",
    })
  }, [])

  // Parse YAML and set entire schema state
  const setSchemaFromYaml = useCallback((yaml: string) => {
    try {
      const parsed = parseYaml(yaml) as Record<string, unknown>
      if (!parsed || typeof parsed !== "object") return

      const jsonSchemaExtra = (parsed.json_schema_extra || {}) as Record<string, unknown>
      const tools = (jsonSchemaExtra.tools || []) as Array<Record<string, string>>

      const newSchema: AgentSchemaState = {
        description: (parsed.description as string) || "",
        properties: (parsed.properties || {}) as Record<string, PropertyDefinition>,
        required: (parsed.required || []) as string[],
        metadata: {
          kind: (jsonSchemaExtra.kind as string) || "agent",
          name: (jsonSchemaExtra.name as string) || "",
          version: (jsonSchemaExtra.version as string) || "1.0.0",
          tools: tools.map(t => ({
            name: t.name || "",
            description: t.description,
            server: t.server,
          })),
          resources: (jsonSchemaExtra.resources || []) as string[],
          structured_output: (jsonSchemaExtra.structured_output as boolean) || false,
          tags: (jsonSchemaExtra.tags || []) as string[],
        },
      }

      setSchema(newSchema)
    } catch {
      // Invalid YAML - ignore
    }
  }, [])

  // Apply JSON Patch (RFC 6902) operations to schema
  // IMPORTANT: Updates schemaRef SYNCHRONOUSLY before calling setSchema
  // so that toYaml() can read the patched value immediately
  const applyJsonPatch = useCallback((patches: Operation[]) => {
    try {
      // Use ref as source of truth for synchronous access
      const cloned = JSON.parse(JSON.stringify(schemaRef.current))
      const result = applyPatch(cloned, patches, undefined, false)
      const newSchema = result.newDocument as AgentSchemaState
      // Update ref IMMEDIATELY (synchronous) - this is the key fix
      schemaRef.current = newSchema
      // Then update React state (async) for re-render
      setSchema(newSchema)
    } catch {
      // Patch failed, don't update anything
    }
  }, [])

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
    setSchemaFromYaml,
    applyJsonPatch,
    focusState,
    setFocusState,
    clearFocus,
    isValid,
    validationErrors,
    toYaml,
  }
}
