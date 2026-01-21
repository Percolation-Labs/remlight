/**
 * Agent Schema Types
 *
 * TypeScript types for agent schemas used in the agent builder.
 */

/**
 * Reference to an MCP tool that an agent can use.
 */
export interface ToolReference {
  name: string
  server?: string
  description?: string
}

/**
 * Reference to an MCP resource.
 */
export interface ResourceReference {
  uri?: string
  uri_pattern?: string
  name?: string
  description?: string
}

/**
 * Schema metadata (json_schema_extra in YAML).
 */
export interface SchemaMetadata {
  kind: "agent"
  name: string
  version: string
  tools: ToolReference[]
  resources: ResourceReference[]
  structured_output: boolean
  override_model?: string
  override_temperature?: number
  override_max_iterations?: number
  tags: string[]
}

/**
 * Recursive property definition for schema fields.
 */
export interface PropertyDefinition {
  type: "string" | "number" | "integer" | "boolean" | "array" | "object"
  description?: string
  properties?: Record<string, PropertyDefinition>
  items?: PropertyDefinition
  required?: string[]
  enum?: string[]
  minimum?: number
  maximum?: number
  default?: unknown
}

/**
 * Complete agent schema state for the builder.
 */
export interface AgentSchemaState {
  description: string
  properties: Record<string, PropertyDefinition>
  required: string[]
  metadata: SchemaMetadata
}

/**
 * Focus state for highlighting sections in the UI.
 */
export type FocusSection =
  | "tools"
  | "system_prompt"
  | "properties"
  | "metadata"
  | null

export interface FocusState {
  section: FocusSection
  propertyPath?: string
  message?: string
  expiresAt?: number
}

/**
 * Schema update event payload from SSE.
 */
export interface SchemaUpdatePayload {
  section: "tools" | "system_prompt" | "properties" | "metadata"
  value: unknown
  operation?: "set" | "append" | "remove"
  path?: string
}

/**
 * Schema focus event payload from SSE.
 */
export interface SchemaFocusPayload {
  section: FocusSection
  property_path?: string
  message?: string
}

/**
 * User edit notification for bi-directional sync.
 */
export interface UserSchemaEdit {
  type: "user_schema_edit"
  section: string
  value: unknown
  summary: string
}

/**
 * Creates an empty default schema state.
 */
export function createEmptySchema(): AgentSchemaState {
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

/**
 * Converts schema state to YAML-compatible object.
 */
export function schemaToYamlObject(schema: AgentSchemaState): Record<string, unknown> {
  return {
    type: "object",
    description: schema.description,
    properties: schema.properties,
    required: schema.required,
    json_schema_extra: schema.metadata,
  }
}

/**
 * Type icons for property types.
 */
export const PROPERTY_TYPE_ICONS: Record<PropertyDefinition["type"], string> = {
  string: "Type",
  number: "Hash",
  integer: "Hash",
  boolean: "ToggleLeft",
  array: "List",
  object: "Braces",
}

/**
 * Type labels for display.
 */
export const PROPERTY_TYPE_LABELS: Record<PropertyDefinition["type"], string> = {
  string: "String",
  number: "Number",
  integer: "Integer",
  boolean: "Boolean",
  array: "Array",
  object: "Object",
}
