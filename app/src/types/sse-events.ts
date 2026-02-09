/**
 * SSE Event Types - Server-Sent Events from the chat API
 *
 * These types define the structure of events streamed from the
 * /api/chat/completions endpoint.
 */

/**
 * OpenAI-compatible text delta event.
 * Contains a chunk of the assistant's response text.
 */
export interface TextDeltaEvent {
  type: "text_delta"
  /** OpenAI completion ID (not the database message ID) */
  id?: string
  choices: Array<{
    delta: {
      content?: string
      role?: string
    }
    index: number
    finish_reason?: string | null
  }>
}

/**
 * Tool call event from the agent.
 * Tracks the lifecycle of a tool invocation.
 */
export interface ToolCallEvent {
  type: "tool_call"
  /** Name of the tool being called */
  tool_name: string
  /** Tool call ID for matching with results (API sends 'tool_id') */
  tool_call_id?: string
  /** Tool call ID (alternative field name from API) */
  tool_id?: string
  /** Current status of the tool call */
  status: "started" | "in_progress" | "executing" | "completed" | "failed"
  /** Arguments passed to the tool */
  arguments?: Record<string, unknown>
  /** Result from the tool (when completed) */
  result?: unknown
  /** Error message (when failed) */
  error?: string
}

/**
 * Metadata event with session and response information.
 * IMPORTANT: The message_id here is the database ID, not the OpenAI completion ID.
 */
export interface MetadataEvent {
  type: "metadata"
  /** Database message ID (use for feedback) */
  message_id?: string
  /** Agent's confidence score */
  confidence?: number
  /** Entity keys referenced */
  sources?: string[]
  /** Agent schema that handled the request */
  agent_schema?: string
  /** Responding agent name */
  responding_agent?: string
  /** Response latency in ms */
  latency_ms?: number
  /** Token count */
  token_count?: number
  /** Model version used */
  model_version?: string
  /** Session display name */
  session_name?: string
  /** Extension fields */
  extra?: Record<string, unknown>
}

/**
 * Reasoning/thinking event from the agent.
 */
export interface ReasoningEvent {
  type: "reasoning"
  /** Reasoning step content */
  content: string
  /** Step number in the chain */
  step?: number
}

/**
 * Stream completion event.
 */
export interface DoneEvent {
  type: "done"
  /** Optional finish reason */
  reason?: string
}

/**
 * Error event.
 */
export interface ErrorEvent {
  type: "error"
  /** Error message */
  message: string
  /** Error code */
  code?: string
}

/**
 * Action event emitted via the action tool.
 * Contains structured data for UI rendering.
 */
export interface ActionEvent {
  type: "action"
  /** Action type (e.g., "observation", "elicit") */
  action_type: string
  /** Action payload */
  payload?: Record<string, unknown>
  /** Whether this is an action event marker */
  _action_event?: boolean
}

/**
 * Child agent content event (from nested agent calls).
 */
export interface ChildContentEvent {
  type: "child_content"
  /** Content chunk from child agent */
  content: string
  /** Name of the child agent */
  agent_name?: string
}

/**
 * Child agent tool call event.
 */
export interface ChildToolEvent {
  type: "child_tool_start" | "child_tool_result"
  /** Tool name */
  tool_name: string
  /** Tool call ID for matching start/result */
  tool_call_id?: string
  /** Tool arguments */
  arguments?: Record<string, unknown>
  /** Tool result (for child_tool_result) */
  result?: unknown
  /** Child agent name */
  agent_name?: string
}

/**
 * Progress event for request processing status.
 */
export interface ProgressEvent {
  type: "progress"
  /** Current step number */
  step?: number
  /** Total steps */
  total_steps?: number
  /** Status label */
  label?: string
  /** Progress status */
  status?: string
}

/**
 * Schema update event from agent-builder.
 * Updates a specific section of the schema being built.
 * Can come either with fields at top level or nested in payload (from ActionEvent).
 */
export interface SchemaUpdateEvent {
  type: "schema_update"
  /** Which section to update (may be at top level or in payload) */
  section?: "tools" | "system_prompt" | "properties" | "metadata"
  /** The new value (may be at top level or in payload) */
  value?: unknown
  /** Operation type */
  operation?: "set" | "append" | "remove"
  /** Optional nested path within section */
  path?: string
  /** Payload from ActionEvent format */
  payload?: {
    section: "tools" | "system_prompt" | "properties" | "metadata"
    value: unknown
    operation?: "set" | "append" | "remove"
    path?: string
  }
  /** Action event marker */
  _action_event?: boolean
  action_type?: "schema_update"
}

/**
 * Schema focus event from agent-builder.
 * Highlights a specific section in the UI.
 * Can come either with fields at top level or nested in payload (from ActionEvent).
 */
export interface SchemaFocusEvent {
  type: "schema_focus"
  /** Which section to focus (may be at top level or in payload) */
  section?: "tools" | "system_prompt" | "properties" | "metadata" | null
  /** Optional property path for properties section */
  property_path?: string
  /** Optional message to display */
  message?: string
  /** Payload from ActionEvent format */
  payload?: {
    section: "tools" | "system_prompt" | "properties" | "metadata" | null
    property_path?: string
    message?: string
  }
  /** Action event marker */
  _action_event?: boolean
  action_type?: "schema_focus"
}

/**
 * Union type of all possible SSE events.
 */
export type SSEEvent =
  | TextDeltaEvent
  | ToolCallEvent
  | MetadataEvent
  | ReasoningEvent
  | DoneEvent
  | ErrorEvent
  | ActionEvent
  | ChildContentEvent
  | ChildToolEvent
  | ProgressEvent
  | SchemaUpdateEvent
  | SchemaFocusEvent

/**
 * Raw SSE event before type discrimination.
 */
export interface RawSSEEvent {
  type?: string
  [key: string]: unknown
}
