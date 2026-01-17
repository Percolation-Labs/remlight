/**
 * Chat Types - Core data structures for the chat interface
 */

/**
 * State of a tool call during agent execution.
 */
export type ToolCallState = "pending" | "in_progress" | "completed" | "failed"

/**
 * Represents a single tool call made by the agent.
 * Tool calls are displayed as expandable cards showing args and results.
 */
export interface ToolCall {
  /** Unique identifier for this tool call */
  id: string
  /** Name of the tool being called (e.g., "search", "action") */
  name: string
  /** Arguments passed to the tool as key-value pairs */
  args: Record<string, unknown>
  /** Current state of the tool call */
  state: ToolCallState
  /** Result returned by the tool (if completed) */
  output?: unknown
  /** Error message (if failed) */
  error?: string
  /** Parent tool call ID (for sub-agent nested calls) */
  parentId?: string
  /** Agent name (for sub-agent calls, e.g., "worker-agent") */
  agentName?: string
}

/**
 * Message status during streaming.
 */
export type MessageStatus = "pending" | "streaming" | "completed" | "failed"

/**
 * Metadata attached to assistant messages.
 */
export interface MessageMetadata {
  /** Agent's confidence score (0.0 - 1.0) */
  confidence?: number
  /** Entity keys referenced in the response */
  sources?: string[]
  /** Name of the responding agent */
  agentName?: string
  /** LLM model version used */
  modelVersion?: string
  /** Response latency in milliseconds */
  latencyMs?: number
  /** Token count for the response */
  tokenCount?: number
  /** Session display name (set by agent) */
  sessionName?: string
}

/**
 * A single message in the chat conversation.
 */
export interface Message {
  /** Unique identifier for the message */
  id: string
  /** Role: user or assistant */
  role: "user" | "assistant"
  /** Text content of the message */
  content: string
  /** Current status during streaming */
  status: MessageStatus
  /** Tool calls made during this message (assistant only) */
  toolCalls?: ToolCall[]
  /** Metadata for the message (assistant only) */
  metadata?: MessageMetadata
  /** When the message was created */
  createdAt: Date
  /** Session ID this message belongs to */
  sessionId?: string
}

/**
 * A chat session containing multiple messages.
 */
export interface Session {
  /** Session UUID */
  id: string
  /** Display name for the session */
  name?: string
  /** Preview of first message in session */
  firstMessage?: string
  /** Number of messages in the session */
  messageCount?: number
  /** User who owns this session */
  userId?: string
  /** When the session was created */
  createdAt: Date
  /** When the session was last updated */
  updatedAt: Date
  /** Optional labels for categorization */
  labels?: string[]
}

/**
 * Agent configuration summary.
 */
export interface Agent {
  /** Agent schema name (e.g., "query-agent") */
  name: string
  /** Display title */
  title?: string
  /** Agent description */
  description?: string
  /** Semantic version */
  version: string
  /** Whether the agent is enabled */
  enabled: boolean
  /** Source: filesystem or database */
  source: "filesystem" | "database"
}

/**
 * Available LLM models.
 */
export interface Model {
  /** Model identifier (e.g., "openai:gpt-4.1") */
  id: string
  /** Display name */
  name: string
  /** Provider (openai, anthropic, etc.) */
  provider: string
}

/**
 * Chat settings stored in the sidebar.
 */
export interface ChatSettings {
  /** Currently selected agent schema */
  agentSchema: string
  /** Currently selected model */
  model: string
}
