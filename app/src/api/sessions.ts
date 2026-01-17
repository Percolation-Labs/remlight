/**
 * Sessions API - Fetch and manage chat sessions
 */

import { apiRequest } from "@/lib/api-client"
import type { Session } from "@/types/chat"

// API returns snake_case for session fields
interface ApiSession {
  id: string
  name?: string | null
  first_message?: string | null
  message_count?: number
  user_id?: string | null
  created_at: string
  updated_at: string
  labels?: string[]
}

interface SessionListResponse {
  sessions: ApiSession[]
}

// API returns snake_case, we need to handle both formats
interface ApiMessage {
  id: string
  role: string
  content: string
  status?: string
  tool_calls?: unknown[]
  toolCalls?: unknown[]
  metadata?: Record<string, unknown>
  created_at?: string
  createdAt?: string | Date
  session_id?: string
  sessionId?: string
}

interface SessionMessagesResponse {
  messages: ApiMessage[]
}

/**
 * Map API session (snake_case) to frontend Session (camelCase).
 */
function mapApiSession(s: ApiSession): Session {
  return {
    id: s.id,
    name: s.name || undefined,
    firstMessage: s.first_message || undefined,
    messageCount: s.message_count,
    userId: s.user_id || undefined,
    createdAt: new Date(s.created_at),
    updatedAt: new Date(s.updated_at),
    labels: s.labels,
  }
}

/**
 * Fetch list of sessions for the current user.
 */
export async function fetchSessions(): Promise<Session[]> {
  try {
    const response = await apiRequest<SessionListResponse>("/v1/sessions")
    return (response.sessions || []).map(mapApiSession)
  } catch (error) {
    console.warn("Failed to fetch sessions:", error)
    return []
  }
}

/**
 * Fetch messages for a specific session.
 * Returns raw API messages - caller should filter/transform as needed.
 */
export async function fetchSessionMessages(sessionId: string): Promise<ApiMessage[]> {
  try {
    const response = await apiRequest<SessionMessagesResponse>(
      `/v1/sessions/${sessionId}/messages`
    )
    return response.messages || []
  } catch (error) {
    console.warn("Failed to fetch session messages:", error)
    return []
  }
}

/**
 * Search sessions by query string.
 */
export async function searchSessions(query: string): Promise<Session[]> {
  try {
    const response = await apiRequest<SessionListResponse>(
      `/v1/sessions?search=${encodeURIComponent(query)}`
    )
    return (response.sessions || []).map(mapApiSession)
  } catch (error) {
    console.warn("Failed to search sessions:", error)
    return []
  }
}

/**
 * Export session as YAML file.
 * Triggers a browser download of the YAML file.
 */
export async function exportSessionAsYaml(sessionId: string): Promise<void> {
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api"
  const url = `${API_BASE_URL}/v1/sessions/${sessionId}/export`

  try {
    const response = await fetch(url)
    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`)
    }

    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = downloadUrl
    a.download = `session-${sessionId}.yaml`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(downloadUrl)
  } catch (error) {
    console.error("Failed to export session:", error)
    throw error
  }
}
