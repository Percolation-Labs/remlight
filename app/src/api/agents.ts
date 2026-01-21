/**
 * Agents API - Fetch available agent schemas
 */

import { apiRequest } from "@/lib/api-client"
import type { Agent } from "@/types/chat"

interface AgentListResponse {
  agents: Agent[]
}

export interface AgentContentResponse {
  name: string
  version: string
  content: string
  source: string
  enabled: boolean
  icon: string | null
  tags: string[]
}

/**
 * Default agents available without API.
 */
const DEFAULT_AGENTS: Agent[] = [
  { name: "default", title: "Default Agent", version: "1.0.0", enabled: true, source: "filesystem" },
]

/**
 * Fetch list of available agents from the API.
 * Falls back to default agents on error.
 */
export async function fetchAgents(): Promise<Agent[]> {
  try {
    const response = await apiRequest<AgentListResponse>("/v1/agents")
    return response.agents || DEFAULT_AGENTS
  } catch (error) {
    console.warn("Failed to fetch agents, using defaults:", error)
    return DEFAULT_AGENTS
  }
}

/**
 * Fetch full agent content (YAML) by name.
 */
export async function fetchAgentContent(agentName: string): Promise<AgentContentResponse | null> {
  try {
    return await apiRequest<AgentContentResponse>(`/v1/agents/${agentName}/content`)
  } catch (error) {
    console.warn(`Failed to fetch agent content for '${agentName}':`, error)
    return null
  }
}

/**
 * Save agent to the database.
 */
export async function saveAgent(content: string, enabled: boolean = true, tags: string[] = []): Promise<{ name: string; created: boolean } | null> {
  try {
    return await apiRequest<{ name: string; version: string; created: boolean; message: string }>("/v1/agents", {
      method: "PUT",
      body: JSON.stringify({ content, enabled, tags }),
    })
  } catch (error) {
    console.warn("Failed to save agent:", error)
    return null
  }
}
