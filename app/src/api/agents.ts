/**
 * Agents API - Fetch available agent schemas
 */

import { apiRequest } from "@/lib/api-client"
import type { Agent } from "@/types/chat"

interface AgentListResponse {
  agents: Agent[]
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
