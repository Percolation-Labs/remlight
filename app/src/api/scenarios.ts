/**
 * Scenarios API - Create and manage evaluation scenarios
 */

import { getApiHeaders } from "@/lib/api-client"

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api"

export interface Scenario {
  id: string
  name: string
  description?: string
  session_id: string
  agent_name?: string
  status: "active" | "archived"
  tags: string[]
  metadata: Record<string, unknown>
  user_id?: string
  created_at: string
  updated_at: string
}

export interface CreateScenarioRequest {
  name: string
  description?: string
  session_id: string
  agent_name?: string
  tags?: string[]
  metadata?: Record<string, unknown>
}

/**
 * List all scenarios.
 */
export async function listScenarios(): Promise<Scenario[]> {
  const response = await fetch(`${API_BASE_URL}/v1/scenarios`, {
    headers: getApiHeaders(),
  })

  if (!response.ok) {
    throw new Error(`Failed to list scenarios: ${response.status}`)
  }

  return response.json()
}

/**
 * Create a new scenario from a session.
 */
export async function createScenario(
  request: CreateScenarioRequest
): Promise<Scenario> {
  const response = await fetch(`${API_BASE_URL}/v1/scenarios`, {
    method: "POST",
    headers: getApiHeaders(),
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    const error = await response.text()
    throw new Error(error || `Failed to create scenario: ${response.status}`)
  }

  return response.json()
}

/**
 * Get a scenario by ID.
 */
export async function getScenario(id: string): Promise<Scenario> {
  const response = await fetch(`${API_BASE_URL}/v1/scenarios/${id}`, {
    headers: getApiHeaders(),
  })

  if (!response.ok) {
    throw new Error(`Failed to get scenario: ${response.status}`)
  }

  return response.json()
}
