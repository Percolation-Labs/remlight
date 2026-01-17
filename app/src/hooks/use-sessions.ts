/**
 * useSessions - Session management hook
 *
 * Handles fetching, searching, and loading chat sessions.
 */

import { useState, useEffect, useCallback } from "react"
import { fetchSessions, searchSessions } from "@/api/sessions"
import type { Session } from "@/types/chat"

export interface UseSessionsOptions {
  /** Auto-fetch sessions on mount */
  autoFetch?: boolean
}

export interface UseSessionsReturn {
  /** List of sessions */
  sessions: Session[]
  /** Currently selected session ID */
  currentSessionId: string | null
  /** Whether loading */
  isLoading: boolean
  /** Fetch sessions */
  refresh: () => Promise<void>
  /** Search sessions */
  search: (query: string) => Promise<void>
  /** Set current session */
  setCurrentSession: (sessionId: string | null) => void
}

export function useSessions(options: UseSessionsOptions = {}): UseSessionsReturn {
  const { autoFetch = true } = options

  const [sessions, setSessions] = useState<Session[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  /**
   * Fetch all sessions.
   */
  const refresh = useCallback(async () => {
    setIsLoading(true)
    try {
      const result = await fetchSessions()
      setSessions(result)
    } finally {
      setIsLoading(false)
    }
  }, [])

  /**
   * Search sessions by query.
   */
  const search = useCallback(async (query: string) => {
    if (!query.trim()) {
      return refresh()
    }

    setIsLoading(true)
    try {
      const result = await searchSessions(query)
      setSessions(result)
    } finally {
      setIsLoading(false)
    }
  }, [refresh])

  /**
   * Set current session ID.
   */
  const setCurrentSession = useCallback((sessionId: string | null) => {
    setCurrentSessionId(sessionId)
  }, [])

  /**
   * Auto-fetch on mount.
   */
  useEffect(() => {
    if (autoFetch) {
      refresh()
    }
  }, [autoFetch, refresh])

  return {
    sessions,
    currentSessionId,
    isLoading,
    refresh,
    search,
    setCurrentSession,
  }
}
