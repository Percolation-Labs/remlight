/**
 * AppLayout - Main application layout with sidebar
 *
 * Provides the overall structure with collapsible sidebar.
 */

import { useCallback, useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { ChatView } from "@/components/chat"
import { useSessions } from "@/hooks/use-sessions"

export function AppLayout() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const {
    sessions,
    currentSessionId,
    isLoading,
    refresh,
    search,
    setCurrentSession,
  } = useSessions()

  /**
   * Handle session selection from sidebar.
   */
  const handleSessionSelect = useCallback((sessionId: string) => {
    setCurrentSession(sessionId)
  }, [setCurrentSession])

  /**
   * Handle new chat creation.
   */
  const handleNewChat = useCallback(() => {
    setCurrentSession(null)
  }, [setCurrentSession])

  /**
   * Handle session change from chat view.
   */
  const handleSessionChange = useCallback((sessionId: string) => {
    setCurrentSession(sessionId)
    // Refresh sessions list to include the new session
    refresh()
  }, [setCurrentSession, refresh])

  /**
   * Navigate to settings (placeholder).
   */
  const handleSettings = useCallback(() => {
    // TODO: Implement settings navigation
    console.log("Settings clicked")
  }, [])

  return (
    <div className="flex h-screen bg-zinc-50">
      {/* Sidebar */}
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        isLoading={isLoading}
        isCollapsed={isCollapsed}
        onSessionSelect={handleSessionSelect}
        onNewChat={handleNewChat}
        onSearch={search}
        onSettings={handleSettings}
        onToggleCollapse={() => setIsCollapsed(!isCollapsed)}
      />

      {/* Main content */}
      <main className="flex-1 flex flex-col overflow-hidden">
        <ChatView
          sessionId={currentSessionId || undefined}
          onSessionChange={handleSessionChange}
        />
      </main>
    </div>
  )
}
