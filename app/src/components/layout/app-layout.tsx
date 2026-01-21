/**
 * AppLayout - Main application layout with sidebar
 *
 * Provides the overall structure with icon rail and expandable panels.
 */

import { useCallback } from "react"
import { Sidebar } from "@/components/sidebar"
import { ChatView } from "@/components/chat"
import { useSessions } from "@/hooks/use-sessions"

export function AppLayout() {
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
   * Navigate to settings.
   */
  const handleSettings = useCallback(() => {
    // TODO: Implement settings navigation
    console.log("Settings clicked")
  }, [])

  /**
   * Handle scenario selection.
   */
  const handleScenarioSelect = useCallback((scenarioId: string) => {
    // TODO: Load scenario into chat
    console.log("Scenario selected:", scenarioId)
  }, [])

  /**
   * Handle schema selection.
   */
  const handleSchemaSelect = useCallback((schemaId: string) => {
    // TODO: Open schema editor
    console.log("Schema selected:", schemaId)
  }, [])

  /**
   * Handle new schema creation.
   */
  const handleNewSchema = useCallback(() => {
    // TODO: Open new schema dialog
    console.log("New schema clicked")
  }, [])

  /**
   * Handle ontology page selection.
   */
  const handleOntologyPageSelect = useCallback((path: string) => {
    // TODO: Display ontology page
    console.log("Ontology page selected:", path)
  }, [])

  return (
    <div className="flex h-screen bg-zinc-50">
      {/* Sidebar with icon rail and panels */}
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        isLoading={isLoading}
        onSessionSelect={handleSessionSelect}
        onNewChat={handleNewChat}
        onSearch={search}
        onSettings={handleSettings}
        onScenarioSelect={handleScenarioSelect}
        onSchemaSelect={handleSchemaSelect}
        onNewSchema={handleNewSchema}
        onOntologyPageSelect={handleOntologyPageSelect}
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
