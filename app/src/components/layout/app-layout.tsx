/**
 * AppLayout - Main application layout with sidebar
 *
 * Provides the overall structure with icon rail and expandable panels.
 */

import { useCallback, useState, useMemo } from "react"
import { useSearchParams } from "react-router-dom"
import ReactMarkdown from "react-markdown"
import { X } from "lucide-react"
import { Sidebar } from "@/components/sidebar"
import { ChatView } from "@/components/chat"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useSessions } from "@/hooks/use-sessions"
import type { PanelType } from "@/components/sidebar/nav-rail"

interface OntologyPage {
  path: string
  name: string
  content: string
}

export function AppLayout() {
  const [searchParams] = useSearchParams()

  const {
    sessions,
    currentSessionId,
    isLoading,
    refresh,
    search,
    setCurrentSession,
  } = useSessions()

  const [ontologyPage, setOntologyPage] = useState<OntologyPage | null>(null)

  // Get initial panel from URL query param
  const initialPanel = useMemo(() => {
    const panel = searchParams.get("panel")
    if (panel === "schema" || panel === "chat" || panel === "scenarios" || panel === "ontology") {
      // Clear the param after reading it
      return panel as PanelType
    }
    return undefined
  }, [searchParams])

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
  const handleOntologyPageSelect = useCallback(async (path: string) => {
    try {
      const response = await fetch(`/api/v1/ontology/content/${encodeURIComponent(path)}`)
      if (response.ok) {
        const data = await response.json()
        setOntologyPage({
          path: data.path,
          name: data.name,
          content: data.content,
        })
      }
    } catch (error) {
      console.error("Failed to load ontology page:", error)
    }
  }, [])

  /**
   * Close ontology page view.
   */
  const handleCloseOntology = useCallback(() => {
    setOntologyPage(null)
  }, [])

  return (
    <div className="flex h-screen bg-zinc-50">
      {/* Sidebar with icon rail and panels */}
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        isLoading={isLoading}
        initialPanel={initialPanel}
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
        {ontologyPage ? (
          /* Ontology page view */
          <div className="flex flex-col h-full bg-white">
            <div className="flex items-center justify-between px-6 py-3 border-b border-zinc-200">
              <h1 className="text-lg font-semibold text-zinc-800">{ontologyPage.name}</h1>
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleCloseOntology}
                className="h-8 w-8 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <ScrollArea className="flex-1">
              <div className="max-w-3xl mx-auto px-6 py-8">
                <article className="prose prose-zinc max-w-none">
                  <ReactMarkdown>{ontologyPage.content}</ReactMarkdown>
                </article>
              </div>
            </ScrollArea>
          </div>
        ) : (
          <ChatView
            sessionId={currentSessionId || undefined}
            onSessionChange={handleSessionChange}
          />
        )}
      </main>
    </div>
  )
}
