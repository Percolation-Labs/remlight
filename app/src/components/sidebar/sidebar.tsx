/**
 * Sidebar - Main sidebar container
 *
 * Contains icon rail navigation and expandable panels.
 */

import { useState, useEffect } from "react"
import { NavRail, type PanelType } from "./nav-rail"
import {
  ChatHistoryPanel,
  ScenariosPanel,
  SchemaBuilderPanel,
  OntologyPanel,
} from "./panels"
import type { Session } from "@/types/chat"

interface SidebarProps {
  sessions: Session[]
  currentSessionId: string | null
  isLoading?: boolean
  initialPanel?: PanelType | null
  onSessionSelect: (sessionId: string) => void
  onNewChat: () => void
  onSearch?: (query: string) => void
  onSettings?: () => void
  onScenarioSelect?: (scenarioId: string) => void
  onSchemaSelect?: (schemaId: string) => void
  onNewSchema?: () => void
  onOntologyPageSelect?: (path: string) => void
}

export function Sidebar({
  sessions,
  currentSessionId,
  isLoading,
  initialPanel,
  onSessionSelect,
  onNewChat,
  onSearch,
  onSettings,
  onScenarioSelect,
  onSchemaSelect,
  onNewSchema,
  onOntologyPageSelect,
}: SidebarProps) {
  const [activePanel, setActivePanel] = useState<PanelType | null>(initialPanel ?? "chat")

  // Update panel when initialPanel changes (e.g., from URL param)
  useEffect(() => {
    if (initialPanel !== undefined) {
      setActivePanel(initialPanel)
    }
  }, [initialPanel])

  const handlePanelSelect = (panel: PanelType) => {
    // Toggle panel if clicking the same one
    setActivePanel((current) => (current === panel ? null : panel))
  }

  const handleClosePanel = () => {
    setActivePanel(null)
  }

  return (
    <div className="flex h-full">
      {/* Icon rail - always visible */}
      <NavRail
        activePanel={activePanel}
        onPanelSelect={handlePanelSelect}
        onSettings={onSettings}
      />

      {/* Active panel */}
      {activePanel === "chat" && (
        <ChatHistoryPanel
          sessions={sessions}
          currentSessionId={currentSessionId}
          isLoading={isLoading}
          onSessionSelect={onSessionSelect}
          onNewChat={onNewChat}
          onSearch={onSearch}
          onClose={handleClosePanel}
        />
      )}

      {activePanel === "scenarios" && (
        <ScenariosPanel
          onClose={handleClosePanel}
          onScenarioSelect={onScenarioSelect}
        />
      )}

      {activePanel === "schema" && (
        <SchemaBuilderPanel
          onClose={handleClosePanel}
          onSchemaSelect={onSchemaSelect}
          onNewSchema={onNewSchema}
        />
      )}

      {activePanel === "ontology" && (
        <OntologyPanel
          onClose={handleClosePanel}
          onPageSelect={onOntologyPageSelect}
        />
      )}
    </div>
  )
}
