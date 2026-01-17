/**
 * Sidebar - Main sidebar container
 *
 * Contains session history, new chat button, and navigation.
 */

import { Plus, Settings, PanelLeftClose, PanelLeft } from "lucide-react"
import { Button } from "@/components/ui/button"
import { SessionList } from "./session-list"
import type { Session } from "@/types/chat"

interface SidebarProps {
  sessions: Session[]
  currentSessionId: string | null
  isLoading?: boolean
  isCollapsed?: boolean
  onSessionSelect: (sessionId: string) => void
  onNewChat: () => void
  onSearch?: (query: string) => void
  onSettings?: () => void
  onToggleCollapse?: () => void
}

export function Sidebar({
  sessions,
  currentSessionId,
  isLoading,
  isCollapsed = false,
  onSessionSelect,
  onNewChat,
  onSearch,
  onSettings,
  onToggleCollapse,
}: SidebarProps) {
  if (isCollapsed) {
    return (
      <div className="w-12 border-r border-zinc-200 bg-white flex flex-col items-center py-2 gap-2">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onToggleCollapse}
          className="h-8 w-8 p-0"
          title="Expand sidebar"
        >
          <PanelLeft className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onNewChat}
          className="h-8 w-8 p-0"
          title="New chat"
        >
          <Plus className="h-4 w-4" />
        </Button>
        {onSettings && (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onSettings}
            className="h-8 w-8 p-0 mt-auto"
            title="Settings"
          >
            <Settings className="h-4 w-4" />
          </Button>
        )}
      </div>
    )
  }

  return (
    <div className="w-64 border-r border-zinc-200 bg-white flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-zinc-100">
        <div className="flex items-center gap-1">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onToggleCollapse}
            className="h-7 w-7 p-0"
            title="Collapse sidebar"
          >
            <PanelLeftClose className="h-3.5 w-3.5" />
          </Button>
          <h2 className="text-sm font-medium text-zinc-700">Sessions</h2>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onNewChat}
          className="h-7 gap-1 text-xs"
        >
          <Plus className="h-3.5 w-3.5" />
          New
        </Button>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-hidden">
        <SessionList
          sessions={sessions}
          currentSessionId={currentSessionId}
          isLoading={isLoading}
          onSessionSelect={onSessionSelect}
          onSearch={onSearch}
        />
      </div>

      {/* Footer */}
      {onSettings && (
        <div className="border-t border-zinc-100 p-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onSettings}
            className="w-full justify-start gap-2 text-xs text-zinc-600"
          >
            <Settings className="h-3.5 w-3.5" />
            Settings
          </Button>
        </div>
      )}
    </div>
  )
}
