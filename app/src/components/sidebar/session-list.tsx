/**
 * SessionList - List of chat sessions in sidebar
 *
 * Displays sessions with search functionality.
 */

import { useState } from "react"
import { Search, Loader2 } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { SessionItem } from "./session-item"
import type { Session } from "@/types/chat"

interface SessionListProps {
  sessions: Session[]
  currentSessionId: string | null
  isLoading?: boolean
  onSessionSelect: (sessionId: string) => void
  onSearch?: (query: string) => void
}

export function SessionList({
  sessions,
  currentSessionId,
  isLoading,
  onSessionSelect,
  onSearch,
}: SessionListProps) {
  const [searchQuery, setSearchQuery] = useState("")

  const handleSearchChange = (value: string) => {
    setSearchQuery(value)
    onSearch?.(value)
  }

  return (
    <div className="flex flex-col h-full">
      {/* Search input */}
      <div className="p-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => handleSearchChange(e.target.value)}
            placeholder="Search sessions..."
            className="w-full h-8 pl-8 pr-3 text-xs bg-zinc-50 border border-zinc-200 rounded-md focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:border-zinc-400"
          />
        </div>
      </div>

      {/* Session list */}
      <ScrollArea className="flex-1">
        <div className="px-2 pb-2 space-y-0.5">
          {isLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
            </div>
          ) : sessions.length === 0 ? (
            <div className="text-center py-8 text-xs text-zinc-400">
              {searchQuery ? "No sessions found" : "No sessions yet"}
            </div>
          ) : (
            sessions.map((session) => (
              <SessionItem
                key={session.id}
                session={session}
                isActive={session.id === currentSessionId}
                onClick={() => onSessionSelect(session.id)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
