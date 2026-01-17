/**
 * SessionItem - Individual session in the sidebar list
 *
 * Displays session name, timestamp, and preview with active state.
 */

import { MessageSquare } from "lucide-react"
import type { Session } from "@/types/chat"
import { formatTimeAgo, truncate, cn } from "@/lib/utils"

interface SessionItemProps {
  session: Session
  isActive?: boolean
  onClick: () => void
}

export function SessionItem({ session, isActive, onClick }: SessionItemProps) {
  const displayName = session.name || session.firstMessage || "New conversation"

  return (
    <button
      onClick={onClick}
      data-testid="session-item"
      className={cn(
        "w-full text-left px-3 py-2 rounded-md transition-colors",
        "hover:bg-zinc-100",
        isActive ? "bg-zinc-100" : "bg-transparent"
      )}
    >
      <div className="flex items-start gap-2">
        <MessageSquare className="h-4 w-4 text-zinc-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span
              className={cn(
                "text-xs font-medium truncate",
                isActive ? "text-zinc-900" : "text-zinc-700"
              )}
            >
              {truncate(displayName, 30)}
            </span>
            <span className="text-xs text-zinc-400 flex-shrink-0">
              {formatTimeAgo(session.updatedAt || session.createdAt)}
            </span>
          </div>
          {session.messageCount !== undefined && (
            <span className="text-xs text-zinc-400">
              {session.messageCount} messages
            </span>
          )}
        </div>
      </div>
    </button>
  )
}
