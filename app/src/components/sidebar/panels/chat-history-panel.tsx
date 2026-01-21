/**
 * ChatHistoryPanel - Chat session history panel
 *
 * Displays list of chat sessions with search and new chat button.
 */

import { MessageSquare, Plus } from "lucide-react"
import { PanelWrapper } from "../panel-wrapper"
import { SessionList } from "../session-list"
import type { Session } from "@/types/chat"

interface ChatHistoryPanelProps {
  sessions: Session[]
  currentSessionId: string | null
  isLoading?: boolean
  onSessionSelect: (sessionId: string) => void
  onNewChat: () => void
  onSearch?: (query: string) => void
  onClose: () => void
}

export function ChatHistoryPanel({
  sessions,
  currentSessionId,
  isLoading,
  onSessionSelect,
  onNewChat,
  onSearch,
  onClose,
}: ChatHistoryPanelProps) {
  return (
    <PanelWrapper
      title="Chat History"
      icon={<MessageSquare className="h-4 w-4" />}
      onClose={onClose}
      width="wide"
    >
      <div className="flex flex-col h-full">
        {/* New Chat button */}
        <div className="p-3 border-b border-zinc-100">
          <button
            onClick={onNewChat}
            className="w-full flex items-center justify-center gap-2 p-3 rounded-lg border border-dashed border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50 transition-colors"
          >
            <Plus className="h-5 w-5 text-zinc-400" />
            <span className="text-xs text-zinc-600">Start New Chat</span>
          </button>
        </div>

        {/* Session list */}
        <SessionList
          sessions={sessions}
          currentSessionId={currentSessionId}
          isLoading={isLoading}
          onSessionSelect={onSessionSelect}
          onSearch={onSearch}
        />
      </div>
    </PanelWrapper>
  )
}
