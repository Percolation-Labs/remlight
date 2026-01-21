/**
 * ChatHistoryPanel - Chat session history panel
 *
 * Displays list of chat sessions with search and new chat button.
 */

import { MessageSquare, Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
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
      actions={
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
      }
    >
      <SessionList
        sessions={sessions}
        currentSessionId={currentSessionId}
        isLoading={isLoading}
        onSessionSelect={onSessionSelect}
        onSearch={onSearch}
      />
    </PanelWrapper>
  )
}
