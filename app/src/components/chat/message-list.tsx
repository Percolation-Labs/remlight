/**
 * MessageList - Scrollable message list with auto-scroll
 *
 * Renders messages and handles auto-scrolling on new messages.
 */

import { useEffect, useRef } from "react"
import { Message } from "./message"
import type { Message as MessageType } from "@/types/chat"

interface MessageListProps {
  messages: MessageType[]
  showFeedback?: boolean
  sessionId?: string
  onExport?: () => void
  onAddScenario?: () => void
}

export function MessageList({
  messages,
  showFeedback = true,
  sessionId,
  onExport,
  onAddScenario,
}: MessageListProps) {
  // Find the last completed assistant message for showing actions
  let lastAssistantIndex = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant" && messages[i].status === "completed") {
      lastAssistantIndex = i
      break
    }
  }
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  /**
   * Auto-scroll to bottom when new messages arrive or content updates.
   */
  useEffect(() => {
    const lastMessage = messages[messages.length - 1]

    // Always scroll on new message or streaming update
    const shouldScroll =
      lastMessage?.status === "streaming" || lastMessage?.status === "pending"

    if (shouldScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages])

  if (messages.length === 0) {
    return null
  }

  return (
    <div ref={scrollRef} className="h-full overflow-y-auto bg-white">
      <div className="max-w-3xl mx-auto py-6">
        {messages.map((message, index) => {
          const isLastAssistant = index === lastAssistantIndex
          return (
            <div key={message.id}>
              {index > 0 && (
                <div className="border-b border-zinc-100 mx-4" />
              )}
              <Message
                message={message}
                showFeedback={showFeedback}
                showActions={isLastAssistant && !!sessionId}
                onExport={isLastAssistant ? onExport : undefined}
                onAddScenario={isLastAssistant ? onAddScenario : undefined}
              />
            </div>
          )
        })}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
