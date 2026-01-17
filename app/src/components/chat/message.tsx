/**
 * Message - Single chat message component
 *
 * Renders user or assistant messages with content, tool calls, and metadata.
 */

import { User, Bot, Loader2, Download, Plus } from "lucide-react"
import { MessageContent } from "./message-content"
import { ToolCallsDisplay } from "./tool-calls-display"
import { FeedbackButtons } from "./feedback-buttons"
import { Button } from "@/components/ui/button"
import type { Message as MessageType } from "@/types/chat"
import { cn } from "@/lib/utils"

interface MessageProps {
  message: MessageType
  showFeedback?: boolean
  showActions?: boolean
  onExport?: () => void
  onAddScenario?: () => void
}

export function Message({
  message,
  showFeedback = true,
  showActions = false,
  onExport,
  onAddScenario,
}: MessageProps) {
  const isUser = message.role === "user"
  const isStreaming = message.status === "streaming"
  const isPending = message.status === "pending"
  const isFailed = message.status === "failed"

  return (
    <div
      className={cn(
        "flex gap-4 px-4 py-4",
        isUser ? "flex-row-reverse" : "flex-row"
      )}
      data-role={message.role}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center mt-1",
          isUser ? "bg-zinc-600" : "bg-zinc-100"
        )}
      >
        {isUser ? (
          <User className="h-4 w-4 text-white" />
        ) : (
          <Bot className="h-4 w-4 text-zinc-500" />
        )}
      </div>

      {/* Message content */}
      <div
        className={cn(
          "flex-1 space-y-3 min-w-0",
          isUser ? "text-right" : "text-left"
        )}
      >
        {/* Role label */}
        <div className={cn(
          "text-xs font-medium text-zinc-500 mb-1",
          isUser ? "text-right" : "text-left"
        )}>
          {isUser ? "You" : "Assistant"}
        </div>

        <div
          className={cn(
            "inline-block text-sm",
            isUser
              ? "text-zinc-700"
              : "text-zinc-800",
            isFailed && "text-red-600"
          )}
        >
          {isPending ? (
            <div className="flex items-center gap-2 text-zinc-400">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Thinking...</span>
            </div>
          ) : (
            <MessageContent content={message.content} />
          )}

          {isStreaming && (
            <span className="inline-block w-1 h-4 bg-zinc-400 animate-pulse ml-0.5 align-middle" />
          )}
        </div>

        {/* Tool calls with nested sub-agent calls */}
        {message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mt-3">
            <ToolCallsDisplay toolCalls={message.toolCalls} />
          </div>
        )}

        {/* Metadata row */}
        {!isUser && message.status === "completed" && (
          <div className="flex items-center gap-3 text-xs text-zinc-400 mt-2">
            {message.metadata?.agentName && (
              <span>{message.metadata.agentName}</span>
            )}
            {message.metadata?.latencyMs && (
              <span>{(message.metadata.latencyMs / 1000).toFixed(1)}s</span>
            )}
            {message.metadata?.tokenCount && (
              <span>{message.metadata.tokenCount} tokens</span>
            )}
          </div>
        )}

        {/* Feedback and actions row - full width, nicely positioned */}
        {!isUser && message.status === "completed" && (showFeedback || showActions) && (
          <div className="mt-3 pt-3 border-t border-zinc-100">
            <div className="flex items-center justify-between">
              {/* Feedback buttons on the left */}
              <div className="relative flex-1">
                {showFeedback && message.id && (
                  <FeedbackButtons messageId={message.id} />
                )}
              </div>

              {/* Action buttons on the right */}
              {showActions && (
                <div className="flex items-center gap-2">
                  {onExport && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        onExport()
                      }}
                      className="h-7 text-xs"
                    >
                      <Download className="h-3 w-3 mr-1" />
                      Export
                    </Button>
                  )}
                  {onAddScenario && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation()
                        onAddScenario()
                      }}
                      className="h-7 text-xs"
                    >
                      <Plus className="h-3 w-3 mr-1" />
                      Add to Scenario
                    </Button>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
