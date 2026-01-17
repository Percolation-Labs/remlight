/**
 * ChatInput - Text input with send button
 *
 * Handles message composition with Shift+Enter for newlines.
 */

import { useState, useRef, useEffect, type KeyboardEvent } from "react"
import { Send, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ChatInputProps {
  onSend: (message: string) => void
  onStop?: () => void
  isLoading?: boolean
  disabled?: boolean
  placeholder?: string
}

export function ChatInput({
  onSend,
  onStop,
  isLoading = false,
  disabled = false,
  placeholder = "Type a message...",
}: ChatInputProps) {
  const [value, setValue] = useState("")
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  /**
   * Auto-resize textarea based on content.
   */
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [value])

  /**
   * Focus textarea on mount.
   */
  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  const handleSend = () => {
    if (value.trim() && !isLoading && !disabled) {
      onSend(value.trim())
      setValue("")
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="border-t border-zinc-200 bg-white p-4">
      <div className="max-w-3xl mx-auto">
        <div className="flex items-end gap-2 bg-zinc-50 border border-zinc-200 rounded-lg p-2 shadow-chat focus-within:shadow-chat-hover focus-within:border-zinc-300 transition-all">
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isLoading}
            rows={1}
            className={cn(
              "flex-1 bg-transparent resize-none text-sm text-zinc-800",
              "placeholder:text-zinc-400 focus:outline-none",
              "min-h-[24px] max-h-[200px]",
              (disabled || isLoading) && "opacity-50 cursor-not-allowed"
            )}
          />

          {isLoading ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={onStop}
              className="h-8 w-8 p-0 text-zinc-500 hover:text-red-600"
            >
              <Square className="h-4 w-4 fill-current" />
            </Button>
          ) : (
            <Button
              variant="ghost"
              size="sm"
              onClick={handleSend}
              disabled={!value.trim() || disabled}
              className={cn(
                "h-8 w-8 p-0",
                value.trim()
                  ? "text-zinc-700 hover:text-zinc-900"
                  : "text-zinc-300"
              )}
            >
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>

        <div className="text-xs text-zinc-400 mt-1 text-center">
          Press Enter to send, Shift+Enter for new line
        </div>
      </div>
    </div>
  )
}
