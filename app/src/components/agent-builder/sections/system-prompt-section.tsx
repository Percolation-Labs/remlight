/**
 * SystemPromptSection - Editable system prompt section for agent builder
 *
 * Displays and allows editing of the agent's system prompt (description).
 */

import { useState, useRef, useEffect } from "react"
import { FileText, Check, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface SystemPromptSectionProps {
  description: string
  isFocused?: boolean
  focusMessage?: string
  onChange: (description: string) => void
}

export function SystemPromptSection({
  description,
  isFocused,
  focusMessage,
  onChange,
}: SystemPromptSectionProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [editValue, setEditValue] = useState(description)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Sync edit value when description changes externally
  useEffect(() => {
    if (!isEditing) {
      setEditValue(description)
    }
  }, [description, isEditing])

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current && isEditing) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 400)}px`
    }
  }, [editValue, isEditing])

  // Focus textarea when entering edit mode
  useEffect(() => {
    if (isEditing && textareaRef.current) {
      textareaRef.current.focus()
      textareaRef.current.setSelectionRange(
        textareaRef.current.value.length,
        textareaRef.current.value.length
      )
    }
  }, [isEditing])

  const handleSave = () => {
    onChange(editValue)
    setIsEditing(false)
  }

  const handleCancel = () => {
    setEditValue(description)
    setIsEditing(false)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      handleCancel()
    } else if (e.key === "Enter" && e.metaKey) {
      handleSave()
    }
  }

  return (
    <div
      className={cn(
        "rounded-lg border transition-all duration-200",
        isFocused ? "border-blue-300 ring-1 ring-blue-100" : "border-zinc-200",
        isEditing && "border-amber-300 ring-1 ring-amber-100",
        "bg-white"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-100">
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-zinc-500" />
          <h3 className="text-sm font-medium text-zinc-800">System Prompt</h3>
        </div>
        {!isEditing ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setIsEditing(true)}
            className="h-7 text-xs"
          >
            Edit
          </Button>
        ) : (
          <div className="flex items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleCancel}
              className="h-7 w-7 p-0"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={handleSave}
              className="h-7 w-7 p-0 text-green-600"
            >
              <Check className="h-3.5 w-3.5" />
            </Button>
          </div>
        )}
      </div>

      {/* Focus message */}
      {isFocused && focusMessage && (
        <div className="px-4 py-2 bg-blue-50 text-xs text-blue-600 border-b border-blue-100">
          {focusMessage}
        </div>
      )}

      {/* Content */}
      <div className="p-4">
        {isEditing ? (
          <div>
            <textarea
              ref={textareaRef}
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter the system prompt for your agent..."
              className="w-full min-h-[120px] p-3 text-sm bg-zinc-50 border border-zinc-200 rounded-md resize-none focus:outline-none focus:ring-1 focus:ring-amber-400 focus:border-amber-400"
            />
            <div className="mt-2 text-[10px] text-zinc-400">
              Press <kbd className="px-1 py-0.5 bg-zinc-100 rounded">⌘ Enter</kbd> to save or{" "}
              <kbd className="px-1 py-0.5 bg-zinc-100 rounded">Esc</kbd> to cancel
            </div>
          </div>
        ) : description ? (
          <div
            className="text-sm text-zinc-700 whitespace-pre-wrap cursor-pointer hover:bg-zinc-50 p-2 -m-2 rounded-md transition-colors"
            onClick={() => setIsEditing(true)}
          >
            {description}
          </div>
        ) : (
          <div
            className="text-sm text-zinc-400 italic cursor-pointer hover:bg-zinc-50 p-2 -m-2 rounded-md transition-colors"
            onClick={() => setIsEditing(true)}
          >
            Click to add a system prompt...
          </div>
        )}
      </div>
    </div>
  )
}
