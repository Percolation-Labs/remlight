/**
 * ToolCallCard - Expandable card displaying tool call details
 *
 * Shows tool name, status, and expandable sections for arguments and results.
 * Uses subtle color accents based on state (in_progress, completed, failed).
 */

import { useState } from "react"
import { ChevronRight, Wrench, CheckCircle, XCircle, Loader2 } from "lucide-react"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import type { ToolCall } from "@/types/chat"
import { cn } from "@/lib/utils"

interface ToolCallCardProps {
  toolCall: ToolCall
  agentName?: string
}

/**
 * Get status icon based on tool call state.
 */
function StatusIcon({ state }: { state: ToolCall["state"] }) {
  switch (state) {
    case "completed":
      return <CheckCircle className="h-4 w-4 text-green-500 shrink-0" />
    case "failed":
      return <XCircle className="h-4 w-4 text-red-500 shrink-0" />
    case "in_progress":
    default:
      return <Loader2 className="h-4 w-4 text-blue-500 animate-spin shrink-0" />
  }
}

/**
 * Get background color classes based on state.
 */
function getStateStyles(state: ToolCall["state"]) {
  switch (state) {
    case "completed":
      return "bg-green-50 border-green-200"
    case "failed":
      return "bg-red-50 border-red-200"
    case "in_progress":
    default:
      return "bg-blue-50 border-blue-200"
  }
}

/**
 * Format result for display - handles JSON parsing for pretty printing.
 */
function formatResult(r: unknown): string {
  if (r === undefined || r === null) return ""
  if (typeof r === "string") {
    try {
      const parsed = JSON.parse(r)
      return JSON.stringify(parsed, null, 2)
    } catch {
      return r
    }
  }
  return JSON.stringify(r, null, 2)
}

export function ToolCallCard({ toolCall, agentName }: ToolCallCardProps) {
  const [isOpen, setIsOpen] = useState(false)
  const hasContent = Object.keys(toolCall.args || {}).length > 0 || toolCall.output !== undefined
  const argCount = Object.keys(toolCall.args || {}).length

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <div
        className={cn(
          "border rounded-lg mb-2",
          getStateStyles(toolCall.state)
        )}
        data-testid="tool-call-card"
      >
        <CollapsibleTrigger asChild>
          <button
            className="flex items-center gap-2 w-full px-3 py-2 text-sm text-left hover:bg-black/5 rounded-lg transition-colors"
            disabled={!hasContent}
          >
            {hasContent ? (
              <ChevronRight
                className={cn(
                  "h-3 w-3 text-gray-500 transition-transform shrink-0",
                  isOpen && "rotate-90"
                )}
              />
            ) : (
              <div className="w-3" />
            )}
            <StatusIcon state={toolCall.state} />
            <Wrench className="h-3.5 w-3.5 text-gray-600 shrink-0" />
            {agentName && (
              <span className="text-xs px-1.5 py-0.5 bg-purple-100 text-purple-700 rounded font-mono shrink-0">
                {agentName}
              </span>
            )}
            <span className="font-medium text-gray-800 truncate">{toolCall.name}</span>
            {argCount > 0 && !isOpen && (
              <span className="text-xs text-gray-500 truncate ml-auto">
                {argCount} arg{argCount !== 1 ? "s" : ""}
              </span>
            )}
          </button>
        </CollapsibleTrigger>

        {hasContent && (
          <CollapsibleContent className="px-3 pb-3">
            {/* Arguments section */}
            {Object.keys(toolCall.args || {}).length > 0 && (
              <div className="mt-1">
                <div className="text-xs font-medium text-gray-600 mb-1">Arguments:</div>
                <pre className="text-xs bg-white/80 rounded p-2 overflow-x-auto text-gray-700 border border-gray-200 whitespace-pre-wrap break-words">
                  {JSON.stringify(toolCall.args, null, 2)}
                </pre>
              </div>
            )}

            {/* Result section */}
            {toolCall.output !== undefined && (
              <div className="mt-2">
                <div className="text-xs font-medium text-gray-600 mb-1">Result:</div>
                <pre className="text-xs bg-white/80 rounded p-2 overflow-x-auto text-gray-700 border border-gray-200 whitespace-pre-wrap break-words">
                  {formatResult(toolCall.output)}
                </pre>
              </div>
            )}

            {/* Error section */}
            {toolCall.error && (
              <div className="mt-2">
                <div className="text-xs font-medium text-red-600 mb-1">Error:</div>
                <pre className="text-xs bg-red-50/50 rounded p-2 overflow-x-auto text-red-700 border border-red-200 whitespace-pre-wrap break-words">
                  {toolCall.error}
                </pre>
              </div>
            )}
          </CollapsibleContent>
        )}
      </div>
    </Collapsible>
  )
}
