/**
 * ToolCallsDisplay - Renders tool calls with nested sub-agent calls
 *
 * Displays tool calls in a tree structure where child agent calls
 * (e.g., from ask_agent) are nested under their parent tool call.
 */

import type { ToolCall } from "@/types/chat"
import { ToolCallCard } from "./tool-call-card"

interface ToolCallsDisplayProps {
  toolCalls: ToolCall[]
}

export function ToolCallsDisplay({ toolCalls }: ToolCallsDisplayProps) {
  if (toolCalls.length === 0) return null

  // Separate top-level and sub-agent tool calls
  const topLevelTools = toolCalls.filter((t) => !t.parentId)
  const subAgentTools = toolCalls.filter((t) => t.parentId)

  // Group sub-agent tools by parent ID
  const subAgentsByParent = subAgentTools.reduce(
    (acc, tool) => {
      const parentId = tool.parentId!
      if (!acc[parentId]) acc[parentId] = []
      acc[parentId].push(tool)
      return acc
    },
    {} as Record<string, ToolCall[]>
  )

  return (
    <div className="w-full space-y-1">
      {topLevelTools.map((tool) => (
        <div key={tool.id}>
          <ToolCallCard toolCall={tool} />
          {/* Render sub-agent tool calls nested under their parent */}
          {subAgentsByParent[tool.id] && subAgentsByParent[tool.id].length > 0 && (
            <div className="ml-4 pl-2 border-l-2 border-gray-200 space-y-1 mt-1">
              {subAgentsByParent[tool.id].map((subTool) => (
                <ToolCallCard
                  key={subTool.id}
                  toolCall={subTool}
                  agentName={subTool.agentName}
                />
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
