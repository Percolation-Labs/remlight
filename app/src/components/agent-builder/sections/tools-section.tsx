/**
 * ToolsSection - Tools configuration section for agent builder
 *
 * Lists configured tools and allows adding/removing from available tools.
 */

import { useState, useEffect, useMemo } from "react"
import { Wrench, Plus, X, Search, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import type { ToolReference } from "@/types/agent-schema"

interface ToolsSectionProps {
  tools: ToolReference[]
  isFocused?: boolean
  focusMessage?: string
  onAddTool: (tool: ToolReference) => void
  onRemoveTool: (toolName: string) => void
}

export function ToolsSection({
  tools,
  isFocused,
  focusMessage,
  onAddTool,
  onRemoveTool,
}: ToolsSectionProps) {
  const [isAdding, setIsAdding] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [availableTools, setAvailableTools] = useState<ToolReference[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Load available tools when opening the add panel
  useEffect(() => {
    if (isAdding && availableTools.length === 0) {
      loadTools()
    }
  }, [isAdding])

  const loadTools = async () => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/v1/tools")
      if (response.ok) {
        const data = await response.json()
        setAvailableTools(
          (data.tools || []).map((t: { name: string; description?: string }) => ({
            name: t.name,
            description: t.description,
          }))
        )
      }
    } catch (error) {
      console.error("Failed to load tools:", error)
    } finally {
      setIsLoading(false)
    }
  }

  // Filter available tools - exclude already added ones and apply search
  const filteredTools = useMemo(() => {
    const addedNames = new Set(tools.map((t) => t.name))
    let filtered = availableTools.filter((t) => !addedNames.has(t.name))

    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(
        (t) =>
          t.name.toLowerCase().includes(query) ||
          t.description?.toLowerCase().includes(query)
      )
    }

    return filtered
  }, [availableTools, tools, searchQuery])

  const handleAddTool = (tool: ToolReference) => {
    onAddTool(tool)
  }

  const handleClose = () => {
    setIsAdding(false)
    setSearchQuery("")
  }

  return (
    <div
      className={cn(
        "rounded-lg border transition-all duration-200 overflow-hidden",
        isFocused ? "border-blue-300 ring-1 ring-blue-100" : "border-zinc-200",
        "bg-white"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-100">
        <div className="flex items-center gap-2">
          <Wrench className="h-4 w-4 text-zinc-500" />
          <h3 className="text-sm font-medium text-zinc-800">Tools</h3>
          <span className="text-xs text-zinc-400">({tools.length})</span>
        </div>
        {!isAdding ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setIsAdding(true)}
            className="h-7 gap-1 text-xs"
          >
            <Plus className="h-3.5 w-3.5" />
            Add
          </Button>
        ) : (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handleClose}
            className="h-7 gap-1 text-xs text-zinc-500"
          >
            <X className="h-3.5 w-3.5" />
            Cancel
          </Button>
        )}
      </div>

      {/* Focus message */}
      {isFocused && focusMessage && (
        <div className="px-4 py-2 bg-blue-50 text-xs text-blue-600 border-b border-blue-100">
          {focusMessage}
        </div>
      )}

      {/* Add tools panel */}
      {isAdding && (
        <div className="border-b border-zinc-100 bg-zinc-50">
          {/* Search input */}
          <div className="p-3 border-b border-zinc-100">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter tools..."
                className="w-full h-8 pl-8 pr-3 text-xs bg-white border border-zinc-200 rounded-md focus:outline-none focus:ring-1 focus:ring-zinc-400"
                autoFocus
              />
            </div>
          </div>

          {/* Available tools list */}
          <ScrollArea className="max-h-[200px]">
            <div className="p-2">
              {isLoading ? (
                <div className="flex items-center justify-center py-6">
                  <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
                </div>
              ) : filteredTools.length === 0 ? (
                <div className="text-center py-4 text-xs text-zinc-400">
                  {availableTools.length === 0
                    ? "No tools registered. Register tools via the API."
                    : searchQuery
                    ? "No matching tools found"
                    : "All available tools have been added"}
                </div>
              ) : (
                <div className="space-y-1">
                  {filteredTools.map((tool) => (
                    <button
                      key={tool.name}
                      onClick={() => handleAddTool(tool)}
                      className="w-full flex items-center gap-2 p-2 text-left rounded-md hover:bg-white transition-colors group"
                    >
                      <div className="h-6 w-6 rounded bg-zinc-200 flex items-center justify-center flex-shrink-0 group-hover:bg-violet-100">
                        <Plus className="h-3 w-3 text-zinc-500 group-hover:text-violet-600" />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium text-zinc-700">{tool.name}</div>
                        {tool.description && (
                          <div className="text-[10px] text-zinc-400 truncate">
                            {tool.description}
                          </div>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      )}

      {/* Configured tools list */}
      <div className="p-3">
        {tools.length === 0 ? (
          <div className="text-center py-4 text-xs text-zinc-400">
            No tools configured. Add tools to give your agent capabilities.
          </div>
        ) : (
          <div className="space-y-1.5">
            {tools.map((tool) => (
              <div
                key={tool.name}
                className="flex items-center gap-2 px-3 py-2 rounded-md bg-zinc-50 group"
                title={tool.description || undefined}
              >
                <div className="h-7 w-7 rounded-md bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                  <Wrench className="h-3.5 w-3.5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-zinc-800">{tool.name}</div>
                  {tool.description && (
                    <div className="text-[10px] text-zinc-500 truncate">
                      {tool.description}
                    </div>
                  )}
                </div>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => onRemoveTool(tool.name)}
                  className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity text-zinc-400 hover:text-red-500"
                >
                  <X className="h-3.5 w-3.5" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
