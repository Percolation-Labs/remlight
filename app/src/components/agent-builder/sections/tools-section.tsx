/**
 * ToolsSection - Tools configuration section for agent builder
 *
 * Lists configured tools and allows adding/removing via REM search.
 */

import { useState } from "react"
import { Wrench, Plus, X, Search, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { ToolReference, FocusSection } from "@/types/agent-schema"

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
  const [isSearching, setIsSearching] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [searchResults, setSearchResults] = useState<ToolReference[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsLoading(true)
    try {
      // TODO: Call REM search API
      const response = await fetch(`/api/tools?search=${encodeURIComponent(searchQuery)}`)
      if (response.ok) {
        const data = await response.json()
        setSearchResults(data.tools || [])
      }
    } catch (error) {
      console.error("Failed to search tools:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleAddTool = (tool: ToolReference) => {
    onAddTool(tool)
    setSearchResults((prev) => prev.filter((t) => t.name !== tool.name))
  }

  return (
    <div
      className={cn(
        "rounded-lg border transition-all duration-200",
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
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={() => setIsSearching(!isSearching)}
          className="h-7 gap-1 text-xs"
        >
          <Plus className="h-3.5 w-3.5" />
          Add
        </Button>
      </div>

      {/* Focus message */}
      {isFocused && focusMessage && (
        <div className="px-4 py-2 bg-blue-50 text-xs text-blue-600 border-b border-blue-100">
          {focusMessage}
        </div>
      )}

      {/* Search panel */}
      {isSearching && (
        <div className="p-3 border-b border-zinc-100 bg-zinc-50">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="Search available tools..."
                className="w-full h-8 pl-8 pr-3 text-xs bg-white border border-zinc-200 rounded-md focus:outline-none focus:ring-1 focus:ring-zinc-400"
              />
            </div>
            <Button type="button" size="sm" onClick={handleSearch} disabled={isLoading}>
              {isLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Search"}
            </Button>
          </div>

          {/* Search results */}
          {searchResults.length > 0 && (
            <div className="mt-2 space-y-1">
              {searchResults.map((tool) => (
                <button
                  key={tool.name}
                  onClick={() => handleAddTool(tool)}
                  className="w-full flex items-center gap-2 p-2 text-left rounded-md hover:bg-white transition-colors"
                >
                  <Plus className="h-3.5 w-3.5 text-zinc-400" />
                  <span className="text-sm font-medium text-zinc-700">{tool.name}</span>
                  {tool.description && (
                    <span className="text-xs text-zinc-400 truncate">{tool.description}</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Tools list */}
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
              >
                <div className="h-7 w-7 rounded-md bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                  <Wrench className="h-3.5 w-3.5 text-white" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-zinc-800">{tool.name}</div>
                  {tool.description && (
                    <div className="text-[10px] text-zinc-500 truncate">{tool.description}</div>
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
