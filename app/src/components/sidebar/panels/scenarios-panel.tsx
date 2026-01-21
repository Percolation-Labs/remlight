/**
 * ScenariosPanel - Scenario search and browsing panel
 *
 * Search scenarios by name, tags, and semantic search.
 */

import { useState } from "react"
import { Layers, Search, Tag, Sparkles, Loader2 } from "lucide-react"
import { ScrollArea } from "@/components/ui/scroll-area"
import { PanelWrapper } from "../panel-wrapper"
import { cn } from "@/lib/utils"

type SearchMode = "name" | "tag" | "semantic"

interface Scenario {
  id: string
  name: string
  description?: string
  tags: string[]
  updatedAt: string
}

interface ScenariosPanelProps {
  onClose: () => void
  onScenarioSelect?: (scenarioId: string) => void
}

const searchModes: { id: SearchMode; icon: typeof Search; label: string }[] = [
  { id: "name", icon: Search, label: "Name" },
  { id: "tag", icon: Tag, label: "Tag" },
  { id: "semantic", icon: Sparkles, label: "Semantic" },
]

export function ScenariosPanel({ onClose, onScenarioSelect }: ScenariosPanelProps) {
  const [searchMode, setSearchMode] = useState<SearchMode>("name")
  const [searchQuery, setSearchQuery] = useState("")
  const [scenarios, setScenarios] = useState<Scenario[]>([])
  const [isLoading, setIsLoading] = useState(false)

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setIsLoading(true)
    try {
      // TODO: Implement API call based on search mode
      // For now, just a placeholder
      const response = await fetch(
        `/api/scenarios/search?mode=${searchMode}&q=${encodeURIComponent(searchQuery)}`
      )
      if (response.ok) {
        const data = await response.json()
        setScenarios(data.scenarios || [])
      }
    } catch (error) {
      console.error("Failed to search scenarios:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const ActiveModeIcon = searchModes.find((m) => m.id === searchMode)?.icon || Search

  return (
    <PanelWrapper
      title="Scenarios"
      icon={<Layers className="h-4 w-4" />}
      onClose={onClose}
    >
      <div className="flex flex-col h-full">
        {/* Search mode tabs */}
        <div className="px-3 pt-3">
          <div className="flex gap-1 p-1 bg-zinc-100 rounded-lg">
            {searchModes.map((mode) => {
              const Icon = mode.icon
              return (
                <button
                  key={mode.id}
                  onClick={() => setSearchMode(mode.id)}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs rounded-md transition-colors",
                    searchMode === mode.id
                      ? "bg-white text-zinc-900 shadow-sm"
                      : "text-zinc-500 hover:text-zinc-700"
                  )}
                >
                  <Icon className="h-3 w-3" />
                  {mode.label}
                </button>
              )
            })}
          </div>
        </div>

        {/* Search input */}
        <div className="p-3">
          <div className="relative">
            <ActiveModeIcon className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder={
                searchMode === "name"
                  ? "Search by name..."
                  : searchMode === "tag"
                  ? "Search by tags..."
                  : "Describe what you're looking for..."
              }
              className="w-full h-8 pl-8 pr-3 text-xs bg-zinc-50 border border-zinc-200 rounded-md focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:border-zinc-400"
            />
          </div>
          {searchMode === "semantic" && (
            <p className="mt-1.5 text-[10px] text-zinc-400 leading-tight">
              Uses AI to find semantically similar scenarios
            </p>
          )}
        </div>

        {/* Results */}
        <ScrollArea className="flex-1">
          <div className="px-2 pb-2 space-y-1">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
              </div>
            ) : scenarios.length === 0 ? (
              <div className="text-center py-8 text-xs text-zinc-400">
                {searchQuery ? "No scenarios found" : "Search for scenarios"}
              </div>
            ) : (
              scenarios.map((scenario) => (
                <button
                  key={scenario.id}
                  onClick={() => onScenarioSelect?.(scenario.id)}
                  className="w-full text-left p-2.5 rounded-lg hover:bg-zinc-50 transition-colors group"
                >
                  <div className="text-sm font-medium text-zinc-800 truncate">
                    {scenario.name}
                  </div>
                  {scenario.description && (
                    <div className="text-xs text-zinc-500 truncate mt-0.5">
                      {scenario.description}
                    </div>
                  )}
                  {scenario.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1.5">
                      {scenario.tags.slice(0, 3).map((tag) => (
                        <span
                          key={tag}
                          className="px-1.5 py-0.5 text-[10px] bg-zinc-100 text-zinc-600 rounded"
                        >
                          {tag}
                        </span>
                      ))}
                      {scenario.tags.length > 3 && (
                        <span className="text-[10px] text-zinc-400">
                          +{scenario.tags.length - 3}
                        </span>
                      )}
                    </div>
                  )}
                </button>
              ))
            )}
          </div>
        </ScrollArea>
      </div>
    </PanelWrapper>
  )
}
