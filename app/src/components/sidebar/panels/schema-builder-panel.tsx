/**
 * SchemaBuilderPanel - Agent schema builder panel
 *
 * Design and edit agent schemas with visual tools.
 */

import { useState, useEffect } from "react"
import { useNavigate } from "react-router-dom"
import { Boxes, Plus, FileJson, ChevronRight, Loader2, ExternalLink } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { PanelWrapper } from "../panel-wrapper"
import { cn } from "@/lib/utils"

interface Schema {
  id: string
  name: string
  description?: string
  version: string
  updatedAt: string
}

interface SchemaBuilderPanelProps {
  onClose: () => void
  onSchemaSelect?: (schemaId: string) => void
  onNewSchema?: () => void
}

export function SchemaBuilderPanel({
  onClose,
  onSchemaSelect,
  onNewSchema,
}: SchemaBuilderPanelProps) {
  const navigate = useNavigate()
  const [schemas, setSchemas] = useState<Schema[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Load schemas from API on mount
  useEffect(() => {
    loadSchemas()
  }, [])

  const loadSchemas = async () => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/v1/agents")
      if (response.ok) {
        const data = await response.json()
        setSchemas(
          (data.agents || []).map((agent: { name: string; description?: string; version?: string; updated_at?: string }) => ({
            id: agent.name,
            name: agent.name,
            description: agent.description,
            version: agent.version || "1.0.0",
            updatedAt: agent.updated_at || new Date().toISOString(),
          }))
        )
      }
    } catch (error) {
      console.error("Failed to load schemas:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSelect = (schemaName: string) => {
    setSelectedId(schemaName)
    // Navigate to agent builder with this schema
    navigate(`/agent-builder/${schemaName}`)
  }

  const handleNewSchema = () => {
    navigate("/agent-builder")
  }

  return (
    <PanelWrapper
      title="Schema Builder"
      icon={<Boxes className="h-4 w-4" />}
      onClose={onClose}
      actions={
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={handleNewSchema}
          className="h-7 gap-1 text-xs"
        >
          <Plus className="h-3.5 w-3.5" />
          New
        </Button>
      }
    >
      <div className="flex flex-col h-full">
        {/* Quick actions */}
        <div className="p-3 border-b border-zinc-100">
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={handleNewSchema}
              className="flex flex-col items-center gap-1.5 p-3 rounded-lg border border-dashed border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50 transition-colors"
            >
              <Plus className="h-5 w-5 text-zinc-400" />
              <span className="text-xs text-zinc-600">New Agent</span>
            </button>
            <button
              onClick={() => navigate("/agent-builder")}
              className="flex flex-col items-center gap-1.5 p-3 rounded-lg border border-dashed border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50 transition-colors"
            >
              <ExternalLink className="h-5 w-5 text-zinc-400" />
              <span className="text-xs text-zinc-600">Open Builder</span>
            </button>
          </div>
        </div>

        {/* Schema list */}
        <div className="px-3 py-2">
          <h3 className="text-[10px] font-medium text-zinc-400 uppercase tracking-wider">
            Your Agents
          </h3>
        </div>

        <ScrollArea className="flex-1">
          <div className="px-2 pb-2 space-y-0.5">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
              </div>
            ) : schemas.length === 0 ? (
              <div className="text-center py-8">
                <Boxes className="h-8 w-8 text-zinc-200 mx-auto mb-2" />
                <p className="text-xs text-zinc-400">No agents yet</p>
                <p className="text-[10px] text-zinc-400 mt-1">
                  Create your first agent to get started
                </p>
              </div>
            ) : (
              schemas.map((schema) => (
                <button
                  key={schema.id}
                  onClick={() => handleSelect(schema.name)}
                  className={cn(
                    "w-full flex items-center gap-2 p-2.5 rounded-lg transition-colors group",
                    selectedId === schema.id
                      ? "bg-zinc-100"
                      : "hover:bg-zinc-50"
                  )}
                >
                  <div className="h-8 w-8 rounded-md bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center flex-shrink-0">
                    <Boxes className="h-4 w-4 text-white" />
                  </div>
                  <div className="flex-1 text-left min-w-0">
                    <div className="text-sm font-medium text-zinc-800 truncate">
                      {schema.name}
                    </div>
                    <div className="text-[10px] text-zinc-400">
                      v{schema.version}
                    </div>
                  </div>
                  <ChevronRight className="h-4 w-4 text-zinc-300 group-hover:text-zinc-400" />
                </button>
              ))
            )}
          </div>
        </ScrollArea>
      </div>
    </PanelWrapper>
  )
}
