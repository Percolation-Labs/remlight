/**
 * ChatToolbar - Agent and model selection toolbar
 *
 * Provides dropdowns for selecting agent schema and model.
 */

import { useEffect, useState } from "react"
import { Bot, Cpu } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { fetchAgents } from "@/api/agents"
import { fetchModels } from "@/api/models"
import type { Agent, Model } from "@/types/chat"

interface ChatToolbarProps {
  selectedAgent?: string
  selectedModel?: string
  onAgentChange: (agent: string) => void
  onModelChange: (model: string) => void
}

export function ChatToolbar({
  selectedAgent,
  selectedModel,
  onAgentChange,
  onModelChange,
}: ChatToolbarProps) {
  const [agents, setAgents] = useState<Agent[]>([])
  const [models, setModels] = useState<Model[]>([])
  const [isLoadingAgents, setIsLoadingAgents] = useState(true)
  const [isLoadingModels, setIsLoadingModels] = useState(true)

  /**
   * Fetch agents and models on mount.
   */
  useEffect(() => {
    async function loadAgents() {
      setIsLoadingAgents(true)
      try {
        const result = await fetchAgents()
        setAgents(result)
        // Validate the current selection exists, or set default
        if (result.length > 0) {
          const currentExists = selectedAgent && result.some((a) => a.name === selectedAgent)
          if (!currentExists) {
            const orchestrator = result.find((a) => a.name === "orchestrator-agent")
            onAgentChange(orchestrator ? orchestrator.name : result[0].name)
          }
        }
      } catch (error) {
        console.error("Failed to load agents:", error)
      } finally {
        setIsLoadingAgents(false)
      }
    }

    async function loadModels() {
      setIsLoadingModels(true)
      try {
        const result = await fetchModels()
        setModels(result)
        // Set default if none selected
        if (!selectedModel && result.length > 0) {
          onModelChange(result[0].id)
        }
      } catch (error) {
        console.error("Failed to load models:", error)
      } finally {
        setIsLoadingModels(false)
      }
    }

    loadAgents()
    loadModels()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="flex items-center justify-between border-b border-zinc-200 bg-white px-4 py-2">
      <div className="flex items-center gap-3">
        {/* Agent selector */}
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-zinc-500" />
          <Select
            value={selectedAgent}
            onValueChange={onAgentChange}
            disabled={isLoadingAgents}
          >
            <SelectTrigger className="h-8 w-40 text-xs">
              <SelectValue placeholder="Select agent" />
            </SelectTrigger>
            <SelectContent>
              {agents.map((agent) => (
                <SelectItem
                  key={agent.name}
                  value={agent.name}
                  title={agent.description || agent.title || agent.name}
                >
                  {agent.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Model selector */}
        <div className="flex items-center gap-2">
          <Cpu className="h-4 w-4 text-zinc-500" />
          <Select
            value={selectedModel}
            onValueChange={onModelChange}
            disabled={isLoadingModels}
          >
            <SelectTrigger className="h-8 w-48 text-xs">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {models.map((model) => (
                <SelectItem key={model.id} value={model.id}>
                  {model.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {/* Empty div to maintain flex spacing */}
      <div />
    </div>
  )
}
