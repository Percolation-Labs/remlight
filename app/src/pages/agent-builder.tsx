/**
 * AgentBuilderPage - Page component for the agent builder
 *
 * Provides the full-page agent builder experience.
 */

import { useParams, useNavigate } from "react-router-dom"
import { ArrowLeft, Save } from "lucide-react"
import { Button } from "@/components/ui/button"
import { AgentBuilderView } from "@/components/agent-builder"

export function AgentBuilderPage() {
  const { agentName } = useParams<{ agentName?: string }>()
  const navigate = useNavigate()

  const handleBack = () => {
    navigate("/")
  }

  return (
    <div className="flex flex-col h-screen bg-zinc-50">
      {/* Top bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-200 bg-white">
        <div className="flex items-center gap-3">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={handleBack}
            className="gap-1"
          >
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
          <div className="h-4 w-px bg-zinc-200" />
          <h1 className="text-sm font-semibold text-zinc-800">
            {agentName ? `Edit: ${agentName}` : "Create New Agent"}
          </h1>
        </div>
        <div className="flex items-center gap-2">
          <Button type="button" variant="outline" size="sm">
            Preview
          </Button>
          <Button type="button" size="sm" className="gap-1">
            <Save className="h-3.5 w-3.5" />
            Save Agent
          </Button>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        <AgentBuilderView initialAgentName={agentName} />
      </div>
    </div>
  )
}
