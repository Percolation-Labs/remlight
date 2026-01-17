/**
 * AddScenarioModal - Modal for adding a session to a scenario
 */

import { useState, useCallback, type KeyboardEvent } from "react"
import { X, Plus, Loader2, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { createScenario } from "@/api/scenarios"

interface AddScenarioModalProps {
  sessionId: string
  agentName?: string
  onClose: () => void
  onSuccess?: (scenarioId: string) => void
}

export function AddScenarioModal({
  sessionId,
  agentName,
  onClose,
  onSuccess,
}: AddScenarioModalProps) {
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [tags, setTags] = useState<string[]>([])
  const [tagInput, setTagInput] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const addTag = useCallback(() => {
    const trimmed = tagInput.trim().toLowerCase()
    if (trimmed && !tags.includes(trimmed)) {
      setTags((prev) => [...prev, trimmed])
      setTagInput("")
    }
  }, [tagInput, tags])

  const removeTag = useCallback((tagToRemove: string) => {
    setTags((prev) => prev.filter((t) => t !== tagToRemove))
  }, [])

  const handleTagKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault()
      addTag()
    } else if (e.key === "Backspace" && !tagInput && tags.length > 0) {
      // Remove last tag on backspace if input is empty
      setTags((prev) => prev.slice(0, -1))
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name.trim()) {
      setError("Name is required")
      return
    }

    setIsSubmitting(true)
    setError(null)

    try {
      const scenario = await createScenario({
        name: name.trim(),
        description: description.trim() || undefined,
        session_id: sessionId,
        agent_name: agentName,
        tags: tags,
      })
      setSuccess(true)
      setTimeout(() => {
        onSuccess?.(scenario.id)
        onClose()
      }, 1000)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create scenario")
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b">
          <h2 className="text-lg font-semibold text-zinc-800">Add to Scenario</h2>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0"
            onClick={onClose}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {/* Session info */}
          <div className="text-xs text-zinc-500 bg-zinc-50 rounded p-2">
            <span className="font-medium">Session:</span>{" "}
            <span className="font-mono">{sessionId.slice(0, 8)}...</span>
            {agentName && (
              <>
                <br />
                <span className="font-medium">Agent:</span> {agentName}
              </>
            )}
          </div>

          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">
              Scenario Name <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Multi-step query test"
              className="w-full px-3 py-2 text-sm border border-zinc-200 rounded-md focus:outline-none focus:ring-2 focus:ring-zinc-400"
              disabled={isSubmitting || success}
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this scenario tests..."
              rows={3}
              className="w-full px-3 py-2 text-sm border border-zinc-200 rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-zinc-400"
              disabled={isSubmitting || success}
            />
          </div>

          {/* Tags */}
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">
              Tags
            </label>
            <div className="flex flex-wrap gap-1.5 p-2 border border-zinc-200 rounded-md min-h-[42px] focus-within:ring-2 focus-within:ring-zinc-400">
              {tags.map((tag) => (
                <span
                  key={tag}
                  className="inline-flex items-center gap-1 px-2 py-0.5 bg-zinc-100 text-zinc-700 rounded text-sm"
                >
                  {tag}
                  <button
                    type="button"
                    onClick={() => removeTag(tag)}
                    className="hover:text-zinc-900"
                    disabled={isSubmitting || success}
                  >
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
              <input
                type="text"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={handleTagKeyDown}
                onBlur={addTag}
                placeholder={tags.length === 0 ? "Type and press Enter..." : ""}
                className="flex-1 min-w-[100px] text-sm outline-none bg-transparent"
                disabled={isSubmitting || success}
              />
            </div>
            <p className="mt-1 text-xs text-zinc-400">Press Enter or comma to add a tag</p>
          </div>

          {/* Error */}
          {error && (
            <div className="text-sm text-red-600 bg-red-50 rounded p-2">
              {error}
            </div>
          )}

          {/* Success */}
          {success && (
            <div className="flex items-center gap-2 text-sm text-green-600 bg-green-50 rounded p-2">
              <Check className="h-4 w-4" />
              Scenario created successfully!
            </div>
          )}

          {/* Actions */}
          <div className="flex justify-end gap-2 pt-2">
            <Button
              type="button"
              variant="outline"
              onClick={onClose}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSubmitting || success}>
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Creating...
                </>
              ) : success ? (
                <>
                  <Check className="h-4 w-4 mr-2" />
                  Created
                </>
              ) : (
                <>
                  <Plus className="h-4 w-4 mr-2" />
                  Create Scenario
                </>
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
