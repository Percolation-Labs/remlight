/**
 * AddScenarioButton - Add session to scenario for evaluations
 *
 * Plus icon that opens a modal to save the session with title, description, and tags.
 */

import { useState } from "react"
import { Plus, X } from "lucide-react"
import { Button } from "@/components/ui/button"

interface AddScenarioButtonProps {
  sessionId: string
  onSubmit?: (scenario: ScenarioData) => Promise<void>
}

export interface ScenarioData {
  sessionId: string
  title: string
  description: string
  tags: string[]
}

export function AddScenarioButton({ sessionId, onSubmit }: AddScenarioButtonProps) {
  const [isOpen, setIsOpen] = useState(false)
  const [title, setTitle] = useState("")
  const [description, setDescription] = useState("")
  const [tagsInput, setTagsInput] = useState("")
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async () => {
    if (!title.trim()) return

    const tags = tagsInput
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean)

    const scenario: ScenarioData = {
      sessionId,
      title: title.trim(),
      description: description.trim(),
      tags,
    }

    setIsSubmitting(true)
    try {
      if (onSubmit) {
        await onSubmit(scenario)
      } else {
        // Default API call
        await fetch("/api/scenarios", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(scenario),
        })
      }
      setSuccess(true)
      setIsOpen(false)
      // Reset form
      setTitle("")
      setDescription("")
      setTagsInput("")
    } catch (error) {
      console.error("Failed to add scenario:", error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    setIsOpen(false)
    setTitle("")
    setDescription("")
    setTagsInput("")
  }

  if (success) {
    return (
      <span className="text-xs text-green-600">Added to scenarios</span>
    )
  }

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        onClick={() => setIsOpen(true)}
        className="h-6 w-6 p-0"
        title="Add to scenario"
      >
        <Plus className="h-3 w-3" />
      </Button>

      {/* Modal */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 bg-black/20 z-40"
            onClick={handleClose}
          />

          {/* Modal content */}
          <div className="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-96 bg-white border border-zinc-200 rounded-lg shadow-container p-4 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-zinc-700">
                Add to Scenario
              </h3>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0"
                onClick={handleClose}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            <div className="space-y-3">
              {/* Title */}
              <div>
                <label className="block text-xs font-medium text-zinc-600 mb-1">
                  Title *
                </label>
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Scenario title"
                  className="w-full text-sm border border-zinc-200 rounded px-3 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                />
              </div>

              {/* Description */}
              <div>
                <label className="block text-xs font-medium text-zinc-600 mb-1">
                  Description
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What is this scenario testing?"
                  rows={3}
                  className="w-full text-sm border border-zinc-200 rounded px-3 py-2 resize-none focus:outline-none focus:ring-1 focus:ring-zinc-400"
                />
              </div>

              {/* Tags */}
              <div>
                <label className="block text-xs font-medium text-zinc-600 mb-1">
                  Tags (comma-separated)
                </label>
                <input
                  type="text"
                  value={tagsInput}
                  onChange={(e) => setTagsInput(e.target.value)}
                  placeholder="e.g. search, tool-use, multi-step"
                  className="w-full text-sm border border-zinc-200 rounded px-3 py-2 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleClose}
                className="text-xs"
              >
                Cancel
              </Button>
              <Button
                size="sm"
                onClick={handleSubmit}
                disabled={!title.trim() || isSubmitting}
                className="text-xs"
              >
                {isSubmitting ? "Adding..." : "Add Scenario"}
              </Button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
