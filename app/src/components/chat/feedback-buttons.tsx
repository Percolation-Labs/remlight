/**
 * FeedbackButtons - Thumbs up/down feedback for messages
 *
 * Allows users to rate messages with optional text feedback.
 */

import { useState } from "react"
import { ThumbsUp, ThumbsDown, X, Send } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface FeedbackButtonsProps {
  messageId: string
  categories?: string[]
  onSubmit?: (feedback: FeedbackData) => Promise<void>
}

export interface FeedbackData {
  messageId: string
  rating: "positive" | "negative"
  text?: string
  category?: string
}

const DEFAULT_CATEGORIES = ["Helpful", "Accurate", "Clear", "Relevant"]

export function FeedbackButtons({
  messageId,
  categories = DEFAULT_CATEGORIES,
  onSubmit,
}: FeedbackButtonsProps) {
  const [rating, setRating] = useState<"positive" | "negative" | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [text, setText] = useState("")
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleRating = (value: "positive" | "negative") => {
    setRating(value)
    setShowForm(true)
  }

  const handleSubmit = async () => {
    if (!rating) return

    const feedback: FeedbackData = {
      messageId,
      rating,
      text: text.trim() || undefined,
      category: selectedCategory || undefined,
    }

    setIsSubmitting(true)
    try {
      if (onSubmit) {
        await onSubmit(feedback)
      } else {
        // Default API call
        await fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(feedback),
        })
      }
      setSubmitted(true)
      setShowForm(false)
    } catch (error) {
      console.error("Failed to submit feedback:", error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleClose = () => {
    setShowForm(false)
    setRating(null)
    setText("")
    setSelectedCategory(null)
  }

  if (submitted) {
    return <span className="text-xs text-zinc-400">Thanks for feedback</span>
  }

  return (
    <div className="relative">
      {/* Rating buttons */}
      <div className="flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            "h-6 w-6 p-0",
            rating === "positive" && "text-green-600"
          )}
          onClick={() => handleRating("positive")}
        >
          <ThumbsUp className="h-3 w-3" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className={cn(
            "h-6 w-6 p-0",
            rating === "negative" && "text-red-600"
          )}
          onClick={() => handleRating("negative")}
        >
          <ThumbsDown className="h-3 w-3" />
        </Button>
      </div>

      {/* Feedback form - full width */}
      {showForm && (
        <div className="absolute left-0 right-0 bottom-full mb-2 z-50 bg-white border border-zinc-200 rounded-lg shadow-lg p-4 space-y-3" style={{ minWidth: "320px" }}>
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-zinc-700">
              {rating === "positive" ? "What was good?" : "What could improve?"}
            </span>
            <Button
              variant="ghost"
              size="sm"
              className="h-5 w-5 p-0"
              onClick={handleClose}
            >
              <X className="h-3 w-3" />
            </Button>
          </div>

          {/* Categories */}
          <div className="flex flex-wrap gap-1">
            {categories.map((cat) => (
              <button
                key={cat}
                className={cn(
                  "px-2 py-0.5 text-xs rounded border transition-colors",
                  selectedCategory === cat
                    ? "bg-zinc-700 text-white border-zinc-700"
                    : "bg-zinc-50 text-zinc-600 border-zinc-200 hover:border-zinc-300"
                )}
                onClick={() =>
                  setSelectedCategory(selectedCategory === cat ? null : cat)
                }
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Text input */}
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Optional: Add more details..."
            className="w-full text-xs border border-zinc-200 rounded p-2 resize-none h-16 focus:outline-none focus:ring-1 focus:ring-zinc-400"
          />

          {/* Submit button */}
          <Button
            size="sm"
            className="w-full h-7 text-xs"
            onClick={handleSubmit}
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              "Sending..."
            ) : (
              <>
                <Send className="h-3 w-3 mr-1" />
                Submit
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  )
}
