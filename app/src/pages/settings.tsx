/**
 * Settings Page - Placeholder for future settings
 */

import { ArrowLeft } from "lucide-react"
import { Button } from "@/components/ui/button"

interface SettingsPageProps {
  onBack?: () => void
}

export function SettingsPage({ onBack }: SettingsPageProps) {
  return (
    <div className="min-h-screen bg-zinc-50 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="flex items-center gap-4 mb-8">
          {onBack && (
            <Button variant="ghost" size="sm" onClick={onBack}>
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back
            </Button>
          )}
          <h1 className="text-xl font-semibold text-zinc-800">Settings</h1>
        </div>

        <div className="bg-white rounded-lg border border-zinc-200 shadow-chat p-8">
          <p className="text-sm text-zinc-500 text-center">
            Settings coming soon.
          </p>
        </div>
      </div>
    </div>
  )
}
