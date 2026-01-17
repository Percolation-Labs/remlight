/**
 * ExportSessionButton - Export session as YAML
 *
 * Download icon that triggers YAML export of the full session.
 */

import { useState } from "react"
import { Download } from "lucide-react"
import { Button } from "@/components/ui/button"
import { exportSessionAsYaml } from "@/api/sessions"

interface ExportSessionButtonProps {
  sessionId: string
}

export function ExportSessionButton({ sessionId }: ExportSessionButtonProps) {
  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleExport = async () => {
    setIsExporting(true)
    setError(null)
    try {
      await exportSessionAsYaml(sessionId)
    } catch (err) {
      setError("Export failed")
      console.error("Failed to export session:", err)
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="relative">
      <Button
        variant="ghost"
        size="sm"
        onClick={handleExport}
        disabled={isExporting}
        className="h-6 w-6 p-0"
        title="Export session as YAML"
      >
        <Download className={`h-3 w-3 ${isExporting ? "animate-pulse" : ""}`} />
      </Button>
      {error && (
        <span className="absolute right-0 top-full text-xs text-red-500 whitespace-nowrap">
          {error}
        </span>
      )}
    </div>
  )
}
