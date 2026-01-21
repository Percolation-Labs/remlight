/**
 * PanelWrapper - Wrapper for sidebar panels
 *
 * Provides consistent header and layout for all panels.
 */

import { X } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { ReactNode } from "react"

interface PanelWrapperProps {
  title: string
  icon: ReactNode
  onClose: () => void
  actions?: ReactNode
  children: ReactNode
}

export function PanelWrapper({
  title,
  icon,
  onClose,
  actions,
  children,
}: PanelWrapperProps) {
  return (
    <div className="w-64 border-r border-zinc-200 bg-white flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-zinc-100">
        <div className="flex items-center gap-2">
          <span className="text-zinc-500">{icon}</span>
          <h2 className="text-sm font-medium text-zinc-800">{title}</h2>
        </div>
        <div className="flex items-center gap-1">
          {actions}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onClose}
            className="h-7 w-7 p-0 text-zinc-400 hover:text-zinc-600"
            title="Close panel"
          >
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden">{children}</div>
    </div>
  )
}
