/**
 * NavRail - Vertical icon navigation bar
 *
 * Always visible, provides navigation between sidebar panels.
 */

import { MessageSquare, Layers, Boxes, BookMarked, Settings } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export type PanelType = "chat" | "scenarios" | "schema" | "ontology"

interface NavRailProps {
  activePanel: PanelType | null
  onPanelSelect: (panel: PanelType) => void
  onSettings?: () => void
}

const navItems: { id: PanelType; icon: typeof MessageSquare; label: string }[] = [
  { id: "chat", icon: MessageSquare, label: "Chat History" },
  { id: "scenarios", icon: Layers, label: "Scenarios" },
  { id: "schema", icon: Boxes, label: "Schema Builder" },
  { id: "ontology", icon: BookMarked, label: "Ontology" },
]

export function NavRail({ activePanel, onPanelSelect, onSettings }: NavRailProps) {
  return (
    <div className="w-12 border-r border-zinc-200 bg-zinc-50 flex flex-col items-center py-3 gap-1">
      {navItems.map((item) => {
        const Icon = item.icon
        const isActive = activePanel === item.id
        return (
          <Button
            key={item.id}
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => onPanelSelect(item.id)}
            className={cn(
              "h-10 w-10 p-0 rounded-lg transition-colors",
              isActive
                ? "bg-zinc-900 text-white hover:bg-zinc-800 hover:text-white"
                : "text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700"
            )}
            title={item.label}
          >
            <Icon className="h-5 w-5" />
          </Button>
        )
      })}

      {/* Settings at bottom */}
      {onSettings && (
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onSettings}
          className="h-10 w-10 p-0 rounded-lg mt-auto text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700"
          title="Settings"
        >
          <Settings className="h-5 w-5" />
        </Button>
      )}
    </div>
  )
}
