/**
 * PropertyNode - Recursive component for rendering schema properties
 *
 * Renders nested properties with collapsible sections and editing support.
 */

import { useState, useEffect, useRef } from "react"
import {
  ChevronRight,
  Type,
  Hash,
  ToggleLeft,
  List,
  Braces,
  Trash2,
  Edit2,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import type { PropertyDefinition } from "@/types/agent-schema"

interface PropertyNodeProps {
  name: string
  path: string
  definition: PropertyDefinition
  depth?: number
  focusedPath?: string | null
  isRequired?: boolean
  onEdit?: (path: string) => void
  onRemove?: (path: string) => void
  className?: string
  defaultCollapsed?: boolean
}

const TYPE_ICONS: Record<PropertyDefinition["type"], typeof Type> = {
  string: Type,
  number: Hash,
  integer: Hash,
  boolean: ToggleLeft,
  array: List,
  object: Braces,
}

const TYPE_COLORS: Record<PropertyDefinition["type"], string> = {
  string: "text-green-600 bg-green-50",
  number: "text-blue-600 bg-blue-50",
  integer: "text-blue-600 bg-blue-50",
  boolean: "text-purple-600 bg-purple-50",
  array: "text-orange-600 bg-orange-50",
  object: "text-zinc-600 bg-zinc-100",
}

export function PropertyNode({
  name,
  path,
  definition,
  depth = 0,
  focusedPath,
  isRequired = false,
  onEdit,
  onRemove,
  className,
  defaultCollapsed = true,
}: PropertyNodeProps) {
  const [isOpen, setIsOpen] = useState(!defaultCollapsed)
  const nodeRef = useRef<HTMLDivElement>(null)
  const [isFlashing, setIsFlashing] = useState(false)

  const hasChildren =
    (definition.type === "object" && definition.properties) ||
    (definition.type === "array" && definition.items)

  const isFocused = focusedPath === path || focusedPath?.startsWith(`${path}.`)
  const isDirectFocus = focusedPath === path

  // Auto-expand when focused
  useEffect(() => {
    if (isFocused && hasChildren && !isOpen) {
      setIsOpen(true)
    }
  }, [isFocused, hasChildren, isOpen])

  // Scroll into view and flash when directly focused
  useEffect(() => {
    if (isDirectFocus && nodeRef.current) {
      nodeRef.current.scrollIntoView({ behavior: "smooth", block: "nearest" })
      setIsFlashing(true)
      const timer = setTimeout(() => setIsFlashing(false), 500)
      return () => clearTimeout(timer)
    }
  }, [isDirectFocus])

  const Icon = TYPE_ICONS[definition.type]
  const typeColor = TYPE_COLORS[definition.type]

  const content = (
    <div
      ref={nodeRef}
      className={cn(
        "rounded-md border transition-all duration-200",
        isDirectFocus && "border-blue-300 ring-1 ring-blue-100",
        isFlashing && "bg-blue-50",
        !isDirectFocus && !isFlashing && "border-zinc-200 bg-white",
        className
      )}
      style={{ marginLeft: depth > 0 ? 12 : 0 }}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-center gap-2 px-3 py-2",
          hasChildren && "cursor-pointer hover:bg-zinc-50"
        )}
        onClick={() => hasChildren && setIsOpen(!isOpen)}
      >
        {/* Expand chevron */}
        {hasChildren ? (
          <ChevronRight
            className={cn(
              "h-3.5 w-3.5 text-zinc-400 transition-transform",
              isOpen && "rotate-90"
            )}
          />
        ) : (
          <span className="w-3.5" />
        )}

        {/* Type icon */}
        <span className={cn("p-1 rounded", typeColor)}>
          <Icon className="h-3 w-3" />
        </span>

        {/* Property name */}
        <span className="font-medium text-sm text-zinc-800">
          {name}
          {isRequired && <span className="text-red-500 ml-0.5">*</span>}
        </span>

        {/* Type badge */}
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-zinc-100 text-zinc-500 font-mono">
          {definition.type}
          {definition.type === "array" && definition.items && (
            <span className="text-zinc-400">[{definition.items.type}]</span>
          )}
        </span>

        {/* Enum indicator */}
        {definition.enum && (
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-50 text-amber-600">
            enum
          </span>
        )}

        {/* Spacer */}
        <span className="flex-1" />

        {/* Actions */}
        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {onEdit && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                onEdit(path)
              }}
              className="h-6 w-6 p-0"
            >
              <Edit2 className="h-3 w-3" />
            </Button>
          )}
          {onRemove && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={(e) => {
                e.stopPropagation()
                onRemove(path)
              }}
              className="h-6 w-6 p-0 text-red-500 hover:text-red-600"
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          )}
        </div>
      </div>

      {/* Description */}
      {definition.description && (
        <div className="px-3 pb-2 -mt-1">
          <p className="text-xs text-zinc-500 leading-relaxed">{definition.description}</p>
        </div>
      )}

      {/* Constraints */}
      {(definition.enum || definition.minimum !== undefined || definition.maximum !== undefined) && (
        <div className="px-3 pb-2 flex flex-wrap gap-1.5">
          {definition.enum && (
            <span className="text-[10px] text-zinc-500">
              Values: {definition.enum.slice(0, 3).join(", ")}
              {definition.enum.length > 3 && ` +${definition.enum.length - 3}`}
            </span>
          )}
          {definition.minimum !== undefined && (
            <span className="text-[10px] text-zinc-500">min: {definition.minimum}</span>
          )}
          {definition.maximum !== undefined && (
            <span className="text-[10px] text-zinc-500">max: {definition.maximum}</span>
          )}
        </div>
      )}

      {/* Children */}
      {hasChildren && isOpen && (
        <div className="px-2 pb-2 border-t border-zinc-100 pt-2 space-y-1.5">
          {definition.type === "object" &&
            definition.properties &&
            Object.entries(definition.properties).map(([childName, childDef]) => (
              <PropertyNode
                key={childName}
                name={childName}
                path={`${path}.${childName}`}
                definition={childDef}
                depth={depth + 1}
                focusedPath={focusedPath}
                isRequired={definition.required?.includes(childName)}
                onEdit={onEdit}
                onRemove={onRemove}
                defaultCollapsed={defaultCollapsed}
              />
            ))}
          {definition.type === "array" && definition.items && (
            <div className="text-xs text-zinc-500 px-2 py-1">
              <span className="font-medium">Array items:</span>
              <PropertyNode
                name="[item]"
                path={`${path}.items`}
                definition={definition.items}
                depth={depth + 1}
                focusedPath={focusedPath}
                onEdit={onEdit}
                onRemove={onRemove}
                defaultCollapsed={defaultCollapsed}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )

  return <div className="group">{content}</div>
}
