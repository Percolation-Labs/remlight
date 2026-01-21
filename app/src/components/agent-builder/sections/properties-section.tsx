/**
 * PropertiesSection - Output schema properties section for agent builder
 *
 * Displays nested property definitions with collapsible tree view.
 */

import { useState } from "react"
import { Braces, Plus, ChevronsUpDown, ChevronsDownUp } from "lucide-react"
import { Button } from "@/components/ui/button"
import { PropertyNode } from "../property-node"
import { cn } from "@/lib/utils"
import type { PropertyDefinition } from "@/types/agent-schema"

interface PropertiesSectionProps {
  properties: Record<string, PropertyDefinition>
  required: string[]
  isFocused?: boolean
  focusMessage?: string
  focusedPropertyPath?: string
  onEditProperty?: (path: string) => void
  onRemoveProperty?: (path: string) => void
  onAddProperty?: () => void
}

export function PropertiesSection({
  properties,
  required,
  isFocused,
  focusMessage,
  focusedPropertyPath,
  onEditProperty,
  onRemoveProperty,
  onAddProperty,
}: PropertiesSectionProps) {
  const propertyCount = Object.keys(properties).length
  const [allCollapsed, setAllCollapsed] = useState(true)
  const [key, setKey] = useState(0) // Force re-render to reset collapsed state

  return (
    <div
      className={cn(
        "rounded-lg border transition-all duration-200",
        isFocused ? "border-blue-300 ring-1 ring-blue-100" : "border-zinc-200",
        "bg-white"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-100">
        <div className="flex items-center gap-2">
          <Braces className="h-4 w-4 text-zinc-500" />
          <h3 className="text-sm font-medium text-zinc-800">Output Schema</h3>
          <span className="text-xs text-zinc-400">({propertyCount} fields)</span>
        </div>
        <div className="flex items-center gap-1">
          {propertyCount > 0 && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => {
                setAllCollapsed(!allCollapsed)
                setKey(k => k + 1) // Force re-render
              }}
              className="h-7 w-7 p-0"
              title={allCollapsed ? "Expand all" : "Collapse all"}
            >
              {allCollapsed ? (
                <ChevronsUpDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronsDownUp className="h-3.5 w-3.5" />
              )}
            </Button>
          )}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={onAddProperty}
            className="h-7 gap-1 text-xs"
          >
            <Plus className="h-3.5 w-3.5" />
            Add Field
          </Button>
        </div>
      </div>

      {/* Focus message */}
      {isFocused && focusMessage && (
        <div className="px-4 py-2 bg-blue-50 text-xs text-blue-600 border-b border-blue-100">
          {focusMessage}
        </div>
      )}

      {/* Properties tree */}
      <div className="p-3">
        {propertyCount === 0 ? (
          <div className="text-center py-6">
            <Braces className="h-8 w-8 text-zinc-200 mx-auto mb-2" />
            <p className="text-xs text-zinc-400">No output schema defined</p>
            <p className="text-[10px] text-zinc-400 mt-1">
              Add fields to define structured output, or leave empty for conversational responses
            </p>
          </div>
        ) : (
          <div className="space-y-1.5" key={key}>
            {Object.entries(properties).map(([name, definition]) => (
              <PropertyNode
                key={name}
                name={name}
                path={name}
                definition={definition}
                depth={0}
                focusedPath={focusedPropertyPath}
                isRequired={required.includes(name)}
                onEdit={onEditProperty}
                onRemove={onRemoveProperty}
                defaultCollapsed={allCollapsed}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
