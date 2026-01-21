/**
 * SchemaPreviewPanel - Left panel showing dynamic schema preview
 *
 * Contains all schema sections with focus highlighting and editing support.
 */

import { ScrollArea } from "@/components/ui/scroll-area"
import { ToolsSection, SystemPromptSection, PropertiesSection } from "./sections"
import type {
  AgentSchemaState,
  FocusState,
  ToolReference,
  PropertyDefinition,
} from "@/types/agent-schema"

interface SchemaPreviewPanelProps {
  schema: AgentSchemaState
  focusState: FocusState
  onDescriptionChange: (description: string) => void
  onAddTool: (tool: ToolReference) => void
  onRemoveTool: (toolName: string) => void
  onEditProperty?: (path: string) => void
  onRemoveProperty?: (path: string) => void
  onAddProperty?: () => void
}

export function SchemaPreviewPanel({
  schema,
  focusState,
  onDescriptionChange,
  onAddTool,
  onRemoveTool,
  onEditProperty,
  onRemoveProperty,
  onAddProperty,
}: SchemaPreviewPanelProps) {
  return (
    <div className="flex flex-col h-full bg-zinc-50">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-200 bg-white">
        <div>
          <h2 className="text-sm font-semibold text-zinc-800">
            {schema.metadata.name || "New Agent"}
          </h2>
          <p className="text-[10px] text-zinc-400">v{schema.metadata.version}</p>
        </div>
        <div className="flex items-center gap-2">
          {schema.metadata.structured_output && (
            <span className="text-[10px] px-2 py-0.5 bg-violet-100 text-violet-600 rounded">
              Structured Output
            </span>
          )}
        </div>
      </div>

      {/* Sections */}
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {/* Tools Section */}
          <ToolsSection
            tools={schema.metadata.tools}
            isFocused={focusState.section === "tools"}
            focusMessage={focusState.section === "tools" ? focusState.message : undefined}
            onAddTool={onAddTool}
            onRemoveTool={onRemoveTool}
          />

          {/* System Prompt Section */}
          <SystemPromptSection
            description={schema.description}
            isFocused={focusState.section === "system_prompt"}
            focusMessage={focusState.section === "system_prompt" ? focusState.message : undefined}
            onChange={onDescriptionChange}
          />

          {/* Properties Section */}
          <PropertiesSection
            properties={schema.properties}
            required={schema.required}
            isFocused={focusState.section === "properties"}
            focusMessage={focusState.section === "properties" ? focusState.message : undefined}
            focusedPropertyPath={focusState.propertyPath}
            onEditProperty={onEditProperty}
            onRemoveProperty={onRemoveProperty}
            onAddProperty={onAddProperty}
          />

          {/* Metadata Section */}
          <div className="rounded-lg border border-zinc-200 bg-white p-4">
            <h3 className="text-sm font-medium text-zinc-800 mb-3">Metadata</h3>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <label className="text-zinc-500">Name</label>
                <p className="font-medium text-zinc-700">
                  {schema.metadata.name || <span className="text-zinc-400 italic">Not set</span>}
                </p>
              </div>
              <div>
                <label className="text-zinc-500">Version</label>
                <p className="font-medium text-zinc-700">{schema.metadata.version}</p>
              </div>
              <div className="col-span-2">
                <label className="text-zinc-500">Tags</label>
                <div className="flex flex-wrap gap-1 mt-1">
                  {schema.metadata.tags.length > 0 ? (
                    schema.metadata.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-1.5 py-0.5 bg-zinc-100 text-zinc-600 rounded"
                      >
                        {tag}
                      </span>
                    ))
                  ) : (
                    <span className="text-zinc-400 italic">No tags</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}
