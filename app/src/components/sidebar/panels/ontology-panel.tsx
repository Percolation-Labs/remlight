/**
 * OntologyPanel - Wiki/Ontology browser panel
 *
 * Mounts and browses the wiki/ontology structure.
 * Clicking a page triggers onPageSelect to show content in main window.
 */

import { useState, useEffect } from "react"
import {
  BookMarked,
  ChevronRight,
  ChevronDown,
  FileText,
  Folder,
  FolderOpen,
  Search,
  Loader2,
  RefreshCw,
  Download,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { PanelWrapper } from "../panel-wrapper"
import { cn } from "@/lib/utils"

interface WikiNode {
  id: string
  name: string
  type: "folder" | "page"
  children?: WikiNode[]
  path: string
}

interface OntologyPanelProps {
  onClose: () => void
  onPageSelect?: (path: string) => void
}

export function OntologyPanel({ onClose, onPageSelect }: OntologyPanelProps) {
  const [nodes, setNodes] = useState<WikiNode[]>([])
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())
  const [selectedPath, setSelectedPath] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [isLoading, setIsLoading] = useState(true)
  const [isImporting, setIsImporting] = useState(false)

  useEffect(() => {
    loadOntology()
  }, [])

  const loadOntology = async () => {
    setIsLoading(true)
    try {
      const response = await fetch("/api/v1/ontology/tree")
      if (response.ok) {
        const data = await response.json()
        setNodes(data.nodes || [])
      }
    } catch (error) {
      console.error("Failed to load ontology:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const importOntology = async () => {
    setIsImporting(true)
    try {
      const response = await fetch("/api/v1/ontology/import", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}), // Uses default path
      })
      if (response.ok) {
        const data = await response.json()
        console.log("Import result:", data)
        // Reload the tree after import
        await loadOntology()
      }
    } catch (error) {
      console.error("Failed to import ontology:", error)
    } finally {
      setIsImporting(false)
    }
  }

  const toggleExpand = (id: string) => {
    setExpandedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }

  const handleSelect = (node: WikiNode) => {
    if (node.type === "folder") {
      toggleExpand(node.id)
    } else {
      setSelectedPath(node.path)
      // Notify parent to show content in main window
      onPageSelect?.(node.path)
    }
  }

  const renderNode = (node: WikiNode, depth: number = 0) => {
    const isExpanded = expandedIds.has(node.id)
    const isSelected = selectedPath === node.path
    const isFolder = node.type === "folder"

    return (
      <div key={node.id}>
        <button
          onClick={() => handleSelect(node)}
          className={cn(
            "w-full flex items-center gap-1.5 py-1.5 px-2 rounded-md text-left transition-colors",
            isSelected
              ? "bg-zinc-100 text-zinc-900"
              : "text-zinc-600 hover:bg-zinc-50 hover:text-zinc-800"
          )}
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
        >
          {isFolder ? (
            <>
              <span className="text-zinc-400">
                {isExpanded ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
              </span>
              {isExpanded ? (
                <FolderOpen className="h-3.5 w-3.5 text-amber-500" />
              ) : (
                <Folder className="h-3.5 w-3.5 text-amber-500" />
              )}
            </>
          ) : (
            <>
              <span className="w-3" />
              <FileText className="h-3.5 w-3.5 text-zinc-400" />
            </>
          )}
          <span className="text-xs truncate flex-1">{node.name}</span>
        </button>
        {isFolder && isExpanded && node.children && (
          <div>
            {node.children.map((child) => renderNode(child, depth + 1))}
          </div>
        )}
      </div>
    )
  }

  return (
    <PanelWrapper
      title="Ontology"
      icon={<BookMarked className="h-4 w-4" />}
      onClose={onClose}
      width="wide"
      actions={
        <>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={importOntology}
            disabled={isImporting}
            className="h-7 w-7 p-0"
            title="Import from default path"
          >
            <Download className={cn("h-3.5 w-3.5", isImporting && "animate-pulse")} />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={loadOntology}
            disabled={isLoading}
            className="h-7 w-7 p-0"
            title="Refresh"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", isLoading && "animate-spin")} />
          </Button>
        </>
      }
    >
      <div className="flex flex-col h-full">
        {/* Search */}
        <div className="p-3">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-zinc-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search ontology..."
              className="w-full h-8 pl-8 pr-3 text-xs bg-zinc-50 border border-zinc-200 rounded-md focus:outline-none focus:ring-1 focus:ring-zinc-400 focus:border-zinc-400"
            />
          </div>
        </div>

        {/* Tree view */}
        <ScrollArea className="flex-1">
          <div className="px-2 pb-2">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
              </div>
            ) : nodes.length === 0 ? (
              <div className="text-center py-8">
                <BookMarked className="h-8 w-8 text-zinc-200 mx-auto mb-2" />
                <p className="text-xs text-zinc-400">No ontology loaded</p>
                <p className="text-[10px] text-zinc-400 mt-1 mb-3">
                  Click import to load from default path
                </p>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={importOntology}
                  disabled={isImporting}
                  className="text-xs gap-1"
                >
                  <Download className="h-3 w-3" />
                  {isImporting ? "Importing..." : "Import Ontology"}
                </Button>
              </div>
            ) : (
              nodes.map((node) => renderNode(node))
            )}
          </div>
        </ScrollArea>
      </div>
    </PanelWrapper>
  )
}
