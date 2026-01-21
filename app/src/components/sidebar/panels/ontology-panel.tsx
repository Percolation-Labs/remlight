/**
 * OntologyPanel - Wiki/Ontology browser panel
 *
 * Mounts and browses the wiki/ontology structure.
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

  useEffect(() => {
    loadOntology()
  }, [])

  const loadOntology = async () => {
    setIsLoading(true)
    try {
      // TODO: Implement API call to load wiki structure
      const response = await fetch("/api/ontology/tree")
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
      actions={
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={loadOntology}
          className="h-7 w-7 p-0"
          title="Refresh"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", isLoading && "animate-spin")} />
        </Button>
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
                <p className="text-[10px] text-zinc-400 mt-1">
                  Configure wiki mount in settings
                </p>
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
