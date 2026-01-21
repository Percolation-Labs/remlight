"""Ontology router - Browse and import ontology/wiki content.

Provides endpoints to:
- Get ontology tree structure from filesystem
- Import ontology from a directory path
- Get ontology content by path
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ontology", tags=["ontology"])

# Default ontology directory - relative to project root
DEFAULT_ONTOLOGY_DIR = Path(__file__).parent.parent.parent.parent / "ontology"


class WikiNode(BaseModel):
    """A node in the wiki/ontology tree."""

    id: str
    name: str
    type: str = "page"  # 'folder' or 'page'
    path: str
    children: list["WikiNode"] = []


class OntologyTreeResponse(BaseModel):
    """Response containing the ontology tree structure."""

    nodes: list[WikiNode]
    source_path: str


class OntologyContentResponse(BaseModel):
    """Response containing ontology page content."""

    path: str
    name: str
    content: str
    frontmatter: dict[str, Any] = {}


class ImportOntologyRequest(BaseModel):
    """Request to import ontology from a path."""

    path: str | None = Field(
        default=None,
        description="Path to ontology directory. Uses default if not specified."
    )


class ImportOntologyResponse(BaseModel):
    """Response after importing ontology."""

    imported_count: int
    source_path: str
    message: str


def build_tree_from_directory(directory: Path, base_path: Path | None = None) -> list[WikiNode]:
    """Recursively build a tree structure from a directory.

    Args:
        directory: The directory to scan
        base_path: The base path for computing relative paths (defaults to directory)

    Returns:
        List of WikiNode objects representing the tree
    """
    if base_path is None:
        base_path = directory

    nodes: list[WikiNode] = []

    if not directory.exists():
        return nodes

    # Sort entries: directories first, then files, alphabetically
    entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))

    for entry in entries:
        # Skip hidden files and common non-content files
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue

        relative_path = str(entry.relative_to(base_path))
        node_id = relative_path.replace("/", "-").replace("\\", "-")

        if entry.is_dir():
            children = build_tree_from_directory(entry, base_path)
            # Only add folder if it has content
            if children:
                nodes.append(WikiNode(
                    id=node_id,
                    name=entry.name,
                    type="folder",
                    path=relative_path,
                    children=children,
                ))
        elif entry.suffix.lower() == ".md":
            # Markdown file
            display_name = entry.stem
            # Clean up name: remove leading numbers, convert dashes to spaces
            if display_name.lower() != "readme":
                nodes.append(WikiNode(
                    id=node_id,
                    name=display_name.replace("-", " ").title(),
                    type="page",
                    path=relative_path,
                    children=[],
                ))

    return nodes


def get_ontology_dir() -> Path:
    """Get the default ontology directory path."""
    return DEFAULT_ONTOLOGY_DIR


@router.get("/tree", response_model=OntologyTreeResponse)
async def get_ontology_tree(
    path: str | None = Query(
        default=None,
        description="Custom path to ontology directory. Uses default if not specified."
    )
) -> OntologyTreeResponse:
    """Get the ontology tree structure.

    Returns a hierarchical tree of folders and markdown pages
    from the ontology directory.
    """
    ontology_dir = Path(path) if path else get_ontology_dir()

    if not ontology_dir.exists():
        logger.warning(f"Ontology directory not found: {ontology_dir}")
        return OntologyTreeResponse(nodes=[], source_path=str(ontology_dir))

    nodes = build_tree_from_directory(ontology_dir)

    return OntologyTreeResponse(
        nodes=nodes,
        source_path=str(ontology_dir),
    )


@router.get("/content/{path:path}", response_model=OntologyContentResponse)
async def get_ontology_content(
    path: str,
    base_path: str | None = Query(
        default=None,
        description="Base ontology directory. Uses default if not specified."
    )
) -> OntologyContentResponse:
    """Get content of a specific ontology page.

    Returns the markdown content and parsed frontmatter.
    """
    ontology_dir = Path(base_path) if base_path else get_ontology_dir()
    file_path = ontology_dir / path

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Page not found: {path}")

    if not file_path.suffix.lower() == ".md":
        raise HTTPException(status_code=400, detail="Only markdown files are supported")

    # Read content
    content = file_path.read_text(encoding="utf-8")

    # Parse frontmatter if present
    frontmatter: dict[str, Any] = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                frontmatter = yaml.safe_load(parts[1]) or {}
                content = parts[2].strip()
            except Exception as e:
                logger.warning(f"Failed to parse frontmatter for {path}: {e}")

    return OntologyContentResponse(
        path=path,
        name=file_path.stem.replace("-", " ").title(),
        content=content,
        frontmatter=frontmatter,
    )


@router.post("/import", response_model=ImportOntologyResponse)
async def import_ontology(request: ImportOntologyRequest) -> ImportOntologyResponse:
    """Import ontology content from a directory into the database.

    Reads markdown files from the specified directory and stores them
    in the ontologies table with embeddings for semantic search.
    """
    from remlight.models.entities import Ontology
    from remlight.services.repository import Repository
    from remlight.services.embeddings import generate_embedding_async

    ontology_dir = Path(request.path) if request.path else get_ontology_dir()

    if not ontology_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Ontology directory not found: {ontology_dir}"
        )

    repo = Repository(Ontology, table_name="ontologies")
    imported_count = 0

    # Recursively find all markdown files
    for md_file in ontology_dir.rglob("*.md"):
        # Skip README files and hidden files
        if md_file.name.lower() == "readme.md" or md_file.name.startswith("."):
            continue

        try:
            content = md_file.read_text(encoding="utf-8")
            relative_path = str(md_file.relative_to(ontology_dir))

            # Parse frontmatter
            properties: dict[str, Any] = {}
            description = ""
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    try:
                        import yaml
                        properties = yaml.safe_load(parts[1]) or {}
                        content = parts[2].strip()
                    except Exception:
                        pass

            # Extract first paragraph as description
            lines = content.strip().split("\n\n")
            if lines:
                first_para = lines[0].strip()
                # Remove markdown headers
                if first_para.startswith("#"):
                    first_para = first_para.lstrip("#").strip()
                description = first_para[:500]

            # Create entity key from path
            entity_key = relative_path.replace("/", ".").replace("\\", ".").replace(".md", "")

            # Determine category from directory structure
            parts = relative_path.split("/")
            category = parts[0] if len(parts) > 1 else "general"

            # Create ontology entry
            ontology = Ontology(
                name=entity_key,
                content=content,
                description=description,
                category=category,
                entity_type="document",
                uri=str(md_file),
                properties=properties,
            )

            await repo.upsert(ontology, conflict_field="name")
            imported_count += 1

        except Exception as e:
            logger.warning(f"Failed to import {md_file}: {e}")

    return ImportOntologyResponse(
        imported_count=imported_count,
        source_path=str(ontology_dir),
        message=f"Successfully imported {imported_count} ontology entries from {ontology_dir}",
    )


@router.get("/default-path")
async def get_default_path() -> dict[str, str]:
    """Get the default ontology directory path."""
    return {
        "path": str(get_ontology_dir()),
        "exists": get_ontology_dir().exists(),
    }
