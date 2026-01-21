"""Servers router - List, retrieve, upsert, and search MCP tool servers.

Provides endpoints to:
- List all registered servers
- Get server information by name
- Create/update servers
- Search servers semantically or by name/tags
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from remlight.models.entities import Server
from remlight.services.repository import Repository

router = APIRouter(prefix="/servers", tags=["servers"])


class ServerInfo(BaseModel):
    """Server information response."""

    name: str
    description: str | None = None
    server_type: str = "mcp"
    endpoint: str | None = None
    enabled: bool = True
    icon: str | None = None
    tags: list[str] = []
    registry_uri: str | None = None


class ServerListResponse(BaseModel):
    """List of servers response."""

    servers: list[ServerInfo]
    total: int


class ServerUpsertRequest(BaseModel):
    """Request to create/update a server."""

    name: str = Field(..., description="Unique server identifier")
    description: str | None = Field(None, description="Server description for search")
    server_type: str = Field(default="mcp", description="Server type: mcp, rest, stdio")
    endpoint: str | None = Field(None, description="URL or command for remote servers")
    config: dict[str, Any] = Field(default_factory=dict, description="Server-specific config")
    enabled: bool = Field(default=True, description="Whether server is active")
    icon: str | None = Field(None, description="Display icon")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    registry_uri: str | None = Field(None, description="Parent registry URI (federation)")


class ServerUpsertResponse(BaseModel):
    """Response after upserting a server."""

    name: str
    created: bool
    message: str


class ServerSearchRequest(BaseModel):
    """Request to search for servers."""

    query: str = Field(..., description="Search query")
    search_type: str = Field(
        default="semantic",
        description="Type of search: semantic, name, tags, all"
    )
    limit: int = Field(default=10, ge=1, le=100)
    include_disabled: bool = Field(default=False)


class ServerSearchResult(BaseModel):
    """Search result for a server."""

    name: str
    description: str | None = None
    server_type: str = "mcp"
    endpoint: str | None = None
    icon: str | None = None
    tags: list[str] = []
    similarity: float | None = None


class ServerSearchResponse(BaseModel):
    """Response for server search."""

    results: list[ServerSearchResult]
    total: int


def server_to_info(server: Server) -> ServerInfo:
    """Convert Server model to ServerInfo response."""
    return ServerInfo(
        name=server.name,
        description=server.description,
        server_type=server.server_type,
        endpoint=server.endpoint,
        enabled=server.enabled,
        icon=server.icon,
        tags=server.tags,
        registry_uri=server.registry_uri,
    )


@router.get("", response_model=ServerListResponse)
async def list_servers(
    include_disabled: bool = Query(default=False, description="Include disabled servers"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> ServerListResponse:
    """List all registered servers."""
    repo = Repository(Server, table_name="servers")

    filters: dict[str, Any] = {}
    if not include_disabled:
        filters["enabled"] = True

    servers = await repo.find(filters, order_by="name ASC", limit=limit, offset=offset)
    total = await repo.count(filters)

    return ServerListResponse(
        servers=[server_to_info(s) for s in servers],
        total=total,
    )


@router.get("/{server_name}", response_model=ServerInfo)
async def get_server(server_name: str) -> ServerInfo:
    """Get information about a specific server."""
    repo = Repository(Server, table_name="servers")

    server = await repo.get_by_name(server_name)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    return server_to_info(server)


@router.put("", response_model=ServerUpsertResponse)
async def upsert_server(request: ServerUpsertRequest) -> ServerUpsertResponse:
    """Create or update a server."""
    repo = Repository(Server, table_name="servers")

    # Check if server exists
    existing = await repo.get_by_name(request.name)
    created = existing is None

    server = Server(
        id=existing.id if existing else None,
        name=request.name,
        description=request.description,
        server_type=request.server_type,
        endpoint=request.endpoint,
        config=request.config,
        enabled=request.enabled,
        icon=request.icon,
        tags=request.tags,
        registry_uri=request.registry_uri,
    )

    await repo.upsert(server, conflict_field="name" if not existing else "id")

    return ServerUpsertResponse(
        name=request.name,
        created=created,
        message=f"Server '{request.name}' {'created' if created else 'updated'} successfully",
    )


@router.delete("/{server_name}")
async def delete_server(server_name: str) -> dict[str, str]:
    """Delete a server (soft delete)."""
    repo = Repository(Server, table_name="servers")

    existing = await repo.get_by_name(server_name)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    await repo.delete(str(existing.id))

    return {"message": f"Server '{server_name}' deleted successfully"}


@router.post("/search", response_model=ServerSearchResponse)
async def search_servers(request: ServerSearchRequest) -> ServerSearchResponse:
    """Search for servers semantically or by name/tags."""
    repo = Repository(Server, table_name="servers")

    try:
        results = await repo.search(
            query=request.query,
            search_type=request.search_type,
            limit=request.limit,
            include_disabled=request.include_disabled,
        )

        search_results = [
            ServerSearchResult(
                name=server.name,
                description=server.description,
                server_type=server.server_type,
                endpoint=server.endpoint,
                icon=server.icon,
                tags=server.tags,
                similarity=similarity,
            )
            for server, similarity in results
        ]

        return ServerSearchResponse(
            results=search_results,
            total=len(search_results),
        )

    except Exception as e:
        logger.error(f"Server search failed: {e}")
        return ServerSearchResponse(results=[], total=0)


@router.get("/{server_name}/tools")
async def get_server_tools(server_name: str) -> list[dict[str, Any]]:
    """Get all tools for a specific server."""
    from remlight.models.entities import Tool

    server_repo = Repository(Server, table_name="servers")
    tool_repo = Repository(Tool, table_name="tools")

    server = await server_repo.get_by_name(server_name)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    tools = await tool_repo.find({"server_id": str(server.id), "enabled": True})

    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
            "icon": t.icon,
            "tags": t.tags,
        }
        for t in tools
    ]
