"""Tools registry router - List, retrieve, upsert, and search registered tools.

Provides endpoints to:
- List all registered tools
- Get tool information by server and name
- Create/update tools
- Search tools semantically or by name/tags

Note: This is for the TOOL REGISTRY (metadata about tools), not tool execution.
Tool execution is handled by the MCP server in mcp_main.py and tools.py.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from remlight.models.entities import Server, Tool
from remlight.services.repository import Repository

router = APIRouter(prefix="/tools", tags=["tools"])


class ToolInfo(BaseModel):
    """Tool information response."""

    name: str
    description: str | None = None
    server_name: str | None = None
    input_schema: dict[str, Any] = {}
    enabled: bool = True
    icon: str | None = None
    tags: list[str] = []


class ToolListResponse(BaseModel):
    """List of tools response."""

    tools: list[ToolInfo]
    total: int


class ToolUpsertRequest(BaseModel):
    """Request to create/update a tool."""

    name: str = Field(..., description="Tool function name")
    server_name: str = Field(..., description="Parent server name")
    description: str | None = Field(None, description="Tool description for search")
    input_schema: dict[str, Any] = Field(default_factory=dict, description="JSON Schema for params")
    enabled: bool = Field(default=True, description="Whether tool is active")
    icon: str | None = Field(None, description="Display icon")
    tags: list[str] = Field(default_factory=list, description="Classification tags")


class ToolUpsertResponse(BaseModel):
    """Response after upserting a tool."""

    name: str
    server_name: str
    created: bool
    message: str


class ToolSearchRequest(BaseModel):
    """Request to search for tools."""

    query: str = Field(..., description="Search query")
    search_type: str = Field(
        default="semantic",
        description="Type of search: semantic, name, tags, all"
    )
    server_name: str | None = Field(None, description="Filter to specific server")
    limit: int = Field(default=10, ge=1, le=100)
    include_disabled: bool = Field(default=False)


class ToolSearchResult(BaseModel):
    """Search result for a tool."""

    name: str
    description: str | None = None
    server_name: str | None = None
    input_schema: dict[str, Any] = {}
    icon: str | None = None
    tags: list[str] = []
    similarity: float | None = None


class ToolSearchResponse(BaseModel):
    """Response for tool search."""

    results: list[ToolSearchResult]
    total: int


# Cache for server_id -> server_name mapping
_server_name_cache: dict[str, str] = {}


async def get_server_name(server_id: str | None) -> str | None:
    """Get server name from server_id with caching."""
    if not server_id:
        return None

    if server_id in _server_name_cache:
        return _server_name_cache[server_id]

    repo = Repository(Server, table_name="servers")
    server = await repo.get_by_id(server_id)
    if server:
        _server_name_cache[server_id] = server.name
        return server.name

    return None


async def tool_to_info(tool: Tool) -> ToolInfo:
    """Convert Tool model to ToolInfo response."""
    server_name = await get_server_name(str(tool.server_id) if tool.server_id else None)
    return ToolInfo(
        name=tool.name,
        description=tool.description,
        server_name=server_name,
        input_schema=tool.input_schema,
        enabled=tool.enabled,
        icon=tool.icon,
        tags=tool.tags,
    )


@router.get("", response_model=ToolListResponse)
async def list_tools(
    server_name: str | None = Query(default=None, description="Filter by server"),
    include_disabled: bool = Query(default=False, description="Include disabled tools"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> ToolListResponse:
    """List all registered tools."""
    tool_repo = Repository(Tool, table_name="tools")

    filters: dict[str, Any] = {}
    if not include_disabled:
        filters["enabled"] = True

    # Filter by server if specified
    if server_name:
        server_repo = Repository(Server, table_name="servers")
        server = await server_repo.get_by_name(server_name)
        if not server:
            raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
        filters["server_id"] = str(server.id)

    tools = await tool_repo.find(filters, order_by="name ASC", limit=limit, offset=offset)
    total = await tool_repo.count(filters)

    # Convert to info objects
    tool_infos = []
    for tool in tools:
        info = await tool_to_info(tool)
        tool_infos.append(info)

    return ToolListResponse(tools=tool_infos, total=total)


@router.get("/{server_name}/{tool_name}", response_model=ToolInfo)
async def get_tool(server_name: str, tool_name: str) -> ToolInfo:
    """Get information about a specific tool."""
    server_repo = Repository(Server, table_name="servers")
    tool_repo = Repository(Tool, table_name="tools")

    server = await server_repo.get_by_name(server_name)
    if not server:
        raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")

    tools = await tool_repo.find({"server_id": str(server.id), "name": tool_name})
    if not tools:
        raise HTTPException(
            status_code=404,
            detail=f"Tool '{tool_name}' not found on server '{server_name}'"
        )

    return await tool_to_info(tools[0])


@router.put("", response_model=ToolUpsertResponse)
async def upsert_tool(request: ToolUpsertRequest) -> ToolUpsertResponse:
    """Create or update a tool."""
    server_repo = Repository(Server, table_name="servers")
    tool_repo = Repository(Tool, table_name="tools")

    # Get parent server
    server = await server_repo.get_by_name(request.server_name)
    if not server:
        raise HTTPException(
            status_code=404,
            detail=f"Server '{request.server_name}' not found"
        )

    # Check if tool exists
    existing_tools = await tool_repo.find({
        "server_id": str(server.id),
        "name": request.name
    })
    existing = existing_tools[0] if existing_tools else None
    created = existing is None

    tool = Tool(
        id=existing.id if existing else None,
        name=request.name,
        description=request.description,
        server_id=server.id,
        input_schema=request.input_schema,
        enabled=request.enabled,
        icon=request.icon,
        tags=request.tags,
    )

    await tool_repo.upsert(tool)

    return ToolUpsertResponse(
        name=request.name,
        server_name=request.server_name,
        created=created,
        message=f"Tool '{request.name}' on '{request.server_name}' "
                f"{'created' if created else 'updated'} successfully",
    )


@router.delete("/{tool_id}")
async def delete_tool(tool_id: str) -> dict[str, str]:
    """Delete a tool (soft delete)."""
    repo = Repository(Tool, table_name="tools")

    existing = await repo.get_by_id(tool_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Tool with ID '{tool_id}' not found")

    await repo.delete(tool_id)

    return {"message": f"Tool '{existing.name}' deleted successfully"}


@router.post("/search", response_model=ToolSearchResponse)
async def search_tools(request: ToolSearchRequest) -> ToolSearchResponse:
    """Search for tools semantically or by name/tags."""
    tool_repo = Repository(Tool, table_name="tools")

    try:
        # If server_name specified, get server_id for filtering
        extra_filters = None
        if request.server_name:
            server_repo = Repository(Server, table_name="servers")
            server = await server_repo.get_by_name(request.server_name)
            if server:
                extra_filters = {"server_id": str(server.id)}

        results = await tool_repo.search(
            query=request.query,
            search_type=request.search_type,
            limit=request.limit,
            include_disabled=request.include_disabled,
            extra_filters=extra_filters,
        )

        search_results = []
        for tool, similarity in results:
            server_name = await get_server_name(
                str(tool.server_id) if tool.server_id else None
            )
            search_results.append(ToolSearchResult(
                name=tool.name,
                description=tool.description,
                server_name=server_name,
                input_schema=tool.input_schema,
                icon=tool.icon,
                tags=tool.tags,
                similarity=similarity,
            ))

        return ToolSearchResponse(
            results=search_results,
            total=len(search_results),
        )

    except Exception as e:
        logger.error(f"Tool search failed: {e}")
        return ToolSearchResponse(results=[], total=0)


@router.post("/register")
async def register_project_tools(
    force: bool = Query(default=False, description="Force re-register even if unchanged"),
) -> dict[str, Any]:
    """Register all project tools from the MCP server.

    This endpoint triggers registration of all tools defined in the project's
    MCP server. Uses MD5 hash to detect changes and skip unchanged items.
    """
    from remlight.services.registration import register_project_tools as do_register

    stats = await do_register(force=force, generate_embeddings=True)
    return stats
