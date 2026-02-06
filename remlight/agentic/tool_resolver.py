"""
Tool Resolver - Load tools from local and remote MCP servers
=============================================================

This module handles resolving tool references in agent schemas to toolsets
for PydanticAI using FastMCPToolset.

SIMPLIFIED ARCHITECTURE (using FastMCPToolset)
----------------------------------------------
FastMCPToolset from pydantic-ai handles:
- Schema fetching from MCP servers
- Type annotation generation
- Tool wrapper creation
- Caching

We just need to:
1. Create FastMCPToolset instances for each server
2. Apply filtering based on agent schema's allowed tools
3. Return toolsets for the Agent

SUPPORTED SERVER TYPES
----------------------
- Local tools: Built-in tools from the project's MCP server
- MCP servers: Remote MCP servers accessed via FastMCPToolset
- REST servers: Legacy REST APIs (still uses manual wrapper)

RESOLUTION FLOW
---------------

1. Agent schema declares tools with optional server:
   ```yaml
   tools:
     - name: search
       server: local
     - name: fetch_data
       server: data-service  # registered MCP server
   ```

2. resolve_tools_as_toolsets() is called with the tool references

3. For each server:
   a. Create FastMCPToolset (local or remote)
   b. Apply .filtered() to restrict to allowed tools
   c. Return list of toolsets for Agent

4. Agent uses toolsets=[] parameter instead of tools=[]
"""

from typing import Any, Callable
import httpx
from loguru import logger

from remlight.agentic.agent_schema import AgentContext, MCPToolReference
from remlight.models.entities import Server
from remlight.services.repository import Repository


# Cache for resolved servers
_server_cache: dict[str, Server] = {}


async def get_server(server_name: str) -> Server | None:
    """Get server by name with caching."""
    if server_name in _server_cache:
        return _server_cache[server_name]

    repo = Repository(Server, table_name="servers")
    server = await repo.get_by_name(server_name)
    if server:
        _server_cache[server_name] = server
    return server


def clear_server_cache():
    """Clear the server cache."""
    _server_cache.clear()


# =============================================================================
# FASTMCP TOOLSET CREATION
# =============================================================================


def create_filtered_mcp_toolset(
    server_or_endpoint: Any,
    allowed_tools: set[str] | None = None,
) -> Any:
    """
    Create a FastMCPToolset with optional tool filtering.

    Args:
        server_or_endpoint: Either a FastMCP server instance or endpoint URL string
        allowed_tools: Set of tool names to allow. If None, all tools are allowed.

    Returns:
        FastMCPToolset (filtered if allowed_tools provided)
    """
    from pydantic_ai.toolsets.fastmcp import FastMCPToolset

    toolset = FastMCPToolset(server_or_endpoint)

    if allowed_tools:
        # Filter to only allowed tools
        return toolset.filtered(
            lambda ctx, tool_def: tool_def.name in allowed_tools
        )

    return toolset


# =============================================================================
# REST TOOL WRAPPER (Legacy - no schema introspection)
# =============================================================================


def create_rest_tool_wrapper(
    server: Server,
    tool_name: str,
    context: AgentContext | None = None,
) -> Callable:
    """
    Create a wrapper for a REST-based remote tool (legacy, no schema).

    WARNING: This creates a wrapper with **kwargs which gives PydanticAI
    no type information. The LLM won't know what parameters to pass.
    Prefer MCP server type which provides proper schemas.

    Args:
        server: Server configuration with endpoint URL
        tool_name: Name of the tool to call
        context: Optional context for user/session info

    Returns:
        Async callable (with poor type info)
    """
    endpoint = server.endpoint.rstrip("/") if server.endpoint else ""

    async def remote_tool(**kwargs: Any) -> Any:
        """Call remote tool via REST API."""
        url = f"{endpoint}/tools/{tool_name}"

        payload = {
            "arguments": kwargs,
            "context": {
                "user_id": context.user_id if context else None,
                "session_id": context.session_id if context else None,
                "trace_id": context.trace_id if context else None,
            },
        }

        headers = {"Content-Type": "application/json"}
        if server.config.get("api_key"):
            headers["Authorization"] = f"Bearer {server.config['api_key']}"
        if server.config.get("headers"):
            headers.update(server.config["headers"])

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
                response.raise_for_status()

                result = response.json()
                if "error" in result and result["error"]:
                    raise Exception(f"Remote tool error: {result['error']}")

                return result.get("result", result)

        except httpx.HTTPError as e:
            logger.error(f"Remote tool {server.name}/{tool_name} failed: {e}")
            raise Exception(f"Failed to call remote tool {tool_name}: {e}")

    remote_tool.__name__ = tool_name
    remote_tool.__doc__ = f"Remote tool '{tool_name}' from server '{server.name}' (no schema)"

    return remote_tool


# =============================================================================
# MAIN RESOLUTION FUNCTIONS
# =============================================================================


async def resolve_tools_as_toolsets(
    tool_refs: list[MCPToolReference | dict],
    local_mcp_server: Any | None = None,
    context: AgentContext | None = None,
) -> tuple[list[Any], list[Callable]]:
    """
    Resolve tool references to FastMCPToolsets and legacy tool callables.

    This is the main entry point for tool resolution. It takes tool references
    from an agent schema and returns:
    - List of FastMCPToolset instances (for MCP servers)
    - List of callable functions (for REST servers - legacy)

    Args:
        tool_refs: List of MCPToolReference or dicts with name/server
        local_mcp_server: Local FastMCP server instance (for "local" tools)
        context: Optional context for REST tool calls

    Returns:
        Tuple of (toolsets, legacy_tools)
        - toolsets: List of FastMCPToolset instances
        - legacy_tools: List of callable functions (for REST servers)
    """
    toolsets: list[Any] = []
    legacy_tools: list[Callable] = []

    # Group tools by server for efficient toolset creation
    tools_by_server: dict[str, set[str]] = {}

    for ref in tool_refs:
        # Duck typing: check for name attribute (supports MCPToolReference from any module)
        if hasattr(ref, "name"):
            tool_name = ref.name
            server_name = getattr(ref, "server", None) or "local"
        elif isinstance(ref, dict):
            tool_name = ref.get("name", "")
            server_name = ref.get("server") or ref.get("mcp_server") or "local"
        else:
            continue

        if not tool_name:
            continue

        if server_name not in tools_by_server:
            tools_by_server[server_name] = set()
        tools_by_server[server_name].add(tool_name)

    # Create toolsets for each server
    for server_name, tool_names in tools_by_server.items():
        if server_name == "local":
            # Use local FastMCP server instance
            if local_mcp_server:
                toolset = create_filtered_mcp_toolset(
                    local_mcp_server,
                    allowed_tools=tool_names,
                )
                toolsets.append(toolset)
            else:
                logger.warning("Local tools requested but no local_mcp_server provided")
        else:
            # Load from remote server
            server = await get_server(server_name)
            if not server:
                logger.warning(f"Server '{server_name}' not found")
                continue

            if not server.enabled:
                logger.warning(f"Server '{server_name}' is disabled")
                continue

            if server.server_type == "mcp" and server.endpoint:
                # MCP server - use FastMCPToolset
                toolset = create_filtered_mcp_toolset(
                    server.endpoint,
                    allowed_tools=tool_names,
                )
                toolsets.append(toolset)
                logger.debug(
                    f"Created FastMCPToolset for '{server_name}' with tools: {tool_names}"
                )

            elif server.server_type == "rest":
                # REST server - legacy wrapper (no schema)
                for tool_name in tool_names:
                    wrapper = create_rest_tool_wrapper(server, tool_name, context)
                    legacy_tools.append(wrapper)
                    logger.warning(
                        f"REST tool '{tool_name}' has no schema - LLM may not know params"
                    )

            elif server.server_type == "stdio":
                # stdio server - use FastMCPToolset with stdio transport
                try:
                    from fastmcp.client.transports import StdioTransport

                    command = server.config.get("command", [])
                    if command:
                        transport = StdioTransport(command=command)
                        toolset = create_filtered_mcp_toolset(
                            transport,
                            allowed_tools=tool_names,
                        )
                        toolsets.append(toolset)
                        logger.debug(
                            f"Created stdio FastMCPToolset for '{server_name}'"
                        )
                    else:
                        logger.warning(
                            f"stdio server '{server_name}' has no command configured"
                        )
                except ImportError:
                    logger.warning(
                        f"stdio transport not available for '{server_name}'"
                    )

            else:
                logger.warning(
                    f"Unknown server type '{server.server_type}' for '{server_name}'"
                )

    return toolsets, legacy_tools


# =============================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# =============================================================================
# These functions maintain backward compatibility with code that expects
# a list of callables instead of toolsets.
# =============================================================================


async def resolve_tools(
    tool_refs: list[MCPToolReference | dict],
    local_tools: dict[str, Any] | list | None = None,
    context: AgentContext | None = None,
) -> list[Callable]:
    """
    Legacy function - resolve tool references to callable tools.

    DEPRECATED: Use resolve_tools_as_toolsets() instead for better performance.

    This function maintains backward compatibility by extracting callables
    from local tools dict/list. For remote tools, it returns REST wrappers only
    (MCP tools should use the toolset approach).

    Args:
        tool_refs: List of MCPToolReference or dicts with name/server
        local_tools: Local tools dict or list (for "local" server)
        context: Optional context for remote tool calls

    Returns:
        List of callable async functions
    """
    resolved_tools: list[Callable] = []

    # Convert local_tools to dict if needed
    local_tools_dict: dict[str, Any] = {}
    if local_tools:
        if isinstance(local_tools, dict):
            local_tools_dict = local_tools
        elif isinstance(local_tools, list):
            for tool in local_tools:
                name = getattr(tool, "__name__", None) or getattr(tool, "name", None)
                if name:
                    local_tools_dict[name] = tool

    # Group tools by server
    tools_by_server: dict[str, list[str]] = {}

    for ref in tool_refs:
        # Duck typing: check for name attribute (supports MCPToolReference from any module)
        if hasattr(ref, "name"):
            tool_name = ref.name
            server_name = getattr(ref, "server", None) or "local"
        elif isinstance(ref, dict):
            tool_name = ref.get("name", "")
            server_name = ref.get("server") or ref.get("mcp_server") or "local"
        else:
            continue

        if not tool_name:
            continue

        if server_name not in tools_by_server:
            tools_by_server[server_name] = []
        tools_by_server[server_name].append(tool_name)

    # Resolve tools by server
    for server_name, tool_names in tools_by_server.items():
        if server_name == "local":
            # Use local tools (already have proper annotations via tool.fn)
            for tool_name in tool_names:
                if tool_name in local_tools_dict:
                    tool = local_tools_dict[tool_name]
                    if hasattr(tool, "fn"):
                        resolved_tools.append(tool.fn)
                    elif callable(tool):
                        resolved_tools.append(tool)
                else:
                    logger.warning(f"Local tool '{tool_name}' not found")
        else:
            # Load from remote server - only REST supported in legacy mode
            server = await get_server(server_name)
            if not server:
                logger.warning(f"Server '{server_name}' not found")
                continue

            if not server.enabled:
                logger.warning(f"Server '{server_name}' is disabled")
                continue

            if server.server_type == "rest":
                # REST server - legacy wrapper
                for tool_name in tool_names:
                    wrapper = create_rest_tool_wrapper(server, tool_name, context)
                    resolved_tools.append(wrapper)
            else:
                logger.warning(
                    f"Legacy resolve_tools() doesn't support '{server.server_type}' servers. "
                    f"Use resolve_tools_as_toolsets() for MCP servers."
                )

    return resolved_tools


async def resolve_tools_for_agent(
    tool_refs: list[MCPToolReference | dict],
    local_tools: dict[str, Any] | list | None = None,
    context: AgentContext | None = None,
    allowed_tool_names: set[str] | None = None,
) -> list[Callable]:
    """
    Legacy function - resolve tools for an agent, applying filtering.

    DEPRECATED: Use resolve_tools_as_toolsets() instead.

    Args:
        tool_refs: List of MCPToolReference or dicts
        local_tools: Local tools dict or list
        context: Optional context for remote tools
        allowed_tool_names: Optional set of allowed tool names (for filtering)

    Returns:
        List of callable async functions, filtered by allowed_tool_names
    """
    if not tool_refs and allowed_tool_names:
        tool_refs = [{"name": name, "server": "local"} for name in allowed_tool_names]

    if not allowed_tool_names:
        return await resolve_tools(tool_refs, local_tools, context)

    filtered_refs = []
    for ref in tool_refs:
        if isinstance(ref, MCPToolReference):
            name = ref.name
        elif isinstance(ref, dict):
            name = ref.get("name", "")
        else:
            continue

        if name in allowed_tool_names:
            filtered_refs.append(ref)

    return await resolve_tools(filtered_refs, local_tools, context)


# =============================================================================
# SIMPLE SCHEMA-BASED RESOLUTION
# =============================================================================


async def resolve_tools_from_schema(schema, mcp_endpoint: str | None = None) -> list | None:
    """
    Resolve toolsets from an AgentSchema.

    Simple helper that returns toolsets or None.

    Args:
        schema: AgentSchema with tools defined
        mcp_endpoint: Optional MCP endpoint URL. If not provided, tries local server
                      then falls back to localhost:8000 (for demonstration - pure local is supported)

    Returns:
        List of toolsets or None if no tools
    """
    if not schema.tools:
        return None

    from remlight.api.mcp_main import get_mcp_server

    local_server = get_mcp_server()

    # If no local server initialized, fall back to HTTP endpoint
    # This is for demonstration purposes - pure local MCP server is also supported
    if local_server is None and mcp_endpoint is None:
        mcp_endpoint = "http://localhost:8000/api/v1/mcp"
        logger.debug(f"No local MCP server, using endpoint: {mcp_endpoint}")

    toolsets, _ = await resolve_tools_as_toolsets(
        schema.tools,
        local_mcp_server=local_server or mcp_endpoint,
    )

    return toolsets if toolsets else None
