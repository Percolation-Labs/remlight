"""
Tool Resolver - Load tools from local and remote MCP servers
=============================================================

This module handles resolving tool references in agent schemas to actual
callable functions with PROPER TYPE ANNOTATIONS for PydanticAI.

KEY INSIGHT: PydanticAI requires properly-annotated functions
--------------------------------------------------------------
PydanticAI extracts tool schemas from function signatures and docstrings.
Remote tools must be wrapped in functions with proper type annotations,
otherwise the LLM receives an empty schema and doesn't know what params to use.

SUPPORTED SERVER TYPES
----------------------
- Local tools: Built-in tools from the project's MCP server (tool.fn)
- MCP servers: Remote MCP servers accessed via FastMCP Client
- REST servers: Remote REST APIs (legacy, no schema introspection)

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

2. resolve_tools() is called with the tool references

3. For each tool:
   a. If server is "local" or None: Use local tool.fn (already annotated)
   b. If server is MCP type: Connect via FastMCP Client, fetch schema, build annotated wrapper
   c. If server is REST type: Create wrapper (no schema - legacy)

4. Return list of properly-annotated callables for PydanticAI

ANNOTATED WRAPPER CREATION
--------------------------
For remote MCP tools, we:
1. Connect to the MCP server via FastMCP Client
2. Call list_tools() to get tool schemas (name, description, inputSchema)
3. Build a Python function with proper type annotations from the JSON Schema
4. The wrapper calls client.call_tool() to execute the remote tool

This ensures PydanticAI generates the correct JSON Schema for the LLM.
"""

from typing import Any, Callable
import httpx
from loguru import logger

from remlight.agentic.schema import MCPToolReference
from remlight.agentic.context import AgentContext
from remlight.models.entities import Server
from remlight.services.repository import Repository


# Cache for resolved servers
_server_cache: dict[str, Server] = {}

# Cache for MCP tool schemas: {server_endpoint: {tool_name: MCPToolInfo}}
_mcp_schema_cache: dict[str, dict[str, Any]] = {}


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
    _mcp_schema_cache.clear()


# =============================================================================
# MCP TOOL SCHEMA FETCHING
# =============================================================================


async def fetch_mcp_tool_schemas(endpoint: str) -> dict[str, Any]:
    """
    Fetch tool schemas from an MCP server via FastMCP Client.

    Args:
        endpoint: MCP server URL (e.g., "http://localhost:8001/mcp")

    Returns:
        Dict mapping tool names to their schema info:
        {
            "search": {
                "name": "search",
                "description": "Search the knowledge base...",
                "inputSchema": {"properties": {...}, "required": [...]}
            }
        }
    """
    if endpoint in _mcp_schema_cache:
        return _mcp_schema_cache[endpoint]

    try:
        from fastmcp import Client

        async with Client(endpoint) as client:
            tools = await client.list_tools()
            schemas = {}
            for tool in tools:
                schemas[tool.name] = {
                    "name": tool.name,
                    "description": tool.description or f"Remote tool: {tool.name}",
                    "inputSchema": tool.inputSchema or {"type": "object", "properties": {}},
                }
            _mcp_schema_cache[endpoint] = schemas
            logger.debug(f"Fetched {len(schemas)} tool schemas from {endpoint}")
            return schemas

    except Exception as e:
        logger.error(f"Failed to fetch MCP tool schemas from {endpoint}: {e}")
        return {}


# =============================================================================
# ANNOTATED WRAPPER CREATION
# =============================================================================


def create_annotated_mcp_wrapper(
    tool_name: str,
    description: str,
    input_schema: dict[str, Any],
    endpoint: str,
    context: AgentContext | None = None,
) -> Callable:
    """
    Create a properly-annotated wrapper for a remote MCP tool.

    This function dynamically creates a Python function with:
    - Proper parameter names and types from the JSON Schema
    - Docstring with description and Args section
    - Type annotations that PydanticAI can parse

    Args:
        tool_name: Name of the tool
        description: Tool description (for docstring)
        input_schema: JSON Schema for the tool's input parameters
        endpoint: MCP server endpoint URL
        context: Optional context for user/session info

    Returns:
        Async callable with proper annotations for PydanticAI
    """
    # Map JSON Schema types to Python types
    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }

    # Parse JSON Schema properties
    props = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    # Build parameter strings and docstring args
    param_strs = []
    docstring_args = []

    for name, prop in props.items():
        prop_type = prop.get("type", "string")
        py_type = type_map.get(prop_type, "Any")

        # Handle nullable (anyOf with null)
        any_of = prop.get("anyOf", [])
        is_nullable = any(t.get("type") == "null" for t in any_of)
        if is_nullable:
            # Get the non-null type
            for t in any_of:
                if t.get("type") != "null":
                    py_type = type_map.get(t.get("type", "string"), "Any")
                    break
            py_type = f"{py_type} | None"

        # Build parameter string
        if name in required:
            param_strs.append(f"{name}: {py_type}")
        else:
            default = prop.get("default")
            default_repr = repr(default)
            param_strs.append(f"{name}: {py_type} = {default_repr}")

        # Build docstring arg
        param_desc = prop.get("description", "")
        if param_desc:
            docstring_args.append(f"        {name}: {param_desc}")

    # Build full docstring
    docstring_parts = [description]
    if docstring_args:
        docstring_parts.append("\n    Args:")
        docstring_parts.extend(docstring_args)

    full_docstring = "\n".join(docstring_parts)

    # Build the wrapper function code
    params_str = ", ".join(param_strs) if param_strs else ""

    # Create unique function to avoid closure issues
    func_code = f'''
async def {tool_name}({params_str}) -> dict:
    """
    {full_docstring}
    """
    # Collect all arguments
    import inspect
    frame = inspect.currentframe()
    args = {{k: v for k, v in frame.f_locals.items() if k not in ("inspect", "frame")}}

    # Call the remote MCP tool
    from fastmcp import Client
    async with Client("{endpoint}") as client:
        result = await client.call_tool("{tool_name}", args)
        # Extract content from MCP response
        if hasattr(result, "content") and result.content:
            content = result.content[0]
            if hasattr(content, "text"):
                import json
                try:
                    return json.loads(content.text)
                except:
                    return {{"result": content.text}}
        return {{"result": str(result)}}
'''

    # Execute to create the function
    exec_globals = {"__builtins__": __builtins__}
    exec(func_code, exec_globals)
    func = exec_globals[tool_name]

    return func


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


async def resolve_tools(
    tool_refs: list[MCPToolReference | dict],
    local_tools: dict[str, Any] | list | None = None,
    context: AgentContext | None = None,
) -> list[Callable]:
    """
    Resolve tool references to properly-annotated callable tools.

    This is the main entry point for tool resolution. It takes tool references
    from an agent schema and returns callable functions with proper type
    annotations for PydanticAI.

    Args:
        tool_refs: List of MCPToolReference or dicts with name/server
        local_tools: Local tools dict or list (for "local" server)
        context: Optional context for remote tool calls

    Returns:
        List of callable async functions with proper annotations
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

    # Group tools by server for efficient schema fetching
    tools_by_server: dict[str, list[str]] = {}

    for ref in tool_refs:
        if isinstance(ref, MCPToolReference):
            tool_name = ref.name
            server_name = ref.server or "local"
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
            # Load from remote server
            server = await get_server(server_name)
            if not server:
                logger.warning(f"Server '{server_name}' not found")
                continue

            if not server.enabled:
                logger.warning(f"Server '{server_name}' is disabled")
                continue

            if server.server_type == "mcp" and server.endpoint:
                # MCP server - fetch schemas and create annotated wrappers
                schemas = await fetch_mcp_tool_schemas(server.endpoint)

                for tool_name in tool_names:
                    if tool_name in schemas:
                        schema_info = schemas[tool_name]
                        wrapper = create_annotated_mcp_wrapper(
                            tool_name=tool_name,
                            description=schema_info["description"],
                            input_schema=schema_info["inputSchema"],
                            endpoint=server.endpoint,
                            context=context,
                        )
                        resolved_tools.append(wrapper)
                    else:
                        logger.warning(
                            f"Tool '{tool_name}' not found on MCP server '{server_name}'"
                        )

            elif server.server_type == "rest":
                # REST server - legacy wrapper (no schema)
                for tool_name in tool_names:
                    wrapper = create_rest_tool_wrapper(server, tool_name, context)
                    resolved_tools.append(wrapper)
                    logger.warning(
                        f"REST tool '{tool_name}' has no schema - LLM may not know params"
                    )

            elif server.server_type == "stdio":
                logger.warning(f"stdio server type not yet implemented for '{server_name}'")

            else:
                logger.warning(f"Unknown server type '{server.server_type}' for '{server_name}'")

    return resolved_tools


async def resolve_tools_for_agent(
    tool_refs: list[MCPToolReference | dict],
    local_tools: dict[str, Any] | list | None = None,
    context: AgentContext | None = None,
    allowed_tool_names: set[str] | None = None,
) -> list[Callable]:
    """
    Resolve tools for an agent, applying filtering.

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
