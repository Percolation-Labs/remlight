"""
Tool and Server Registration Utility
====================================

This module provides utilities for registering MCP tool servers and their
tools into the database. It supports:

- Auto-registration on API startup (disabled by default)
- MD5 hash-based change detection to avoid unnecessary embedding regeneration
- Deterministic IDs from URIs for idempotent upserts
- Project tool discovery from the MCP server module

DETERMINISTIC IDS
-----------------
IDs are generated deterministically from URIs to enable idempotent upserts:
- Servers: UUID from hash of endpoint URL (or name for local servers)
- Tools: UUID from hash of server_id + tool name

This ensures the same server/tool always gets the same ID, making
upserts work correctly without needing ON CONFLICT clauses.

CHANGE DETECTION
----------------
Each server and tool stores a description_hash in its metadata. On registration:
1. Compute MD5 hash of current description
2. Compare with stored hash
3. Skip update if unchanged (saves embedding API costs)
4. Update and regenerate embedding if changed

USAGE
-----
    # Manual registration
    stats = await register_project_tools(force=False)
    print(f"Registered {stats['servers_registered']} servers, {stats['tools_registered']} tools")

    # In API startup (if TOOLS__AUTO_REGISTER=true)
    if settings.tools.auto_register:
        await register_project_tools()
"""

import hashlib
from typing import Any
from uuid import UUID

from loguru import logger

from remlight.models.entities import Server, Tool
from remlight.services.repository import Repository


def generate_server_id(endpoint: str | None, name: str) -> UUID:
    """
    Generate a deterministic UUID for a server from its endpoint or name.

    For remote servers, the ID is derived from the endpoint URL.
    For local servers (no endpoint), the ID is derived from the name.

    Args:
        endpoint: Server endpoint URL (for remote servers)
        name: Server name (used if no endpoint)

    Returns:
        Deterministic UUID based on the server identity
    """
    # Use endpoint if available, otherwise use name
    identity = endpoint or f"local://{name}"
    # Create a UUID from the hash (using UUID5-style approach with fixed namespace)
    hash_bytes = hashlib.sha256(identity.encode()).digest()[:16]
    return UUID(bytes=hash_bytes)


def generate_tool_id(server_id: UUID | str, tool_name: str) -> UUID:
    """
    Generate a deterministic UUID for a tool from its server and name.

    The tool ID is derived from the combination of server_id and tool_name,
    ensuring uniqueness within a server while being deterministic.

    Args:
        server_id: The parent server's UUID
        tool_name: The tool's name

    Returns:
        Deterministic UUID based on the tool identity
    """
    identity = f"{server_id}:{tool_name}"
    hash_bytes = hashlib.sha256(identity.encode()).digest()[:16]
    return UUID(bytes=hash_bytes)


def generate_agent_id(registry_uri: str | None, agent_name: str) -> UUID:
    """
    Generate a deterministic UUID for an agent from its registry and name.

    For local agents (no registry), the ID is derived from "local" + name.
    For remote agents, the ID is derived from registry_uri + name.

    Args:
        registry_uri: Registry URI (None = "local")
        agent_name: Agent name

    Returns:
        Deterministic UUID based on the agent identity
    """
    registry = registry_uri or "local"
    identity = f"{registry}:{agent_name}"
    hash_bytes = hashlib.sha256(identity.encode()).digest()[:16]
    return UUID(bytes=hash_bytes)


def compute_description_hash(description: str | None) -> str:
    """
    Compute MD5 hash of description for change detection.

    This enables skipping re-registration when descriptions haven't changed,
    avoiding unnecessary embedding regeneration costs.

    Args:
        description: The description text to hash (None treated as empty)

    Returns:
        MD5 hex digest of the description
    """
    content = description or ""
    return hashlib.md5(content.encode()).hexdigest()


async def register_server(
    server: Server,
    force: bool = False,
    generate_embeddings: bool = True,
) -> tuple[bool, bool]:
    """
    Register or update a server in the database.

    Uses deterministic ID generation from endpoint/name for idempotent upserts.
    Uses MD5 hash to detect description changes and skip unnecessary updates.

    Args:
        server: Server model to register
        force: If True, update even if unchanged
        generate_embeddings: If True, generate embeddings for description

    Returns:
        Tuple of (was_registered, was_created)
        - was_registered: True if server was inserted/updated
        - was_created: True if new server (vs update)
    """
    repo = Repository(Server, table_name="servers")

    # Generate deterministic ID from endpoint or name
    server.id = generate_server_id(server.endpoint, server.name)

    # Check if server exists by ID
    existing = await repo.get_by_id(str(server.id))

    # Compute description hash
    new_hash = compute_description_hash(server.description)

    if existing:
        # Check if description changed
        existing_hash = existing.metadata.get("description_hash", "")

        if not force and new_hash == existing_hash:
            logger.debug(f"Server '{server.name}' unchanged, skipping")
            return (False, False)

        # Update existing server - preserve existing metadata, add new hash
        server.metadata = {**existing.metadata, "description_hash": new_hash}

    else:
        # New server
        server.metadata = {**server.metadata, "description_hash": new_hash}

    # Upsert by ID (always works because ID is deterministic)
    await repo.upsert(server, generate_embeddings=generate_embeddings)

    logger.info(f"{'Updated' if existing else 'Registered'} server: {server.name}")
    return (True, existing is None)


async def register_tool(
    tool: Tool,
    server_id: UUID | str,
    force: bool = False,
    generate_embeddings: bool = True,
) -> tuple[bool, bool]:
    """
    Register or update a tool in the database.

    Uses deterministic ID generation from server_id + tool name for idempotent upserts.
    Uses MD5 hash to detect description changes and skip unnecessary updates.

    Args:
        tool: Tool model to register
        server_id: ID of the parent server
        force: If True, update even if unchanged
        generate_embeddings: If True, generate embeddings for description

    Returns:
        Tuple of (was_registered, was_created)
    """
    tool_repo = Repository(Tool, table_name="tools")

    # Set server_id and generate deterministic tool ID
    tool.server_id = server_id
    tool.id = generate_tool_id(server_id, tool.name)

    # Check if tool exists by ID
    existing = await tool_repo.get_by_id(str(tool.id))

    # Compute description hash
    new_hash = compute_description_hash(tool.description)

    if existing:
        existing_hash = existing.metadata.get("description_hash", "")

        if not force and new_hash == existing_hash:
            logger.debug(f"Tool '{tool.name}' unchanged, skipping")
            return (False, False)

        # Update existing tool - preserve existing metadata, add new hash
        tool.metadata = {**existing.metadata, "description_hash": new_hash}
    else:
        tool.metadata = {**tool.metadata, "description_hash": new_hash}

    # Upsert by ID (always works because ID is deterministic)
    await tool_repo.upsert(tool, generate_embeddings=generate_embeddings)

    logger.info(f"{'Updated' if existing else 'Registered'} tool: {tool.name}")
    return (True, existing is None)


async def get_project_servers_and_tools() -> tuple[list[Server], dict[str, list[Tool]]]:
    """
    Discover servers and tools from the project's MCP server module.

    This introspects the remlight.api.mcp_main module to find registered
    tools and creates Server/Tool models for them.

    Returns:
        Tuple of (servers, tools_by_server)
        - servers: List of Server models
        - tools_by_server: Dict mapping server name to list of Tool models
    """
    # Import MCP main to get registered tools
    from remlight.api.mcp_main import get_mcp_tools

    # Create the local MCP server with deterministic ID
    local_server = Server(
        name="local",
        description="Built-in REMLight MCP server with search, action, and agent tools. "
                    "Provides semantic search over knowledge bases, structured action emission, "
                    "and multi-agent delegation capabilities.",
        server_type="mcp",
        icon="🏠",
        tags=["builtin", "mcp", "local"],
    )
    # Set deterministic ID
    local_server.id = generate_server_id(None, "local")

    # Get tools from FastMCP via the proper API
    tools: list[Tool] = []
    try:
        mcp_tools = await get_mcp_tools()

        # mcp_tools is a dict of {name: FunctionTool}
        if isinstance(mcp_tools, dict):
            for name, tool_info in mcp_tools.items():
                # Extract description from the tool function
                func = tool_info.fn if hasattr(tool_info, 'fn') else None
                description = func.__doc__ if func and func.__doc__ else f"MCP tool: {name}"

                # Extract input schema if available
                input_schema: dict[str, Any] = {}
                if hasattr(tool_info, 'parameters'):
                    input_schema = tool_info.parameters
                elif hasattr(tool_info, 'inputSchema'):
                    input_schema = tool_info.inputSchema

                tools.append(Tool(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    icon=_get_tool_icon(name),
                    tags=_get_tool_tags(name),
                ))
    except Exception as e:
        logger.warning(f"Failed to discover tools from MCP server: {e}")

    return ([local_server], {"local": tools})


def _get_tool_icon(name: str) -> str:
    """Get icon for a tool based on its name."""
    icons = {
        "search": "🔍",
        "action": "⚡",
        "ask_agent": "🤖",
        "lookup": "📖",
        "fetch": "🌐",
        "read": "📄",
        "write": "✏️",
    }
    return icons.get(name, "🔧")


def _get_tool_tags(name: str) -> list[str]:
    """Get tags for a tool based on its name."""
    tags_map = {
        "search": ["search", "knowledge", "semantic"],
        "action": ["action", "metadata", "events"],
        "ask_agent": ["agent", "delegation", "multi-agent"],
        "lookup": ["lookup", "key-value", "fast"],
    }
    return tags_map.get(name, ["tool"])


async def register_project_tools(
    force: bool = False,
    generate_embeddings: bool = True,
) -> dict[str, int]:
    """
    Register all project tools and servers in the database.

    This is the main entry point for tool registration. It:
    1. Discovers servers and tools from the MCP server module
    2. Registers servers with MD5 hash change detection
    3. Registers tools with MD5 hash change detection

    Args:
        force: If True, re-register even if unchanged
        generate_embeddings: If True, generate embeddings for descriptions

    Returns:
        Stats dict with keys:
        - servers_registered: Number of servers created/updated
        - servers_created: Number of new servers
        - tools_registered: Number of tools created/updated
        - tools_created: Number of new tools
        - skipped: Number of unchanged items skipped
    """
    stats = {
        "servers_registered": 0,
        "servers_created": 0,
        "tools_registered": 0,
        "tools_created": 0,
        "skipped": 0,
    }

    servers, tools_by_server = await get_project_servers_and_tools()

    # Register servers
    for server in servers:
        was_registered, was_created = await register_server(
            server, force=force, generate_embeddings=generate_embeddings
        )
        if was_registered:
            stats["servers_registered"] += 1
            if was_created:
                stats["servers_created"] += 1
        else:
            stats["skipped"] += 1

    # Build server name -> id mapping
    server_ids = {s.name: s.id for s in servers}

    # Register tools
    for server_name, tools in tools_by_server.items():
        server_id = server_ids.get(server_name)
        if not server_id:
            logger.warning(f"Server '{server_name}' not found, skipping its tools")
            continue

        for tool in tools:
            was_registered, was_created = await register_tool(
                tool, server_id, force=force, generate_embeddings=generate_embeddings
            )
            if was_registered:
                stats["tools_registered"] += 1
                if was_created:
                    stats["tools_created"] += 1
            else:
                stats["skipped"] += 1

    logger.info(
        f"Registration complete: "
        f"{stats['servers_registered']} servers ({stats['servers_created']} new), "
        f"{stats['tools_registered']} tools ({stats['tools_created']} new), "
        f"{stats['skipped']} skipped"
    )

    return stats
