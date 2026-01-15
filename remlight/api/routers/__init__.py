"""API routers for REMLight.

This module contains FastAPI routers and tool definitions.
Tools are defined as router-style endpoints that can be used by both
the REST API and the MCP server.

Routers:
- chat_router: OpenAI-compatible chat completions
- query_router: REM query execution
- tools_router: Tool REST endpoints

Tool Functions (can be called directly or via MCP):
- search: Execute REM queries
- action: Emit typed action events for SSE streaming
- ask_agent: Multi-agent orchestration
"""

from remlight.api.routers.chat import router as chat_router
from remlight.api.routers.query import router as query_router
from remlight.api.routers.tools import router as tools_router
from remlight.api.routers.tools import (
    search,
    action,
    ask_agent,
    get_metadata,
    clear_metadata,
    init_tools,
    get_agent_schema,
    get_user_profile,
    get_user_profile_hint,
    format_user_profile,
)

__all__ = [
    # Routers
    "chat_router",
    "query_router",
    "tools_router",
    # Tool functions
    "search",
    "action",
    "ask_agent",
    "get_metadata",
    "clear_metadata",
    "init_tools",
    "get_agent_schema",
    "get_user_profile",
    "get_user_profile_hint",
    "format_user_profile",
]
