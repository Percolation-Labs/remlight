"""DEPRECATED: Use remlight.api.mcp_main instead.

This module is kept for backward compatibility only.
All functionality has been migrated to remlight.api.mcp_main and remlight.api.routers.tools.
"""

import warnings

warnings.warn(
    "remlight.mcp_server is deprecated. Use remlight.api.mcp_main instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from remlight.api.mcp_main import (
    create_mcp_server,
    get_mcp_server,
    get_mcp_tools,
    init_mcp as init_services,
    mcp,
)

from remlight.api.routers.tools import (
    search,
    action,
    ask_agent,
    get_metadata,
    clear_metadata,
    get_agent_schema,
    get_user_profile,
    get_user_profile_hint,
    format_user_profile,
    register_agent_schema,
)

__all__ = [
    "create_mcp_server",
    "get_mcp_server",
    "get_mcp_tools",
    "init_services",
    "mcp",
    "search",
    "action",
    "ask_agent",
    "get_metadata",
    "clear_metadata",
    "get_agent_schema",
    "get_user_profile",
    "get_user_profile_hint",
    "format_user_profile",
    "register_agent_schema",
]
