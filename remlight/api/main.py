"""REMLight FastAPI Server.

Clean FastAPI application that mounts:
- /api/v1/chat - Chat completions (OpenAI-compatible)
- /api/v1/query - REM query execution
- /api/v1/tools - Tool REST endpoints
- /api/v1/mcp - MCP server HTTP endpoint

Architecture:
```
main.py (FastAPI app)
    ├── routers/
    │   ├── chat.py      - OpenAI-compatible chat completions
    │   ├── query.py     - REM query endpoint
    │   └── tools.py     - Tool functions (shared with MCP)
    │
    └── mcp_main.py      - MCP server (mounted at /api/v1/mcp)
```
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from remlight import __version__
from remlight.services.database import get_db

# Import routers
from remlight.api.routers.chat import router as chat_router
from remlight.api.routers.query import router as query_router
from remlight.api.routers.tools import router as tools_router, init_tools

# Import MCP server
from remlight.api.mcp_main import get_mcp_server, init_mcp


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # Connect to database
    db = get_db()
    await db.connect()

    # Initialize tools and MCP with database
    init_tools(db)
    init_mcp(db)

    yield

    # Cleanup
    await db.disconnect()


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="REMLight API",
        version=__version__,
        description="Minimal declarative agent framework with PostgreSQL memory",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount MCP server at /api/v1/mcp
    mcp = get_mcp_server()
    mcp_http = mcp.http_app(path="/", transport="http")
    app.mount("/api/v1/mcp", mcp_http)

    # Include routers with /api/v1 prefix
    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")
    app.include_router(tools_router, prefix="/api/v1")

    # Health check
    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    # Root info
    @app.get("/")
    async def root() -> dict[str, Any]:
        """API information endpoint."""
        return {
            "name": "REMLight API",
            "version": __version__,
            "endpoints": {
                "health": "/health",
                "chat": "/api/v1/chat/completions",
                "query": "/api/v1/query",
                "tools": "/api/v1/tools",
                "mcp": "/api/v1/mcp",
                "docs": "/docs",
            },
        }

    return app


# Create default app instance
app = create_app()
