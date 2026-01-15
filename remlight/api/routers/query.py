"""Query router for REM dialect queries.

Provides direct query execution endpoints separate from chat.
"""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from remlight.api.routers.tools import search

router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    """REM query request."""
    query: str
    limit: int = 20
    user_id: str | None = None


class QueryResponse(BaseModel):
    """REM query response."""
    status: str
    query_type: str | None = None
    results: list[dict[str, Any]] | None = None
    result: dict[str, Any] | None = None
    count: int | None = None
    error: str | None = None


@router.post("")
async def execute_query(request: QueryRequest) -> dict[str, Any]:
    """
    Execute a REM query.

    Supports all REM query types:
    - LOOKUP <key>: O(1) exact entity lookup
    - FUZZY <text>: Fuzzy text matching
    - SEARCH <text> IN <table>: Semantic search
    - TRAVERSE <key> [DEPTH n]: Graph traversal

    Examples:
        {"query": "LOOKUP sarah-chen"}
        {"query": "FUZZY project alpha", "limit": 10}
        {"query": "SEARCH machine learning IN resources"}
        {"query": "TRAVERSE project-alpha DEPTH 2"}
    """
    return await search(
        query=request.query,
        limit=request.limit,
        user_id=request.user_id,
    )


@router.get("/health")
async def query_health() -> dict[str, str]:
    """Query service health check."""
    return {"status": "ok", "service": "query"}
