"""
REM Query Models

REM provides schema-agnostic query operations optimized for LLM-augmented
iterated retrieval. Unlike traditional SQL, REM queries work with natural
language labels instead of UUIDs and support multi-turn exploration.

Query Types (Performance Contract):
- LOOKUP: O(1) schema-agnostic entity resolution
- FUZZY: Indexed fuzzy text matching across all entities
- SEARCH: Indexed semantic vector search
- SQL: Direct table queries (provider dialect)
- TRAVERSE: Iterative O(1) lookups on graph edges
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    """REM query types with specific performance contracts."""

    LOOKUP = "LOOKUP"
    FUZZY = "FUZZY"
    SEARCH = "SEARCH"
    SQL = "SQL"
    TRAVERSE = "TRAVERSE"


class LookupParameters(BaseModel):
    """
    LOOKUP query parameters.

    Performance: O(1) per key
    Schema: Agnostic - No table name required
    """

    key: str | list[str] = Field(
        ..., description="Entity identifier(s) - single key or list of keys"
    )
    user_id: str | None = Field(default=None, description="Optional user ID filter")


class FuzzyParameters(BaseModel):
    """
    FUZZY query parameters.

    Performance: Indexed - Trigram index required
    Schema: Agnostic - Searches across all entity names
    """

    query_text: str = Field(..., description="Fuzzy search text")
    threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Similarity threshold"
    )
    limit: int = Field(default=10, gt=0, description="Maximum results")


class SearchParameters(BaseModel):
    """
    SEARCH query parameters.

    Performance: Indexed - Vector index required (IVFFlat, HNSW)
    Schema: Table-specific - Requires table name
    """

    query_text: str = Field(..., description="Semantic search query")
    table_name: str = Field(
        default="resources", description="Table to search (ontologies, resources, messages)"
    )
    limit: int = Field(default=10, gt=0, description="Maximum results")
    min_similarity: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum similarity score"
    )


class SQLParameters(BaseModel):
    """
    SQL query parameters.

    Performance: O(n) - Table scan with optional indexes
    Schema: Table-specific

    Supports two modes:
    1. Raw: raw_query (full SQL statement)
    2. Structured: table_name + where_clause + order_by + limit
    """

    raw_query: str | None = Field(
        default=None, description="Raw SQL query (e.g., SELECT * FROM resources WHERE...)"
    )
    table_name: str | None = Field(default=None, description="Table to query (structured mode)")
    where_clause: str | None = Field(default=None, description="SQL WHERE clause")
    order_by: str | None = Field(default=None, description="SQL ORDER BY clause")
    limit: int | None = Field(default=None, description="SQL LIMIT")


class TraverseParameters(BaseModel):
    """
    TRAVERSE query parameters.

    Performance: O(k) where k = number of keys traversed
    Schema: Agnostic - Follows graph edges across tables

    Depth Modes:
    - 0: PLAN mode (analyze edges without traversal)
    - 1: Single-hop traversal (default)
    - N: Multi-hop traversal
    """

    initial_query: str = Field(
        ..., description="Initial query to find entry nodes (entity key or LOOKUP key)"
    )
    edge_types: list[str] | None = Field(
        default=None,
        description="Edge types to follow (e.g., ['manages', 'reports_to']). None = all",
    )
    max_depth: int = Field(
        default=1, ge=0, description="Maximum traversal depth. 0 = PLAN mode"
    )
    limit: int = Field(default=10, gt=0, description="Maximum nodes to return")


QueryParameters = (
    LookupParameters
    | FuzzyParameters
    | SearchParameters
    | SQLParameters
    | TraverseParameters
)


class RemQuery(BaseModel):
    """
    REM query plan combining query type with parameters.
    """

    query_type: QueryType = Field(..., description="REM query type")
    parameters: QueryParameters = Field(..., description="Query parameters")
    user_id: str | None = Field(default=None, description="User identifier for data isolation")


class RemQueryResult(BaseModel):
    """Result from a REM query execution."""

    query_type: QueryType
    data: list[dict[str, Any]] = Field(default_factory=list)
    count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
