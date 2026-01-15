"""
REM Service

High-level service for executing REM queries. Combines the parser,
database functions, and embedding generation.
"""

from typing import Any

from remlight.models.rem_query import (
    QueryType,
    LookupParameters,
    FuzzyParameters,
    SearchParameters,
    SQLParameters,
    TraverseParameters,
    RemQuery,
    RemQueryResult,
)
from remlight.services.database import DatabaseService
from remlight.services.rem.parser import RemQueryParser


class RemService:
    """
    REM query execution service.

    Provides both string-based and structured query interfaces.
    """

    # Allowed SQL keywords for safety
    ALLOWED_SQL_PREFIXES = ("SELECT", "WITH")
    BLOCKED_SQL_PREFIXES = ("DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE")

    def __init__(self, db: DatabaseService, embed_fn=None):
        """
        Initialize RemService.

        Args:
            db: Database service instance
            embed_fn: Optional async function to generate embeddings
                      Signature: async def embed_fn(text: str) -> list[float]
        """
        self.db = db
        self.parser = RemQueryParser()
        self.embed_fn = embed_fn

    async def execute(self, query_string: str, user_id: str | None = None) -> RemQueryResult:
        """
        Execute a REM query from a string.

        Args:
            query_string: REM query string (e.g., 'LOOKUP "sarah-chen"')
            user_id: Optional user ID for data isolation

        Returns:
            RemQueryResult with data and metadata
        """
        query_type, params = self.parser.parse(query_string)
        return await self._execute_query(query_type, params, user_id)

    async def execute_query(self, query: RemQuery) -> RemQueryResult:
        """
        Execute a structured REM query.

        Args:
            query: RemQuery model with type and parameters

        Returns:
            RemQueryResult with data and metadata
        """
        params = query.parameters.model_dump()
        return await self._execute_query(query.query_type, params, query.user_id)

    async def _execute_query(
        self, query_type: QueryType, params: dict[str, Any], user_id: str | None
    ) -> RemQueryResult:
        """Execute query based on type."""
        if query_type == QueryType.LOOKUP:
            return await self._execute_lookup(params, user_id)
        elif query_type == QueryType.FUZZY:
            return await self._execute_fuzzy(params, user_id)
        elif query_type == QueryType.SEARCH:
            return await self._execute_search(params, user_id)
        elif query_type == QueryType.SQL:
            return await self._execute_sql(params, user_id)
        elif query_type == QueryType.TRAVERSE:
            return await self._execute_traverse(params, user_id)
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    async def _execute_lookup(
        self, params: dict[str, Any], user_id: str | None
    ) -> RemQueryResult:
        """Execute LOOKUP query."""
        key = params.get("key")
        if not key:
            raise ValueError("LOOKUP requires 'key' parameter")

        # Handle single key or list of keys
        keys = [key] if isinstance(key, str) else key
        results = []

        for k in keys:
            data = await self.db.rem_lookup(k, user_id)
            # rem_lookup returns JSONB which may be empty dict {} or string '{}'
            if data and data != {} and data != '{}':
                # Ensure it's a dict
                if isinstance(data, str):
                    import json
                    data = json.loads(data)
                results.append(data)

        return RemQueryResult(
            query_type=QueryType.LOOKUP,
            data=results,
            count=len(results),
            metadata={"keys": keys},
        )

    async def _execute_fuzzy(
        self, params: dict[str, Any], user_id: str | None
    ) -> RemQueryResult:
        """Execute FUZZY query."""
        query_text = params.get("query_text")
        if not query_text:
            raise ValueError("FUZZY requires 'query_text' parameter")

        threshold = params.get("threshold", 0.3)
        limit = params.get("limit", 10)

        rows = await self.db.rem_fuzzy(query_text, user_id, threshold, limit)

        return RemQueryResult(
            query_type=QueryType.FUZZY,
            data=rows,
            count=len(rows),
            metadata={"query_text": query_text, "threshold": threshold},
        )

    async def _execute_search(
        self, params: dict[str, Any], user_id: str | None
    ) -> RemQueryResult:
        """Execute SEARCH query (requires embedding function)."""
        query_text = params.get("query_text")
        if not query_text:
            raise ValueError("SEARCH requires 'query_text' parameter")

        table_name = params.get("table_name", "resources")
        limit = params.get("limit", 10)
        min_similarity = params.get("min_similarity", 0.3)

        # Generate embedding if embed_fn is available
        if self.embed_fn:
            embedding = await self.embed_fn(query_text)
            rows = await self.db.rem_search(
                embedding, table_name, limit, min_similarity, user_id
            )
        else:
            # Fallback to fuzzy search if no embedding function
            rows = await self.db.rem_fuzzy(query_text, user_id, 0.2, limit)

        return RemQueryResult(
            query_type=QueryType.SEARCH,
            data=rows,
            count=len(rows),
            metadata={
                "query_text": query_text,
                "table_name": table_name,
                "used_embedding": self.embed_fn is not None,
            },
        )

    async def _execute_sql(
        self, params: dict[str, Any], user_id: str | None
    ) -> RemQueryResult:
        """Execute SQL query with safety checks."""
        raw_query = params.get("raw_query")

        if raw_query:
            # Safety check
            query_upper = raw_query.strip().upper()
            if any(query_upper.startswith(p) for p in self.BLOCKED_SQL_PREFIXES):
                raise ValueError(f"Blocked SQL operation. Only SELECT/WITH allowed.")

            if not any(query_upper.startswith(p) for p in self.ALLOWED_SQL_PREFIXES):
                raise ValueError(f"Only SELECT and WITH queries are allowed.")

            # Add user_id filter if present (basic injection into WHERE)
            # Note: For production, use parameterized queries
            rows = await self.db.fetch(raw_query)
        else:
            # Structured mode
            table_name = params.get("table_name")
            if not table_name:
                raise ValueError("SQL requires either 'raw_query' or 'table_name'")

            where_clause = params.get("where_clause", "1=1")
            order_by = params.get("order_by", "created_at DESC")
            limit = params.get("limit", 100)

            query = f"""
                SELECT * FROM {table_name}
                WHERE {where_clause} AND deleted_at IS NULL
                ORDER BY {order_by}
                LIMIT {limit}
            """
            rows = await self.db.fetch(query)

        return RemQueryResult(
            query_type=QueryType.SQL,
            data=rows,
            count=len(rows),
            metadata={"raw_query": raw_query or "structured"},
        )

    async def _execute_traverse(
        self, params: dict[str, Any], user_id: str | None
    ) -> RemQueryResult:
        """Execute TRAVERSE query."""
        initial_query = params.get("initial_query")
        if not initial_query:
            raise ValueError("TRAVERSE requires 'initial_query' parameter")

        edge_types = params.get("edge_types")
        max_depth = params.get("max_depth", 1)

        rows = await self.db.rem_traverse(
            initial_query, edge_types, max_depth, user_id
        )

        return RemQueryResult(
            query_type=QueryType.TRAVERSE,
            data=rows,
            count=len(rows),
            metadata={
                "initial_query": initial_query,
                "edge_types": edge_types,
                "max_depth": max_depth,
            },
        )

    # Convenience methods for direct query type access

    async def lookup(self, key: str | list[str], user_id: str | None = None) -> list[dict]:
        """Direct LOOKUP query."""
        result = await self._execute_lookup({"key": key}, user_id)
        return result.data

    async def fuzzy(
        self,
        query_text: str,
        threshold: float = 0.3,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[dict]:
        """Direct FUZZY query."""
        result = await self._execute_fuzzy(
            {"query_text": query_text, "threshold": threshold, "limit": limit},
            user_id,
        )
        return result.data

    async def search(
        self,
        query_text: str,
        table_name: str = "resources",
        limit: int = 10,
        min_similarity: float = 0.3,
        user_id: str | None = None,
    ) -> list[dict]:
        """Direct SEARCH query."""
        result = await self._execute_search(
            {
                "query_text": query_text,
                "table_name": table_name,
                "limit": limit,
                "min_similarity": min_similarity,
            },
            user_id,
        )
        return result.data

    async def traverse(
        self,
        entity_key: str,
        edge_types: list[str] | None = None,
        max_depth: int = 1,
        user_id: str | None = None,
    ) -> list[dict]:
        """Direct TRAVERSE query."""
        result = await self._execute_traverse(
            {
                "initial_query": entity_key,
                "edge_types": edge_types,
                "max_depth": max_depth,
            },
            user_id,
        )
        return result.data
