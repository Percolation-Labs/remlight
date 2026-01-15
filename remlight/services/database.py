"""Database service with connection pooling and REM functions."""

from typing import Any
from pathlib import Path

import asyncpg

from remlight.settings import settings


class DatabaseService:
    """PostgreSQL database service with REM query functions."""

    def __init__(self, connection_string: str | None = None):
        self.connection_string = connection_string or settings.postgres.connection_string
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Create connection pool."""
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=2,
                max_size=10,
            )

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None

    async def execute(self, query: str, *args) -> str:
        """Execute a query."""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> list[dict]:
        """Fetch rows as dicts."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetchrow(self, query: str, *args) -> dict | None:
        """Fetch single row as dict."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch single value."""
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    # REM Functions

    async def rem_lookup(self, key: str, user_id: str | None = None) -> dict:
        """O(1) KV store lookup."""
        result = await self.fetchval(
            "SELECT rem_lookup($1, $2)", key, user_id
        )
        return result or {}

    async def rem_search(
        self,
        embedding: list[float],
        table_name: str,
        limit: int = 10,
        min_similarity: float = 0.3,
        user_id: str | None = None,
    ) -> list[dict]:
        """Semantic vector search."""
        rows = await self.fetch(
            """
            SELECT * FROM rem_search($1::vector, $2, $3, $4, $5)
            """,
            embedding,
            table_name,
            limit,
            min_similarity,
            user_id,
        )
        return rows

    async def rem_fuzzy(
        self,
        query_text: str,
        user_id: str | None = None,
        threshold: float = 0.3,
        limit: int = 20,
    ) -> list[dict]:
        """Fuzzy text search."""
        rows = await self.fetch(
            "SELECT * FROM rem_fuzzy($1, $2, $3, $4)",
            query_text,
            user_id,
            threshold,
            limit,
        )
        return rows

    async def rem_traverse(
        self,
        entity_key: str,
        edge_types: list[str] | None = None,
        max_depth: int = 1,
        user_id: str | None = None,
    ) -> list[dict]:
        """Graph traversal."""
        rows = await self.fetch(
            "SELECT * FROM rem_traverse($1, $2, $3, $4)",
            entity_key,
            edge_types,
            max_depth,
            user_id,
        )
        return rows

    async def install_schema(self) -> None:
        """Run the install.sql script."""
        sql_path = Path(__file__).parent.parent.parent / "sql" / "install.sql"
        sql = sql_path.read_text()
        async with self.pool.acquire() as conn:
            await conn.execute(sql)


# Singleton instance
_db: DatabaseService | None = None


def get_db() -> DatabaseService:
    """Get or create database service singleton."""
    global _db
    if _db is None:
        _db = DatabaseService()
    return _db
