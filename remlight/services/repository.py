"""
Repository layer for entity CRUD operations.

Provides a clean abstraction over database operations for
ontologies, resources, sessions, messages, and kv_store.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from loguru import logger

from remlight.services.database import DatabaseService
from remlight.services.embeddings import generate_embedding_async


class Repository:
    """Generic repository for entity CRUD operations."""

    def __init__(self, db: DatabaseService, table_name: str):
        self.db = db
        self.table_name = table_name

    async def get(self, id: UUID | str, user_id: str | None = None) -> dict | None:
        """Get entity by ID."""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE id = $1 AND deleted_at IS NULL
        """
        args = [id]
        if user_id:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE id = $1 AND deleted_at IS NULL
                  AND (user_id = $2 OR user_id IS NULL)
            """
            args.append(user_id)
        return await self.db.fetchrow(query, *args)

    async def get_by_name(self, name: str, user_id: str | None = None) -> dict | None:
        """Get entity by name."""
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE name = $1 AND deleted_at IS NULL
        """
        args = [name]
        if user_id:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE name = $1 AND deleted_at IS NULL
                  AND (user_id = $2 OR user_id IS NULL)
            """
            args.append(user_id)
        return await self.db.fetchrow(query, *args)

    async def list(
        self,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at DESC",
    ) -> list[dict]:
        """List entities with pagination."""
        if user_id:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE deleted_at IS NULL
                  AND (user_id = $1 OR user_id IS NULL)
                ORDER BY {order_by}
                LIMIT $2 OFFSET $3
            """
            return await self.db.fetch(query, user_id, limit, offset)
        else:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE deleted_at IS NULL
                ORDER BY {order_by}
                LIMIT $1 OFFSET $2
            """
            return await self.db.fetch(query, limit, offset)

    async def create(self, data: dict[str, Any]) -> dict:
        """Create a new entity."""
        columns = list(data.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        values = list(data.values())

        query = f"""
            INSERT INTO {self.table_name} ({", ".join(columns)})
            VALUES ({", ".join(placeholders)})
            RETURNING *
        """
        return await self.db.fetchrow(query, *values)

    async def update(self, id: UUID | str, data: dict[str, Any]) -> dict | None:
        """Update an entity."""
        if not data:
            return await self.get(id)

        set_clauses = [f"{k} = ${i+2}" for i, k in enumerate(data.keys())]
        values = [id] + list(data.values())

        query = f"""
            UPDATE {self.table_name}
            SET {", ".join(set_clauses)}, updated_at = NOW()
            WHERE id = $1 AND deleted_at IS NULL
            RETURNING *
        """
        return await self.db.fetchrow(query, *values)

    async def delete(self, id: UUID | str, soft: bool = True) -> bool:
        """Delete an entity (soft delete by default)."""
        if soft:
            query = f"""
                UPDATE {self.table_name}
                SET deleted_at = NOW()
                WHERE id = $1 AND deleted_at IS NULL
            """
        else:
            query = f"DELETE FROM {self.table_name} WHERE id = $1"
        result = await self.db.execute(query, id)
        return "UPDATE 1" in result or "DELETE 1" in result

    async def search_by_field(
        self,
        field: str,
        value: Any,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search entities by a specific field."""
        if user_id:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE {field} = $1 AND deleted_at IS NULL
                  AND (user_id = $2 OR user_id IS NULL)
                LIMIT $3
            """
            return await self.db.fetch(query, value, user_id, limit)
        else:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE {field} = $1 AND deleted_at IS NULL
                LIMIT $2
            """
            return await self.db.fetch(query, value, limit)


class OntologyRepository(Repository):
    """Repository for ontology entities."""

    def __init__(self, db: DatabaseService):
        super().__init__(db, "ontologies")

    async def upsert(
        self,
        name: str,
        description: str,
        category: str | None = None,
        tags: list[str] | None = None,
        properties: dict | None = None,
        generate_embeddings: bool = True,
    ) -> dict:
        """
        Upsert ontology entity with automatic embedding generation.

        Args:
            name: Entity key/name (used for LOOKUP)
            description: Full content (markdown)
            category: Optional category
            tags: Optional tags list
            properties: Optional properties dict
            generate_embeddings: Whether to generate embeddings (default: True)

        Returns:
            Upserted record dict
        """
        # Generate embedding if requested
        embedding = None
        if generate_embeddings:
            logger.debug(f"Generating embedding for {name}")
            embedding_list = await generate_embedding_async(description)
            # Convert to pgvector string format
            embedding = "[" + ",".join(str(x) for x in embedding_list) + "]"

        query = """
            INSERT INTO ontologies (name, description, category, tags, properties, embedding)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (name) DO UPDATE SET
                description = EXCLUDED.description,
                category = EXCLUDED.category,
                tags = EXCLUDED.tags,
                properties = EXCLUDED.properties,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            RETURNING *
        """
        return await self.db.fetchrow(
            query,
            name,
            description,
            category,
            tags or [],
            json.dumps(properties or {}),
            embedding,
        )

    async def get_by_category(
        self, category: str, user_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        """Get ontologies by category."""
        return await self.search_by_field("category", category, user_id, limit)


class ResourceRepository(Repository):
    """Repository for resource entities."""

    def __init__(self, db: DatabaseService):
        super().__init__(db, "resources")

    async def upsert(
        self,
        name: str,
        content: str,
        uri: str | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
        generate_embeddings: bool = True,
    ) -> dict:
        """
        Upsert resource entity with automatic embedding generation.

        Args:
            name: Entity key/name
            content: Full content
            uri: Optional URI
            category: Optional category
            tags: Optional tags list
            metadata: Optional metadata dict
            generate_embeddings: Whether to generate embeddings (default: True)

        Returns:
            Upserted record dict
        """
        # Generate embedding if requested
        embedding = None
        if generate_embeddings:
            logger.debug(f"Generating embedding for {name}")
            embedding_list = await generate_embedding_async(content)
            # Convert to pgvector string format
            embedding = "[" + ",".join(str(x) for x in embedding_list) + "]"

        query = """
            INSERT INTO resources (name, content, uri, category, tags, metadata, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (name) DO UPDATE SET
                content = EXCLUDED.content,
                uri = EXCLUDED.uri,
                category = EXCLUDED.category,
                tags = EXCLUDED.tags,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
            RETURNING *
        """
        return await self.db.fetchrow(
            query,
            name,
            content,
            uri,
            category,
            tags or [],
            json.dumps(metadata or {}),
            embedding,
        )

    async def get_by_uri(self, uri: str, user_id: str | None = None) -> list[dict]:
        """Get resources by URI (may have multiple ordinals)."""
        if user_id:
            query = """
                SELECT * FROM resources
                WHERE uri = $1 AND deleted_at IS NULL
                  AND (user_id = $2 OR user_id IS NULL)
                ORDER BY ordinal
            """
            return await self.db.fetch(query, uri, user_id)
        else:
            query = """
                SELECT * FROM resources
                WHERE uri = $1 AND deleted_at IS NULL
                ORDER BY ordinal
            """
            return await self.db.fetch(query, uri)


class SessionRepository(Repository):
    """Repository for session entities."""

    def __init__(self, db: DatabaseService):
        super().__init__(db, "sessions")

    async def get_active(self, user_id: str | None = None, limit: int = 50) -> list[dict]:
        """Get active sessions."""
        if user_id:
            query = """
                SELECT * FROM sessions
                WHERE status = 'active' AND deleted_at IS NULL
                  AND (user_id = $1 OR user_id IS NULL)
                ORDER BY updated_at DESC
                LIMIT $2
            """
            return await self.db.fetch(query, user_id, limit)
        else:
            query = """
                SELECT * FROM sessions
                WHERE status = 'active' AND deleted_at IS NULL
                ORDER BY updated_at DESC
                LIMIT $1
            """
            return await self.db.fetch(query, limit)


class MessageRepository(Repository):
    """Repository for message entities."""

    def __init__(self, db: DatabaseService):
        super().__init__(db, "messages")

    async def get_by_session(
        self, session_id: UUID | str, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        """Get messages for a session."""
        query = """
            SELECT * FROM messages
            WHERE session_id = $1 AND deleted_at IS NULL
            ORDER BY created_at ASC
            LIMIT $2 OFFSET $3
        """
        return await self.db.fetch(query, session_id, limit, offset)

    async def get_recent(
        self, session_id: UUID | str, limit: int = 10
    ) -> list[dict]:
        """Get recent messages for a session (most recent first)."""
        query = """
            SELECT * FROM messages
            WHERE session_id = $1 AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, session_id, limit)
        return list(reversed(rows))  # Return in chronological order


class KVStoreRepository:
    """Repository for KV store operations."""

    def __init__(self, db: DatabaseService):
        self.db = db

    async def get(self, entity_key: str, user_id: str | None = None) -> dict | None:
        """Get entity by key."""
        if user_id:
            query = """
                SELECT * FROM kv_store
                WHERE entity_key = $1
                  AND (user_id = $2 OR user_id IS NULL)
            """
            return await self.db.fetchrow(query, entity_key, user_id)
        else:
            query = "SELECT * FROM kv_store WHERE entity_key = $1"
            return await self.db.fetchrow(query, entity_key)

    async def set(
        self,
        entity_key: str,
        entity_type: str,
        data: dict[str, Any],
        user_id: str | None = None,
        tenant_id: str | None = None,
    ) -> dict:
        """Set or update an entity in the KV store."""
        query = """
            INSERT INTO kv_store (entity_key, entity_type, table_name, data, user_id, tenant_id)
            VALUES ($1, $2, $2, $3::jsonb, $4, $5)
            ON CONFLICT (entity_key) DO UPDATE SET
                data = $3::jsonb,
                updated_at = NOW()
            RETURNING *
        """
        return await self.db.fetchrow(
            query, entity_key, entity_type, json.dumps(data), user_id, tenant_id
        )

    async def delete(self, entity_key: str) -> bool:
        """Delete an entity from the KV store."""
        result = await self.db.execute(
            "DELETE FROM kv_store WHERE entity_key = $1", entity_key
        )
        return "DELETE 1" in result

    async def list_by_type(
        self, entity_type: str, user_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        """List entities by type."""
        if user_id:
            query = """
                SELECT * FROM kv_store
                WHERE entity_type = $1
                  AND (user_id = $2 OR user_id IS NULL)
                ORDER BY updated_at DESC
                LIMIT $3
            """
            return await self.db.fetch(query, entity_type, user_id, limit)
        else:
            query = """
                SELECT * FROM kv_store
                WHERE entity_type = $1
                ORDER BY updated_at DESC
                LIMIT $2
            """
            return await self.db.fetch(query, entity_type, limit)
