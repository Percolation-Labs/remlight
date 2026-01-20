"""
Generic Repository Pattern for Entity Persistence
===================================================

This module implements the MODEL-BASED REPOSITORY pattern that enables database
operations on ANY Pydantic model type without writing model-specific code.

THE REPOSITORY PATTERN
----------------------
Instead of writing separate repository classes for each entity type:

    class MessageRepository:
        async def save(self, message: Message): ...
        async def find(self, filters): ...

    class OntologyRepository:
        async def save(self, ontology: Ontology): ...
        async def find(self, filters): ...

We have ONE generic class that works with any Pydantic model:

    repo = Repository(Message)      # Works with Message entities
    repo = Repository(Ontology)     # Works with Ontology entities
    repo = Repository(Session)      # Works with Session entities

BENEFITS OF THIS APPROACH
-------------------------
1. **Zero Boilerplate**: No need to write entity-specific repository code
2. **Type Safety**: TypeVar[T] ensures methods return correct types
3. **Consistency**: All entities get the same CRUD operations
4. **Embedding Integration**: Automatic vector embedding for searchable fields
5. **Soft Delete**: All entities use deleted_at for recoverable deletion
6. **Lazy Initialization**: Database connection happens on first use

KEY DESIGN DECISIONS
--------------------

1. **Singleton Database Pattern**:
   Repository uses _get_db() for lazy DB initialization. This avoids:
   - Circular imports (DB depends on settings, settings imported everywhere)
   - Premature connection (tests don't need real DB)
   - Explicit DB passing (cleaner API)

   But supports explicit DB for testing:
       repo = Repository(Message, db=mock_db)

2. **Table Name Inference**:
   By default, table name = model class name lowercase + 's':
   - Message → messages
   - Ontology → ontologys (override to "ontology" if needed)
   - Session → sessions

3. **Model-Driven Embedding Generation**:
   Models that have an 'embedding' field will automatically get embeddings
   generated. The source field is determined by model_config['embedding_field']
   with fallback to 'description' then 'content'. This enables semantic search.

4. **Upsert as Primary Operation**:
   We use upsert (INSERT ON CONFLICT UPDATE) instead of separate create/update.
   This simplifies client code - just call upsert() regardless of whether
   the record exists.

USAGE EXAMPLES
--------------

    from remlight.services.repository import Repository
    from remlight.models.entities import Message, Ontology

    # Basic CRUD
    repo = Repository(Message)
    message = await repo.upsert(Message(content="Hello", session_id="sess-123"))
    messages = await repo.find({"session_id": "sess-123"})
    await repo.delete(message.id)

    # Batch operations (upsert accepts list)
    messages = [Message(...), Message(...), Message(...)]
    await repo.upsert(messages)  # All upserted in sequence

    # Convenience methods
    recent = await repo.get_recent(session_id, limit=10)
    count = await repo.count({"user_id": "user-123"})

KV REPOSITORY
-------------
In addition to the generic Repository, this module provides KVRepository
for key-value storage operations. This enables O(1) lookups via entity_key
(the REM LOOKUP operation).

    kv = KVRepository()
    await kv.set("architecture", "ontology", {"title": "..."})
    data = await kv.get("architecture")  # O(1) lookup
"""

from __future__ import annotations

import json
from typing import Any, Generic, Type, TypeVar

from loguru import logger
from pydantic import BaseModel

from remlight.services.embeddings import generate_embedding_async
from remlight.services.sql_builder import (
    build_count,
    build_delete,
    build_select,
    build_upsert,
)

# TypeVar bound to BaseModel ensures we only work with Pydantic models
# This enables type inference: Repository[Message].find() returns list[Message]
T = TypeVar("T", bound=BaseModel)

# Default fields to check for embedding content (in priority order)
# Used when model doesn't specify embedding_field in model_config
DEFAULT_EMBEDDING_FIELDS = ["description", "content"]


def _get_db():
    """
    Get the database singleton with late import to avoid circular dependencies.

    WHY LATE IMPORT?
    ----------------
    The database module imports settings, and settings may be imported by
    modules that also import repository. Late import breaks this cycle.

    Additionally, this enables:
    - Testing without real database (just don't call methods that need DB)
    - Deferred connection (DB only connects when first used)
    - Clean separation (repository doesn't depend on DB at import time)

    Returns:
        DatabaseService singleton instance
    """
    from remlight.services.database import get_db
    return get_db()


class Repository(Generic[T]):
    """
    Generic repository for any Pydantic model type.

    The heart of the model-based persistence pattern. This single class
    replaces the need for entity-specific repository implementations.

    TYPE PARAMETER T
    ----------------
    The generic parameter T (bound to BaseModel) enables type inference:

        repo = Repository(Message)
        messages = await repo.find({...})  # Returns list[Message], not list[Any]

    This provides IDE autocomplete and type checking across all operations.

    DATABASE ACCESS PATTERN
    ----------------------
    - Uses singleton by default (no DB parameter needed)
    - Accepts explicit db for testing (mock injection)
    - Lazy connection (connects on first operation, not init)

    Example:
        # Production: uses singleton
        repo = Repository(Message)

        # Testing: inject mock
        repo = Repository(Message, db=mock_database)
    """

    def __init__(
        self,
        model_class: Type[T],
        table_name: str | None = None,
        db=None,  # Only for testing/mocks - uses singleton by default
    ):
        """
        Initialize repository for a specific model type.

        The repository infers the database table name from the model class:
        - Message → messages
        - Ontology → ontologys (may need explicit override)
        - Session → sessions

        Override table_name when the default pluralization is wrong.

        Args:
            model_class: Pydantic model class (e.g., Message, Resource)
                        Used for type inference and table name derivation
            table_name: Optional explicit table name (overrides inference)
            db: Optional DatabaseService for testing/mocking
                When None, uses the global singleton via _get_db()
        """
        self._db = db  # Store explicit db if provided (for testing)
        self.model_class = model_class
        # Default table name: lowercase class name + 's' (simple pluralization)
        self.table_name = table_name or f"{model_class.__name__.lower()}s"

    @property
    def db(self):
        """
        Get database service - uses singleton if not explicitly set.

        This property pattern enables:
        - Default singleton for production (no constructor arg needed)
        - Explicit mock injection for testing (pass db to constructor)
        - Lazy evaluation (DB not accessed until property used)
        """
        if self._db is not None:
            return self._db
        return _get_db()

    async def _ensure_connected(self):
        """
        Ensure database connection pool is established.

        Called before each database operation. If pool doesn't exist,
        initiates connection. This enables lazy initialization - the
        repository can be created without immediately connecting.
        """
        if not self.db.pool:
            await self.db.connect()

    async def upsert(
        self,
        records: T | list[T],
        generate_embeddings: bool = True,
        conflict_field: str = "id",
    ) -> T | list[T]:
        """
        Upsert (create or update) single record or list of records.

        UPSERT PATTERN
        --------------
        This is the PRIMARY persistence operation. Instead of separate create/update:

            # Traditional (don't do this)
            if await repo.exists(id):
                await repo.update(record)
            else:
                await repo.create(record)

            # With upsert (do this instead)
            await repo.upsert(record)

        The database handles conflict resolution via ON CONFLICT DO UPDATE.

        POLYMORPHIC INPUT
        -----------------
        Accepts both single records and lists. Returns matching type:
        - Input: Message → Returns: Message
        - Input: list[Message] → Returns: list[Message]

        This simplifies client code - no need for separate batch methods.

        MODEL-DRIVEN EMBEDDING GENERATION
        ----------------------------------
        When generate_embeddings=True (default), embeddings are generated ONLY
        if the model has 'embedding_field' in model_config. Models without this
        config are not affected (backward compatible).

        Config options:
        - embedding_field: True → use default precedence (description → content)
        - embedding_field: "fieldname" → use that field, fallback to defaults

        Example models:
            class Ontology(CoreModel):
                model_config = {"embedding_field": True}  # description → content

            class Resource(CoreModel):
                model_config = {"embedding_field": "content"}  # content only

        This enables the SEARCH REM query operation:
            REM SEARCH "machine learning" IN ontology
            → Uses cosine similarity on the embedding column

        CONFLICT RESOLUTION
        ------------------
        By default, conflicts are detected on the 'id' field. When a record
        with the same ID exists, it's updated. Override conflict_field for
        different uniqueness constraints (e.g., "email" for users).

        Args:
            records: Single model instance or list of instances
            generate_embeddings: If True, generate vector embeddings for models
                                that have an 'embedding' field defined
            conflict_field: Field to detect conflicts (default: "id")

        Returns:
            Same structure as input: single record or list with generated IDs
        """
        await self._ensure_connected()

        # Coerce single item to list for uniform processing
        # Track input type to return matching output type
        is_single = not isinstance(records, list)
        records_list: list[T] = [records] if is_single else records  # type: ignore

        for record in records_list:
            # =================================================================
            # EMBEDDING GENERATION (Model-Driven)
            # =================================================================
            # Only generate embeddings if:
            # 1. generate_embeddings=True (default)
            # 2. The model has an 'embedding' field defined
            #
            # Source field priority:
            # 1. model_config['embedding_field'] if specified
            # 2. Fallback: 'description' then 'content'
            # =================================================================
            embedding = None
            if generate_embeddings:
                # Check if model has embedding_field config - only then generate
                model_config = getattr(record, 'model_config', {}) or {}
                embedding_field = model_config.get('embedding_field')

                if embedding_field:
                    # Determine which fields to try:
                    # - True: use default precedence (description → content)
                    # - "fieldname": use that specific field, then defaults
                    if embedding_field is True:
                        fields_to_try = DEFAULT_EMBEDDING_FIELDS
                    else:
                        fields_to_try = [embedding_field] + DEFAULT_EMBEDDING_FIELDS

                    for field_name in fields_to_try:
                        content = getattr(record, field_name, None)
                        if content and isinstance(content, str):
                            logger.debug(f"Generating embedding for {field_name}")
                            embedding_list = await generate_embedding_async(content)
                            # Format as PostgreSQL array literal for pgvector
                            embedding = "[" + ",".join(str(x) for x in embedding_list) + "]"
                            break  # Use first field with content

            # =================================================================
            # SQL GENERATION AND EXECUTION
            # =================================================================
            # build_upsert creates: INSERT INTO ... ON CONFLICT (id) DO UPDATE
            # If embedding was generated, we patch the SQL to include it
            # =================================================================
            sql, params = build_upsert(record, self.table_name, conflict_field)

            # Add embedding to params if generated
            if embedding:
                params.append(embedding)
                # Modify SQL to include embedding column
                # This is SQL surgery - adding embedding to INSERT and UPDATE clauses
                sql = sql.replace(
                    ") VALUES (",
                    ", embedding) VALUES ("
                ).replace(
                    f") ON CONFLICT",
                    f", ${len(params)}) ON CONFLICT"
                ).replace(
                    "SET ",
                    "SET embedding = EXCLUDED.embedding, "
                )

            # Execute and capture returned ID
            row = await self.db.fetchrow(sql, *params)
            if row and "id" in row:
                record.id = row["id"]  # type: ignore[attr-defined]

        # Return matching type: single if input was single, list if input was list
        return records_list[0] if is_single else records_list

    async def get_by_id(self, record_id: str, user_id: str | None = None) -> T | None:
        """
        Get a single record by primary key ID.

        DATA ISOLATION PATTERN
        ----------------------
        When user_id is provided, the query includes a user isolation clause:
            (user_id = $2 OR user_id IS NULL)

        This means:
        - User sees their own records (user_id matches)
        - User sees shared records (user_id IS NULL = public data)
        - User cannot see other users' records

        SOFT DELETE
        -----------
        All queries filter out soft-deleted records (deleted_at IS NULL).
        Records are never hard-deleted; they have deleted_at set instead.

        Args:
            record_id: The UUID primary key
            user_id: Optional user filter for data isolation

        Returns:
            Model instance if found, None otherwise
        """
        await self._ensure_connected()

        if user_id:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE id = $1 AND deleted_at IS NULL
                  AND (user_id = $2 OR user_id IS NULL)
            """
            row = await self.db.fetchrow(query, record_id, user_id)
        else:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE id = $1 AND deleted_at IS NULL
            """
            row = await self.db.fetchrow(query, record_id)

        if not row:
            return None

        # model_validate converts the DB row dict to a typed Pydantic model
        return self.model_class.model_validate(dict(row))

    async def get_by_name(self, name: str, user_id: str | None = None) -> T | None:
        """
        Get a single record by name field.

        This is a convenience method for entities with a 'name' column.
        Uses the same data isolation pattern as get_by_id.

        Args:
            name: The name field value to match
            user_id: Optional user filter for data isolation

        Returns:
            Model instance if found, None otherwise
        """
        await self._ensure_connected()

        if user_id:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE name = $1 AND deleted_at IS NULL
                  AND (user_id = $2 OR user_id IS NULL)
            """
            row = await self.db.fetchrow(query, name, user_id)
        else:
            query = f"""
                SELECT * FROM {self.table_name}
                WHERE name = $1 AND deleted_at IS NULL
            """
            row = await self.db.fetchrow(query, name)

        if not row:
            return None

        return self.model_class.model_validate(dict(row))

    async def find(
        self,
        filters: dict[str, Any],
        order_by: str = "created_at ASC",
        limit: int | None = None,
        offset: int = 0,
    ) -> list[T]:
        """
        Find records matching filter criteria.

        This is the primary QUERY method. Filters are a simple dict that
        gets converted to WHERE clauses:

            {"session_id": "abc", "user_id": "123"}
            → WHERE session_id = $1 AND user_id = $2 AND deleted_at IS NULL

        ORDERING AND PAGINATION
        ----------------------
        - order_by: SQL ORDER BY clause (default: chronological)
        - limit: Max records to return (None = all)
        - offset: Skip first N records (for pagination)

        Args:
            filters: Dict of field=value conditions (all must match)
            order_by: SQL ORDER BY clause
            limit: Maximum records to return
            offset: Number of records to skip

        Returns:
            List of matching model instances (empty list if none match)
        """
        await self._ensure_connected()

        # build_select generates SELECT ... WHERE ... with parameterized values
        sql, params = build_select(
            self.model_class,
            self.table_name,
            filters,
            order_by=order_by,
            limit=limit,
            offset=offset,
        )

        rows = await self.db.fetch(sql, *params)
        # Convert each row to a typed Pydantic model
        return [self.model_class.model_validate(dict(row)) for row in rows]

    async def find_one(self, filters: dict[str, Any]) -> T | None:
        """
        Find single record matching filters.

        Convenience wrapper around find() with limit=1.
        Use when you expect exactly one match (or none).

        Args:
            filters: Dict of field=value conditions

        Returns:
            Single model instance or None
        """
        results = await self.find(filters, limit=1)
        return results[0] if results else None

    async def list(
        self,
        user_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at DESC",
    ) -> list[T]:
        """
        List entities with pagination.

        Higher-level convenience method for listing with common defaults:
        - Optional user filtering
        - Sensible pagination defaults
        - Newest first ordering

        Args:
            user_id: Optional filter to user's records
            limit: Max records (default: 100)
            offset: Skip first N (default: 0)
            order_by: Order clause (default: newest first)

        Returns:
            List of model instances
        """
        filters: dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id

        return await self.find(filters, order_by=order_by, limit=limit, offset=offset)

    async def delete(self, record_id: str, tenant_id: str | None = None) -> bool:
        """
        Soft delete a record (sets deleted_at timestamp).

        SOFT DELETE PATTERN
        ------------------
        Records are never permanently deleted. Instead, deleted_at is set
        to the current timestamp. All queries filter out deleted records.

        Benefits:
        - Audit trail (deleted data still available)
        - Easy recovery (just clear deleted_at)
        - Referential integrity (foreign keys still valid)

        Args:
            record_id: UUID of record to delete
            tenant_id: Optional tenant for multi-tenant isolation

        Returns:
            True if record was deleted, False if not found
        """
        await self._ensure_connected()

        sql, params = build_delete(self.table_name, record_id, tenant_id)
        row = await self.db.fetchrow(sql, *params)

        return row is not None

    async def count(self, filters: dict[str, Any]) -> int:
        """
        Count records matching filters.

        Useful for pagination (total count) and analytics.

        Args:
            filters: Dict of field=value conditions

        Returns:
            Count of matching records
        """
        await self._ensure_connected()

        sql, params = build_count(self.table_name, filters)
        row = await self.db.fetchrow(sql, *params)

        return row[0] if row else 0

    # =========================================================================
    # CONVENIENCE METHODS FOR SPECIFIC USE CASES
    # =========================================================================
    # These methods are tailored for common patterns like session-based
    # message retrieval. They're built on top of the core CRUD methods.
    # =========================================================================

    async def get_by_session(
        self, session_id: str, user_id: str | None = None
    ) -> list[T]:
        """
        Get all records for a session (designed for Message model).

        SESSION-BASED RETRIEVAL
        ----------------------
        Messages belong to sessions. This method retrieves all messages
        for a given session in chronological order.

        Used by:
        - SessionMessageStore.load_session_messages()
        - Multi-turn conversation reconstruction

        Args:
            session_id: UUID of the session
            user_id: Optional user filter

        Returns:
            List of records in chronological order (oldest first)
        """
        filters: dict[str, Any] = {"session_id": session_id}
        if user_id:
            filters["user_id"] = user_id

        return await self.find(filters, order_by="created_at ASC")

    async def get_recent(self, session_id: str, limit: int = 10) -> list[T]:
        """
        Get recent messages for a session in chronological order.

        CONTEXT WINDOW LOADING
        ---------------------
        For LLM context, we often need the most recent N messages.
        This method:
        1. Fetches last N by timestamp (DESC)
        2. Reverses to chronological order (ASC)

        The reversal is important - LLMs expect chronological order,
        but we want the LAST N messages, not the FIRST N.

        Args:
            session_id: UUID of the session
            limit: Number of recent messages to return

        Returns:
            List of recent messages in chronological order
        """
        await self._ensure_connected()

        # Fetch most recent first, then reverse for chronological order
        query = f"""
            SELECT * FROM {self.table_name}
            WHERE session_id = $1 AND deleted_at IS NULL
            ORDER BY created_at DESC
            LIMIT $2
        """
        rows = await self.db.fetch(query, session_id, limit)
        results = [self.model_class.model_validate(dict(row)) for row in rows]
        # Reverse: [newest, ..., oldest] → [oldest, ..., newest]
        return list(reversed(results))


class KVRepository:
    """
    Repository for Key-Value store operations.

    THE KV STORE PATTERN
    -------------------
    The kv_store table provides O(1) lookup by entity_key. This is the backing
    store for the REM LOOKUP operation:

        REM LOOKUP architecture
        → KVRepository.get("architecture")
        → Returns ontology document in O(1) time

    Unlike the generic Repository (which works with tables), KVRepository
    works specifically with the kv_store table that stores:
    - entity_key: Unique identifier for lookup (e.g., "architecture", "rem-query")
    - entity_type: Type classification (e.g., "ontology", "agent")
    - data: JSONB payload with the actual content
    - user_id/tenant_id: For data isolation

    WHY A SEPARATE KV STORE?
    -----------------------
    While we have entity tables (ontology, resources, etc.), the kv_store
    provides a unified lookup mechanism:

    1. **Fast LOOKUP**: entity_key is indexed, O(1) retrieval
    2. **Schema-agnostic**: Don't need to know which table an entity is in
    3. **Denormalized**: Data is copied for fast access
    4. **Cross-entity**: Can lookup any entity type by key

    The kv_store is populated by triggers when entities are created/updated,
    or explicitly by the ingest process.

    USAGE:
        kv = KVRepository()

        # Store an entity
        await kv.set("architecture", "ontology", {"title": "Architecture", ...})

        # Lookup by key (O(1))
        data = await kv.get("architecture")

        # List all ontology entities
        entities = await kv.list_by_type("ontology")
    """

    def __init__(self, db=None):
        """
        Initialize KV repository.

        Like Repository, uses singleton DB by default but accepts
        explicit DB for testing.

        Args:
            db: Optional DatabaseService for testing/mocking
        """
        self._db = db

    @property
    def db(self):
        """Get database service - uses singleton if not explicitly set."""
        if self._db is not None:
            return self._db
        return _get_db()

    async def _ensure_connected(self):
        """Ensure database connection pool is established."""
        if not self.db.pool:
            await self.db.connect()

    async def get(self, entity_key: str, user_id: str | None = None) -> dict | None:
        """
        Get entity by key - the REM LOOKUP operation.

        This is O(1) lookup via the indexed entity_key column.
        It's the fastest way to retrieve a known entity.

        DATA ISOLATION
        -------------
        When user_id is provided, returns:
        - User's own entities (user_id matches)
        - Shared entities (user_id IS NULL)

        Args:
            entity_key: Unique key for the entity (e.g., "architecture")
            user_id: Optional user filter for data isolation

        Returns:
            Row dict if found, None otherwise
        """
        await self._ensure_connected()

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
        """
        Set or update an entity in the KV store.

        Uses upsert (INSERT ON CONFLICT UPDATE) to handle both
        creation and updates in a single operation.

        INDEXED STORAGE
        ---------------
        The entity_key is the primary lookup key. The data payload
        is stored as JSONB, enabling:
        - Flexible schema per entity type
        - JSON path queries if needed
        - Efficient storage of nested structures

        Args:
            entity_key: Unique key for lookup (e.g., "architecture")
            entity_type: Type classification (e.g., "ontology", "agent")
            data: JSONB payload containing entity content
            user_id: Optional owner for data isolation
            tenant_id: Optional tenant for multi-tenancy

        Returns:
            The inserted/updated row
        """
        await self._ensure_connected()

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
        """
        Delete an entity from the KV store (hard delete).

        Note: Unlike Repository.delete(), this is a HARD delete.
        The kv_store is a cache/index, so soft delete isn't needed.

        Args:
            entity_key: Key of entity to delete

        Returns:
            True if deleted, False if not found
        """
        await self._ensure_connected()

        result = await self.db.execute(
            "DELETE FROM kv_store WHERE entity_key = $1", entity_key
        )
        return "DELETE 1" in result

    async def list_by_type(
        self, entity_type: str, user_id: str | None = None, limit: int = 100
    ) -> list[dict]:
        """
        List all entities of a given type.

        Useful for:
        - Listing all ontology documents
        - Listing all agent schemas
        - Browse/discovery UIs

        Results are ordered by updated_at DESC (newest first).

        Args:
            entity_type: Type to filter by (e.g., "ontology", "agent")
            user_id: Optional user filter for data isolation
            limit: Max results (default: 100)

        Returns:
            List of row dicts matching the type
        """
        await self._ensure_connected()

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
