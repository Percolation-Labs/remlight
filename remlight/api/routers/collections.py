"""Collections router - Manage session collections for evaluation.

Collections group sessions together for:
- Batch evaluation: Run evaluator agents across all sessions
- Test suites: Create reproducible test sets from real sessions
- Analysis: Compare agent performance across session groups

This supports the REMLight evaluation workflow where:
1. Sessions are captured during normal operation
2. Interesting sessions are added to collections (manually or via query)
3. Evaluator agents review collections and provide ratings
4. Results are stored as Feedback records
"""

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Header, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from remlight.services.database import get_db

router = APIRouter(prefix="/collections", tags=["collections"])

# Database instance (initialized via dependency)
_db = None


def init_collections(db):
    """Initialize collections router with database."""
    global _db
    _db = db


# =============================================================================
# Request/Response Models
# =============================================================================


class CollectionCreate(BaseModel):
    """Request model for creating a collection."""

    name: str
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    query_filter: dict[str, Any] | None = None  # Saved search for auto-population


class CollectionUpdate(BaseModel):
    """Request model for updating a collection."""

    name: str | None = None
    description: str | None = None
    status: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    query_filter: dict[str, Any] | None = None


class CollectionResponse(BaseModel):
    """Response model for a collection."""

    id: str | UUID
    name: str
    description: str | None = None
    session_count: int = 0
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    query_filter: dict[str, Any] | None = None
    user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_metadata(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return {}
        return v if v else {}

    @field_validator("query_filter", mode="before")
    @classmethod
    def parse_query_filter(cls, v: Any) -> dict[str, Any] | None:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except (json.JSONDecodeError, TypeError):
                return None
        return v


class SessionInCollection(BaseModel):
    """Session within a collection."""

    id: str | UUID
    session_id: str | UUID
    name: str | None = None
    description: str | None = None
    agent_name: str | None = None
    message_count: int = 0
    ordinal: int = 0
    notes: str | None = None
    created_at: datetime | None = None


class CollectionSessionsResponse(BaseModel):
    """Response with collection and its sessions."""

    collection: CollectionResponse
    sessions: list[SessionInCollection]
    total: int
    limit: int
    offset: int


class AddSessionRequest(BaseModel):
    """Request to add a session to a collection."""

    session_id: str | UUID
    notes: str | None = None
    ordinal: int | None = None


class AddSessionsRequest(BaseModel):
    """Request to add multiple sessions to a collection."""

    session_ids: list[str | UUID]
    notes: str | None = None


class AddSessionsFromQueryRequest(BaseModel):
    """Request to add sessions matching a query to a collection."""

    query: str | None = None  # Semantic search on session content
    agent_name: str | None = None
    tags: list[str] | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: int = 100


class CollectionSearch(BaseModel):
    """Request model for searching collections."""

    query: str | None = None  # Semantic search on description
    tags: list[str] | None = None  # Filter by tags
    tag_match: str = "any"  # "any" or "all"
    name_contains: str | None = None  # Fuzzy name search
    status: str | None = None  # Filter by status
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: int = 20
    offset: int = 0


# =============================================================================
# Collection CRUD Endpoints
# =============================================================================


@router.post("", response_model=CollectionResponse)
async def create_collection(
    collection: CollectionCreate,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
) -> CollectionResponse:
    """Create a new collection for grouping sessions."""
    db = get_db()
    await db.connect()

    try:
        # Convert dicts to JSON strings for asyncpg JSONB
        metadata_json = json.dumps(collection.metadata) if collection.metadata else "{}"
        query_filter_json = json.dumps(collection.query_filter) if collection.query_filter else None

        result = await db.fetchrow(
            """
            INSERT INTO collections (name, description, tags, metadata, query_filter, user_id, tenant_id)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7)
            RETURNING *
            """,
            collection.name,
            collection.description,
            collection.tags,
            metadata_json,
            query_filter_json,
            x_user_id,
            x_tenant_id or x_user_id,
        )
        return CollectionResponse(**dict(result))
    except Exception as e:
        if "collections_name_unique" in str(e):
            raise HTTPException(status_code=409, detail="Collection name already exists")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=list[CollectionResponse])
async def list_collections(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status: str | None = None,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> list[CollectionResponse]:
    """List all collections with optional status filter."""
    db = get_db()
    await db.connect()

    conditions = ["deleted_at IS NULL"]
    params: list[Any] = []
    param_idx = 1

    if x_user_id:
        conditions.append(f"(user_id = ${param_idx} OR user_id IS NULL)")
        params.append(x_user_id)
        param_idx += 1

    if status:
        conditions.append(f"status = ${param_idx}")
        params.append(status)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT * FROM collections
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """
    params.extend([limit, offset])

    rows = await db.fetch(query, *params)
    return [CollectionResponse(**dict(row)) for row in rows]


@router.post("/search", response_model=list[CollectionResponse])
async def search_collections(
    search: CollectionSearch,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> list[CollectionResponse]:
    """
    Search collections with multiple filter options.

    Search supports:
    - Semantic search via `query` (searches description with embeddings)
    - Tag filtering with `tags` (match any or all)
    - Name/title fuzzy filter
    - Status filter
    - Date range filters (created_after, created_before)
    - Pagination (limit, offset)
    """
    from remlight.services.embeddings import generate_embedding_async

    db = get_db()
    await db.connect()

    # Build the query dynamically
    conditions = ["deleted_at IS NULL"]
    params: list[Any] = []
    param_idx = 1

    # User isolation
    if x_user_id:
        conditions.append(f"(user_id = ${param_idx} OR user_id IS NULL)")
        params.append(x_user_id)
        param_idx += 1

    # Tag filtering
    if search.tags:
        if search.tag_match == "all":
            conditions.append(f"tags @> ${param_idx}::text[]")
        else:
            conditions.append(f"tags && ${param_idx}::text[]")
        params.append(search.tags)
        param_idx += 1

    # Name contains (fuzzy)
    if search.name_contains:
        conditions.append(f"name ILIKE ${param_idx}")
        params.append(f"%{search.name_contains}%")
        param_idx += 1

    # Status filter
    if search.status:
        conditions.append(f"status = ${param_idx}")
        params.append(search.status)
        param_idx += 1

    # Date range filters
    if search.created_after:
        conditions.append(f"created_at >= ${param_idx}")
        params.append(search.created_after)
        param_idx += 1

    if search.created_before:
        conditions.append(f"created_at <= ${param_idx}")
        params.append(search.created_before)
        param_idx += 1

    where_clause = " AND ".join(conditions)

    # If semantic search query provided, use vector similarity
    if search.query:
        query_embedding = await generate_embedding_async(search.query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = f"""
            SELECT *,
                   1 - (embedding <=> ${param_idx}::vector) as similarity
            FROM collections
            WHERE {where_clause}
              AND embedding IS NOT NULL
              AND 1 - (embedding <=> ${param_idx}::vector) >= 0.3
            ORDER BY embedding <=> ${param_idx}::vector
            LIMIT ${param_idx + 1}
            OFFSET ${param_idx + 2}
        """
        params.extend([embedding_str, search.limit, search.offset])
    else:
        sql = f"""
            SELECT * FROM collections
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx}
            OFFSET ${param_idx + 1}
        """
        params.extend([search.limit, search.offset])

    rows = await db.fetch(sql, *params)
    return [CollectionResponse(**dict(row)) for row in rows]


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> CollectionResponse:
    """Get a specific collection by ID."""
    db = get_db()
    await db.connect()

    result = await db.fetchrow(
        """
        SELECT * FROM collections
        WHERE id = $1 AND deleted_at IS NULL
          AND (user_id = $2 OR user_id IS NULL OR $2 IS NULL)
        """,
        collection_id,
        x_user_id,
    )

    if not result:
        raise HTTPException(status_code=404, detail="Collection not found")

    return CollectionResponse(**dict(result))


@router.put("/{collection_id}", response_model=CollectionResponse)
async def update_collection(
    collection_id: str,
    update: CollectionUpdate,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> CollectionResponse:
    """Update an existing collection."""
    db = get_db()
    await db.connect()

    # Build dynamic update
    updates = []
    params: list[Any] = [collection_id]
    param_idx = 2

    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            updates.append(f"{key} = ${param_idx}")
            params.append(value)
            param_idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clause = ", ".join(updates)
    params.append(x_user_id)

    result = await db.fetchrow(
        f"""
        UPDATE collections
        SET {set_clause}, updated_at = NOW()
        WHERE id = $1 AND deleted_at IS NULL
          AND (user_id = ${param_idx} OR user_id IS NULL OR ${param_idx} IS NULL)
        RETURNING *
        """,
        *params,
    )

    if not result:
        raise HTTPException(status_code=404, detail="Collection not found")

    return CollectionResponse(**dict(result))


@router.delete("/{collection_id}")
async def delete_collection(
    collection_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> dict[str, str]:
    """Soft delete a collection."""
    db = get_db()
    await db.connect()

    result = await db.execute(
        """
        UPDATE collections
        SET deleted_at = NOW()
        WHERE id = $1 AND deleted_at IS NULL
          AND (user_id = $2 OR user_id IS NULL OR $2 IS NULL)
        """,
        collection_id,
        x_user_id,
    )

    if result == "UPDATE 0":
        raise HTTPException(status_code=404, detail="Collection not found")

    return {"status": "success", "message": "Collection deleted"}


# =============================================================================
# Collection Sessions Endpoints
# =============================================================================


@router.get("/{collection_id}/sessions", response_model=CollectionSessionsResponse)
async def get_collection_sessions(
    collection_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> CollectionSessionsResponse:
    """Get all sessions in a collection with pagination.

    Returns the collection metadata and a paginated list of sessions.
    Sessions include message counts for context.
    """
    db = get_db()
    await db.connect()

    # Get collection
    collection = await db.fetchrow(
        """
        SELECT * FROM collections
        WHERE id = $1 AND deleted_at IS NULL
          AND (user_id = $2 OR user_id IS NULL OR $2 IS NULL)
        """,
        collection_id,
        x_user_id,
    )

    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Get sessions with message counts
    sessions_query = """
        SELECT
            cs.id,
            cs.session_id,
            s.name,
            s.description,
            s.agent_name,
            cs.ordinal,
            cs.notes,
            cs.created_at,
            COUNT(m.id) as message_count
        FROM collection_sessions cs
        JOIN sessions s ON cs.session_id = s.id
        LEFT JOIN messages m ON s.id = m.session_id
        WHERE cs.collection_id = $1 AND cs.deleted_at IS NULL
          AND (cs.user_id = $2 OR cs.user_id IS NULL OR $2 IS NULL)
        GROUP BY cs.id, cs.session_id, s.name, s.description, s.agent_name, cs.ordinal, cs.notes, cs.created_at
        ORDER BY cs.ordinal ASC, cs.created_at ASC
        LIMIT $3 OFFSET $4
    """
    session_rows = await db.fetch(sessions_query, collection_id, x_user_id, limit, offset)

    # Get total count
    count_result = await db.fetchrow(
        """
        SELECT COUNT(*) as total FROM collection_sessions
        WHERE collection_id = $1 AND deleted_at IS NULL
          AND (user_id = $2 OR user_id IS NULL OR $2 IS NULL)
        """,
        collection_id,
        x_user_id,
    )
    total = count_result["total"] if count_result else 0

    return CollectionSessionsResponse(
        collection=CollectionResponse(**dict(collection)),
        sessions=[SessionInCollection(**dict(row)) for row in session_rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/{collection_id}/sessions")
async def add_session_to_collection(
    collection_id: str,
    request: AddSessionRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
) -> dict[str, Any]:
    """Add a single session to a collection."""
    db = get_db()
    await db.connect()

    # Verify collection exists
    collection = await db.fetchrow(
        "SELECT id FROM collections WHERE id = $1 AND deleted_at IS NULL",
        collection_id,
    )
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Verify session exists
    session = await db.fetchrow(
        "SELECT id FROM sessions WHERE id = $1",
        str(request.session_id),
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        # Get next ordinal if not specified
        ordinal = request.ordinal
        if ordinal is None:
            max_ord = await db.fetchrow(
                "SELECT COALESCE(MAX(ordinal), -1) + 1 as next_ord FROM collection_sessions WHERE collection_id = $1",
                collection_id,
            )
            ordinal = max_ord["next_ord"] if max_ord else 0

        await db.execute(
            """
            INSERT INTO collection_sessions (collection_id, session_id, ordinal, notes, user_id, tenant_id)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (collection_id, session_id) DO UPDATE SET
                ordinal = EXCLUDED.ordinal,
                notes = COALESCE(EXCLUDED.notes, collection_sessions.notes),
                updated_at = NOW()
            """,
            collection_id,
            str(request.session_id),
            ordinal,
            request.notes,
            x_user_id,
            x_tenant_id or x_user_id,
        )

        return {"status": "success", "message": "Session added to collection"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{collection_id}/sessions/batch")
async def add_sessions_batch(
    collection_id: str,
    request: AddSessionsRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
) -> dict[str, Any]:
    """Add multiple sessions to a collection."""
    db = get_db()
    await db.connect()

    # Verify collection exists
    collection = await db.fetchrow(
        "SELECT id FROM collections WHERE id = $1 AND deleted_at IS NULL",
        collection_id,
    )
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    added = 0
    for session_id in request.session_ids:
        try:
            await db.execute(
                """
                INSERT INTO collection_sessions (collection_id, session_id, notes, user_id, tenant_id)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (collection_id, session_id) DO NOTHING
                """,
                collection_id,
                str(session_id),
                request.notes,
                x_user_id,
                x_tenant_id or x_user_id,
            )
            added += 1
        except Exception:
            pass  # Skip invalid sessions

    return {"status": "success", "added": added, "total_requested": len(request.session_ids)}


@router.post("/{collection_id}/sessions/from-query")
async def add_sessions_from_query(
    collection_id: str,
    request: AddSessionsFromQueryRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
) -> dict[str, Any]:
    """Add sessions matching a query to a collection.

    This enables building test collections from production sessions
    that match certain criteria.
    """
    db = get_db()
    await db.connect()

    # Verify collection exists
    collection = await db.fetchrow(
        "SELECT id FROM collections WHERE id = $1 AND deleted_at IS NULL",
        collection_id,
    )
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Build session query
    conditions = ["s.deleted_at IS NULL"]
    params: list[Any] = []
    param_idx = 1

    if request.agent_name:
        conditions.append(f"s.agent_name = ${param_idx}")
        params.append(request.agent_name)
        param_idx += 1

    if request.tags:
        conditions.append(f"s.tags && ${param_idx}::text[]")
        params.append(request.tags)
        param_idx += 1

    if request.created_after:
        conditions.append(f"s.created_at >= ${param_idx}")
        params.append(request.created_after)
        param_idx += 1

    if request.created_before:
        conditions.append(f"s.created_at <= ${param_idx}")
        params.append(request.created_before)
        param_idx += 1

    where_clause = " AND ".join(conditions)
    params.append(request.limit)

    query = f"""
        SELECT s.id FROM sessions s
        WHERE {where_clause}
        ORDER BY s.created_at DESC
        LIMIT ${param_idx}
    """

    session_rows = await db.fetch(query, *params)
    session_ids = [str(row["id"]) for row in session_rows]

    # Add sessions to collection
    added = 0
    for session_id in session_ids:
        try:
            await db.execute(
                """
                INSERT INTO collection_sessions (collection_id, session_id, user_id, tenant_id)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (collection_id, session_id) DO NOTHING
                """,
                collection_id,
                session_id,
                x_user_id,
                x_tenant_id or x_user_id,
            )
            added += 1
        except Exception:
            pass

    return {"status": "success", "added": added, "matched": len(session_ids)}


@router.delete("/{collection_id}/sessions/{session_id}")
async def remove_session_from_collection(
    collection_id: str,
    session_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> dict[str, str]:
    """Remove a session from a collection."""
    db = get_db()
    await db.connect()

    result = await db.execute(
        """
        DELETE FROM collection_sessions
        WHERE collection_id = $1 AND session_id = $2
          AND (user_id = $3 OR user_id IS NULL OR $3 IS NULL)
        """,
        collection_id,
        session_id,
        x_user_id,
    )

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Session not in collection")

    return {"status": "success", "message": "Session removed from collection"}


# =============================================================================
# Collection Export for Evaluation
# =============================================================================


@router.get("/{collection_id}/export")
async def export_collection(
    collection_id: str,
    include_messages: bool = True,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> dict[str, Any]:
    """Export a collection as JSON for evaluation.

    Returns the full collection with all sessions and optionally messages.
    This format can be consumed by evaluator agents or external tools.
    """
    db = get_db()
    await db.connect()

    # Get collection
    collection = await db.fetchrow(
        """
        SELECT * FROM collections
        WHERE id = $1 AND deleted_at IS NULL
          AND (user_id = $2 OR user_id IS NULL OR $2 IS NULL)
        """,
        collection_id,
        x_user_id,
    )

    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Get sessions
    sessions_query = """
        SELECT
            cs.session_id,
            s.name,
            s.description,
            s.agent_name,
            s.metadata,
            s.tags,
            cs.ordinal,
            cs.notes
        FROM collection_sessions cs
        JOIN sessions s ON cs.session_id = s.id
        WHERE cs.collection_id = $1 AND cs.deleted_at IS NULL
        ORDER BY cs.ordinal ASC, cs.created_at ASC
    """
    session_rows = await db.fetch(sessions_query, collection_id)

    sessions_data = []
    for row in session_rows:
        session_data = {
            "session_id": str(row["session_id"]),
            "name": row["name"],
            "description": row["description"],
            "agent_name": row["agent_name"],
            "metadata": row["metadata"],
            "tags": row["tags"],
            "ordinal": row["ordinal"],
            "notes": row["notes"],
        }

        if include_messages:
            messages = await db.fetch(
                """
                SELECT role, content, tool_calls, metadata, created_at
                FROM messages
                WHERE session_id = $1
                ORDER BY created_at ASC
                """,
                str(row["session_id"]),
            )
            session_data["messages"] = [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "tool_calls": m["tool_calls"],
                    "metadata": m["metadata"],
                    "created_at": m["created_at"].isoformat() if m["created_at"] else None,
                }
                for m in messages
            ]

        sessions_data.append(session_data)

    return {
        "collection": {
            "id": str(collection["id"]),
            "name": collection["name"],
            "description": collection["description"],
            "session_count": collection["session_count"],
            "metadata": collection["metadata"],
            "tags": collection["tags"],
        },
        "sessions": sessions_data,
        "exported_at": datetime.utcnow().isoformat(),
    }
