"""Sessions router - List and retrieve chat sessions.

Provides endpoints to list sessions and retrieve session messages
from the database.
"""

from datetime import datetime
from typing import Any

import yaml
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from remlight.services.database import get_db

router = APIRouter(prefix="/sessions", tags=["sessions"])

# Database instance (initialized in lifespan)
_db = None


def init_sessions(db):
    """Initialize sessions router with database."""
    global _db
    _db = db


class SessionInfo(BaseModel):
    """Session information."""

    id: str
    name: str | None = None
    first_message: str | None = None
    message_count: int = 0
    user_id: str | None = None
    created_at: datetime
    updated_at: datetime
    labels: list[str] = []


class SessionListResponse(BaseModel):
    """List of sessions response."""

    sessions: list[SessionInfo]


class MessageInfo(BaseModel):
    """Message information."""

    id: str
    role: str
    content: str
    status: str = "completed"
    tool_calls: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    created_at: datetime
    session_id: str


class SessionMessagesResponse(BaseModel):
    """Session messages response."""

    messages: list[MessageInfo]


@router.get("", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> SessionListResponse:
    """List chat sessions.

    Args:
        limit: Maximum number of sessions to return
        offset: Offset for pagination
        search: Search query for session names/content
        x_user_id: Filter by user ID
    """
    if not _db:
        return SessionListResponse(sessions=[])

    try:
        # Build query with positional parameters for asyncpg
        params = []
        conditions = []
        param_idx = 1

        if x_user_id:
            conditions.append(f"m.user_id = ${param_idx}")
            params.append(x_user_id)
            param_idx += 1

        if search:
            conditions.append(f"m.content ILIKE ${param_idx}")
            params.append(f"%{search}%")
            param_idx += 1

        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT
                m.session_id as id,
                s.name as session_name,
                MIN(m.content) FILTER (WHERE m.role = 'user') as first_message,
                COUNT(*) as message_count,
                m.user_id,
                MIN(m.created_at) as created_at,
                MAX(m.created_at) as updated_at
            FROM messages m
            LEFT JOIN sessions s ON m.session_id = s.id
            {where_clause}
            GROUP BY m.session_id, m.user_id, s.name
            ORDER BY MAX(m.created_at) DESC
            LIMIT ${param_idx} OFFSET ${param_idx + 1}
        """
        params.extend([limit, offset])

        rows = await _db.fetch(query, *params)

        sessions = []
        for row in rows:
            # Use session name if set, otherwise None (UI will fall back to first_message)
            session_name = row.get("session_name")
            # Don't use UUID as name (that's the default)
            if session_name and session_name == row["id"]:
                session_name = None

            sessions.append(
                SessionInfo(
                    id=str(row["id"]) if row["id"] else "",
                    name=session_name,
                    first_message=(
                        row["first_message"][:100] if row["first_message"] else None
                    ),
                    message_count=row["message_count"] or 0,
                    user_id=row["user_id"],
                    created_at=row["created_at"] or datetime.now(),
                    updated_at=row["updated_at"] or datetime.now(),
                )
            )

        return SessionListResponse(sessions=sessions)

    except Exception as e:
        print(f"Failed to list sessions: {e}")
        return SessionListResponse(sessions=[])


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
async def get_session_messages(
    session_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> SessionMessagesResponse:
    """Get messages for a specific session.

    Args:
        session_id: The session UUID
        x_user_id: Optional user ID filter
    """
    if not _db:
        return SessionMessagesResponse(messages=[])

    try:
        params = [session_id]
        param_idx = 2

        user_filter = ""
        if x_user_id:
            user_filter = f" AND user_id = ${param_idx}"
            params.append(x_user_id)

        query = f"""
            SELECT
                id,
                role,
                content,
                tool_calls,
                metadata,
                created_at,
                session_id
            FROM messages
            WHERE session_id = $1
            {user_filter}
            ORDER BY created_at ASC
        """

        rows = await _db.fetch(query, *params)

        messages = []
        for row in rows:
            # Parse JSONB fields that may come as strings from asyncpg
            import json

            tool_calls_raw = row["tool_calls"]
            if isinstance(tool_calls_raw, str):
                try:
                    tool_calls_raw = json.loads(tool_calls_raw)
                except json.JSONDecodeError:
                    tool_calls_raw = None
            # Handle both formats: list directly or {"items": [...]} from JSONB
            if tool_calls_raw is None:
                tool_calls = []
            elif isinstance(tool_calls_raw, list):
                tool_calls = tool_calls_raw
            elif isinstance(tool_calls_raw, dict) and "items" in tool_calls_raw:
                tool_calls = tool_calls_raw["items"]
            else:
                tool_calls = []

            metadata_raw = row["metadata"]
            if isinstance(metadata_raw, str):
                try:
                    metadata_raw = json.loads(metadata_raw)
                except json.JSONDecodeError:
                    metadata_raw = {}
            metadata = metadata_raw if isinstance(metadata_raw, dict) else {}

            messages.append(
                MessageInfo(
                    id=str(row["id"]) if row["id"] else "",
                    role=row["role"] or "user",
                    content=row["content"] or "",
                    status="completed",
                    tool_calls=tool_calls,
                    metadata=metadata,
                    created_at=row["created_at"] or datetime.now(),
                    session_id=str(row["session_id"]) if row["session_id"] else "",
                )
            )

        return SessionMessagesResponse(messages=messages)

    except Exception as e:
        print(f"Failed to get session messages: {e}")
        return SessionMessagesResponse(messages=[])


@router.get("/{session_id}/export")
async def export_session(
    session_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> Response:
    """Export a session as YAML.

    Returns the full session data as a downloadable YAML file.

    Args:
        session_id: The session UUID
        x_user_id: Optional user ID filter
    """
    if not _db:
        raise HTTPException(status_code=500, detail="Database not available")

    try:
        # Get session info
        params = [session_id]
        param_idx = 2

        user_filter = ""
        if x_user_id:
            user_filter = f" AND m.user_id = ${param_idx}"
            params.append(x_user_id)

        session_query = f"""
            SELECT
                m.session_id as id,
                MIN(m.content) FILTER (WHERE m.role = 'user') as first_message,
                COUNT(*) as message_count,
                m.user_id,
                MIN(m.created_at) as created_at,
                MAX(m.created_at) as updated_at
            FROM messages m
            WHERE m.session_id = $1
            {user_filter}
            GROUP BY m.session_id, m.user_id
        """

        session_row = await _db.fetchrow(session_query, *params)

        if not session_row:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get all messages - build params fresh for this query
        msg_params = [session_id]
        msg_user_filter = ""
        if x_user_id:
            msg_user_filter = " AND user_id = $2"
            msg_params.append(x_user_id)

        messages_query = f"""
            SELECT
                id,
                role,
                content,
                tool_calls,
                metadata,
                created_at,
                session_id
            FROM messages
            WHERE session_id = $1
            {msg_user_filter}
            ORDER BY created_at ASC
        """

        rows = await _db.fetch(messages_query, *msg_params)

        # Build export structure
        messages_data = []
        for row in rows:
            msg_data = {
                "id": str(row["id"]) if row["id"] else None,
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            }
            if row["tool_calls"]:
                msg_data["tool_calls"] = row["tool_calls"]
            if row["metadata"]:
                msg_data["metadata"] = row["metadata"]
            messages_data.append(msg_data)

        export_data = {
            "session": {
                "id": str(session_row["id"]),
                "user_id": session_row["user_id"],
                "first_message": session_row["first_message"],
                "message_count": session_row["message_count"],
                "created_at": session_row["created_at"].isoformat() if session_row["created_at"] else None,
                "updated_at": session_row["updated_at"].isoformat() if session_row["updated_at"] else None,
            },
            "messages": messages_data,
        }

        # Convert to YAML
        yaml_content = yaml.dump(
            export_data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

        return Response(
            content=yaml_content,
            media_type="application/x-yaml",
            headers={
                "Content-Disposition": f'attachment; filename="session-{session_id}.yaml"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Failed to export session: {e}")
        raise HTTPException(status_code=500, detail="Failed to export session")
