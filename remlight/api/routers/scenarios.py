"""Scenarios router for labeled sessions search and replay.

Scenarios allow users to:
- Label sessions with descriptive metadata for later retrieval
- Search by description (semantic), tags, dates, and title
- Replay old sessions by loading the associated session
- Build context by finding relevant past interactions
"""

from datetime import datetime
from typing import Any
from uuid import UUID

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

from remlight.models import Scenario
from remlight.services.database import get_db
from remlight.services.embeddings import generate_embedding_async
from remlight.services.repository import Repository
from remlight.settings import settings

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


# =============================================================================
# Request/Response Models
# =============================================================================


class ScenarioCreate(BaseModel):
    """Request model for creating a scenario."""

    name: str | None = None
    description: str | None = None
    session_id: str | UUID | None = None
    agent_name: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScenarioUpdate(BaseModel):
    """Request model for updating a scenario."""

    name: str | None = None
    description: str | None = None
    agent_name: str | None = None
    status: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class ScenarioSearch(BaseModel):
    """Request model for searching scenarios."""

    query: str | None = None  # Semantic search on description
    tags: list[str] | None = None  # Filter by tags (AND)
    tag_match: str = "any"  # "any" or "all"
    name_contains: str | None = None  # Title/name filter
    agent_name: str | None = None  # Filter by agent
    status: str | None = None  # Filter by status
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: int = 20
    offset: int = 0


class ScenarioResponse(BaseModel):
    """Response model for a scenario."""

    id: str | UUID
    name: str | None = None
    description: str | None = None
    session_id: str | UUID | None = None
    agent_name: str | None = None
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("metadata", mode="before")
    @classmethod
    def parse_jsonb(cls, v: Any) -> Any:
        """Parse JSONB strings from asyncpg into Python objects."""
        if isinstance(v, str):
            import json
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback via Phoenix."""

    trace_id: str | None = None
    span_id: str | None = None
    name: str = "user_feedback"  # Annotation name
    score: float | None = None  # 0.0 to 1.0
    label: str | None = None  # e.g., "thumbs_up", "thumbs_down", "relevant", "not_relevant"
    comment: str | None = None  # Free text feedback
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackResponse(BaseModel):
    """Response model for feedback submission."""

    status: str
    message: str
    annotation_id: str | None = None


# =============================================================================
# Scenario Endpoints
# =============================================================================


@router.post("", response_model=ScenarioResponse)
async def create_scenario(
    scenario: ScenarioCreate,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
) -> ScenarioResponse:
    """
    Create a new scenario linked to a session.

    Scenarios allow labeling sessions with tags and descriptions
    for later search and replay. The description field is automatically
    embedded for semantic search.
    """
    repo = Repository(Scenario)

    new_scenario = Scenario(
        name=scenario.name,
        description=scenario.description,
        session_id=scenario.session_id,
        agent_name=scenario.agent_name,
        tags=scenario.tags,
        metadata=scenario.metadata,
        user_id=x_user_id,
        tenant_id=x_tenant_id or x_user_id,
    )

    result = await repo.upsert(new_scenario, generate_embeddings=True)
    return ScenarioResponse(**result.model_dump())


@router.get("/{scenario_id}", response_model=ScenarioResponse)
async def get_scenario(
    scenario_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> ScenarioResponse:
    """Get a specific scenario by ID."""
    repo = Repository(Scenario)
    result = await repo.get_by_id(scenario_id, user_id=x_user_id)

    if not result:
        raise HTTPException(status_code=404, detail="Scenario not found")

    return ScenarioResponse(**result.model_dump())


@router.put("/{scenario_id}", response_model=ScenarioResponse)
async def update_scenario(
    scenario_id: str,
    update: ScenarioUpdate,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
) -> ScenarioResponse:
    """Update an existing scenario."""
    repo = Repository(Scenario)
    existing = await repo.get_by_id(scenario_id, user_id=x_user_id)

    if not existing:
        raise HTTPException(status_code=404, detail="Scenario not found")

    # Update only provided fields
    update_data = update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        if value is not None:
            setattr(existing, key, value)

    existing.updated_at = datetime.utcnow()

    # Re-generate embeddings if description changed
    generate_embeddings = "description" in update_data and update_data["description"]
    result = await repo.upsert(existing, generate_embeddings=generate_embeddings)
    return ScenarioResponse(**result.model_dump())


@router.delete("/{scenario_id}")
async def delete_scenario(
    scenario_id: str,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
) -> dict[str, Any]:
    """Soft delete a scenario."""
    repo = Repository(Scenario)
    success = await repo.delete(scenario_id, tenant_id=x_tenant_id)

    if not success:
        raise HTTPException(status_code=404, detail="Scenario not found")

    return {"status": "success", "message": "Scenario deleted"}


@router.post("/search", response_model=list[ScenarioResponse])
async def search_scenarios(
    search: ScenarioSearch,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> list[ScenarioResponse]:
    """
    Search scenarios with multiple filter options.

    Search supports:
    - Semantic search via `query` (searches description with embeddings)
    - Tag filtering with `tags` (match any or all)
    - Name/title contains filter
    - Agent name filter
    - Status filter
    - Date range filters (created_after, created_before)
    - Pagination (limit, offset)
    """
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
            # All tags must be present
            conditions.append(f"tags @> ${param_idx}::text[]")
        else:
            # Any tag must be present
            conditions.append(f"tags && ${param_idx}::text[]")
        params.append(search.tags)
        param_idx += 1

    # Name contains
    if search.name_contains:
        conditions.append(f"name ILIKE ${param_idx}")
        params.append(f"%{search.name_contains}%")
        param_idx += 1

    # Agent name filter
    if search.agent_name:
        conditions.append(f"agent_name = ${param_idx}")
        params.append(search.agent_name)
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
        # Generate embedding for the query
        query_embedding = await generate_embedding_async(search.query)
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = f"""
            SELECT *,
                   1 - (embedding <=> ${param_idx}::vector) as similarity
            FROM scenarios
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
            SELECT * FROM scenarios
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ${param_idx}
            OFFSET ${param_idx + 1}
        """
        params.extend([search.limit, search.offset])

    rows = await db.fetch(sql, *params)
    return [ScenarioResponse(**dict(row)) for row in rows]


@router.get("", response_model=list[ScenarioResponse])
async def list_scenarios(
    limit: int = 20,
    offset: int = 0,
    status: str | None = None,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> list[ScenarioResponse]:
    """List scenarios with optional status filter."""
    repo = Repository(Scenario)
    filters: dict[str, Any] = {}

    if x_user_id:
        filters["user_id"] = x_user_id
    if status:
        filters["status"] = status

    results = await repo.find(
        filters,
        order_by="created_at DESC",
        limit=limit,
        offset=offset,
    )
    return [ScenarioResponse(**r.model_dump()) for r in results]


# =============================================================================
# Feedback Endpoint (Phoenix Integration)
# =============================================================================


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    x_user_id: str | None = Header(None, alias="X-User-Id"),
) -> FeedbackResponse:
    """
    Submit feedback on a trace/span via Phoenix.

    This endpoint sends annotation data to Phoenix's span annotation API.
    Feedback can include:
    - score: Numeric rating (0.0 to 1.0)
    - label: Categorical label (e.g., "thumbs_up", "relevant")
    - comment: Free text feedback

    Either trace_id or span_id should be provided to associate
    the feedback with a specific trace or span.
    """
    if not settings.otel.enabled:
        return FeedbackResponse(
            status="skipped",
            message="OpenTelemetry/Phoenix integration is disabled",
        )

    if not feedback.trace_id and not feedback.span_id:
        raise HTTPException(
            status_code=400,
            detail="Either trace_id or span_id must be provided",
        )

    # Build annotation payload for Phoenix
    annotation_data: dict[str, Any] = {
        "name": feedback.name,
        "annotator_kind": "HUMAN",
    }

    if feedback.trace_id:
        annotation_data["trace_id"] = feedback.trace_id
    if feedback.span_id:
        annotation_data["span_id"] = feedback.span_id
    if feedback.score is not None:
        annotation_data["score"] = feedback.score
    if feedback.label:
        annotation_data["label"] = feedback.label
    if feedback.comment:
        annotation_data["explanation"] = feedback.comment

    # Add metadata
    metadata = feedback.metadata.copy()
    if x_user_id:
        metadata["user_id"] = x_user_id
    if metadata:
        annotation_data["metadata"] = metadata

    # Send to Phoenix annotation API
    phoenix_endpoint = settings.otel.collector_endpoint.rstrip("/")
    annotation_url = f"{phoenix_endpoint}/v1/span_annotations"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                annotation_url,
                json={"data": [annotation_data]},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code in (200, 201):
                result = response.json()
                annotation_id = None
                if isinstance(result, dict) and "data" in result:
                    data = result["data"]
                    if isinstance(data, list) and len(data) > 0:
                        annotation_id = data[0].get("id")

                return FeedbackResponse(
                    status="success",
                    message="Feedback submitted to Phoenix",
                    annotation_id=annotation_id,
                )
            else:
                return FeedbackResponse(
                    status="error",
                    message=f"Phoenix returned {response.status_code}: {response.text}",
                )

    except httpx.RequestError as e:
        return FeedbackResponse(
            status="error",
            message=f"Failed to connect to Phoenix: {str(e)}",
        )
    except Exception as e:
        return FeedbackResponse(
            status="error",
            message=f"Unexpected error: {str(e)}",
        )
