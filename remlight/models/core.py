"""Core model base class for all REMLight entities."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class CoreModel(BaseModel):
    """Base model for all REMLight entities with system fields."""

    id: UUID | str | None = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: datetime | None = None

    tenant_id: str | None = None
    user_id: str | None = None

    graph_edges: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    model_config = {"from_attributes": True}
