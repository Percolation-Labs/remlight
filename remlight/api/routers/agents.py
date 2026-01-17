"""Agents router - List available agent schemas.

Provides endpoints to list and retrieve agent schema information
from the filesystem schemas directory.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from remlight.agentic.schema import schema_from_yaml_file

router = APIRouter(prefix="/agents", tags=["agents"])

# Schemas directory - relative to project root (remlight/remlight/api/routers/agents.py -> remlight/)
SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "schemas"


class AgentInfo(BaseModel):
    """Agent information response."""

    name: str
    title: str | None = None
    version: str
    enabled: bool = True
    source: str = "filesystem"
    description: str | None = None
    tags: list[str] = []


class AgentListResponse(BaseModel):
    """List of agents response."""

    agents: list[AgentInfo]


def get_schemas_dir() -> Path:
    """Get the schemas directory path."""
    return SCHEMAS_DIR


def load_agent_info(file_path: Path) -> AgentInfo | None:
    """Load agent info from a YAML schema file."""
    try:
        schema = schema_from_yaml_file(file_path)
        meta = schema.json_schema_extra

        # Get first line of description as title
        desc_lines = schema.description.strip().split("\n")
        title = desc_lines[0][:50] if desc_lines else meta.name

        return AgentInfo(
            name=meta.name,
            title=title,
            version=meta.version,
            enabled=True,
            source="filesystem",
            description=schema.description[:200] if schema.description else None,
            tags=meta.tags,
        )
    except Exception as e:
        print(f"Failed to load schema {file_path}: {e}")
        return None


@router.get("", response_model=AgentListResponse)
async def list_agents() -> AgentListResponse:
    """List all available agent schemas.

    Returns agent schemas from the filesystem schemas directory.
    """
    schemas_dir = get_schemas_dir()
    agents: list[AgentInfo] = []

    if schemas_dir.exists():
        for yaml_file in schemas_dir.glob("*.yaml"):
            info = load_agent_info(yaml_file)
            if info:
                agents.append(info)

    # Sort by name
    agents.sort(key=lambda a: a.name)

    return AgentListResponse(agents=agents)


@router.get("/{agent_name}", response_model=AgentInfo)
async def get_agent(agent_name: str) -> AgentInfo:
    """Get information about a specific agent.

    Args:
        agent_name: The agent schema name (without .yaml extension)
    """
    schemas_dir = get_schemas_dir()
    yaml_file = schemas_dir / f"{agent_name}.yaml"

    if not yaml_file.exists():
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    info = load_agent_info(yaml_file)
    if not info:
        raise HTTPException(
            status_code=500, detail=f"Failed to load agent '{agent_name}'"
        )

    return info
