"""Agents router - List, retrieve, upsert, and search agent schemas.

Provides endpoints to:
- List all available agents (merged from filesystem and database)
- Get agent information by name
- Upsert agents to the database
- Search agents semantically or by name/tags

Loading precedence (default: file-first):
- File-based agents from schemas directory take priority
- Database agents are used if not found in filesystem
- database_first=True reverses this, preferring database over files
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from remlight.agentic.schema import schema_from_yaml, schema_from_yaml_file
from remlight.models.entities import Agent
from remlight.services.embeddings import generate_embedding_async
from remlight.services.repository import Repository

router = APIRouter(prefix="/agents", tags=["agents"])

# Schemas directory - relative to project root (remlight/remlight/api/routers/agents.py -> remlight/)
SCHEMAS_DIR = Path(__file__).parent.parent.parent.parent / "schemas"


class AgentInfo(BaseModel):
    """Agent information response."""

    name: str
    title: str | None = None
    version: str
    enabled: bool = True
    source: str = "filesystem"  # 'filesystem' or 'database'
    description: str | None = None
    icon: str | None = None
    tags: list[str] = []


class AgentListResponse(BaseModel):
    """List of agents response."""

    agents: list[AgentInfo]


class AgentUpsertRequest(BaseModel):
    """Request to upsert an agent."""

    content: str = Field(..., description="Full YAML content of the agent schema")
    enabled: bool = Field(default=True, description="Whether the agent is active")
    icon: str | None = Field(default=None, description="Icon URL or emoji")
    tags: list[str] = Field(default_factory=list, description="Classification tags")
    save_to_file: bool = Field(default=True, description="Also save to schemas/ directory")


class AgentUpsertResponse(BaseModel):
    """Response after upserting an agent."""

    name: str
    version: str
    created: bool  # True if new, False if updated
    message: str


class AgentSearchRequest(BaseModel):
    """Request to search for agents."""

    query: str = Field(..., description="Search query (semantic, name, or tags)")
    search_type: str = Field(
        default="semantic",
        description="Type of search: 'semantic', 'name', 'tags', or 'all'"
    )
    limit: int = Field(default=10, ge=1, le=100)
    include_disabled: bool = Field(default=False)


class AgentSearchResult(BaseModel):
    """Search result for an agent."""

    name: str
    title: str | None = None
    version: str
    description: str | None = None
    icon: str | None = None
    tags: list[str] = []
    source: str
    similarity: float | None = None  # Only for semantic search


class AgentSearchResponse(BaseModel):
    """Response for agent search."""

    results: list[AgentSearchResult]
    total: int


class AgentContentResponse(BaseModel):
    """Response with full agent content."""

    name: str
    version: str
    content: str
    source: str
    enabled: bool = True
    icon: str | None = None
    tags: list[str] = []


def get_schemas_dir() -> Path:
    """Get the schemas directory path."""
    return SCHEMAS_DIR


def load_agent_info_from_file(file_path: Path) -> AgentInfo | None:
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
        logger.warning(f"Failed to load schema {file_path}: {e}")
        return None


def load_agent_info_from_db(agent: Agent) -> AgentInfo:
    """Convert database Agent to AgentInfo."""
    # Parse YAML to get metadata
    try:
        schema = schema_from_yaml(agent.content)
        meta = schema.json_schema_extra
        desc_lines = schema.description.strip().split("\n")
        title = desc_lines[0][:50] if desc_lines else meta.name
        tags = meta.tags
    except Exception:
        title = agent.name
        tags = agent.tags

    return AgentInfo(
        name=agent.name,
        title=title,
        version=agent.version,
        enabled=agent.enabled,
        source="database",
        description=agent.description[:200] if agent.description else None,
        icon=agent.icon,
        tags=tags,
    )


async def get_file_agents() -> dict[str, AgentInfo]:
    """Get all agents from filesystem."""
    schemas_dir = get_schemas_dir()
    agents: dict[str, AgentInfo] = {}

    if schemas_dir.exists():
        for yaml_file in schemas_dir.glob("*.yaml"):
            info = load_agent_info_from_file(yaml_file)
            if info:
                agents[info.name] = info

    return agents


async def get_db_agents(include_disabled: bool = False) -> dict[str, AgentInfo]:
    """Get all agents from database."""
    repo = Repository(Agent, table_name="agents")
    agents: dict[str, AgentInfo] = {}

    try:
        filters: dict[str, Any] = {}
        if not include_disabled:
            filters["enabled"] = True

        db_agents = await repo.find(filters)
        for agent in db_agents:
            info = load_agent_info_from_db(agent)
            agents[info.name] = info
    except Exception as e:
        logger.warning(f"Failed to load agents from database: {e}")

    return agents


@router.get("", response_model=AgentListResponse)
async def list_agents(
    database_first: bool = Query(
        default=False,
        description="If True, prefer database agents over filesystem"
    )
) -> AgentListResponse:
    """List all available agent schemas.

    Merges agents from filesystem and database with deduplication.
    Default precedence: filesystem > database (override with database_first=True)
    """
    file_agents = await get_file_agents()
    db_agents = await get_db_agents()

    # Merge with appropriate precedence
    if database_first:
        # Database takes priority - start with file, overlay with db
        merged = {**file_agents, **db_agents}
    else:
        # File takes priority (default) - start with db, overlay with file
        merged = {**db_agents, **file_agents}

    agents = list(merged.values())
    agents.sort(key=lambda a: a.name)

    return AgentListResponse(agents=agents)


@router.get("/{agent_name}", response_model=AgentInfo)
async def get_agent(
    agent_name: str,
    database_first: bool = Query(
        default=False,
        description="If True, check database before filesystem"
    )
) -> AgentInfo:
    """Get information about a specific agent.

    Checks filesystem first (default), falls back to database.
    Use database_first=True to reverse precedence.
    """
    schemas_dir = get_schemas_dir()
    yaml_file = schemas_dir / f"{agent_name}.yaml"

    file_info = None
    db_info = None

    # Load from filesystem
    if yaml_file.exists():
        file_info = load_agent_info_from_file(yaml_file)

    # Load from database
    repo = Repository(Agent, table_name="agents")
    try:
        db_agent = await repo.get_by_name(agent_name)
        if db_agent:
            db_info = load_agent_info_from_db(db_agent)
    except Exception as e:
        logger.warning(f"Failed to check database for agent '{agent_name}': {e}")

    # Apply precedence
    if database_first:
        info = db_info or file_info
    else:
        info = file_info or db_info

    if not info:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return info


@router.get("/{agent_name}/content", response_model=AgentContentResponse)
async def get_agent_content(
    agent_name: str,
    database_first: bool = Query(
        default=False,
        description="If True, check database before filesystem"
    )
) -> AgentContentResponse:
    """Get full YAML content of an agent.

    Returns the complete YAML schema content.
    """
    schemas_dir = get_schemas_dir()
    yaml_file = schemas_dir / f"{agent_name}.yaml"

    file_content = None
    db_agent = None

    # Load from filesystem
    if yaml_file.exists():
        try:
            file_content = yaml_file.read_text()
            schema = schema_from_yaml(file_content)
            meta = schema.json_schema_extra
            file_result = AgentContentResponse(
                name=meta.name,
                version=meta.version,
                content=file_content,
                source="filesystem",
                enabled=True,
                tags=meta.tags,
            )
        except Exception as e:
            logger.warning(f"Failed to load file content for '{agent_name}': {e}")
            file_result = None
    else:
        file_result = None

    # Load from database
    repo = Repository(Agent, table_name="agents")
    try:
        db_agent = await repo.get_by_name(agent_name)
        if db_agent:
            db_result = AgentContentResponse(
                name=db_agent.name,
                version=db_agent.version,
                content=db_agent.content,
                source="database",
                enabled=db_agent.enabled,
                icon=db_agent.icon,
                tags=db_agent.tags,
            )
        else:
            db_result = None
    except Exception as e:
        logger.warning(f"Failed to check database for agent '{agent_name}': {e}")
        db_result = None

    # Apply precedence
    if database_first:
        result = db_result or file_result
    else:
        result = file_result or db_result

    if not result:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

    return result


@router.put("", response_model=AgentUpsertResponse)
async def upsert_agent(request: AgentUpsertRequest) -> AgentUpsertResponse:
    """Create or update an agent in the database and optionally filesystem.

    The agent name is extracted from the YAML content (json_schema_extra.name).
    If an agent with that name exists, it will be updated.
    Time machine trigger automatically records version history on changes.

    When save_to_file=True (default), also writes to schemas/{name}.yaml.
    This ensures filesystem-first agents get updated.
    """
    # Parse YAML to extract metadata
    try:
        schema = schema_from_yaml(request.content)
        meta = schema.json_schema_extra
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid YAML content: {e}"
        )

    # Get first line of description for searchable description
    desc_lines = schema.description.strip().split("\n")
    description = desc_lines[0][:200] if desc_lines else None

    repo = Repository(Agent, table_name="agents")

    # Check if agent exists
    existing = await repo.get_by_name(meta.name)
    created = existing is None

    # Create/update agent in database
    agent = Agent(
        id=existing.id if existing else None,
        name=meta.name,
        description=description,
        content=request.content,
        version=meta.version,
        enabled=request.enabled,
        icon=request.icon,
        tags=request.tags or meta.tags,
    )

    await repo.upsert(agent, conflict_field="name" if not existing else "id")

    # Also save to filesystem if requested
    if request.save_to_file:
        schemas_dir = get_schemas_dir()
        schemas_dir.mkdir(parents=True, exist_ok=True)
        yaml_file = schemas_dir / f"{meta.name}.yaml"
        try:
            yaml_file.write_text(request.content)
            logger.info(f"Saved agent '{meta.name}' to {yaml_file}")
        except Exception as e:
            logger.warning(f"Failed to save agent to file: {e}")
            # Don't fail the request - database save succeeded

    return AgentUpsertResponse(
        name=meta.name,
        version=meta.version,
        created=created,
        message=f"Agent '{meta.name}' {'created' if created else 'updated'} successfully"
    )


@router.post("/search", response_model=AgentSearchResponse)
async def search_agents(request: AgentSearchRequest) -> AgentSearchResponse:
    """Search for agents semantically or by name/tags.

    Search types:
    - semantic: Vector similarity search on description
    - name: Fuzzy/exact name matching
    - tags: Match by tags
    - all: Combine all search methods
    """
    from remlight.services.database import get_db

    results: list[AgentSearchResult] = []
    db = get_db()

    # Ensure connection
    if not db.pool:
        await db.connect()

    try:
        if request.search_type in ("semantic", "all"):
            # Generate embedding for query
            embedding = await generate_embedding_async(request.query)

            # Search via rem_search
            search_results = await db.rem_search(
                embedding=embedding,
                table_name="agents",
                limit=request.limit,
                min_similarity=0.3,
                user_id=None,
            )

            for row in search_results:
                data = row.get("data", {})
                results.append(AgentSearchResult(
                    name=row.get("name") or data.get("name", ""),
                    title=data.get("name"),
                    version=data.get("version", "1.0.0"),
                    description=row.get("content"),
                    tags=data.get("tags", []),
                    source="database",
                    similarity=row.get("similarity"),
                ))

        if request.search_type in ("name", "all"):
            # Fuzzy name search via rem_fuzzy
            fuzzy_results = await db.rem_fuzzy(
                query_text=request.query,
                user_id=None,
                threshold=0.3,
                limit=request.limit,
            )

            for row in fuzzy_results:
                if row.get("entity_type") == "agents":
                    data = row.get("data", {})
                    name = data.get("name", row.get("entity_key", ""))
                    # Avoid duplicates
                    if not any(r.name == name for r in results):
                        results.append(AgentSearchResult(
                            name=name,
                            title=data.get("name"),
                            version=data.get("version", "1.0.0"),
                            description=data.get("description"),
                            tags=data.get("tags", []),
                            source="database",
                            similarity=row.get("similarity"),
                        ))

        if request.search_type in ("tags", "all"):
            # Tag-based search
            tag_query = """
                SELECT * FROM agents
                WHERE $1 = ANY(tags)
                  AND deleted_at IS NULL
                  AND (enabled = TRUE OR $2 = TRUE)
                LIMIT $3
            """
            tag_results = await db.fetch(
                tag_query,
                request.query.lower(),
                request.include_disabled,
                request.limit,
            )

            for row in tag_results:
                name = row.get("name", "")
                # Avoid duplicates
                if not any(r.name == name for r in results):
                    results.append(AgentSearchResult(
                        name=name,
                        title=name,
                        version=row.get("version", "1.0.0"),
                        description=row.get("description"),
                        tags=row.get("tags", []),
                        source="database",
                        similarity=None,
                    ))

    except Exception as e:
        logger.error(f"Agent search failed: {e}")
        # Fall back to basic file search
        file_agents = await get_file_agents()
        query_lower = request.query.lower()
        for info in file_agents.values():
            if query_lower in info.name.lower() or (
                info.description and query_lower in info.description.lower()
            ) or any(query_lower in tag.lower() for tag in info.tags):
                results.append(AgentSearchResult(
                    name=info.name,
                    title=info.title,
                    version=info.version,
                    description=info.description,
                    tags=info.tags,
                    source="filesystem",
                    similarity=None,
                ))

    # Deduplicate and limit
    seen = set()
    unique_results = []
    for r in results:
        if r.name not in seen:
            seen.add(r.name)
            unique_results.append(r)
            if len(unique_results) >= request.limit:
                break

    return AgentSearchResponse(
        results=unique_results,
        total=len(unique_results),
    )


@router.delete("/{agent_name}")
async def delete_agent(agent_name: str) -> dict[str, str]:
    """Delete an agent from the database (soft delete).

    Note: This only affects database agents. Filesystem agents cannot be deleted via API.
    """
    repo = Repository(Agent, table_name="agents")

    existing = await repo.get_by_name(agent_name)
    if not existing:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found in database"
        )

    await repo.delete(str(existing.id))

    return {"message": f"Agent '{agent_name}' deleted successfully"}


@router.get("/{agent_name}/history")
async def get_agent_history(
    agent_name: str,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Get version history for an agent from the time machine.

    Returns historical versions with timestamps and change types.
    """
    from remlight.services.database import get_db

    db = get_db()
    if not db.pool:
        await db.connect()

    query = """
        SELECT id, agent_id, agent_name, version, content_hash, change_type,
               metadata, user_id, created_at
        FROM agent_timemachine
        WHERE agent_name = $1
        ORDER BY created_at DESC
        LIMIT $2
    """

    rows = await db.fetch(query, agent_name, limit)

    return [
        {
            "id": str(row["id"]),
            "agent_id": str(row["agent_id"]),
            "version": row["version"],
            "content_hash": row["content_hash"],
            "change_type": row["change_type"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
        for row in rows
    ]


@router.get("/{agent_name}/history/{history_id}/content")
async def get_agent_history_content(
    agent_name: str,
    history_id: str,
) -> AgentContentResponse:
    """Get the full content of a specific historical version.

    Use this to view or restore a previous version of an agent.
    """
    from remlight.services.database import get_db

    db = get_db()
    if not db.pool:
        await db.connect()

    query = """
        SELECT * FROM agent_timemachine
        WHERE id = $1 AND agent_name = $2
    """

    row = await db.fetchrow(query, history_id, agent_name)

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"History entry '{history_id}' not found for agent '{agent_name}'"
        )

    return AgentContentResponse(
        name=row["agent_name"],
        version=row["version"] or "1.0.0",
        content=row["content"],
        source="timemachine",
        enabled=True,
        tags=[],
    )
