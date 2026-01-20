"""
Integration tests for agent database functionality.

Tests:
- Agent CRUD operations via Repository
- Time machine versioning
- Semantic search for agents
- API endpoints for agents
"""

import pytest

from remlight.models.entities import Agent
from remlight.services.repository import Repository
from remlight.agentic.schema import schema_from_yaml


TEST_AGENT_YAML = """
type: object
description: |
  Test Agent for Database Integration

  This is a test agent used to verify database operations.
  It should support semantic search via the description field.

properties:
  answer:
    type: string
    description: The response from the test agent

required:
  - answer

json_schema_extra:
  kind: agent
  name: test-db-agent
  version: "1.0.0"
  tags:
    - test
    - integration
  tools:
    - name: search
"""

TEST_AGENT_YAML_V2 = """
type: object
description: |
  Updated Test Agent for Database Integration

  This is version 2 of the test agent with enhanced capabilities.
  It demonstrates the time machine versioning feature.

properties:
  answer:
    type: string
    description: The response from the updated test agent
  confidence:
    type: number
    description: Confidence score for the response

required:
  - answer

json_schema_extra:
  kind: agent
  name: test-db-agent
  version: "2.0.0"
  tags:
    - test
    - integration
    - v2
  tools:
    - name: search
    - name: action
"""


@pytest.mark.asyncio
async def test_create_agent(db):
    """Test creating a new agent."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    schema = schema_from_yaml(TEST_AGENT_YAML)
    meta = schema.json_schema_extra

    agent = Agent(
        name=meta.name,
        description="Test Agent for Database Integration",
        content=TEST_AGENT_YAML,
        version=meta.version,
        enabled=True,
        tags=meta.tags,
    )

    result = await repo.upsert(agent)

    assert result.id is not None
    assert result.name == "test-db-agent"
    assert result.version == "1.0.0"
    assert result.enabled is True

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_get_agent_by_name(db):
    """Test retrieving an agent by name."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    agent = Agent(
        name="test-get-agent",
        description="Test getting agent by name",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-get-agent"),
        version="1.0.0",
        enabled=True,
    )
    await repo.upsert(agent)

    # Retrieve
    retrieved = await repo.get_by_name("test-get-agent")

    assert retrieved is not None
    assert retrieved.name == "test-get-agent"
    assert "Test getting agent by name" in retrieved.description

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_update_agent(db):
    """Test updating an existing agent."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    agent = Agent(
        name="test-update-agent",
        description="Initial description",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-update-agent"),
        version="1.0.0",
        enabled=True,
    )
    await repo.upsert(agent)

    # Get and update
    existing = await repo.get_by_name("test-update-agent")
    existing.description = "Updated description"
    existing.version = "1.1.0"
    await repo.upsert(existing)

    # Verify update
    updated = await repo.get_by_name("test-update-agent")
    assert updated.description == "Updated description"
    assert updated.version == "1.1.0"

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_list_enabled_agents(db):
    """Test listing only enabled agents."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    # Create enabled agent
    enabled = Agent(
        name="test-enabled-agent",
        description="Enabled agent",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-enabled-agent"),
        enabled=True,
    )
    await repo.upsert(enabled)

    # Create disabled agent
    disabled = Agent(
        name="test-disabled-agent",
        description="Disabled agent",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-disabled-agent"),
        enabled=False,
    )
    await repo.upsert(disabled)

    # List enabled only
    agents = await repo.find({"enabled": True})
    names = [a.name for a in agents]

    assert "test-enabled-agent" in names
    assert "test-disabled-agent" not in names

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_soft_delete_agent(db):
    """Test soft deleting an agent."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    agent = Agent(
        name="test-delete-agent",
        description="Agent to delete",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-delete-agent"),
    )
    result = await repo.upsert(agent)

    # Soft delete
    deleted = await repo.delete(str(result.id))
    assert deleted is True

    # Should not be found
    retrieved = await repo.get_by_name("test-delete-agent")
    assert retrieved is None

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_time_machine_on_create(db):
    """Test that time machine records creation."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    agent = Agent(
        name="test-tm-create",
        description="Time machine create test",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-tm-create"),
        version="1.0.0",
    )
    await repo.upsert(agent)

    # Check time machine
    rows = await db.fetch(
        "SELECT * FROM agent_timemachine WHERE agent_name = $1",
        "test-tm-create"
    )

    assert len(rows) == 1
    assert rows[0]["change_type"] == "created"
    assert rows[0]["version"] == "1.0.0"

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_time_machine_on_update(db):
    """Test that time machine records updates with content changes."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    # Create initial
    agent = Agent(
        name="test-tm-update",
        description="Time machine update test v1",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-tm-update"),
        version="1.0.0",
    )
    await repo.upsert(agent)

    # Update content
    existing = await repo.get_by_name("test-tm-update")
    existing.content = TEST_AGENT_YAML_V2.replace("test-db-agent", "test-tm-update")
    existing.description = "Time machine update test v2"
    existing.version = "2.0.0"
    await repo.upsert(existing)

    # Check time machine has both versions
    rows = await db.fetch(
        "SELECT * FROM agent_timemachine WHERE agent_name = $1 ORDER BY created_at",
        "test-tm-update"
    )

    assert len(rows) == 2
    assert rows[0]["change_type"] == "created"
    assert rows[0]["version"] == "1.0.0"
    assert rows[1]["change_type"] == "updated"
    assert rows[1]["version"] == "2.0.0"

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_time_machine_no_duplicate_on_same_content(db):
    """Test that time machine doesn't create entries for unchanged content."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    agent = Agent(
        name="test-tm-nodupe",
        description="No duplicate test",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-tm-nodupe"),
        version="1.0.0",
    )
    await repo.upsert(agent)

    # Update without changing content
    existing = await repo.get_by_name("test-tm-nodupe")
    existing.description = "Different description"  # Only description changes
    await repo.upsert(existing)

    # Should only have one entry (content hash unchanged)
    rows = await db.fetch(
        "SELECT * FROM agent_timemachine WHERE agent_name = $1",
        "test-tm-nodupe"
    )

    assert len(rows) == 1  # Only the creation entry

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
@pytest.mark.llm  # Requires embedding API
async def test_semantic_search_finds_agent(db):
    """Test semantic search can find agents by description."""
    from remlight.services.embeddings import generate_embedding_async

    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    # Create agent with specific description
    agent = Agent(
        name="test-search-ml-agent",
        description="Machine learning assistant for data science tasks",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-search-ml-agent"),
        tags=["ml", "data-science"],
    )
    await repo.upsert(agent, generate_embeddings=True)

    # Generate query embedding
    query = "help with machine learning"
    embedding = await generate_embedding_async(query)

    # Convert embedding list to pgvector format string
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    # Search
    results = await db.rem_search(
        embedding=embedding_str,
        table_name="agents",
        limit=5,
        min_similarity=0.3,
    )

    names = [r.get("name") for r in results]
    assert "test-search-ml-agent" in names

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_fuzzy_name_search(db):
    """Test fuzzy name search via rem_fuzzy."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    agent = Agent(
        name="test-fuzzy-query-agent",
        description="Agent for fuzzy search testing",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-fuzzy-query-agent"),
    )
    await repo.upsert(agent)

    # Fuzzy search
    results = await db.rem_fuzzy(
        query_text="fuzzy-query",
        threshold=0.3,
        limit=10,
    )

    keys = [r.get("entity_key") for r in results]
    assert any("fuzzy" in k for k in keys)

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")


@pytest.mark.asyncio
async def test_agent_synced_to_kv_store(db):
    """Test that agents are automatically synced to kv_store."""
    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")

    repo = Repository(Agent, table_name="agents", db=db)
    agent = Agent(
        name="test-kv-agent",
        description="KV store sync test",
        content=TEST_AGENT_YAML.replace("test-db-agent", "test-kv-agent"),
    )
    await repo.upsert(agent)

    # Check kv_store - rem_lookup returns dict or empty dict
    kv_data = await db.rem_lookup("test-kv-agent")

    assert kv_data is not None
    assert kv_data != {}
    # kv_data should be a dict with agent data
    if isinstance(kv_data, dict):
        assert kv_data.get("name") == "test-kv-agent"
    else:
        # If it's a string (JSON), parse it
        import json
        data = json.loads(kv_data) if kv_data else {}
        assert data.get("name") == "test-kv-agent"

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name LIKE 'test-%'")
    await db.execute("DELETE FROM kv_store WHERE entity_key LIKE 'test-%'")
    await db.execute("DELETE FROM agents WHERE name LIKE 'test-%'")
