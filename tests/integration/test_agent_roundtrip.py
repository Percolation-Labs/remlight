"""
Test that agent YAML content round-trips correctly through the database.

This verifies that:
1. Full YAML content is stored as-is
2. Loading from database gives identical content to loading from file
3. The schema can be parsed identically from both sources
"""

import pytest
from pathlib import Path

from remlight.models.entities import Agent
from remlight.services.repository import Repository
from remlight.agentic.schema import schema_from_yaml, schema_from_yaml_file


SCHEMAS_DIR = Path(__file__).parent.parent.parent / "schemas"


@pytest.mark.asyncio
async def test_agent_yaml_roundtrip(db):
    """Test that agent YAML content is stored and retrieved exactly as-is."""
    # Read actual agent file
    yaml_file = SCHEMAS_DIR / "query-agent.yaml"
    original_content = yaml_file.read_text()

    # Parse original
    original_schema = schema_from_yaml_file(yaml_file)

    # Modify name to avoid conflicts
    test_content = original_content.replace("name: query-agent", "name: roundtrip-test-agent")
    test_schema = schema_from_yaml(test_content)
    meta = test_schema.json_schema_extra

    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name = 'roundtrip-test-agent'")
    await db.execute("DELETE FROM kv_store WHERE entity_key = 'roundtrip-test-agent'")
    await db.execute("DELETE FROM agents WHERE name = 'roundtrip-test-agent'")

    # Save to database - content is the full YAML
    repo = Repository(Agent, table_name='agents', db=db)

    # description field is just for search - extract first line
    desc_line = test_schema.description.strip().split('\n')[0][:200]

    agent = Agent(
        name=meta.name,
        description=desc_line,  # For search indexing only
        content=test_content,   # Full YAML - source of truth
        version=meta.version,
        enabled=True,
        tags=meta.tags,
    )

    await repo.upsert(agent)

    # Load from database
    loaded = await repo.get_by_name("roundtrip-test-agent")

    # Content should be identical
    assert loaded.content == test_content, "Content should match exactly"

    # Parse loaded content - should work exactly like from file
    loaded_schema = schema_from_yaml(loaded.content)

    # Verify all schema fields match
    assert loaded_schema.json_schema_extra.name == meta.name
    assert loaded_schema.json_schema_extra.version == meta.version
    assert loaded_schema.description == test_schema.description
    assert len(loaded_schema.json_schema_extra.tools) == len(meta.tools)

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name = 'roundtrip-test-agent'")
    await db.execute("DELETE FROM kv_store WHERE entity_key = 'roundtrip-test-agent'")
    await db.execute("DELETE FROM agents WHERE name = 'roundtrip-test-agent'")


@pytest.mark.asyncio
async def test_agent_full_prompt_preserved(db):
    """Test that the full prompt/description is preserved in content."""
    yaml_content = '''
type: object
description: |
  You are a specialized assistant with a very long and detailed prompt.

  This prompt contains multiple paragraphs and specific instructions
  that must be preserved exactly when stored in the database.

  ## Instructions
  1. First instruction
  2. Second instruction
  3. Third instruction

  ## Context
  You have access to various tools and should use them appropriately.

  Remember: This entire prompt must be preserved character-for-character.

properties:
  answer:
    type: string
    description: Your response

required:
  - answer

json_schema_extra:
  kind: agent
  name: prompt-preservation-test
  version: "1.0.0"
  tags:
    - test
  tools:
    - name: search
'''

    # Clean before
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name = 'prompt-preservation-test'")
    await db.execute("DELETE FROM kv_store WHERE entity_key = 'prompt-preservation-test'")
    await db.execute("DELETE FROM agents WHERE name = 'prompt-preservation-test'")

    schema = schema_from_yaml(yaml_content)
    meta = schema.json_schema_extra

    repo = Repository(Agent, table_name='agents', db=db)

    agent = Agent(
        name=meta.name,
        description=schema.description.strip().split('\n')[0][:200],
        content=yaml_content,  # Full YAML with complete prompt
        version=meta.version,
        enabled=True,
    )

    await repo.upsert(agent)

    # Load and verify
    loaded = await repo.get_by_name("prompt-preservation-test")
    loaded_schema = schema_from_yaml(loaded.content)

    # The full description/prompt must be preserved
    assert loaded_schema.description == schema.description
    assert "## Instructions" in loaded_schema.description
    assert "## Context" in loaded_schema.description
    assert "character-for-character" in loaded_schema.description

    # Clean after
    await db.execute("DELETE FROM agent_timemachine WHERE agent_name = 'prompt-preservation-test'")
    await db.execute("DELETE FROM kv_store WHERE entity_key = 'prompt-preservation-test'")
    await db.execute("DELETE FROM agents WHERE name = 'prompt-preservation-test'")
