"""
Integration tests for REM dialect queries.

These tests require a real PostgreSQL database with pgvector extension.
Run with: pytest tests/integration/ -v

For dockerized tests:
    docker compose -f docker-compose.test.yml up -d
    POSTGRES__CONNECTION_STRING="postgresql://remlight:remlight@localhost:5433/remlight" pytest tests/integration/
    docker compose -f docker-compose.test.yml down -v
"""

import pytest


@pytest.mark.asyncio
async def test_lookup_existing_entity(db, rem_service, clean_test_data):
    """Test LOOKUP finds an existing entity."""
    # Insert test data
    await db.execute("""
        INSERT INTO ontologies (name, description, category, user_id)
        VALUES ('Sarah Chen', 'A software engineer', 'person', 'test-user')
    """)

    # Query via REM service
    result = await rem_service.execute('LOOKUP "sarah-chen"')

    assert result.count == 1
    assert result.data[0]["name"] == "Sarah Chen"


@pytest.mark.asyncio
async def test_lookup_nonexistent_entity(db, rem_service):
    """Test LOOKUP returns empty for nonexistent entity."""
    result = await rem_service.execute('LOOKUP "definitely-does-not-exist-xyz"')

    assert result.count == 0
    assert result.data == []


@pytest.mark.asyncio
async def test_fuzzy_search(db, rem_service, clean_test_data):
    """Test FUZZY finds similar entities."""
    # Insert test data
    await db.execute("""
        INSERT INTO ontologies (name, description, category, user_id)
        VALUES
            ('Project Alpha', 'Main project', 'project', 'test-user'),
            ('Project Beta', 'Secondary project', 'project', 'test-user'),
            ('Something Else', 'Not a project', 'other', 'test-user')
    """)

    # Fuzzy search should find project-related entries
    result = await rem_service.execute('FUZZY "project" THRESHOLD 0.2 LIMIT 10')

    assert result.count >= 2
    # Results should be ordered by similarity
    entity_keys = [r.get("entity_key", "") for r in result.data]
    assert any("project" in key for key in entity_keys)


@pytest.mark.asyncio
async def test_traverse_graph(db, rem_service, clean_test_data):
    """Test TRAVERSE follows graph edges."""
    # Insert entities with graph edges
    await db.execute("""
        INSERT INTO ontologies (name, description, category, graph_edges, user_id)
        VALUES
            ('Manager', 'A manager', 'person',
             '[{"target": "employee-1", "type": "manages"}]'::jsonb, 'test-user'),
            ('Employee 1', 'An employee', 'person', '[]'::jsonb, 'test-user')
    """)

    # Traverse from manager
    result = await rem_service.execute('TRAVERSE "manager" DEPTH 1')

    assert result.count >= 1


@pytest.mark.asyncio
async def test_sql_select_query(db, rem_service, clean_test_data):
    """Test SQL SELECT query execution."""
    # Insert test data
    await db.execute("""
        INSERT INTO ontologies (name, description, category, user_id)
        VALUES ('SQL Test Entity', 'For SQL testing', 'test', 'test-user')
    """)

    # Run SQL query
    result = await rem_service.execute(
        "SELECT name, category FROM ontologies WHERE user_id = 'test-user' AND category = 'test'"
    )

    assert result.count >= 1
    assert result.data[0]["name"] == "SQL Test Entity"


@pytest.mark.asyncio
async def test_kv_store_trigger(db, clean_test_data):
    """Test that KV store is automatically populated by trigger."""
    # Insert into ontologies
    await db.execute("""
        INSERT INTO ontologies (name, description, category, user_id)
        VALUES ('Trigger Test', 'Testing KV trigger', 'test', 'test-user')
    """)

    # Check KV store was populated
    row = await db.fetchrow(
        "SELECT * FROM kv_store WHERE entity_key = 'trigger-test'"
    )

    assert row is not None
    assert row["entity_type"] == "ontologies"


@pytest.mark.asyncio
async def test_multiple_queries_same_session(db, rem_service, clean_test_data):
    """Test multiple queries in same session work correctly."""
    # Insert test data
    await db.execute("""
        INSERT INTO ontologies (name, description, category, user_id)
        VALUES ('Multi Query Test', 'Testing multiple queries', 'test', 'test-user')
    """)

    # Run multiple queries
    lookup_result = await rem_service.execute('LOOKUP "multi-query-test"')
    fuzzy_result = await rem_service.execute('FUZZY "multi" THRESHOLD 0.2')
    sql_result = await rem_service.execute(
        "SELECT * FROM ontologies WHERE name = 'Multi Query Test'"
    )

    assert lookup_result.count == 1
    assert fuzzy_result.count >= 1
    assert sql_result.count >= 1
