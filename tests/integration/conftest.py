"""
Pytest configuration for integration tests.

Integration tests may require real external services:
- Database (PostgreSQL with pgvector)
- LLM APIs (optional, marked with @pytest.mark.llm)

Tests marked with `llm` require actual API calls and are skipped in pre-push hooks.

Usage:
    # Run all integration tests (requires database)
    POSTGRES__CONNECTION_STRING="postgresql://..." pytest tests/integration/

    # Run integration tests excluding LLM tests
    pytest tests/integration/ -m "not llm"

    # Run only LLM tests (requires API keys)
    pytest tests/integration/ -m "llm"
"""

import asyncio

import pytest


@pytest.fixture(scope="session")
def event_loop():
    """Create a session-scoped event loop.

    This ensures all async tests and fixtures share the same event loop,
    preventing 'Event loop is closed' errors when mixing sync and async tests.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db():
    """Provide a connected database service for integration tests.

    Uses get_db() singleton to ensure ContentService and other services
    share the same connection pool.
    """
    from remlight.services.database import get_db

    service = get_db()
    await service.connect()

    # Drop problematic triggers that may not have their functions
    await service.execute("DROP TRIGGER IF EXISTS ontologies_embedding_queue ON ontologies")
    await service.execute("DROP TRIGGER IF EXISTS resources_embedding_queue ON resources")
    await service.execute("DROP TRIGGER IF EXISTS messages_embedding_queue ON messages")
    await service.execute("DROP TRIGGER IF EXISTS agents_embedding_queue ON agents")

    yield service

    await service.disconnect()


@pytest.fixture
async def rem_service(db):
    """Provide a RemService instance for integration tests."""
    from remlight.services.rem.service import RemService

    return RemService(db)


@pytest.fixture
async def clean_test_data(db):
    """Fixture that cleans up test data after each test."""
    yield

    # Cleanup any test data
    await db.execute("DELETE FROM ontologies WHERE user_id = 'test-user'")
    await db.execute("DELETE FROM resources WHERE user_id = 'test-user'")
    await db.execute("DELETE FROM kv_store WHERE user_id = 'test-user'")


def pytest_collection_modifyitems(items):
    """Automatically add 'integration' marker to all tests in /integration/."""
    for item in items:
        if "/integration/" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
