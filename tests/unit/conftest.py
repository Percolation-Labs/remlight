"""
Pytest configuration and fixtures for remlight unit tests.

Unit tests MUST be isolated from external dependencies:
- No database connections
- No LLM API calls
- No external services

All external dependencies must be mocked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def mock_database_service():
    """
    Mock DatabaseService for all unit tests.

    This prevents any database connection attempts during unit testing.
    The mock is applied at the module level where DatabaseService is imported.
    """
    mock_service = MagicMock()
    mock_service.connect = AsyncMock()
    mock_service.disconnect = AsyncMock()
    mock_service.execute = AsyncMock(return_value="OK")
    mock_service.fetch = AsyncMock(return_value=[])
    mock_service.fetchrow = AsyncMock(return_value=None)
    mock_service.fetchval = AsyncMock(return_value=None)
    mock_service.pool = MagicMock()

    # Mock REM functions
    mock_service.rem_lookup = AsyncMock(return_value={})
    mock_service.rem_fuzzy = AsyncMock(return_value=[])
    mock_service.rem_search = AsyncMock(return_value=[])
    mock_service.rem_traverse = AsyncMock(return_value=[])

    with patch("remlight.services.database.DatabaseService", return_value=mock_service):
        with patch("remlight.services.database.get_db", return_value=mock_service):
            yield mock_service


@pytest.fixture
def mock_embed_fn():
    """Mock embedding function for SEARCH tests."""
    async def embed(text: str) -> list[float]:
        # Return a fake 1536-dim embedding
        return [0.1] * 1536

    return embed


def pytest_collection_modifyitems(items):
    """Automatically add 'unit' marker to all tests in /unit/."""
    for item in items:
        if "/unit/" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
