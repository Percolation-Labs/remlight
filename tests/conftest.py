"""
Pytest configuration and shared fixtures for remlight tests.

Test Organization:
- tests/unit/     - Mock-only tests, no external dependencies
- tests/integration/ - Tests requiring database/external services
"""

from pathlib import Path

import pytest


@pytest.fixture
def tests_data_dir() -> Path:
    """Path to tests/data directory for test fixtures."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def sample_ontology() -> dict:
    """Sample ontology entity for testing."""
    return {
        "name": "Test Entity",
        "description": "A test entity for unit tests",
        "category": "test",
        "properties": {"key": "value"},
        "tags": ["test", "sample"],
    }


@pytest.fixture
def sample_resource() -> dict:
    """Sample resource entity for testing."""
    return {
        "name": "Test Resource",
        "uri": "file://test/document.txt",
        "content": "This is test content for the resource.",
        "category": "document",
        "tags": ["test"],
    }
