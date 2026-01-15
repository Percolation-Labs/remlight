# REMLight Tests

This directory contains the test suite for REMLight, organized into unit and integration tests.

## Test Organization

```
tests/
├── README.md           # This file
├── conftest.py         # Shared fixtures
├── unit/               # Mock-only tests (no external dependencies)
│   ├── conftest.py     # Unit test mocks
│   ├── test_rem_parser.py
│   ├── test_rem_models.py
│   └── test_rem_service.py
└── integration/        # Tests requiring database/external services
    ├── conftest.py     # Integration test fixtures
    ├── test_rem_queries.py
    └── test_llm_queries.py  # LLM tests (skipped by default)
```

## Test Types

### Unit Tests (`tests/unit/`)

- **MUST be mock-only** - no database, no API calls, no external services
- Run fast (< 1 second each)
- All external dependencies are mocked via `conftest.py`
- Run on every commit via pre-commit hook

### Integration Tests (`tests/integration/`)

- Require real PostgreSQL database with pgvector
- Test actual query execution and data flow
- Run on push via pre-push hook with dockerized database

### LLM Tests (marked with `@pytest.mark.llm`)

- Require actual LLM API keys (OpenAI, Anthropic, etc.)
- **Skipped by default** - incur API costs
- Must be explicitly enabled with `-m "llm"`

## Running Tests

### Quick Start

```bash
# Run unit tests only (no database required)
uv run pytest tests/unit/ -v

# Run all tests except LLM (requires database)
uv run pytest -m "not llm" -v

# Run integration tests with dockerized database
docker compose -f docker-compose.test.yml up -d
POSTGRES__CONNECTION_STRING="postgresql://remlight:remlight@localhost:5433/remlight" \
    uv run pytest tests/integration/ -v -m "not llm"
docker compose -f docker-compose.test.yml down -v
```

### Test Commands

| Command | Description |
|---------|-------------|
| `pytest tests/unit/` | Run unit tests only |
| `pytest tests/integration/` | Run integration tests |
| `pytest -m "not llm"` | Run all tests except LLM |
| `pytest -m "llm"` | Run only LLM tests |
| `pytest -m "slow"` | Run only slow tests |
| `pytest --cov=remlight` | Run with coverage |

## Git Hooks

Git hooks are configured in `.githooks/`. To enable:

```bash
git config core.hooksPath .githooks
```

### Pre-commit Hook

Runs **unit tests** before every commit:
- Fast feedback loop
- No external dependencies required
- Blocks commit if tests fail

### Pre-push Hook

Runs **integration tests** before push:
- Starts dockerized PostgreSQL database
- Runs full integration test suite
- Skips LLM tests (expensive)
- Cleans up database after tests

## Pytest Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.unit` | Unit test (auto-applied in `tests/unit/`) |
| `@pytest.mark.integration` | Integration test (auto-applied in `tests/integration/`) |
| `@pytest.mark.llm` | Requires LLM API calls (skipped by default) |
| `@pytest.mark.slow` | Slow-running test |
| `@pytest.mark.asyncio` | Async test (auto-configured) |

## Writing Tests

### Unit Test Guidelines

```python
# tests/unit/test_example.py

import pytest
from unittest.mock import AsyncMock, MagicMock

class TestMyFeature:
    """Unit tests for MyFeature."""

    @pytest.fixture
    def mock_dependency(self):
        """Mock external dependency."""
        mock = MagicMock()
        mock.method = AsyncMock(return_value="result")
        return mock

    @pytest.mark.asyncio
    async def test_feature_works(self, mock_dependency):
        """Test that feature works with mocked dependency."""
        # Arrange
        service = MyService(mock_dependency)

        # Act
        result = await service.do_something()

        # Assert
        assert result == "expected"
        mock_dependency.method.assert_called_once()
```

### Integration Test Guidelines

```python
# tests/integration/test_example.py

import pytest

@pytest.mark.asyncio
async def test_database_query(db, clean_test_data):
    """Test actual database query."""
    # Insert test data
    await db.execute("INSERT INTO ...")

    # Query and verify
    result = await db.fetch("SELECT ...")
    assert len(result) == 1
```

### LLM Test Guidelines

```python
# tests/integration/test_llm_example.py

import pytest

@pytest.mark.llm
@pytest.mark.asyncio
async def test_with_real_llm():
    """Test requiring actual LLM API call.

    Skipped by default. Run with: pytest -m "llm"
    """
    # Test implementation
    pass
```

## Docker Test Database

The `docker-compose.test.yml` provides an isolated test database:

- **Port**: 5433 (different from dev port 5432)
- **Container**: `remlight-postgres-test`
- **Image**: `pgvector/pgvector:pg17`
- **Auto-schema**: SQL files mounted to `/docker-entrypoint-initdb.d/`

```bash
# Start test database
docker compose -f docker-compose.test.yml up -d

# Wait for ready
docker exec remlight-postgres-test pg_isready -U remlight

# Run tests
POSTGRES__CONNECTION_STRING="postgresql://remlight:remlight@localhost:5433/remlight" \
    pytest tests/integration/

# Stop and cleanup
docker compose -f docker-compose.test.yml down -v
```

## Coverage

Generate coverage report:

```bash
uv run pytest --cov=remlight --cov-report=html tests/
open htmlcov/index.html
```
