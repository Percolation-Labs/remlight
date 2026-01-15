"""
Unit tests for REM Service.

These tests verify the RemService correctly processes queries
using mocked database service (no real DB connection).
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from remlight.models.rem_query import QueryType, RemQuery, LookupParameters, FuzzyParameters
from remlight.services.rem.service import RemService


class TestRemServiceExecution:
    """Tests for RemService query execution with mocks."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database service."""
        db = MagicMock()
        db.rem_lookup = AsyncMock(return_value={"name": "Test Entity", "id": "123"})
        db.rem_fuzzy = AsyncMock(return_value=[
            {"entity_key": "test-1", "similarity": 0.8},
            {"entity_key": "test-2", "similarity": 0.6},
        ])
        db.rem_search = AsyncMock(return_value=[
            {"id": "1", "name": "Result 1", "similarity": 0.9},
        ])
        db.rem_traverse = AsyncMock(return_value=[
            {"entity_key": "node-1", "depth": 0},
            {"entity_key": "node-2", "depth": 1},
        ])
        db.fetch = AsyncMock(return_value=[{"id": "1", "name": "SQL Result"}])
        return db

    @pytest.mark.asyncio
    async def test_execute_lookup(self, mock_db):
        """Test LOOKUP query execution."""
        service = RemService(mock_db)
        result = await service.execute('LOOKUP "test-entity"')

        assert result.query_type == QueryType.LOOKUP
        assert result.count == 1
        mock_db.rem_lookup.assert_called_once_with("test-entity", None)

    @pytest.mark.asyncio
    async def test_execute_lookup_empty(self, mock_db):
        """Test LOOKUP with no results."""
        mock_db.rem_lookup = AsyncMock(return_value={})
        service = RemService(mock_db)
        result = await service.execute('LOOKUP "nonexistent"')

        assert result.query_type == QueryType.LOOKUP
        assert result.count == 0

    @pytest.mark.asyncio
    async def test_execute_fuzzy(self, mock_db):
        """Test FUZZY query execution."""
        service = RemService(mock_db)
        result = await service.execute('FUZZY "search term" THRESHOLD 0.5 LIMIT 10')

        assert result.query_type == QueryType.FUZZY
        assert result.count == 2
        mock_db.rem_fuzzy.assert_called_once_with("search term", None, 0.5, 10)

    @pytest.mark.asyncio
    async def test_execute_search_without_embeddings(self, mock_db):
        """Test SEARCH falls back to fuzzy without embed_fn."""
        service = RemService(mock_db, embed_fn=None)
        result = await service.execute('SEARCH "semantic query" IN resources')

        assert result.query_type == QueryType.SEARCH
        # Falls back to fuzzy when no embed_fn
        mock_db.rem_fuzzy.assert_called()

    @pytest.mark.asyncio
    async def test_execute_search_with_embeddings(self, mock_db):
        """Test SEARCH uses vector search with embed_fn."""
        async def mock_embed(text):
            return [0.1] * 1536

        service = RemService(mock_db, embed_fn=mock_embed)
        result = await service.execute('SEARCH "semantic query" IN resources')

        assert result.query_type == QueryType.SEARCH
        mock_db.rem_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_traverse(self, mock_db):
        """Test TRAVERSE query execution."""
        service = RemService(mock_db)
        result = await service.execute('TRAVERSE "start-node" DEPTH 2')

        assert result.query_type == QueryType.TRAVERSE
        assert result.count == 2
        mock_db.rem_traverse.assert_called_once_with("start-node", None, 2, None)

    @pytest.mark.asyncio
    async def test_execute_sql(self, mock_db):
        """Test SQL query execution."""
        service = RemService(mock_db)
        result = await service.execute("SELECT * FROM resources LIMIT 5")

        assert result.query_type == QueryType.SQL
        assert result.count == 1
        mock_db.fetch.assert_called()

    @pytest.mark.asyncio
    async def test_execute_sql_blocked(self, mock_db):
        """Test that dangerous SQL is blocked."""
        service = RemService(mock_db)

        with pytest.raises(ValueError, match="Blocked SQL"):
            await service.execute("DROP TABLE resources")

        with pytest.raises(ValueError, match="Blocked SQL"):
            await service.execute("DELETE FROM ontologies")

        with pytest.raises(ValueError, match="Blocked SQL"):
            await service.execute("TRUNCATE messages")


class TestRemServiceDirectMethods:
    """Tests for RemService direct convenience methods."""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database service."""
        db = MagicMock()
        db.rem_lookup = AsyncMock(return_value={"name": "Entity"})
        db.rem_fuzzy = AsyncMock(return_value=[{"entity_key": "test"}])
        db.rem_traverse = AsyncMock(return_value=[{"entity_key": "node"}])
        return db

    @pytest.mark.asyncio
    async def test_lookup_method(self, mock_db):
        """Test direct lookup() method."""
        service = RemService(mock_db)
        data = await service.lookup("entity-key")

        assert len(data) == 1
        mock_db.rem_lookup.assert_called_with("entity-key", None)

    @pytest.mark.asyncio
    async def test_fuzzy_method(self, mock_db):
        """Test direct fuzzy() method."""
        service = RemService(mock_db)
        data = await service.fuzzy("search", threshold=0.4, limit=5)

        assert len(data) == 1
        mock_db.rem_fuzzy.assert_called_with("search", None, 0.4, 5)

    @pytest.mark.asyncio
    async def test_traverse_method(self, mock_db):
        """Test direct traverse() method."""
        service = RemService(mock_db)
        data = await service.traverse("start", edge_types=["manages"], max_depth=3)

        assert len(data) == 1
        mock_db.rem_traverse.assert_called_with("start", ["manages"], 3, None)


class TestRemServiceStructuredQueries:
    """Tests for RemService with structured RemQuery objects."""

    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.rem_lookup = AsyncMock(return_value={"name": "Entity"})
        db.rem_fuzzy = AsyncMock(return_value=[])
        return db

    @pytest.mark.asyncio
    async def test_execute_query_lookup(self, mock_db):
        """Test execute_query with RemQuery object."""
        service = RemService(mock_db)
        query = RemQuery(
            query_type=QueryType.LOOKUP,
            parameters=LookupParameters(key="test-key"),
            user_id="user-123",
        )
        result = await service.execute_query(query)

        assert result.query_type == QueryType.LOOKUP
        mock_db.rem_lookup.assert_called_with("test-key", "user-123")

    @pytest.mark.asyncio
    async def test_execute_query_fuzzy(self, mock_db):
        """Test execute_query with FUZZY RemQuery."""
        service = RemService(mock_db)
        query = RemQuery(
            query_type=QueryType.FUZZY,
            parameters=FuzzyParameters(query_text="search", threshold=0.6, limit=20),
        )
        result = await service.execute_query(query)

        assert result.query_type == QueryType.FUZZY
        mock_db.rem_fuzzy.assert_called_with("search", None, 0.6, 20)
