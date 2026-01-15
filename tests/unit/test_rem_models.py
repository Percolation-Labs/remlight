"""
Unit tests for REM Query Models.

These tests verify the Pydantic models work correctly
without requiring any database connection.
"""

import pytest

from remlight.models.rem_query import (
    QueryType,
    LookupParameters,
    FuzzyParameters,
    SearchParameters,
    SQLParameters,
    TraverseParameters,
    RemQuery,
    RemQueryResult,
)


class TestQueryType:
    """Tests for QueryType enum."""

    def test_all_query_types_exist(self):
        """Verify all expected query types are defined."""
        assert QueryType.LOOKUP == "LOOKUP"
        assert QueryType.FUZZY == "FUZZY"
        assert QueryType.SEARCH == "SEARCH"
        assert QueryType.SQL == "SQL"
        assert QueryType.TRAVERSE == "TRAVERSE"

    def test_query_type_from_string(self):
        """Test creating QueryType from string."""
        assert QueryType("LOOKUP") == QueryType.LOOKUP
        assert QueryType("FUZZY") == QueryType.FUZZY


class TestLookupParameters:
    """Tests for LookupParameters model."""

    def test_single_key(self):
        """Test with single string key."""
        params = LookupParameters(key="test-key")
        assert params.key == "test-key"
        assert params.user_id is None

    def test_multiple_keys(self):
        """Test with list of keys."""
        params = LookupParameters(key=["key1", "key2", "key3"])
        assert params.key == ["key1", "key2", "key3"]

    def test_with_user_id(self):
        """Test with user_id filter."""
        params = LookupParameters(key="test", user_id="user-123")
        assert params.user_id == "user-123"


class TestFuzzyParameters:
    """Tests for FuzzyParameters model."""

    def test_defaults(self):
        """Test default values."""
        params = FuzzyParameters(query_text="test")
        assert params.query_text == "test"
        assert params.threshold == 0.3
        assert params.limit == 10

    def test_custom_values(self):
        """Test with custom values."""
        params = FuzzyParameters(query_text="search", threshold=0.7, limit=50)
        assert params.threshold == 0.7
        assert params.limit == 50

    def test_threshold_validation(self):
        """Test threshold bounds validation."""
        # Valid bounds
        FuzzyParameters(query_text="test", threshold=0.0)
        FuzzyParameters(query_text="test", threshold=1.0)

        # Invalid bounds
        with pytest.raises(ValueError):
            FuzzyParameters(query_text="test", threshold=-0.1)
        with pytest.raises(ValueError):
            FuzzyParameters(query_text="test", threshold=1.1)


class TestSearchParameters:
    """Tests for SearchParameters model."""

    def test_defaults(self):
        """Test default values."""
        params = SearchParameters(query_text="semantic query")
        assert params.query_text == "semantic query"
        assert params.table_name == "resources"
        assert params.limit == 10
        assert params.min_similarity == 0.3

    def test_custom_table(self):
        """Test with custom table name."""
        params = SearchParameters(query_text="query", table_name="ontologies")
        assert params.table_name == "ontologies"


class TestSQLParameters:
    """Tests for SQLParameters model."""

    def test_raw_query(self):
        """Test with raw SQL query."""
        params = SQLParameters(raw_query="SELECT * FROM resources")
        assert params.raw_query == "SELECT * FROM resources"

    def test_structured_query(self):
        """Test with structured query parameters."""
        params = SQLParameters(
            table_name="ontologies",
            where_clause="category = 'person'",
            order_by="name ASC",
            limit=50,
        )
        assert params.table_name == "ontologies"
        assert params.where_clause == "category = 'person'"
        assert params.order_by == "name ASC"
        assert params.limit == 50


class TestTraverseParameters:
    """Tests for TraverseParameters model."""

    def test_defaults(self):
        """Test default values."""
        params = TraverseParameters(initial_query="entity-key")
        assert params.initial_query == "entity-key"
        assert params.edge_types is None
        assert params.max_depth == 1
        assert params.limit == 10

    def test_with_edge_types(self):
        """Test with edge type filter."""
        params = TraverseParameters(
            initial_query="sarah-chen",
            edge_types=["manages", "reports_to"],
            max_depth=2,
        )
        assert params.edge_types == ["manages", "reports_to"]
        assert params.max_depth == 2

    def test_depth_zero_plan_mode(self):
        """Test DEPTH 0 for plan mode."""
        params = TraverseParameters(initial_query="entity", max_depth=0)
        assert params.max_depth == 0


class TestRemQuery:
    """Tests for RemQuery model."""

    def test_lookup_query(self):
        """Test creating LOOKUP query."""
        query = RemQuery(
            query_type=QueryType.LOOKUP,
            parameters=LookupParameters(key="test-key"),
        )
        assert query.query_type == QueryType.LOOKUP
        assert query.parameters.key == "test-key"
        assert query.user_id is None

    def test_fuzzy_query_with_user(self):
        """Test creating FUZZY query with user_id."""
        query = RemQuery(
            query_type=QueryType.FUZZY,
            parameters=FuzzyParameters(query_text="search"),
            user_id="user-123",
        )
        assert query.query_type == QueryType.FUZZY
        assert query.user_id == "user-123"

    def test_search_query(self):
        """Test creating SEARCH query."""
        query = RemQuery(
            query_type=QueryType.SEARCH,
            parameters=SearchParameters(query_text="semantic", table_name="resources"),
        )
        assert query.query_type == QueryType.SEARCH
        assert query.parameters.table_name == "resources"


class TestRemQueryResult:
    """Tests for RemQueryResult model."""

    def test_empty_result(self):
        """Test empty result."""
        result = RemQueryResult(query_type=QueryType.LOOKUP)
        assert result.query_type == QueryType.LOOKUP
        assert result.data == []
        assert result.count == 0
        assert result.metadata == {}

    def test_with_data(self):
        """Test result with data."""
        result = RemQueryResult(
            query_type=QueryType.FUZZY,
            data=[{"name": "Entity 1"}, {"name": "Entity 2"}],
            count=2,
            metadata={"query_text": "test"},
        )
        assert result.count == 2
        assert len(result.data) == 2
        assert result.metadata["query_text"] == "test"
