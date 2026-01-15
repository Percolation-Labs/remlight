"""
Unit tests for REM Query Parser.

These tests verify the parser correctly handles all query types
without requiring any database connection.
"""

import pytest

from remlight.models.rem_query import QueryType
from remlight.services.rem.parser import RemQueryParser


class TestRemQueryParser:
    """Unit tests for RemQueryParser."""

    def setup_method(self):
        self.parser = RemQueryParser()

    # LOOKUP tests

    def test_lookup_simple(self):
        """Test simple LOOKUP query."""
        query_type, params = self.parser.parse('LOOKUP "sarah-chen"')
        assert query_type == QueryType.LOOKUP
        assert params["key"] == "sarah-chen"

    def test_lookup_unquoted(self):
        """Test LOOKUP with unquoted key."""
        query_type, params = self.parser.parse("LOOKUP sarah-chen")
        assert query_type == QueryType.LOOKUP
        assert params["key"] == "sarah-chen"

    def test_lookup_multi_word(self):
        """Test LOOKUP with multi-word quoted key."""
        query_type, params = self.parser.parse('LOOKUP "Sarah Chen"')
        assert query_type == QueryType.LOOKUP
        assert params["key"] == "Sarah Chen"

    def test_lookup_multiple_keys(self):
        """Test LOOKUP with comma-separated keys."""
        query_type, params = self.parser.parse("LOOKUP key1, key2, key3")
        assert query_type == QueryType.LOOKUP
        assert params["key"] == ["key1", "key2", "key3"]

    # FUZZY tests

    def test_fuzzy_simple(self):
        """Test simple FUZZY query."""
        query_type, params = self.parser.parse('FUZZY "project alpha"')
        assert query_type == QueryType.FUZZY
        assert params["query_text"] == "project alpha"

    def test_fuzzy_with_threshold(self):
        """Test FUZZY with threshold parameter."""
        query_type, params = self.parser.parse('FUZZY "test" THRESHOLD 0.5')
        assert query_type == QueryType.FUZZY
        assert params["query_text"] == "test"
        assert params["threshold"] == 0.5

    def test_fuzzy_with_limit(self):
        """Test FUZZY with limit parameter."""
        query_type, params = self.parser.parse('FUZZY "test" LIMIT 20')
        assert query_type == QueryType.FUZZY
        assert params["query_text"] == "test"
        assert params["limit"] == 20

    def test_fuzzy_with_options(self):
        """Test FUZZY with threshold and limit."""
        query_type, params = self.parser.parse('FUZZY "test" THRESHOLD 0.5 LIMIT 20')
        assert query_type == QueryType.FUZZY
        assert params["query_text"] == "test"
        assert params["threshold"] == 0.5
        assert params["limit"] == 20

    # SEARCH tests

    def test_search_simple(self):
        """Test simple SEARCH query."""
        query_type, params = self.parser.parse('SEARCH "machine learning"')
        assert query_type == QueryType.SEARCH
        assert params["query_text"] == "machine learning"

    def test_search_with_table(self):
        """Test SEARCH with IN clause."""
        query_type, params = self.parser.parse('SEARCH "api design" IN resources')
        assert query_type == QueryType.SEARCH
        assert params["query_text"] == "api design"
        assert params["table_name"] == "resources"

    def test_search_with_table_and_limit(self):
        """Test SEARCH with IN and LIMIT."""
        query_type, params = self.parser.parse('SEARCH "api design" IN resources LIMIT 5')
        assert query_type == QueryType.SEARCH
        assert params["query_text"] == "api design"
        assert params["table_name"] == "resources"
        assert params["limit"] == 5

    # TRAVERSE tests

    def test_traverse_simple(self):
        """Test simple TRAVERSE query."""
        query_type, params = self.parser.parse('TRAVERSE "sarah-chen"')
        assert query_type == QueryType.TRAVERSE
        assert params["initial_query"] == "sarah-chen"

    def test_traverse_with_depth(self):
        """Test TRAVERSE with depth."""
        query_type, params = self.parser.parse('TRAVERSE "entity-key" DEPTH 3')
        assert query_type == QueryType.TRAVERSE
        assert params["initial_query"] == "entity-key"
        assert params["max_depth"] == 3

    def test_traverse_depth_zero(self):
        """Test TRAVERSE with DEPTH 0 (plan mode)."""
        query_type, params = self.parser.parse('TRAVERSE "entity" DEPTH 0')
        assert query_type == QueryType.TRAVERSE
        assert params["initial_query"] == "entity"
        assert params["max_depth"] == 0

    # SQL tests

    def test_sql_raw_select(self):
        """Test raw SQL SELECT query."""
        query_type, params = self.parser.parse("SELECT * FROM resources WHERE category = 'docs'")
        assert query_type == QueryType.SQL
        assert "SELECT * FROM resources" in params["raw_query"]

    def test_sql_prefixed(self):
        """Test SQL-prefixed query."""
        query_type, params = self.parser.parse("SQL SELECT id, name FROM ontologies LIMIT 10")
        assert query_type == QueryType.SQL
        assert "SELECT id, name FROM ontologies" in params["raw_query"]

    def test_sql_with_clause(self):
        """Test SQL WITH (CTE) query."""
        query_type, params = self.parser.parse("WITH x AS (SELECT 1) SELECT * FROM x")
        assert query_type == QueryType.SQL
        assert "WITH x AS" in params["raw_query"]

    # Error handling tests

    def test_empty_query_raises(self):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="Empty query string"):
            self.parser.parse("")

    def test_whitespace_only_raises(self):
        """Test that whitespace-only query raises ValueError."""
        with pytest.raises(ValueError, match="Empty query string"):
            self.parser.parse("   ")

    def test_unclosed_quote_raises(self):
        """Test that unclosed quote raises ValueError."""
        with pytest.raises(ValueError, match="Failed to parse"):
            self.parser.parse('LOOKUP "unclosed')
