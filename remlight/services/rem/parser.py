"""
REM Query Parser

Parses REM query strings into structured QueryType and parameters.
Uses shlex for proper quoting support.
"""

import shlex
from typing import Any

from remlight.models.rem_query import QueryType


class RemQueryParser:
    """
    Parser for REM query language.

    Supports:
    - LOOKUP "entity-name"
    - FUZZY "partial text" THRESHOLD 0.5 LIMIT 10
    - SEARCH "semantic query" IN resources LIMIT 10
    - SQL SELECT * FROM resources WHERE ...
    - TRAVERSE "entity-key" DEPTH 2
    """

    def parse(self, query_string: str) -> tuple[QueryType, dict[str, Any]]:
        """
        Parse a REM query string into QueryType and parameters.

        Args:
            query_string: The raw query string (e.g., 'LOOKUP "Sarah Chen"').

        Returns:
            Tuple of (QueryType, parameters_dict).

        Raises:
            ValueError: If the query string is empty or invalid.
        """
        if not query_string or not query_string.strip():
            raise ValueError("Empty query string")

        try:
            tokens = shlex.split(query_string)
        except ValueError as e:
            raise ValueError(f"Failed to parse query string: {e}")

        if not tokens:
            raise ValueError("Empty query string")

        query_type_str = tokens[0].upper()

        # Try to match REM query types
        try:
            query_type = QueryType(query_type_str)
        except ValueError:
            # If not a known REM query type, treat as raw SQL
            query_type = QueryType.SQL
            return query_type, {"raw_query": query_string.strip()}

        params: dict[str, Any] = {}
        positional_args: list[str] = []

        # For SQL queries, preserve the raw query
        if query_type == QueryType.SQL:
            raw_sql = query_string[3:].strip()  # Skip "SQL" prefix
            return query_type, {"raw_query": raw_sql}

        # Process remaining tokens
        i = 1
        while i < len(tokens):
            token = tokens[i]
            token_upper = token.upper()

            # Handle REM keywords
            if token_upper in self._keyword_map and i + 1 < len(tokens):
                key = self._keyword_map[token_upper]
                value = tokens[i + 1]
                params[key] = self._convert_value(key, value)
                i += 2
                continue
            elif "=" in token:
                key, value = token.split("=", 1)
                mapped_key = self._map_alias(key)
                params[mapped_key] = self._convert_value(mapped_key, value)
            else:
                positional_args.append(token)
            i += 1

        # Map positional arguments
        self._map_positional_args(query_type, positional_args, params)

        return query_type, params

    @property
    def _keyword_map(self) -> dict[str, str]:
        return {
            "LIMIT": "limit",
            "DEPTH": "max_depth",
            "THRESHOLD": "threshold",
            "TYPE": "edge_types",
            "FROM": "initial_query",
            "WITH": "initial_query",
            "TABLE": "table_name",
            "IN": "table_name",
            "WHERE": "where_clause",
        }

    def _map_alias(self, key: str) -> str:
        """Map common aliases to internal field names."""
        aliases = {
            "table": "table_name",
            "depth": "max_depth",
            "rel_type": "edge_types",
            "rel_types": "edge_types",
        }
        return aliases.get(key.lower(), key)

    def _convert_value(self, key: str, value: str) -> str | int | float | list[str]:
        """Convert string values to appropriate types."""
        # Integer fields
        if key in ("limit", "max_depth"):
            try:
                return int(value)
            except ValueError:
                return value

        # Float fields
        if key in ("threshold", "min_similarity"):
            try:
                return float(value)
            except ValueError:
                return value

        # List fields (comma-separated)
        if key == "edge_types":
            return [v.strip() for v in value.split(",")]

        return value

    def _map_positional_args(
        self, query_type: QueryType, positional_args: list[str], params: dict[str, Any]
    ) -> None:
        """Map positional arguments to the primary field for the query type."""
        if not positional_args:
            return

        combined_value = " ".join(positional_args)

        if query_type == QueryType.LOOKUP:
            if "," in combined_value:
                params["key"] = [k.strip() for k in combined_value.split(",")]
            else:
                params["key"] = combined_value

        elif query_type == QueryType.FUZZY:
            params["query_text"] = combined_value

        elif query_type == QueryType.SEARCH:
            params["query_text"] = combined_value

        elif query_type == QueryType.TRAVERSE:
            params["initial_query"] = combined_value

        elif query_type == QueryType.SQL:
            params["raw_query"] = combined_value
