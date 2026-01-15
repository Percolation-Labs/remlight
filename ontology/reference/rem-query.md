---
entity_key: rem-query
title: REM Query Language
tags: [reference, query]
---

# REM Query Language

Schema-agnostic query operations for LLM retrieval.

## Query Types

### LOOKUP - O(1)

Exact key-value lookup.

```bash
rem query "LOOKUP my-entity"
rem query "LOOKUP user-123"
```

### SEARCH - O(log n)

Semantic vector search using pgvector.

```bash
rem query "SEARCH machine learning IN ontology"
rem query "SEARCH project planning IN resources"
```

### FUZZY - O(n)

Trigram text matching. Handles typos.

```bash
rem query "FUZZY transfomrer"  # finds "transformer"
rem query "FUZZY backprop"
```

### SQL (Python API only)

Direct SQL access via Python API.

```python
from remlight.models.rem_query import RemQuery, QueryType, SQLParameters

query = RemQuery(
    query_type=QueryType.SQL,
    parameters=SQLParameters(raw_query="SELECT * FROM resources WHERE created_at > '2024-01-01'")
)
```

### TRAVERSE - O(k)

Graph traversal across entity relationships.

```bash
rem query "TRAVERSE user-123 DEPTH 2"
```

## Performance Contracts

| Type | Complexity | Index |
| ---- | ---------- | ----- |
| LOOKUP | O(1) | Primary key |
| SEARCH | O(log n) | Vector (HNSW) |
| FUZZY | O(n) | Trigram |
| TRAVERSE | O(k) | Primary key |

## Python API

```python
from remlight.models.rem_query import RemQuery, QueryType, LookupParameters

query = RemQuery(
    query_type=QueryType.LOOKUP,
    parameters=LookupParameters(key="my-entity")
)
```

## See also

- `REM LOOKUP cli` - CLI commands for queries
- `REM LOOKUP entities` - Entity types that can be queried
- `REM LOOKUP mcp-tools` - Tools that use REM queries
