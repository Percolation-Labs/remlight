---
entity_key: entities
title: Entity Types
tags: [design, database]
---

# Entity Types

REMLight stores data in PostgreSQL tables with pgvector embeddings.

## Tables

| Table | Purpose |
| ----- | ------- |
| `ontology` | Knowledge base entities (markdown with embeddings) |
| `resources` | Documents and content chunks |
| `sessions` | Conversation metadata |
| `messages` | Chat history |
| `kv_store` | O(1) key-value lookup cache |

## Ontology

Domain entities for wiki-style knowledge bases.

```python
class Ontology(CoreModel):
    name: str           # entity_key for LOOKUP queries
    content: str        # Markdown content for SEARCH/FUZZY
    properties: dict    # YAML frontmatter metadata
    tags: list[str]     # Categorical tags
```

## Resource

Documents and content chunks with embeddings.

```python
class Resource(CoreModel):
    name: str
    uri: str            # Source file/URL
    content: str        # Text content
    category: str
```

## Session & Message

Conversation persistence.

```python
class Session(CoreModel):
    name: str
    agent_name: str
    status: str         # active, completed, etc.

class Message(CoreModel):
    session_id: UUID
    role: str           # user, assistant, tool
    content: str
    tool_calls: dict    # Structured tool metadata
```

## Embedding Generation

Content is embedded on insert/update via repository:

```python
await repo.upsert(entity, embeddable_fields=["content"], generate_embeddings=True)
```

Uses `text-embedding-3-small` (1536 dimensions).

## See also

- `REM LOOKUP architecture` - System architecture
- `REM LOOKUP messages` - Message persistence
- `REM LOOKUP rem-query` - Query language for entities
