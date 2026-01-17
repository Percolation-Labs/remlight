# REMLight Database Schema

Entity Relationship Diagram for the REMLight database (`sql/install.sql`).

```mermaid
erDiagram
    sessions ||--o{ messages : "has many"
    ontologies ||--o| kv_store : "triggers sync"
    resources ||--o| kv_store : "triggers sync"
    ontologies ||--o{ embedding_queue : "queues"
    resources ||--o{ embedding_queue : "queues"
    messages ||--o{ embedding_queue : "queues"

    kv_store {
        VARCHAR(512) entity_key PK
        VARCHAR(128) entity_type
        VARCHAR(128) table_name
        JSONB data
        VARCHAR(256) user_id
        VARCHAR(256) tenant_id
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
    }

    ontologies {
        UUID id PK
        VARCHAR(512) name UK
        TEXT description
        VARCHAR(256) category
        VARCHAR(128) entity_type
        JSONB properties
        JSONB graph_edges
        JSONB metadata
        TEXT[] tags
        VECTOR(1536) embedding
        VARCHAR(256) user_id
        VARCHAR(256) tenant_id
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        TIMESTAMPTZ deleted_at
    }

    resources {
        UUID id PK
        VARCHAR(512) name UK
        VARCHAR(1024) uri
        INTEGER ordinal
        TEXT content
        VARCHAR(256) category
        JSONB related_entities
        JSONB graph_edges
        JSONB metadata
        TEXT[] tags
        VECTOR(1536) embedding
        VARCHAR(256) user_id
        VARCHAR(256) tenant_id
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        TIMESTAMPTZ deleted_at
    }

    users {
        UUID id PK
        VARCHAR(512) name
        VARCHAR(256) email UK
        TEXT summary
        TEXT[] interests
        TEXT[] preferred_topics
        VARCHAR(64) activity_level
        JSONB metadata
        VARCHAR(256) user_id
        VARCHAR(256) tenant_id
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        TIMESTAMPTZ deleted_at
    }

    sessions {
        UUID id PK
        VARCHAR(512) name
        TEXT description
        VARCHAR(256) agent_name
        VARCHAR(64) status
        JSONB metadata
        TEXT[] tags
        VARCHAR(256) user_id
        VARCHAR(256) tenant_id
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        TIMESTAMPTZ deleted_at
    }

    messages {
        UUID id PK
        UUID session_id FK
        VARCHAR(64) role
        TEXT content
        JSONB tool_calls
        JSONB metadata
        VECTOR(1536) embedding
        VARCHAR(256) user_id
        VARCHAR(256) tenant_id
        TIMESTAMPTZ created_at
        TIMESTAMPTZ updated_at
        TIMESTAMPTZ deleted_at
    }

    embedding_queue {
        UUID id PK
        VARCHAR(128) table_name
        UUID record_id
        TEXT content
        VARCHAR(64) status
        TEXT error_message
        TIMESTAMPTZ created_at
        TIMESTAMPTZ processed_at
    }
```

## Tables

| Table | Purpose |
|-------|---------|
| `kv_store` | O(1) lookup cache, auto-synced via triggers |
| `ontologies` | Domain entities (people, projects, concepts) with embeddings |
| `resources` | Documents/content chunks with embeddings |
| `users` | User profiles with AI-generated summaries |
| `sessions` | Conversation sessions |
| `messages` | Chat messages linked to sessions |
| `embedding_queue` | Async queue for embedding generation |

## Relationships

- **messages → sessions**: Foreign key with `ON DELETE CASCADE`
- **ontologies/resources → kv_store**: Trigger-based sync for fast lookups
- **ontologies/resources/messages → embedding_queue**: Trigger-based queue for async embedding generation

## REM Functions

The schema includes PostgreSQL functions for the REM query language:

| Function | Purpose |
|----------|---------|
| `rem_lookup(key)` | O(1) KV store lookup |
| `rem_search(embedding, table, limit)` | Semantic vector search |
| `rem_fuzzy(text)` | Fuzzy text search using trigrams |
| `rem_traverse(key, depth)` | Graph traversal following edges |
