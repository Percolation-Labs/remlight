# REMLight Ontology

A sample knowledge base for testing REM queries. This ontology covers AI/ML concepts with interlinked entities.

## Structure

```
ontology/
├── README.md           # This file
├── scripts/
│   └── verify_links.py # Link validation utility
├── concepts/           # Core AI/ML concepts
│   ├── machine-learning.md
│   ├── deep-learning.md
│   ├── neural-network.md
│   └── ...
├── models/             # Specific model architectures
│   ├── transformer.md
│   ├── gpt.md
│   ├── bert.md
│   └── ...
└── techniques/         # Training and optimization techniques
    ├── backpropagation.md
    ├── attention.md
    └── ...
```

## Entity Format

Each markdown file uses YAML frontmatter for metadata:

```yaml
---
entity_key: unique-identifier    # Unique key for LOOKUP queries
title: Human Readable Title      # Display name
parent: parent-entity-key        # Hierarchical parent
children: [child-1, child-2]     # Direct children
related: [related-1, related-2]  # Related concepts
tags: [tag1, tag2]               # Categorical tags
---

# Title

Content with links to other entities using [[entity-key|Display Text]] syntax.
```

## Link Syntax

Entities link to each other using wiki-style syntax:

```markdown
[[entity-key|Display Text]]
```

Examples:
- `[[transformer|Transformer]]` → Links to transformer.md
- `[[deep-learning|deep learning]]` → Links to deep-learning.md
- `[[attention|attention mechanism]]` → Links to attention.md

## Ingestion

Load ontology into database:

```bash
# Ingest all ontology files
rem ingest ontology/

# Preview what would be ingested
rem ingest ontology/ --dry-run

# Ingest with custom pattern
rem ingest ontology/ --pattern "concepts/*.md"
```

## REM Query Examples

After ingestion, query the knowledge base:

### LOOKUP - Exact Key Match

```bash
# O(1) lookup by entity key
rem query "LOOKUP transformer"
rem query "LOOKUP deep-learning"
rem query "LOOKUP attention"
```

### SEARCH - Semantic Vector Search

```bash
# Vector similarity search
rem query "SEARCH neural networks IN ontology"
rem query "SEARCH language models IN ontology"
rem query "SEARCH gradient descent IN ontology"
```

### FUZZY - Trigram Text Matching

```bash
# Fuzzy text search across all content
rem query "FUZZY transfomer architecture"  # Handles typos
rem query "FUZZY bert model"
rem query "FUZZY backprop algorithm"
```

## Link Verification

Validate all entity links:

```bash
python ontology/scripts/verify_links.py
```

This checks:
- All `[[entity-key|...]]` references resolve to existing entities
- Parent/child relationships are bidirectional
- No orphaned entities (except root)
- No duplicate entity keys

## How the Loader Works

The ontology loader (`rem ingest`) processes markdown files:

1. **Parse Frontmatter**: Extract YAML metadata (entity_key, title, parent, etc.)
2. **Generate Entity Key**: Uses `entity_key` from frontmatter, or filename as fallback
3. **Store Content**: Full markdown content stored in `ontology` table
4. **Generate Embeddings**: Content is embedded for semantic search
5. **Index for LOOKUP**: Entity key indexed in `kv_store` for O(1) lookup

### Database Schema

```sql
-- Ontology entities
CREATE TABLE ontology (
    id UUID PRIMARY KEY,
    name VARCHAR(255),        -- entity_key from frontmatter
    content TEXT,             -- full markdown content
    embedding vector(1536),   -- for SEARCH queries
    metadata JSONB,           -- frontmatter as JSON
    tags TEXT[],
    user_id VARCHAR(255),
    tenant_id VARCHAR(255),
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ
);

-- KV store for O(1) LOOKUP
CREATE TABLE kv_store (
    key VARCHAR(512) PRIMARY KEY,
    value JSONB,
    source_table VARCHAR(255),
    source_id UUID
);
```

### Query Flow

```
LOOKUP transformer
    └── kv_store.get("transformer")
        └── Returns: {name, content, metadata}

SEARCH "attention mechanism" IN ontology
    └── rem_search(embed("attention mechanism"), "ontology", 10)
        └── Returns: Top 10 by cosine similarity

FUZZY "transfomrer"
    └── rem_fuzzy("transfomrer", threshold=0.3)
        └── Returns: Matches by trigram similarity
```

## Adding New Entities

1. Create markdown file in appropriate directory
2. Add YAML frontmatter with `entity_key`
3. Link to related entities using `[[entity-key|Display Text]]`
4. Run `verify_links.py` to check references
5. Run `rem ingest ontology/` to load into database
