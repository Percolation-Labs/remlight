# REMLight

Minimal declarative agent framework with PostgreSQL memory and multi-agent support.

## Features

- **Declarative Agents**: Define agents in YAML with JSON Schema
- **PostgreSQL Memory**: Vector search, graph traversal, fuzzy matching
- **MCP Server**: Tools for `search`, `action`, and `ask_agent`
- **Multi-Agent Orchestration**: Child agents stream through parent
- **OpenAI-compatible API**: Streaming SSE responses
- **CLI**: `ask`, `query`, and `ingest` commands
- **Sample Ontology**: AI/ML knowledge base for testing

## Quick Start

### Docker

```bash
# Set your API key
export OPENAI_API_KEY=your-key

# Start services
docker-compose up -d

# API available at http://localhost:8000
```

### Local Development

```bash
# Install
pip install -e .

# Start PostgreSQL with pgvector
docker run -d --name remlight-pg \
  -e POSTGRES_DB=remlight \
  -e POSTGRES_USER=remlight \
  -e POSTGRES_PASSWORD=remlight \
  -p 5432:5432 \
  pgvector/pgvector:pg18

# Install schema
rem install

# Ingest sample ontology
rem ingest ontology/

# Start server
rem serve
```

## CLI Commands

```bash
# Ask an agent
rem ask "What is machine learning?"
rem ask "Find documents about AI" --schema schemas/query-agent.yaml

# Execute REM query
rem query "LOOKUP transformer"
rem query "SEARCH neural networks IN ontology"
rem query "FUZZY backprop algorithm"

# Ingest ontology files
rem ingest ontology/
rem ingest ontology/ --dry-run

# Start API server
rem serve

# Install database schema
rem install
```

## REM Query Examples

After ingesting the sample ontology, test REM queries:

### LOOKUP - O(1) Exact Key Match

```bash
# Lookup by entity_key (from YAML frontmatter)
rem query "LOOKUP transformer"
rem query "LOOKUP deep-learning"
rem query "LOOKUP gpt"
rem query "LOOKUP attention"
```

Returns exact match from kv_store in O(1) time.

### SEARCH - Semantic Vector Search

```bash
# Vector similarity search in ontology table
rem query "SEARCH neural networks IN ontology"
rem query "SEARCH language models IN ontology"
rem query "SEARCH gradient descent optimization IN ontology"
rem query "SEARCH attention mechanism transformers IN ontology"
```

Returns top results ranked by embedding cosine similarity.

### FUZZY - Trigram Text Matching

```bash
# Fuzzy text search (handles typos)
rem query "FUZZY transfomer architecture"  # typo: transfomer
rem query "FUZZY bert language model"
rem query "FUZZY backprop algorithm"
rem query "FUZZY llama meta ai"
```

Uses PostgreSQL trigram similarity for approximate matching.

## Sample Ontology

REMLight includes a sample AI/ML ontology with interlinked entities:

```
ontology/
├── README.md              # Ontology documentation
├── scripts/
│   └── verify_links.py    # Link validation
├── concepts/              # Core concepts
│   ├── machine-learning.md
│   ├── deep-learning.md
│   ├── neural-network.md
│   ├── supervised-learning.md
│   └── unsupervised-learning.md
├── models/                # Model architectures
│   ├── transformer.md
│   ├── gpt.md
│   ├── bert.md
│   └── llama.md
└── techniques/            # Training techniques
    ├── backpropagation.md
    └── attention.md
```

### Entity Linking

Entities link to each other using wiki syntax:

```markdown
[[entity-key|Display Text]]
```

Example from `transformer.md`:
```markdown
The [[attention|attention mechanism]] allows models to focus on
relevant parts of the input. This powers [[gpt|GPT]] and [[bert|BERT]].
```

### Verify Links

```bash
python ontology/scripts/verify_links.py
```

## How the Ontology Loader Works

The `rem ingest` command processes markdown files:

1. **Parse Frontmatter**: Extract YAML metadata (entity_key, title, parent, etc.)
2. **Generate Entity Key**: Uses `entity_key` from frontmatter, or filename
3. **Store Content**: Full markdown stored in `ontology` table
4. **Generate Embeddings**: Content embedded for SEARCH queries
5. **Index for LOOKUP**: Entity key indexed in `kv_store` for O(1) access

```
rem ingest ontology/
    │
    ├── Parse: machine-learning.md
    │   └── entity_key: machine-learning
    │   └── content: Full markdown
    │   └── properties: {parent, children, related, tags}
    │
    ├── Store in ontology table
    │   └── Generate embedding for content
    │
    └── Index in kv_store
        └── key: machine-learning → {name, content, metadata}
```

## Agent Schema (YAML)

```yaml
type: object
description: |
  You are a helpful assistant...

properties:
  answer:
    type: string
    description: Your response

required:
  - answer

json_schema_extra:
  kind: agent
  name: my-agent
  version: 1.0.0
  tools:
    - name: search
    - name: action
    - name: ask_agent
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/api/v1/query` | POST | Execute REM query |
| `/api/v1/mcp` | POST | MCP protocol |

## MCP Tools

### `search`

Execute REM queries:

```
LOOKUP <key>              - O(1) exact lookup
SEARCH <text> IN <table>  - Semantic search
FUZZY <text>              - Fuzzy text matching
TRAVERSE <key>            - Graph traversal
```

### `action`

Emit typed action events:

```python
action(
    type="observation",
    payload={
        "confidence": 0.85,
        "sources": ["transformer", "attention"],
        "session_name": "ML Research"
    }
)
```

### `ask_agent`

Invoke another agent (multi-agent orchestration):

```python
ask_agent(
    agent_name="query-agent",
    input_text="Find documents about machine learning"
)
```

Features:
- Child agents inherit parent's context (user_id, session_id)
- Child agent content streams through parent's SSE connection
- Supports orchestrator, workflow, and ensemble patterns

## Multi-Agent Streaming

When a parent agent calls `ask_agent`, child events are streamed in real-time:

```
Parent Agent
    │
    ▼
ask_agent("query-agent", "Find docs")
    │
    ├── child_tool_start ──────► SSE: tool_call event
    ├── child_content ─────────► SSE: content chunks
    └── child_tool_result ─────► SSE: tool_call complete
```

The streaming architecture prevents content duplication - when child content is streamed, parent text output is skipped.

## Database Tables

- `ontology` - Domain entities (concepts, models, techniques)
- `resources` - Documents and content chunks
- `sessions` - Conversation sessions
- `messages` - Chat messages
- `kv_store` - Key-value lookup cache for O(1) LOOKUP
- `embedding_queue` - Queue for async embedding generation

## REM Functions (PostgreSQL)

- `rem_lookup(key, user_id)` - O(1) KV store lookup
- `rem_search(embedding, table, limit, min_sim, user_id)` - Vector search
- `rem_fuzzy(text, user_id, threshold, limit)` - Trigram matching
- `rem_traverse(key, edge_types, depth, user_id)` - Graph traversal

## Environment Variables

```bash
POSTGRES__CONNECTION_STRING=postgresql://user:pass@host:5432/db
LLM__DEFAULT_MODEL=openai:gpt-4o-mini
LLM__TEMPERATURE=0.5
LLM__MAX_ITERATIONS=20
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ENVIRONMENT=development
```

## Project Structure

```
remlight/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── ontology/                # Sample AI/ML knowledge base
│   ├── concepts/
│   ├── models/
│   ├── techniques/
│   └── scripts/verify_links.py
├── sql/
│   └── install.sql          # Tables, triggers, functions
├── schemas/
│   └── query-agent.yaml     # Sample agent
└── remlight/
    ├── settings.py          # Environment config
    ├── models/
    │   ├── core.py          # CoreModel base
    │   └── entities.py      # Ontology, Resource, Session, Message
    ├── agentic/
    │   ├── schema.py        # AgentSchema + YAML
    │   ├── provider.py      # Pydantic AI agent creation
    │   └── context.py       # AgentContext + event sink
    ├── api/
    │   ├── main.py          # FastAPI app
    │   ├── mcp_main.py      # MCP server with search, action, ask_agent
    │   ├── streaming.py     # SSE with child streaming
    │   └── routers/         # API endpoints (chat, query, tools)
    ├── cli/
    │   └── main.py          # ask, query, ingest, serve, install
    └── services/
        └── database.py      # PostgreSQL + rem_* functions
```

## License

MIT
