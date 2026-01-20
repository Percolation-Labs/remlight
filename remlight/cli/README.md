# REMLight CLI

Command-line interface for REMLight - a minimal declarative agent framework.

## Installation

```bash
pip install remlight
```

## Quick Start

```bash
# Start the API server
rem serve

# Ask an agent a question
rem ask "What is machine learning?"

# Parse a document
rem parse document.pdf

# Search the knowledge base
rem query "SEARCH agents IN ontology"
```

## Commands

### `rem serve`

Start the API server (includes REST API and MCP endpoint).

```bash
rem serve                          # Default: http://0.0.0.0:8000
rem serve --port 3000              # Custom port
rem serve --host 127.0.0.1         # Localhost only
rem serve --no-reload              # Disable auto-reload
```

**Endpoints available:**
- API docs: http://localhost:8000/docs
- MCP endpoint: http://localhost:8000/api/v1/mcp

---

### `rem ask`

Ask an agent a question with optional streaming.

```bash
# Basic usage
rem ask "What is machine learning?"

# Use a specific agent schema
rem ask "Find documents about AI" --schema query-agent
rem ask "Analyze this data" --schema schemas/custom-agent.yaml

# Use a specific model
rem ask "Explain transformers" --model openai:gpt-4.1
rem ask "Summarize this" --model anthropic:claude-sonnet-4-5-20250929

# Multi-turn conversation (session must be a valid UUID)
rem ask "What is REM?" --session 550e8400-e29b-41d4-a716-446655440000
rem ask "Tell me more" --session 550e8400-e29b-41d4-a716-446655440000

# Disable streaming
rem ask "Quick question" --no-stream
```

---

### `rem parse`

Parse a file and extract content. Supports PDF, DOCX, PPTX, XLSX, images, and text files.

```bash
# Local files
rem parse document.pdf
rem parse ./reports/quarterly.docx
rem parse /absolute/path/to/file.pptx

# S3 files
rem parse s3://bucket/documents/report.pdf
rem parse s3://my-bucket/data/spreadsheet.xlsx

# HTTP URLs
rem parse https://example.com/paper.pdf

# Output formats
rem parse document.pdf                    # JSON output (default)
rem parse document.pdf --output text      # Text content only

# Don't save to database
rem parse document.pdf --no-save
```

**Note**: Files are stored globally by default. Only use `--user-id` if you specifically need per-user file isolation (this prevents file sharing).

**Supported formats:**
- Documents: PDF, DOCX, PPTX, XLSX (via Kreuzberg with OCR fallback)
- Images: PNG, JPG, GIF, WEBP (OCR text extraction)
- Text: Markdown, JSON, YAML, code files (UTF-8 extraction)

---

### `rem query`

Execute a REM query directly against the knowledge base.

```bash
# Exact key lookup
rem query "LOOKUP sarah-chen"
rem query "LOOKUP project-alpha"

# Semantic search
rem query "SEARCH machine learning IN ontologies"
rem query "SEARCH project management IN resources"

# Fuzzy text match
rem query "FUZZY project alpha"

# Graph traversal
rem query "TRAVERSE sarah-chen"

# With limit
rem query "SEARCH agents" --limit 50
```

---

### `rem ingest`

Ingest markdown files into the REM database with embeddings for semantic search.

```bash
# Ingest a directory
rem ingest ontology/

# Preview without making changes
rem ingest ontology/ --dry-run

# Custom glob pattern
rem ingest ontology/concepts/ --pattern "*.md"

# Target a different table
rem ingest docs/ --table resources
```

**Default behavior:**
- Pattern: `**/*.md`
- Table: `ontology`
- Excludes: `README.md` files and `scripts/` directories

---

### `rem eval`

Evaluate an agent's self-awareness (identity, tools, structure).

```bash
# Evaluate by agent name
rem eval query-agent

# Evaluate from YAML file
rem eval schemas/my-agent.yaml

# Verbose output with issues
rem eval query-agent --verbose

# JSON output for programmatic use
rem eval query-agent --json

# Use a specific model
rem eval query-agent --model openai:gpt-4.1
```

**Evaluation categories:**
- Identity: Does the agent know its name and purpose?
- Structure: Does it know its output properties?
- Tools: Does it know its available tools?
- Context: Does it know the current date/time?

---

### `rem install`

Install the database schema (tables, triggers, functions).

```bash
rem install
```

This creates all required PostgreSQL tables and extensions. Run this after setting up a new database.

---

## Model Selection

Models are selected with this priority (highest first):

1. **Agent schema `override_model`** - Forces a specific model for the agent
2. **Request/CLI `--model`** - Override from API request or CLI flag
3. **Environment `LLM__DEFAULT_MODEL`** - Default fallback

Example agent schema with forced model:
```yaml
json_schema_extra:
  name: vision-agent
  override_model: openai:gpt-4.1  # Always uses this model
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES__HOST` | PostgreSQL host | `localhost` |
| `POSTGRES__PORT` | PostgreSQL port | `5432` |
| `POSTGRES__DATABASE` | Database name | `remlight` |
| `POSTGRES__USER` | Database user | `postgres` |
| `POSTGRES__PASSWORD` | Database password | - |
| `LLM__DEFAULT_MODEL` | Default LLM model | `openai:gpt-4.1-mini` |
| `S3__ENDPOINT_URL` | S3 endpoint (for MinIO) | - |
| `AWS_ACCESS_KEY_ID` | AWS access key | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | - |

## Examples

### Complete Workflow

```bash
# 1. Start services
docker compose up -d postgres

# 2. Install schema
rem install

# 3. Ingest knowledge base
rem ingest ontology/

# 4. Parse some documents
rem parse reports/q4-summary.pdf
rem parse s3://company-docs/handbook.docx

# 5. Start the server
rem serve

# 6. Query the knowledge base
rem ask "What are our key projects?"
rem query "SEARCH quarterly results"
```

### Document Processing Pipeline

```bash
# Parse multiple documents
for file in documents/*.pdf; do
    rem parse "$file"
done

# Parse from S3 bucket
rem parse s3://my-bucket/reports/annual-2024.pdf
rem parse s3://my-bucket/presentations/deck.pptx

# Extract text only (no database)
rem parse document.pdf --output text --no-save > extracted.txt
```

### Multi-turn Conversation

```bash
# Generate a session UUID
SESSION=$(python -c "import uuid; print(uuid.uuid4())")

# Have a conversation
rem ask "What is REMLight?" --session $SESSION
rem ask "How do agents work?" --session $SESSION
rem ask "Show me an example" --session $SESSION
```
