# REMLight

A lightweight agentic framework built on [REM](https://github.com/mr-saoirse/rem) - the declarative memory and query system for PostgreSQL.

## Quick Start

```bash
# Install
pip install -e .
docker compose up postgres -d

# Test with CLI
rem ask "What can you help me with?"

# Or start API server
rem serve --port 8001
```

## What is REMLight?

- **Declarative Agents**: Define agents in YAML with JSON Schema
- **Multi-Agent Orchestration**: Child agents stream through parent SSE connections
- **OpenAI-compatible API**: Drop-in replacement for chat completions
- **MCP Tools**: `search`, `action`, `ask_agent` via FastMCP

## Documentation

| Document | Description |
|----------|-------------|
| [code-walkthrough.md](code-walkthrough.md) | Agent construction, streaming, tool signatures |
| [remlight/api/README.md](remlight/api/README.md) | API endpoints |
| [schemas/](schemas/) | Agent YAML examples |
| [app/README.md](app/README.md) | React chat client |

## Environment

```bash
export OPENAI_API_KEY=sk-...
```

## License

MIT
