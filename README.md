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

## Blog Series

| Part | Description |
|------|-------------|
| [Part I](https://medium.com/@mrsirsh/a-really-simply-declarative-agent-framework-part-i-6ae2b05fb2a1) | Introduction & architecture |
| [Part II](https://medium.com/@mrsirsh/part-ii-of-a-really-simply-declarative-agent-framework-320da34e5b4d) | Agent construction & tool signatures |
| [Part III](https://medium.com/@mrsirsh/part-iii-of-a-really-simply-declarative-agent-framework-fc96cc950c11) | Streaming, SSE events & database persistence |

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
