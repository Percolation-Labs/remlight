# REMLight Documentation

Knowledge base for REMLight - a declarative agent framework with PostgreSQL memory.

## Overview

- **Declarative agents** - YAML schemas with system prompts, tools, output structure
- **REM queries** - LOOKUP, SEARCH, FUZZY, SQL, TRAVERSE with performance contracts
- **PostgreSQL memory** - Sessions, messages, embeddings persisted automatically
- **MCP integration** - Tools: search, action, ask_agent

## Contents

### [Design](./design/)

| Document | Description |
| -------- | ----------- |
| [Declarative Agents](./design/declarative-agents.md) | Why agents are trivially declarative |
| [Architecture](./design/architecture.md) | System components and data flow |
| [Entities](./design/entities.md) | Database entity types |

### [Reference](./reference/)

| Document | Description |
| -------- | ----------- |
| [REM Query](./reference/rem-query.md) | Query language specification |
| [Agent Schema](./reference/agent-schema.md) | YAML schema format |
| [MCP Tools](./reference/mcp-tools.md) | Available tools |
| [Messages](./reference/messages.md) | Message persistence |

### [Guides](./guides/)

| Document | Description |
| -------- | ----------- |
| [Quick Start](./guides/quick-start.md) | Installation and setup |
| [CLI](./guides/cli.md) | Command line usage |
| [Multi-Agent](./guides/multi-agent.md) | Multi-agent orchestration |

## Quick Start

```bash
rem install                    # Install database schema
rem serve                      # Start API server
rem ask "What is REMLight?"    # Ask an agent
rem query "LOOKUP my-entity"   # Execute REM query
rem ingest ontology/           # Ingest markdown files
```
