---
entity_key: cli
title: CLI Commands
tags: [guides, cli]
---

# CLI Commands

REMLight CLI via the `rem` command.

## rem ask

Ask an agent a question.

```bash
rem ask "What is machine learning?"
rem ask "Find docs" --schema query-agent
rem ask "Hello" --model openai:gpt-4o
rem ask "Hello" --no-stream
```

Options:

- `--schema, -s`: Agent YAML file or name
- `--user-id, -u`: User ID (default: cli-user)
- `--model, -m`: Model identifier
- `--stream/--no-stream`: Streaming (default: on)

## rem query

Execute [REM queries](../reference/rem-query.md) directly.

```bash
rem query "LOOKUP transformer"
rem query "SEARCH neural networks IN ontology"
rem query "FUZZY backprop"
```

Options:

- `--user-id, -u`: User ID for scoping
- `--limit, -l`: Result limit (default: 20)

## rem serve

Start the API server.

```bash
rem serve
rem serve --host 0.0.0.0 --port 8000
rem serve --no-reload
```

## rem ingest

Ingest markdown files.

```bash
rem ingest ontology/
rem ingest ontology/ --dry-run
rem ingest docs/ --table resources
```

Options:

- `--table, -t`: Target table (default: ontology)
- `--pattern, -p`: Glob pattern (default: **/*.md)
- `--dry-run`: Preview only

## rem install

Install database schema.

```bash
rem install
```

Creates tables: ontology, resources, sessions, messages, kv_store.

## See also

- `REM LOOKUP rem-query` - Query language reference
- `REM LOOKUP agent-schema` - Agent YAML schema format
- `REM LOOKUP quick-start` - Installation guide
- `REM LOOKUP multi-agent` - Multi-agent examples
