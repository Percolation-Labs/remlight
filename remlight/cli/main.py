"""REMLight CLI - Command Line Interface."""

import asyncio
import re
import sys
from pathlib import Path

import click
import yaml

from remlight import __version__


@click.group()
@click.version_option(__version__)
def cli():
    """REM - Minimal declarative agent framework."""
    pass


@cli.command()
@click.argument("query")
@click.option("--schema", "-s", help="Path to agent schema YAML file or agent name")
@click.option("--user-id", "-u", default="cli-user", help="User ID for queries")
@click.option("--session", help="Session UUID for multi-turn conversations")
@click.option("--model", "-m", help="Model to use (e.g., openai:gpt-4.1)")
@click.option("--stream/--no-stream", default=True, help="Stream output")
def ask(query: str, schema: str | None, user_id: str, session: str | None, model: str | None, stream: bool):
    """
    Ask an agent a question.

    Examples:
        rem ask "What is machine learning?"
        rem ask "Find documents about AI" --schema query-agent
        rem ask "Search for projects" --model openai:gpt-4.1

    Multi-turn conversations (session must be a UUID):
        rem ask "What is REM?" --session 550e8400-e29b-41d4-a716-446655440000
        rem ask "Tell me more" --session 550e8400-e29b-41d4-a716-446655440000
    """
    asyncio.run(_ask_async(query, schema, user_id, session, model, stream))


async def _ask_async(
    query: str,
    schema_path: str | None,
    user_id: str,
    session_id: str | None,
    model: str | None,
    stream: bool,
):
    """Async implementation of ask command."""
    import uuid as uuid_module

    from remlight.agentic import (
        AgentContext,
        create_agent,
        run_streaming,
        run_sync,
        schema_from_yaml_file,
    )
    from remlight.api.mcp_main import get_mcp_tools, init_mcp
    from remlight.api.routers.tools import get_agent_schema, init_tools
    from remlight.services.database import get_db
    from remlight.settings import settings

    # Validate session_id is a UUID if provided
    if session_id:
        try:
            uuid_module.UUID(session_id)
        except ValueError:
            click.echo(f"Error: Session must be a valid UUID, got: {session_id}", err=True)
            return

    # Connect to database
    db = get_db()
    try:
        await db.connect()
        init_tools(db)
        init_mcp(db)
    except Exception as e:
        click.echo(f"Warning: Could not connect to database: {e}", err=True)
        click.echo("Running without database access.", err=True)

    # Load schema
    if schema_path:
        if Path(schema_path).exists():
            schema = schema_from_yaml_file(schema_path)
        else:
            schema = get_agent_schema(schema_path)
            if schema is None:
                click.echo(f"Error: Agent '{schema_path}' not found", err=True)
                return
    else:
        schema = _default_cli_schema()

    # Create context and agent
    context = AgentContext(user_id=user_id, session_id=session_id)
    tools = await get_mcp_tools()
    agent_runtime = await create_agent(
        schema=schema,
        model_name=model or settings.llm.default_model,
        tools=tools,
        context=context,
    )

    click.echo()

    if stream:
        async for chunk in run_streaming(
            agent=agent_runtime.agent,
            prompt=query,
            context=context,
            model=model,
            user_id=user_id,
            session_id=session_id,
            agent_schema=schema,  # Pass schema for system prompt extraction
            persist_messages=settings.postgres.enabled,
            output_format="plain",
        ):
            click.echo(chunk, nl=False)
        click.echo()
    else:
        result = await run_sync(
            agent=agent_runtime.agent,
            prompt=query,
            context=context,
            user_id=user_id,
            session_id=session_id,
            agent_schema=schema,  # Pass schema for system prompt extraction
            persist_messages=settings.postgres.enabled,
        )
        output = result.output if hasattr(result, "output") else result
        click.echo(output)

    try:
        await db.disconnect()
    except Exception:
        pass


def _default_cli_schema() -> dict:
    """Default agent schema for CLI."""
    return {
        "type": "object",
        "description": """You are a helpful assistant with access to a knowledge base.
Use the search tool to find relevant information.
Provide clear, concise answers.""",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "json_schema_extra": {
            "kind": "agent",
            "name": "cli-agent",
            "version": "1.0.0",
            "tools": [{"name": "search"}, {"name": "action"}],
        },
    }


@cli.command()
@click.argument("query")
@click.option("--user-id", "-u", help="User ID for scoping")
@click.option("--limit", "-l", default=20, help="Result limit")
def query(query: str, user_id: str | None, limit: int):
    """
    Execute a REM query directly.

    Query types:
        LOOKUP <key>              - Exact key lookup
        SEARCH <text> IN <table>  - Semantic search
        FUZZY <text>              - Fuzzy text match
        TRAVERSE <key>            - Graph traversal

    Examples:
        rem query "LOOKUP sarah-chen"
        rem query "SEARCH machine learning IN ontologies"
        rem query "FUZZY project alpha"
    """
    asyncio.run(_query_async(query, user_id, limit))


async def _query_async(query_str: str, user_id: str | None, limit: int):
    """Async implementation of query command."""
    import json

    from remlight.api.routers.tools import init_tools, search
    from remlight.services.database import get_db

    db = get_db()
    try:
        await db.connect()
    except Exception as e:
        click.echo(f"Error: Could not connect to database: {e}", err=True)
        click.echo("The query command requires a running PostgreSQL database.", err=True)
        click.echo("Start with: docker compose up -d postgres", err=True)
        return

    init_tools(db)

    result = await search(query=query_str, limit=limit, user_id=user_id)
    click.echo(json.dumps(result, indent=2, default=str))

    await db.disconnect()


@cli.command()
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind")
@click.option("--port", "-p", default=8000, help="Port to bind")
@click.option("--reload/--no-reload", default=True, help="Enable auto-reload")
def serve(host: str, port: int, reload: bool):
    """Start the REM API server (includes MCP endpoint)."""
    import uvicorn

    click.echo(f"Starting REM server v{__version__} on http://{host}:{port}")
    click.echo(f"  API docs: http://{host}:{port}/docs")
    click.echo(f"  MCP endpoint: http://{host}:{port}/api/v1/mcp")
    uvicorn.run(
        "remlight.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--table", "-t", default="ontology", help="Target table (default: ontology)")
@click.option("--pattern", "-p", default="**/*.md", help="Glob pattern (default: **/*.md)")
@click.option("--dry-run", is_flag=True, help="Preview without making changes")
def ingest(path: str, table: str, pattern: str, dry_run: bool):
    """
    Ingest markdown files into REM database.

    Parses markdown files with YAML frontmatter and stores them with embeddings
    for semantic search.

    Examples:
        rem ingest ontology/
        rem ingest ontology/ --dry-run
        rem ingest ontology/concepts/ --pattern "*.md"
        rem ingest docs/ --table resources
    """
    asyncio.run(_ingest_async(path, table, pattern, dry_run))


async def _ingest_async(path: str, table: str, pattern: str, dry_run: bool):
    """Async implementation of ingest command."""
    from remlight.services.database import get_db
    from remlight.services.repository import OntologyRepository, ResourceRepository

    input_path = Path(path)

    # Collect files to process
    if input_path.is_dir():
        files = list(input_path.glob(pattern))
        files = [
            f for f in files
            if f.name != "README.md" and "scripts" not in f.parts
        ]
    else:
        files = [input_path]

    if not files:
        click.echo(f"No files matching '{pattern}' found in {input_path}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(files)} files")

    if dry_run:
        click.echo("\nDRY RUN - Would ingest:")
        for f in files[:20]:
            entity_key = _extract_entity_key(f)
            click.echo(f"  {f.name} → {table} (key: {entity_key})")
        if len(files) > 20:
            click.echo(f"  ... and {len(files) - 20} more")
        return

    # Connect to database
    db = get_db()
    await db.connect()

    try:
        # Create appropriate repository
        if table in ("ontology", "ontologies"):
            repo = OntologyRepository(db)
        elif table == "resources":
            repo = ResourceRepository(db)
        else:
            click.echo(f"Unknown table: {table}", err=True)
            sys.exit(1)

        processed = 0
        failed = 0

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
                entity_key = _extract_entity_key(file_path, content)
                frontmatter = _parse_frontmatter(content)

                category = frontmatter.get("parent") if frontmatter else None
                tags = frontmatter.get("tags", []) if frontmatter else []
                properties = frontmatter or {}

                # Use repository upsert (handles embeddings automatically)
                if isinstance(repo, OntologyRepository):
                    await repo.upsert(
                        name=entity_key,
                        description=content,
                        category=category,
                        tags=tags,
                        properties=properties,
                    )
                else:
                    await repo.upsert(
                        name=entity_key,
                        content=content,
                        uri=f"file://{file_path.absolute()}",
                        category=category,
                        tags=tags,
                        metadata=properties,
                    )

                processed += 1
                click.echo(f"  ✓ {entity_key}")

            except Exception as e:
                failed += 1
                click.echo(f"  ✗ {file_path.name}: {e}", err=True)

        click.echo(f"\nCompleted: {processed} succeeded, {failed} failed")

    finally:
        await db.disconnect()


def _extract_entity_key(file_path: Path, content: str | None = None) -> str:
    """Extract entity key from frontmatter or filename."""
    if content:
        frontmatter = _parse_frontmatter(content)
        if frontmatter and frontmatter.get("entity_key"):
            return frontmatter["entity_key"]
    return file_path.stem


def _parse_frontmatter(content: str) -> dict | None:
    """Parse YAML frontmatter from markdown content."""
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except Exception:
        return None


@cli.command()
def install():
    """Install database schema (tables, triggers, functions)."""
    asyncio.run(_install_async())


async def _install_async():
    """Async implementation of install command."""
    from remlight.services.database import get_db

    click.echo("Installing REM database schema...")

    db = get_db()
    await db.connect()

    try:
        await db.install_schema()
        click.echo("Database schema installed successfully!")
    except Exception as e:
        click.echo(f"Error installing schema: {e}", err=True)
        sys.exit(1)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    cli()
