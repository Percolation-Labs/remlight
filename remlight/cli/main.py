"""REMLight CLI - Command Line Interface."""

import asyncio
import sys
from pathlib import Path

import click

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
@click.option("--model", "-m", help="Model to use (e.g., openai:gpt-4o-mini)")
@click.option("--stream/--no-stream", default=True, help="Stream output")
def ask(query: str, schema: str | None, user_id: str, model: str | None, stream: bool):
    """
    Ask an agent a question.

    Examples:
        rem ask "What is machine learning?"
        rem ask "Find documents about AI" --schema query-agent
        rem ask "Search for projects" --model openai:gpt-4o
    """
    asyncio.run(_ask_async(query, schema, user_id, model, stream))


async def _ask_async(
    query: str,
    schema_path: str | None,
    user_id: str,
    model: str | None,
    stream: bool,
):
    """Async implementation of ask command."""
    from remlight.services.database import get_db
    from remlight.api.mcp_main import init_mcp, get_mcp_tools
    from remlight.api.routers.tools import get_agent_schema, init_tools
    from remlight.agentic import (
        AgentContext,
        create_agent,
        schema_from_yaml_file,
        run_streaming,
        run_sync,
    )
    from remlight.settings import settings

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
        # Check if it's a file path
        if Path(schema_path).exists():
            schema = schema_from_yaml_file(schema_path)
        else:
            # Try as agent name
            schema = get_agent_schema(schema_path)
            if schema is None:
                click.echo(f"Error: Agent '{schema_path}' not found", err=True)
                return
    else:
        schema = _default_cli_schema()

    # Create context
    context = AgentContext(user_id=user_id)

    # Get tools from MCP server
    tools = await get_mcp_tools()

    # Create agent
    agent_runtime = await create_agent(
        schema=schema,
        model_name=model or settings.llm.default_model,
        tools=tools,
        context=context,
    )

    click.echo()  # Newline before response

    if stream:
        # Stream response using unified runner (plain text for CLI)
        async for chunk in run_streaming(
            agent=agent_runtime.agent,
            prompt=query,
            context=context,
            model=model,
            user_id=user_id,
            persist_messages=settings.postgres.enabled,
            output_format="plain",
        ):
            click.echo(chunk, nl=False)
        click.echo()  # Final newline
    else:
        # Non-streaming response using unified runner
        result = await run_sync(
            agent=agent_runtime.agent,
            prompt=query,
            context=context,
            user_id=user_id,
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
    from remlight.services.database import get_db
    from remlight.api.routers.tools import search, init_tools

    db = get_db()
    await db.connect()
    init_tools(db)

    # Call search tool directly
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
