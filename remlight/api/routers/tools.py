"""MCP Tools as Router-style endpoints.

This module defines tools that can be:
1. Registered with FastMCP as MCP tools
2. Called directly as async functions
3. Exposed via FastAPI router endpoints

The pattern allows unified tool definitions shared between
MCP server and REST API.
"""

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from remlight.services.database import DatabaseService, get_db

# Disable schema caching when SCHEMA_CACHE_DISABLED=true
_schema_cache_disabled = os.getenv("SCHEMA_CACHE_DISABLED", "").lower() in ("true", "1", "yes")


# Request models for REST API endpoints
class SearchRequest(BaseModel):
    """Request body for search endpoint."""
    query: str
    limit: int = 20
    user_id: str | None = None


class ParseFileRequest(BaseModel):
    """Request body for parse-file endpoint.

    Files are stored globally by default. Avoid setting user_id unless you
    specifically need per-user file isolation.
    """
    uri: str
    user_id: str | None = None
    save_to_db: bool = True


from remlight.settings import settings

# Router for REST API exposure (MCP tool execution endpoints)
router = APIRouter(prefix="/mcp-tools", tags=["mcp-tools"])

# Module-level state
_db: DatabaseService | None = None
_metadata_store: dict[str, Any] = {}
_agent_schemas: dict[str, dict] = {}


def init_tools(db: DatabaseService | None = None) -> None:
    """Initialize tools with database connection."""
    global _db
    _db = db or get_db()


def get_tools_db() -> DatabaseService:
    """Get database service for tools."""
    global _db
    if _db is None:
        _db = get_db()
    return _db


def register_agent_schema(name: str, schema: dict) -> None:
    """Register an agent schema for use with ask_agent."""
    _agent_schemas[name] = schema


def get_agent_schema(name: str) -> dict | None:
    """Get a registered agent schema.

    Set SCHEMA_CACHE_DISABLED=true to always reload from disk.
    """
    # Check cache first (unless disabled)
    if not _schema_cache_disabled and name in _agent_schemas:
        return _agent_schemas[name]

    # Try to load from schemas directory (project root/schemas/)
    schema_path = Path(__file__).parent.parent.parent.parent / "schemas" / f"{name}.yaml"
    if schema_path.exists():
        import yaml
        schema = yaml.safe_load(schema_path.read_text())
        if not _schema_cache_disabled:
            _agent_schemas[name] = schema
        return schema
    return None


async def get_user_profile(user_id: str) -> dict | None:
    """Load user profile from database.

    Args:
        user_id: User identifier - can be user_id field, UUID, or email

    Returns:
        User profile dict or None if not found
    """
    db = get_tools_db()
    try:
        row = await db.fetchrow(
            "SELECT * FROM users WHERE user_id = $1 OR id::text = $1 OR email = $1",
            user_id
        )
        return dict(row) if row else None
    except Exception:
        return None


async def get_user_profile_hint(user_id: str | None = None) -> str:
    """Get formatted user profile hint for agent context.

    Always includes current date/time. If user_id provided,
    also includes user profile information.
    """
    hints = []

    # Always inject current date/time (critical for agent self-awareness)
    now = datetime.now(timezone.utc)
    hints.append(f"Date: {now.strftime('%Y-%m-%d')}")
    hints.append(f"Time: {now.strftime('%H:%M:%S')} UTC")

    # Add user profile if available
    if user_id:
        profile = await get_user_profile(user_id)
        if profile:
            if profile.get("name"):
                hints.append(f"User: {profile['name']}")
            if profile.get("summary"):
                hints.append(f"Profile: {profile['summary']}")
            if profile.get("interests"):
                interests = profile["interests"]
                if isinstance(interests, list) and interests:
                    hints.append(f"Interests: {', '.join(interests[:5])}")
            if profile.get("preferred_topics"):
                topics = profile["preferred_topics"]
                if isinstance(topics, list) and topics:
                    hints.append(f"Topics: {', '.join(topics[:5])}")

    return "\n".join(hints)


def format_user_profile(profile: dict) -> str:
    """Format user profile as markdown for MCP resource."""
    output = [f"# User Profile: {profile.get('name') or profile.get('email') or 'Unknown'}", ""]

    if profile.get("email"):
        output.append(f"**Email:** {profile['email']}")
    if profile.get("summary"):
        output.append(f"\n## Summary\n{profile['summary']}")
    if profile.get("interests"):
        interests = profile["interests"]
        if isinstance(interests, list) and interests:
            output.append(f"\n## Interests\n- " + "\n- ".join(interests[:10]))
    if profile.get("preferred_topics"):
        topics = profile["preferred_topics"]
        if isinstance(topics, list) and topics:
            output.append(f"\n## Preferred Topics\n- " + "\n- ".join(topics[:10]))
    if profile.get("activity_level"):
        output.append(f"\n**Activity Level:** {profile['activity_level']}")

    return "\n".join(output)


async def get_project(project_key: str) -> dict | None:
    """
    Load project details by key.

    TODO: In future, save project lookups to database for analytics.
    Currently fetches from ontologies table where entity_type = 'project'.

    Args:
        project_key: Project identifier (e.g., 'project-alpha')

    Returns:
        Project dict with name, description, properties, etc.
    """
    import json as json_lib

    db = get_tools_db()
    try:
        # Try to find in ontologies table (projects are stored as ontology entities)
        row = await db.fetchrow(
            """
            SELECT id, name, description, category, entity_type, properties, tags, metadata
            FROM ontologies
            WHERE name = $1 AND entity_type = 'project' AND deleted_at IS NULL
            """,
            project_key
        )
        if row:
            result = dict(row)
            # Merge properties into top-level for easier access
            # Properties may be a JSON string or dict depending on driver
            props = result.get("properties")
            if props:
                if isinstance(props, str):
                    props = json_lib.loads(props)
                result.update(props)
            return result

        # Fallback: try kv_store lookup
        row = await db.fetchrow(
            "SELECT data FROM kv_store WHERE entity_key = $1",
            project_key
        )
        if row and row["data"]:
            data = row["data"]
            if isinstance(data, str):
                data = json_lib.loads(data)
            return dict(data)

        return None
    except Exception:
        return None


def format_project(project: dict) -> str:
    """
    Format project as JSON string for MCP resource.

    Returns structured JSON that agents can parse and use.
    """
    import json

    # Build clean project output
    output = {
        "name": project.get("name"),
        "description": project.get("description"),
        "category": project.get("category"),
        "status": project.get("status"),
        "lead": project.get("lead"),
        "team_size": project.get("team_size"),
        "start_date": project.get("start_date"),
        "budget": project.get("budget"),
        "priority": project.get("priority"),
        "tags": project.get("tags", []),
    }

    # Remove None values for cleaner output
    output = {k: v for k, v in output.items() if v is not None}

    return json.dumps(output, indent=2)


def get_metadata(request_id: str = "default") -> dict[str, Any]:
    """Retrieve stored metadata for a request."""
    return _metadata_store.get(request_id, {})


def clear_metadata(request_id: str = "default") -> None:
    """Clear stored metadata for a request."""
    _metadata_store.pop(request_id, None)


# =============================================================================
# Tool Functions (can be registered with MCP or called directly)
# =============================================================================

async def search(
    query: str,
    limit: int = 20,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute REM queries to search the knowledge base.

    Query Syntax:
    - LOOKUP <key>: O(1) exact entity lookup by key
    - SEARCH <text> IN <table>: Semantic vector search in table
    - FUZZY <text>: Fuzzy text matching across all entities
    - TRAVERSE <key> [DEPTH n]: Graph traversal from entity

    Tables: ontologies, resources, users, messages

    Examples:
    - search("LOOKUP sarah-chen")
    - search("SEARCH machine learning IN ontologies")
    - search("FUZZY project alpha")
    - search("TRAVERSE project-alpha DEPTH 2")

    Args:
        query: REM query string
        limit: Maximum results (default 20)
        user_id: Optional user scope

    Returns:
        Query results with entities and metadata
    """
    db = get_tools_db()

    try:
        query = query.strip()
        query_upper = query.upper()

        # LOOKUP query
        if query_upper.startswith("LOOKUP"):
            key = query[6:].strip().strip('"\'')
            result = await db.rem_lookup(key, user_id)
            return {
                "status": "success",
                "query_type": "LOOKUP",
                "key": key,
                "result": result,
                "found": bool(result and result != {} and result != '{}'),
            }

        # SEARCH query
        search_match = re.match(
            r"SEARCH\s+[\"']?(.+?)[\"']?\s+IN\s+(\w+)",
            query,
            re.IGNORECASE
        )
        if search_match:
            search_text = search_match.group(1).strip()
            table_name = search_match.group(2).lower()
            results = await db.rem_fuzzy(search_text, user_id, 0.2, limit)
            return {
                "status": "success",
                "query_type": "SEARCH",
                "search_text": search_text,
                "table": table_name,
                "results": results,
                "count": len(results),
            }

        # FUZZY query
        if query_upper.startswith("FUZZY"):
            text = query[5:].strip().strip('"\'')
            results = await db.rem_fuzzy(text, user_id, 0.2, limit)
            return {
                "status": "success",
                "query_type": "FUZZY",
                "text": text,
                "results": results,
                "count": len(results),
            }

        # TRAVERSE query
        traverse_match = re.match(
            r"TRAVERSE\s+[\"']?(\S+)[\"']?(?:\s+DEPTH\s+(\d+))?",
            query,
            re.IGNORECASE
        )
        if traverse_match:
            entity_key = traverse_match.group(1).strip('"\'')
            depth = int(traverse_match.group(2) or 1)
            results = await db.rem_traverse(entity_key, None, depth, user_id)
            return {
                "status": "success",
                "query_type": "TRAVERSE",
                "entity_key": entity_key,
                "depth": depth,
                "results": results,
                "count": len(results),
            }

        # SQL query
        if query_upper.startswith("SQL"):
            raw_sql = query[3:].strip()
            # Safety check - only allow SELECT/WITH
            sql_upper = raw_sql.upper().strip()
            blocked = ("DROP", "DELETE", "TRUNCATE", "ALTER", "CREATE", "INSERT", "UPDATE")
            if any(sql_upper.startswith(p) for p in blocked):
                return {
                    "status": "error",
                    "error": "Blocked SQL operation. Only SELECT/WITH allowed.",
                    "query": query,
                }
            if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
                return {
                    "status": "error",
                    "error": "Only SELECT and WITH queries are allowed.",
                    "query": query,
                }
            rows = await db.fetch(raw_sql)
            return {
                "status": "success",
                "query_type": "SQL",
                "raw_query": raw_sql,
                "results": [dict(r) for r in rows],
                "count": len(rows),
            }

        # Default to fuzzy search
        results = await db.rem_fuzzy(query, user_id, 0.2, limit)
        return {
            "status": "success",
            "query_type": "FUZZY",
            "text": query,
            "results": results,
            "count": len(results),
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "query": query,
        }


async def action(
    type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Emit a typed action event for SSE streaming and UI updates.

    This tool sends structured events to the client. Use it to update UI, record metadata, or request input.

    Action Types:
        - "set_schema": Replace the entire schema with new YAML. Payload:
            - yaml: Complete agent schema as YAML string
        - "schema_focus": Highlight a section in the UI. Payload:
            - section: which section to highlight
            - message: optional message to display
        - "observation": Record metadata about the response (confidence, sources, etc.)
        - "elicit": Request additional information from the user

    Args:
        type: Action type (see above)
        payload: Action-specific data (see above for each type)

    Returns:
        Action result confirming the event was emitted

    Examples:
        action(type="schema_update", payload={"section": "system_prompt", "value": "You are a helpful assistant", "operation": "set"})
        action(type="schema_focus", payload={"section": "tools", "message": "Add tools here"})
        action(type="observation", payload={"confidence": 0.85, "sources": ["doc-1"]})
    """
    global _metadata_store

    result: dict[str, Any] = {
        "_action_event": True,
        "action_type": type,
        "status": "success",
    }

    if payload:
        result["payload"] = payload

        # For observation type, also set _metadata_event for backward compatibility
        if type == "observation":
            result["_metadata_event"] = True
            # Flatten payload fields for metadata event
            result.update({k: v for k, v in payload.items() if v is not None})

    request_id = (payload or {}).get("request_id", "default")
    _metadata_store[request_id] = result

    return result


async def ask_agent(
    agent_name: str,
    input_text: str,
    input_data: dict[str, Any] | None = None,
    user_id: str | None = None,
    timeout_seconds: int = 300,
    structured_output_override: bool | None = None,
) -> dict[str, Any]:
    """
    Invoke another agent by name and return its response.

    Enables multi-agent orchestration by allowing one agent to call another.
    Child agents inherit parent context (user_id, session_id, tenant_id).

    Args:
        agent_name: Name of the agent to invoke (must be registered)
        input_text: The user message/query to send to the agent
        input_data: Optional structured input data for the agent
        user_id: Optional user override (defaults to parent's user_id)
        timeout_seconds: Maximum execution time (default: 300s)
        structured_output_override: Optional override for structured_output mode.
            When provided, overrides the agent schema's structured_output setting.
            Only set this if intentional - often better to let the agent schema decide.

    Returns:
        Dict with status, output, text_response, agent_schema, and is_structured_output
    """
    from remlight.agentic import create_agent
    from remlight.agentic.context import (
        AgentContext,
        get_current_context,
        get_event_sink,
        agent_context_scope,
    )

    parent_context = get_current_context()

    if parent_context is not None:
        effective_user_id = user_id or parent_context.user_id
    else:
        effective_user_id = user_id or "anonymous"

    if parent_context is not None:
        child_context = parent_context.child_context(agent_schema_uri=agent_name)
        if user_id is not None:
            child_context = AgentContext(
                user_id=user_id,
                tenant_id=parent_context.tenant_id,
                session_id=parent_context.session_id,
                default_model=parent_context.default_model,
                agent_schema_uri=agent_name,
            )
    else:
        child_context = AgentContext(
            user_id=effective_user_id,
            tenant_id=effective_user_id,
            default_model=settings.llm.default_model,
            agent_schema_uri=agent_name,
        )

    schema = get_agent_schema(agent_name)
    if schema is None:
        return {
            "status": "error",
            "error": f"Agent not found: {agent_name}",
            "hint": "Available agents can be found in the schemas/ directory",
        }

    # Get tools - import here to avoid circular imports
    from remlight.api.mcp_main import get_mcp_tools
    tools = await get_mcp_tools()

    agent_runtime = await create_agent(
        schema=schema,
        model_name=child_context.default_model,
        tools=tools,
        context=child_context,
        structured_output_override=structured_output_override,
    )

    # Load session history for context continuity
    pydantic_message_history = None
    session_id = child_context.session_id
    if session_id:
        try:
            from remlight.services.session import (
                SessionMessageStore,
                session_to_pydantic_messages,
            )

            store = SessionMessageStore(user_id=child_context.user_id or "default")
            raw_session_history = await store.load_session_messages(
                session_id=session_id,
                user_id=child_context.user_id,
                compress_on_load=True,
            )

            if raw_session_history:
                from remlight.agentic.schema import get_system_prompt
                system_prompt = get_system_prompt(schema)
                pydantic_message_history = session_to_pydantic_messages(
                    raw_session_history,
                    system_prompt=system_prompt,
                )
        except Exception:
            pass

    prompt = input_text
    if input_data:
        prompt = f"{input_text}\n\nInput data: {json.dumps(input_data)}"

    event_sink = get_event_sink()
    use_streaming = event_sink is not None
    streamed_content = ""

    try:
        with agent_context_scope(child_context):
            if use_streaming:
                async def run_with_streaming():
                    accumulated_content = []
                    iter_kwargs = {}
                    if pydantic_message_history:
                        iter_kwargs["message_history"] = pydantic_message_history

                    async with agent_runtime.agent.iter(prompt, **iter_kwargs) as agent_run:
                        async for node in agent_run:
                            from pydantic_ai.agent import Agent

                            if Agent.is_model_request_node(node):
                                async with node.stream(agent_run.ctx) as request_stream:
                                    async for event in request_stream:
                                        event_type = type(event).__name__

                                        # PartStartEvent: Beginning of a new part (text or thinking)
                                        if event_type == "PartStartEvent":
                                            if hasattr(event, "part"):
                                                part_type = type(event.part).__name__
                                                if part_type in ("TextPart", "ThinkingPart"):
                                                    content = getattr(event.part, "content", None)
                                                    if content:
                                                        accumulated_content.append(content)
                                                        await event_sink.put({
                                                            "type": "child_content",
                                                            "agent_name": agent_name,
                                                            "content": content,
                                                        })

                                        # PartDeltaEvent: Incremental content
                                        elif hasattr(event, "delta") and hasattr(event.delta, "content_delta"):
                                            content = event.delta.content_delta
                                            if content:
                                                accumulated_content.append(content)
                                                await event_sink.put({
                                                    "type": "child_content",
                                                    "agent_name": agent_name,
                                                    "content": content,
                                                })

                            elif Agent.is_call_tools_node(node):
                                async with node.stream(agent_run.ctx) as tools_stream:
                                    async for tool_event in tools_stream:
                                        event_type = type(tool_event).__name__
                                        if event_type == "FunctionToolCallEvent":
                                            tool_args = None
                                            if hasattr(tool_event, "part") and hasattr(tool_event.part, "args"):
                                                raw_args = tool_event.part.args
                                                if isinstance(raw_args, str):
                                                    try:
                                                        tool_args = json.loads(raw_args)
                                                    except json.JSONDecodeError:
                                                        tool_args = {"raw": raw_args}
                                                elif isinstance(raw_args, dict):
                                                    tool_args = raw_args

                                            tool_call_id = getattr(tool_event.part, "tool_call_id", None) if hasattr(tool_event, "part") else None
                                            await event_sink.put({
                                                "type": "child_tool_start",
                                                "agent_name": agent_name,
                                                "tool_name": tool_event.part.tool_name if hasattr(tool_event, "part") else "unknown",
                                                "tool_call_id": tool_call_id,
                                                "arguments": tool_args,
                                            })

                                        elif event_type == "FunctionToolResultEvent":
                                            result_content = tool_event.result.content if hasattr(tool_event.result, "content") else tool_event.result
                                            tool_name = getattr(tool_event.result, "tool_name", "tool")
                                            tool_call_id = getattr(tool_event.result, "tool_call_id", None)
                                            await event_sink.put({
                                                "type": "child_tool_result",
                                                "agent_name": agent_name,
                                                "tool_name": tool_name,
                                                "tool_call_id": tool_call_id,
                                                "result": result_content,
                                            })

                        return agent_run.result, "".join(accumulated_content)

                result, streamed_content = await asyncio.wait_for(
                    run_with_streaming(),
                    timeout=timeout_seconds
                )
            else:
                run_kwargs = {}
                if pydantic_message_history:
                    run_kwargs["message_history"] = pydantic_message_history

                result = await asyncio.wait_for(
                    agent_runtime.agent.run(prompt, **run_kwargs),
                    timeout=timeout_seconds
                )

    except asyncio.TimeoutError:
        return {
            "status": "error",
            "error": f"Agent '{agent_name}' timed out after {timeout_seconds}s",
            "agent_schema": agent_name,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "agent_schema": agent_name,
        }

    # Import serialization utilities
    from remlight.agentic.serialization import serialize_agent_result, is_pydantic_model

    raw_output = result.output if hasattr(result, "output") else result

    # Detect if this is structured output (Pydantic model)
    is_structured_output = is_pydantic_model(raw_output)

    # Serialize output for response
    output = serialize_agent_result(raw_output)

    # Structured output tool ID for events and DB storage
    structured_tool_id = f"{agent_name}_structured_output"

    # If child agent returned structured output, emit as tool_call SSE event
    # This allows the frontend to render structured results (forms, cards, etc.)
    if use_streaming and is_structured_output and event_sink is not None:
        await event_sink.put({
            "type": "tool_call",
            "tool_name": agent_name,  # Use agent name as tool name for clarity
            "tool_id": structured_tool_id,
            "status": "completed",
            "arguments": {"input_text": input_text},
            "result": output,  # Serialized Pydantic model as dict
        })

    # Save structured output as a tool message in the database
    # This makes structured output agents look like tool calls in session history
    if is_structured_output and child_context and child_context.session_id:
        try:
            from remlight.services.session import SessionMessageStore

            store = SessionMessageStore(user_id=child_context.user_id or "default")

            # Build tool message in the same format as regular tool calls
            tool_message = {
                "role": "tool",
                "content": json.dumps(output, default=str),  # Structured output as JSON
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_call_id": structured_tool_id,
                "tool_name": agent_name,  # Agent name as tool name
                "tool_arguments": {"input_text": input_text},
            }

            # Store as a single message
            await store.store_session_messages(
                session_id=child_context.session_id,
                messages=[tool_message],
                user_id=child_context.user_id,
                compress=False,  # Don't compress tool results
            )
        except Exception:
            pass  # Best-effort, don't fail the request

    response = {
        "status": "success",
        "output": output,
        "agent_schema": agent_name,
        "input_text": input_text,
        "is_structured_output": is_structured_output,  # Flag for caller to know result type
    }

    # IMPORTANT: Only include text_response if content was NOT streamed.
    # When streaming, child_content events already delivered the content to the client.
    # Including text_response here would cause duplication.
    if not use_streaming or not streamed_content:
        if hasattr(result, "output") and result.output is not None:
            response["text_response"] = str(result.output)

    return response


# =============================================================================
# REST API Endpoints (wrap tool functions for HTTP access)
# =============================================================================


async def parse_file(
    uri: str,
    user_id: str | None = None,
    save_to_db: bool = True,
) -> dict[str, Any]:
    """
    Parse a file and extract content.

    Reads files from local filesystem, S3, or HTTP URLs and extracts text content.
    Uses Kreuzberg for document parsing (PDF, DOCX, PPTX, XLSX, images).
    Saves the parsed result to the files table with rich metadata.

    Files are stored globally by default. Avoid setting user_id unless you
    specifically need per-user file isolation (this prevents file sharing).

    Supported formats:
    - Documents: PDF, DOCX, PPTX, XLSX (via Kreuzberg with OCR fallback)
    - Images: PNG, JPG, GIF, WEBP (OCR text extraction)
    - Text: Markdown, JSON, YAML, code files (UTF-8 extraction)

    Args:
        uri: File URI - local path, file://, s3://, or http(s):// URL
        user_id: Optional user ID (rarely needed - makes file user-specific)
        save_to_db: Whether to save File entity to database (default: True)

    Returns:
        Dict with:
            - file_id: UUID of saved File entity
            - uri: Original URI
            - uri_hash: SHA256 hash of URI for deduplication
            - name: Filename
            - content: Extracted text content
            - parsed_output: Full parsing result with metadata
            - status: 'completed' or 'failed'

    Examples:
        parse_file("./document.pdf")
        parse_file("s3://bucket/report.docx")
        parse_file("https://example.com/paper.pdf")
        parse_file("/path/to/image.png", save_to_db=False)
    """
    from remlight.services.content import get_content_service

    service = get_content_service()

    try:
        result = await service.parse_file(
            uri=uri,
            user_id=user_id,
            save_to_db=save_to_db,
        )
        return result
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": f"File not found: {e}",
            "uri": uri,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "uri": uri,
        }


async def save_agent(
    name: str | None = None,
    schema_yaml: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """
    Save an agent schema to the database.

    When called with no arguments, emits a 'trigger_save' action event that tells
    the frontend to save the current schema state. This is the recommended usage
    in the agent-builder context.

    When called with name and schema_yaml, saves directly to the database.

    Args:
        name: Unique agent name (optional - if not provided, triggers frontend save)
        schema_yaml: Complete agent schema as YAML string (optional)
        description: Optional short description for listing
        tags: Optional tags for discovery
        overwrite: Whether to update if agent already exists

    Returns:
        Dict with action event to trigger frontend save, or save result

    Examples:
        save_agent()  # Triggers frontend to save current schema
        save_agent("my-agent", "type: object\\ndescription: ...", tags=["custom"])
    """
    # If no args provided, emit action event for frontend to handle
    if name is None or schema_yaml is None:
        return {
            "status": "pending",
            "message": "Save triggered - frontend will complete the save",
            "_action_event": True,
            "action_type": "trigger_save",
        }
    import yaml

    db = get_tools_db()

    try:
        # Parse and validate YAML
        try:
            schema = yaml.safe_load(schema_yaml)
        except yaml.YAMLError as e:
            return {
                "status": "error",
                "error": f"Invalid YAML: {e}",
                "_action_event": True,
                "action_type": "agent_save_error",
            }

        # Validate basic schema structure
        if not isinstance(schema, dict):
            return {
                "status": "error",
                "error": "Schema must be a YAML object",
                "_action_event": True,
                "action_type": "agent_save_error",
            }

        if "description" not in schema:
            return {
                "status": "error",
                "error": "Schema must have a 'description' field (system prompt)",
                "_action_event": True,
                "action_type": "agent_save_error",
            }

        # Get version from schema
        json_schema_extra = schema.get("json_schema_extra", {})
        version = json_schema_extra.get("version", "1.0.0")

        # Check if agent exists
        existing = await db.fetchrow(
            "SELECT id FROM agents WHERE name = $1 AND deleted_at IS NULL",
            name
        )

        if existing and not overwrite:
            return {
                "status": "error",
                "error": f"Agent '{name}' already exists. Use overwrite=True to update.",
                "_action_event": True,
                "action_type": "agent_save_error",
            }

        # Prepare agent data
        schema_tags = json_schema_extra.get("tags", [])
        all_tags = list(set((tags or []) + schema_tags))

        if existing:
            # Update existing agent
            await db.execute(
                """
                UPDATE agents
                SET content = $2,
                    description = $3,
                    tags = $4,
                    updated_at = NOW()
                WHERE name = $1 AND deleted_at IS NULL
                """,
                name,
                schema_yaml,
                description or schema.get("description", "")[:200],
                all_tags,
            )
            created = False
        else:
            # Insert new agent
            await db.execute(
                """
                INSERT INTO agents (name, content, description, tags, enabled)
                VALUES ($1, $2, $3, $4, TRUE)
                """,
                name,
                schema_yaml,
                description or schema.get("description", "")[:200],
                all_tags,
            )
            created = True

        return {
            "status": "success",
            "agent_name": name,
            "version": version,
            "created": created,
            "_action_event": True,
            "action_type": "agent_saved",
            "payload": {
                "name": name,
                "version": version,
                "created": created,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "_action_event": True,
            "action_type": "agent_save_error",
        }


# =============================================================================
# REST API Endpoints (wrap tool functions for HTTP access)
# =============================================================================


async def analyze_pages(
    file_uri: str,
    start_page: int = 1,
    end_page: int | None = None,
    prompt: str | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    page_batch_size: int = 5,
) -> dict[str, Any]:
    """
    Analyze document pages using vision AI models.

    Converts PDF pages or images to vision-ready format and analyzes them
    using multimodal LLMs (Claude, GPT-4, Gemini).

    Args:
        file_uri: File path or URI (local path, s3://, https://)
        start_page: First page to analyze (1-indexed, default: 1)
        end_page: Last page to analyze (default: all pages)
        prompt: Analysis prompt (default: generic page description)
        provider: Vision provider - 'anthropic', 'openai', or 'gemini' (default: anthropic)
        model: Model override (e.g., 'claude-sonnet-4.5', 'gpt-4.1')
        page_batch_size: Number of pages per API call (default: 5)

    Returns:
        Dict with:
            - status: 'success' or 'error'
            - description: Combined analysis text
            - page_count: Number of pages analyzed
            - provider: Vision provider used
            - model: Model used
            - usage: Token usage stats

    Examples:
        analyze_pages("invoice.pdf")
        analyze_pages("s3://bucket/doc.pdf", start_page=1, end_page=3)
        analyze_pages("image.png", prompt="Extract all text from this image")
        analyze_pages("report.pdf", provider="openai", model="gpt-4.1")
    """
    import tempfile
    from pathlib import Path

    from loguru import logger

    from remlight.services.vision import (
        VisionProvider,
        analyze_image_async,
        analyze_images_async,
    )

    # Default prompt for generic page analysis
    default_prompt = """Analyze this page and provide a detailed description of its contents.
Include:
- Document type and structure
- Key text and data
- Tables, charts, or images present
- Any notable formatting or layout"""

    analysis_prompt = prompt or default_prompt

    # Map provider string to enum
    provider_map = {
        "anthropic": VisionProvider.ANTHROPIC,
        "openai": VisionProvider.OPENAI,
        "gemini": VisionProvider.GEMINI,
    }
    vision_provider = provider_map.get(provider.lower(), VisionProvider.ANTHROPIC)

    try:
        # Resolve file URI to local path
        local_path = None

        if file_uri.startswith("s3://"):
            # Download from S3
            import aioboto3

            parts = file_uri[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""

            session = aioboto3.Session()
            async with session.client("s3") as s3:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(key).suffix) as f:
                    await s3.download_fileobj(bucket, key, f)
                    local_path = f.name

        elif file_uri.startswith(("http://", "https://")):
            # Download from URL
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(file_uri)
                response.raise_for_status()

                suffix = Path(file_uri.split("?")[0]).suffix or ".pdf"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    f.write(response.content)
                    local_path = f.name

        else:
            # Local file path
            if file_uri.startswith("file://"):
                local_path = file_uri[7:]  # Remove "file://" prefix
            else:
                local_path = file_uri

        path = Path(local_path)
        if not path.exists():
            return {
                "status": "error",
                "error": f"File not found: {local_path}",
                "file_uri": file_uri,
            }

        # Check file type
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # PDF - convert pages to images using PyMuPDF
            import fitz  # pymupdf

            doc = fitz.open(str(path))
            total_pages = len(doc)

            # Determine page range
            first_page = max(1, start_page) - 1  # Convert to 0-indexed
            last_page = min(end_page or total_pages, total_pages)

            logger.info(f"Analyzing PDF pages {first_page + 1}-{last_page} of {total_pages}")

            # Collect all page images
            page_images: list[tuple[bytes, str]] = []

            for page_num in range(first_page, last_page):
                page = doc[page_num]
                # Render at 150 DPI for good quality without excessive size
                pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
                img_bytes = pix.tobytes("png")
                page_images.append((img_bytes, "image/png"))

            doc.close()

            # Analyze in batches
            all_descriptions = []
            total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "requests": 0}

            for i in range(0, len(page_images), page_batch_size):
                batch = page_images[i:i + page_batch_size]
                batch_start = first_page + i + 1
                batch_end = batch_start + len(batch) - 1

                batch_prompt = f"{analysis_prompt}\n\nAnalyzing pages {batch_start}-{batch_end}."

                if len(batch) == 1:
                    result = await analyze_image_async(
                        image_data=batch[0][0],
                        prompt=batch_prompt,
                        provider=vision_provider,
                        media_type=batch[0][1],
                        model=model,
                    )
                else:
                    result = await analyze_images_async(
                        images=batch,
                        prompt=batch_prompt,
                        provider=vision_provider,
                        model=model,
                    )

                all_descriptions.append(result.description)

                # Accumulate usage
                for key in total_usage:
                    total_usage[key] += result.usage.get(key, 0)

            combined_description = "\n\n---\n\n".join(all_descriptions)

            return {
                "status": "success",
                "description": combined_description,
                "page_count": len(page_images),
                "total_pages": total_pages,
                "provider": vision_provider.value,
                "model": result.model if 'result' in dir() else model,
                "usage": total_usage,
                "file_uri": file_uri,
            }

        elif suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            # Single image file
            image_data = path.read_bytes()
            media_type = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".gif": "image/gif",
                ".webp": "image/webp",
            }.get(suffix, "image/png")

            result = await analyze_image_async(
                image_data=image_data,
                prompt=analysis_prompt,
                provider=vision_provider,
                media_type=media_type,
                model=model,
            )

            return {
                "status": "success",
                "description": result.description,
                "page_count": 1,
                "provider": result.provider.value,
                "model": result.model,
                "usage": result.usage,
                "file_uri": file_uri,
            }

        else:
            return {
                "status": "error",
                "error": f"Unsupported file type: {suffix}. Supported: PDF, PNG, JPG, GIF, WEBP",
                "file_uri": file_uri,
            }

    except Exception as e:
        logger.exception(f"Vision analysis failed for {file_uri}")
        return {
            "status": "error",
            "error": str(e),
            "file_uri": file_uri,
        }

    finally:
        # Cleanup temp files (only if we downloaded from S3 or URL)
        original_path = file_uri[7:] if file_uri.startswith("file://") else file_uri
        if local_path and local_path != original_path:
            try:
                Path(local_path).unlink(missing_ok=True)
            except Exception:
                pass


@router.post("/search")
async def search_endpoint(request: SearchRequest) -> dict[str, Any]:
    """REST endpoint for search tool."""
    return await search(query=request.query, limit=request.limit, user_id=request.user_id)


@router.post("/parse-file")
async def parse_file_endpoint(request: ParseFileRequest) -> dict[str, Any]:
    """REST endpoint for parse_file tool."""
    return await parse_file(uri=request.uri, user_id=request.user_id, save_to_db=request.save_to_db)
