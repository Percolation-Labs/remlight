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
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from remlight.services.database import DatabaseService, get_db
from remlight.settings import settings

# Router for REST API exposure
router = APIRouter(prefix="/tools", tags=["tools"])

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
    """Get a registered agent schema."""
    if name in _agent_schemas:
        return _agent_schemas[name]

    # Try to load from schemas directory (project root/schemas/)
    schema_path = Path(__file__).parent.parent.parent.parent / "schemas" / f"{name}.yaml"
    if schema_path.exists():
        import yaml
        schema = yaml.safe_load(schema_path.read_text())
        _agent_schemas[name] = schema
        return schema
    return None


async def get_user_profile(user_id: str) -> dict | None:
    """Load user profile from database."""
    db = get_tools_db()
    try:
        row = await db.fetchrow(
            "SELECT * FROM users WHERE user_id = $1 OR id::text = $1",
            user_id
        )
        return dict(row) if row else None
    except Exception:
        return None


async def get_user_profile_hint(user_id: str) -> str:
    """Get formatted user profile hint for agent context."""
    profile = await get_user_profile(user_id)
    if not profile:
        return ""

    hints = []
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

    return "\n".join(hints) if hints else ""


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
    Emit a typed action event for SSE streaming.

    This is the generic action tool for structured agent outputs.
    Clients handle different action types appropriately.

    Action Types:
        - "observation": Record metadata about the response (confidence, sources, etc.)
        - "elicit": Request additional information from the user
        - "delegate": Signal delegation to another agent (internal use)

    Args:
        type: Action type ("observation", "elicit", "delegate", etc.)
        payload: Action-specific data. For "observation":
            - confidence: float (0.0-1.0)
            - sources: list[str] - Entity keys used
            - session_name: str - Short name for UI
            - references: list[str] - Doc links
            - flags: list[str] - Special handling flags
            - risk_level: str - "low"/"moderate"/"high"/"critical"
            - risk_score: int - 0-100
            - extra: dict - Additional fields

    Returns:
        Action result with _action_event marker for SSE streaming layer

    Examples:
        action(type="observation", payload={"confidence": 0.85, "sources": ["doc-1"]})
        action(type="elicit", payload={"question": "What format?", "options": ["PDF", "CSV"]})
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

    Returns:
        Dict with status, output, text_response, and agent_schema
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
                                        if hasattr(event, "delta") and hasattr(event.delta, "content_delta"):
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

                                            await event_sink.put({
                                                "type": "child_tool_start",
                                                "agent_name": agent_name,
                                                "tool_name": tool_event.part.tool_name if hasattr(tool_event, "part") else "unknown",
                                                "arguments": tool_args,
                                            })

                                        elif event_type == "FunctionToolResultEvent":
                                            result_content = tool_event.result.content if hasattr(tool_event.result, "content") else tool_event.result
                                            await event_sink.put({
                                                "type": "child_tool_result",
                                                "agent_name": agent_name,
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

    output = result.output if hasattr(result, "output") else result
    if hasattr(output, "model_dump"):
        output = output.model_dump()

    response = {
        "status": "success",
        "output": output,
        "agent_schema": agent_name,
        "input_text": input_text,
    }

    # Add text_response: prefer streamed content, fallback to result.output
    if use_streaming and streamed_content:
        response["text_response"] = streamed_content
    elif hasattr(result, "output") and result.output is not None:
        response["text_response"] = str(result.output)

    return response


# =============================================================================
# REST API Endpoints (wrap tool functions for HTTP access)
# =============================================================================


@router.post("/search")
async def search_endpoint(
    query: str,
    limit: int = 20,
    user_id: str | None = None,
) -> dict[str, Any]:
    """REST endpoint for search tool."""
    return await search(query=query, limit=limit, user_id=user_id)
