"""Test agent-builder schema update flow.

The agent-builder should call the action tool with type="patch_schema"
when asked to change the draft schema. This test verifies that behavior.
"""

import asyncio
import json
import uuid
import pytest

from remlight.agentic import AgentContext, create_agent, run_streaming
from remlight.api.mcp_main import get_mcp_tools, init_mcp
from remlight.api.routers.tools import get_agent_schema, init_tools
from remlight.services.database import get_db
from remlight.settings import settings


@pytest.fixture
async def db_connected():
    """Connect database for tests."""
    db = get_db()
    await db.connect()
    init_tools(db)
    init_mcp(db)
    yield db
    await db.disconnect()


def parse_sse_events(chunks: list[str]) -> list[dict]:
    """Parse SSE chunks into event dicts."""
    events = []
    for chunk in chunks:
        lines = chunk.split("\n")
        event_type = None
        data_line = None

        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:].strip()
            elif line.startswith("data: ") and not line.startswith("data: [DONE]"):
                data_line = line[6:].strip()

        if data_line:
            try:
                data = json.loads(data_line)
                if event_type:
                    data["_sse_type"] = event_type
                events.append(data)
            except json.JSONDecodeError:
                pass
    return events


def find_tool_call_events(events: list[dict], tool_name: str) -> list[dict]:
    """Find tool_call events for a specific tool."""
    return [
        e for e in events
        if e.get("type") == "tool_call" and e.get("tool_name") == tool_name
    ]


def find_action_events(events: list[dict], action_type: str) -> list[dict]:
    """Find action events of a specific type."""
    return [
        e for e in events
        if e.get("type") == "action" and e.get("action_type") == action_type
    ]


@pytest.mark.asyncio
async def test_agent_builder_calls_action_tool_for_patch_schema(db_connected):
    """Test that agent-builder calls action tool when asked to update schema."""
    # Clear cached schema to ensure fresh load
    from remlight.api.routers.tools import _agent_schemas
    _agent_schemas.pop("agent-builder", None)

    schema = get_agent_schema("agent-builder")
    assert schema is not None, "agent-builder schema not found"

    context = AgentContext(
        user_id="test-user",
        session_id=str(uuid.uuid4()),
    )

    tools = await get_mcp_tools()
    agent_runtime = await create_agent(
        schema=schema,
        model_name=settings.llm.default_model,
        tools=tools,
        context=context,
        use_cache=False,  # Don't use cache for test
    )

    # Collect streamed chunks
    chunks = []

    async for chunk in run_streaming(
        agent=agent_runtime.agent,
        prompt="set the system prompt to: You are a helpful coding assistant",
        context=context,
        model=settings.llm.default_model,
        session_id=context.session_id,
        user_id=context.user_id,
        persist_messages=False,
        output_format="sse",
    ):
        chunks.append(chunk)

    # Parse events
    events = parse_sse_events(chunks)

    # Check for action tool calls
    action_tool_calls = find_tool_call_events(events, "action")

    # Debug output
    print(f"\n=== EVENTS ({len(events)}) ===")
    for e in events:
        print(json.dumps(e, indent=2, default=str)[:200])

    print(f"\n=== ACTION TOOL CALLS ({len(action_tool_calls)}) ===")
    for tc in action_tool_calls:
        print(json.dumps(tc, indent=2, default=str))

    # Verify the agent called action tool with patch_schema
    assert len(action_tool_calls) >= 1, (
        f"Expected agent to call 'action' tool, but found no action tool calls. "
        f"Total events: {len(events)}. "
        f"Event types: {set(e.get('type') for e in events)}"
    )

    # Check for action events directly (emitted by the streaming core)
    action_events = [e for e in events if e.get("type") == "action"]

    # Also check tool_call results for patch_schema
    patch_schema_from_result = [
        tc for tc in action_tool_calls
        if tc.get("result", {}).get("action_type") == "patch_schema"
    ]

    assert len(action_events) >= 1 or len(patch_schema_from_result) >= 1, (
        f"Expected at least one patch_schema action. "
        f"Action events: {len(action_events)}, patch_schema in results: {len(patch_schema_from_result)}"
    )

    # Get the payload from either source
    if action_events:
        payload = action_events[0].get("payload", {})
    else:
        payload = patch_schema_from_result[0].get("result", {}).get("payload", {})

    # patch_schema uses JSON Patch format with 'patches' array
    patches = payload.get("patches", [])
    assert len(patches) >= 1, (
        f"Expected at least one patch operation, got: {patches}"
    )

    # Check that one of the patches updates the description (system prompt)
    description_patch = next(
        (p for p in patches if p.get("path") == "/description"),
        None
    )
    assert description_patch is not None, (
        f"Expected a patch for /description path, got patches: {patches}"
    )
    assert "coding assistant" in str(description_patch.get("value", "")).lower(), (
        f"Expected value to contain the prompt text, got: {description_patch.get('value')}"
    )


@pytest.mark.asyncio
async def test_agent_builder_patch_schema_returns_action_event(db_connected):
    """Test that patch_schema action returns an ActionEvent for SSE streaming."""
    from remlight.api.routers.tools import action

    # Directly call the action tool with JSON Patch format
    result = await action(
        type="patch_schema",
        payload={
            "patches": [
                {"op": "replace", "path": "/description", "value": "Test prompt"}
            ]
        }
    )

    # Verify the result has _action_event marker for SSE
    assert result.get("_action_event") is True, "action should return _action_event=True"
    assert result.get("action_type") == "patch_schema", "action_type should be patch_schema"
    assert result.get("payload", {}).get("patches") is not None


@pytest.mark.asyncio
async def test_agent_builder_has_correct_tools(db_connected):
    """Test that agent-builder schema has the required tools configured."""
    # Clear cached schema
    from remlight.api.routers.tools import _agent_schemas
    _agent_schemas.pop("agent-builder", None)

    schema = get_agent_schema("agent-builder")
    assert schema is not None, "agent-builder schema not found"

    extra = schema.get("json_schema_extra", {})
    tools = extra.get("tools", [])
    tool_names = [t.get("name") for t in tools]

    assert "action" in tool_names, f"agent-builder should have 'action' tool, got: {tool_names}"
    assert "search" in tool_names, f"agent-builder should have 'search' tool, got: {tool_names}"


@pytest.mark.asyncio
async def test_agent_builder_prompt_describes_action_tool(db_connected):
    """Test that agent-builder system prompt describes how to use action tool."""
    from remlight.api.routers.tools import _agent_schemas
    _agent_schemas.pop("agent-builder", None)

    schema = get_agent_schema("agent-builder")
    assert schema is not None

    description = schema.get("description", "")

    # Check that the prompt mentions key concepts
    assert "action" in description.lower(), "Prompt should mention action tool"
    assert "patch_schema" in description, "Prompt should mention patch_schema type"
    assert "patches" in description.lower(), "Prompt should mention patches parameter"
