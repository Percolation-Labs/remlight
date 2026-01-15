"""Test multi-agent delegation with streaming and persistence."""

import asyncio
import json
import uuid
import pytest

from remlight.agentic import AgentContext, create_agent, run_streaming
from remlight.api.mcp_main import get_mcp_tools, init_mcp, get_mcp_server
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


@pytest.mark.asyncio
async def test_orchestrator_delegates_to_action_agent(db_connected):
    """Test that orchestrator agent can delegate to action agent."""
    schema = get_agent_schema("orchestrator-agent")
    assert schema is not None, "orchestrator-agent schema not found"

    context = AgentContext(
        user_id="test-user",
        session_id=str(uuid.uuid4()),  # Use proper UUID
    )

    tools = await get_mcp_tools()
    agent_runtime = await create_agent(
        schema=schema,
        model_name=settings.llm.default_model,
        tools=tools,
        context=context,
    )

    # Collect streamed events
    events = []
    content_chunks = []
    current_event_type = None
    all_chunks = []  # Debug: capture all raw chunks

    async for chunk in run_streaming(
        agent=agent_runtime.agent,
        prompt="Use ask_agent to invoke action-agent with message: observe multi-agent test working",
        context=context,
        model=settings.llm.default_model,
        session_id=context.session_id,
        user_id=context.user_id,
        persist_messages=True,
        output_format="sse",
    ):
        all_chunks.append(chunk[:80])  # Capture first 80 chars of each chunk

        # SSE chunks may contain "event: X\ndata: Y" combined - split them
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

                # Track content chunks
                if "choices" in data and data["choices"]:
                    delta = data["choices"][0].get("delta", {})
                    if delta.get("content"):
                        content_chunks.append(delta["content"])

                # Track tool calls
                if data.get("type") == "tool_call":
                    print(f"Tool call: {data.get('tool_name')} - {data.get('status')}")

            except json.JSONDecodeError:
                pass

    # Verify we got SSE events
    assert len(events) > 0, "No SSE events received"

    # Check for ask_agent tool call (delegation)
    tool_events = [e for e in events if e.get("type") == "tool_call" or e.get("_sse_type") == "tool_call"]
    ask_agent_calls = [e for e in tool_events if e.get("tool_name") == "ask_agent"]

    print(f"\nReceived {len(events)} events")
    print(f"Tool calls: {len(tool_events)}")
    print(f"ask_agent calls: {len(ask_agent_calls)}")
    print(f"Content: {''.join(content_chunks)[:200]}")

    # Debug: show all events
    for i, e in enumerate(events[:10]):
        etype = e.get('type') or e.get('_sse_type', 'content')
        print(f"  Event {i}: {etype} - {str(e)[:100]}")

    # Debug: show raw chunks
    print(f"\nRaw chunks ({len(all_chunks)} total):")
    for i, c in enumerate(all_chunks[:15]):
        print(f"  Chunk {i}: {repr(c)}")


@pytest.mark.asyncio
async def test_action_agent_registers_metadata(db_connected):
    """Test that action agent calls action(type='observation')."""
    schema = get_agent_schema("action-agent")
    assert schema is not None, "action-agent schema not found"

    context = AgentContext(user_id="test-user")
    tools = await get_mcp_tools()

    agent_runtime = await create_agent(
        schema=schema,
        model_name=settings.llm.default_model,
        tools=tools,
        context=context,
    )

    # Collect metadata events
    metadata_events = []

    async for chunk in run_streaming(
        agent=agent_runtime.agent,
        prompt="Observe that the test is running successfully",
        context=context,
        model=settings.llm.default_model,
        persist_messages=False,
        output_format="sse",
    ):
        if chunk.startswith("data: "):
            try:
                data = json.loads(chunk[6:].strip())
                if data.get("type") == "metadata":
                    metadata_events.append(data)
            except json.JSONDecodeError:
                pass

    print(f"\nMetadata events: {len(metadata_events)}")
    for m in metadata_events:
        print(f"  {m}")


async def run_direct_test():
    """Run test directly without pytest."""
    db = get_db()
    await db.connect()
    init_tools(db)
    init_mcp(db)
    try:
        await test_action_agent_registers_metadata(db)
        await test_orchestrator_delegates_to_action_agent(db)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(run_direct_test())
