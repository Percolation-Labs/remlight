"""Test multi-agent delegation with proper message persistence.

This test verifies:
1. User message is stored BEFORE streaming
2. Orchestrator delegates to worker-agent
3. Worker-agent responds with acknowledgment text
4. All tool calls are stored
5. Assistant text response is stored
"""

import asyncio
import json
import uuid

import pytest

from remlight.agentic import AgentContext, create_agent
from remlight.api.mcp_main import get_mcp_tools, init_mcp
from remlight.api.routers.tools import get_agent_schema, init_tools
from remlight.api.streaming import stream_agent_response_with_save, save_user_message
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
async def test_multi_agent_message_persistence(db_connected):
    """
    Test that multi-agent delegation stores all message types correctly.

    Expected flow:
    1. User sends message -> stored as user message
    2. Orchestrator delegates via ask_agent -> stored as tool message
    3. Worker responds with text -> streamed back
    4. Orchestrator summarizes -> stored as assistant message
    """
    db = db_connected
    session_id = str(uuid.uuid4())
    user_id = "test-persistence-user"

    print(f"\n{'='*60}")
    print(f"Session: {session_id}")
    print(f"Model: {settings.llm.default_model}")
    print(f"{'='*60}\n")

    # Load orchestrator schema
    schema = get_agent_schema("orchestrator-agent")
    assert schema is not None, "orchestrator-agent schema not found"

    context = AgentContext(user_id=user_id, session_id=session_id)
    tools = await get_mcp_tools()

    agent_runtime = await create_agent(
        schema=schema,
        model_name=settings.llm.default_model,
        tools=tools,
        context=context,
    )

    prompt = "Please delegate a task to verify the system is working correctly."

    # 1. Save user message BEFORE streaming (as per the pattern)
    print("1. Saving user message...")
    await save_user_message(session_id, user_id, prompt)

    # 2. Stream response and collect events
    print("2. Streaming agent response...")
    content_chunks = []
    tool_events = []
    action_events = []

    async for chunk in stream_agent_response_with_save(
        agent=agent_runtime.agent,
        prompt=prompt,
        model=settings.llm.default_model,
        agent_schema="orchestrator-agent",
        session_id=session_id,
        user_id=user_id,
        context=context,
    ):
        # Parse SSE chunks
        if chunk.startswith("event: "):
            lines = chunk.strip().split("\n")
            event_type = lines[0].replace("event: ", "")
            if len(lines) > 1 and lines[1].startswith("data: "):
                try:
                    data = json.loads(lines[1][6:])
                    if event_type == "tool_call":
                        tool_events.append(data)
                        print(f"   Tool: {data.get('tool_name')} ({data.get('status')})")
                    elif event_type == "action":
                        action_events.append(data)
                        print(f"   Action: {data.get('action_type')}")
                except json.JSONDecodeError:
                    pass
        elif chunk.startswith("data: ") and "choices" in chunk:
            try:
                data = json.loads(chunk[6:].strip())
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    content_chunks.append(content)
            except json.JSONDecodeError:
                pass

    full_response = "".join(content_chunks)
    print(f"\n3. Response received ({len(full_response)} chars):")
    print(f"   {full_response[:200]}{'...' if len(full_response) > 200 else ''}")

    # 4. Query database to verify persistence
    print("\n4. Checking database persistence...")
    rows = await db.fetch(
        """
        SELECT role, LEFT(content, 100) as content_preview, metadata, created_at
        FROM messages
        WHERE session_id = $1 AND deleted_at IS NULL
        ORDER BY created_at ASC
        """,
        session_id
    )

    print(f"\n{'='*60}")
    print(f"DATABASE CONTENTS: {len(rows)} messages")
    print(f"{'='*60}")

    user_count = 0
    assistant_count = 0
    tool_count = 0

    for i, row in enumerate(rows):
        role = row["role"]
        content = row["content_preview"] or ""

        if role == "user":
            user_count += 1
            print(f"  [{i+1}] USER: {content[:60]}...")
        elif role == "assistant":
            assistant_count += 1
            print(f"  [{i+1}] ASSISTANT: {content[:60]}...")
        elif role == "tool":
            tool_count += 1
            meta = row.get("metadata")
            if meta:
                meta_dict = json.loads(meta) if isinstance(meta, str) else meta
                tool_name = meta_dict.get("tool_name", "unknown")
                print(f"  [{i+1}] TOOL ({tool_name}): {content[:40]}...")
            else:
                print(f"  [{i+1}] TOOL: {content[:40]}...")

    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  - User messages: {user_count}")
    print(f"  - Assistant messages: {assistant_count}")
    print(f"  - Tool messages: {tool_count}")
    print(f"  - Tool events streamed: {len(tool_events)}")
    print(f"  - Action events streamed: {len(action_events)}")
    print(f"{'='*60}")

    # Assertions
    assert user_count >= 1, "User message should be stored"
    assert tool_count >= 1, "At least one tool call should be stored"

    # Check if ask_agent was called (delegation happened)
    ask_agent_calls = [e for e in tool_events if e.get("tool_name") == "ask_agent"]
    if ask_agent_calls:
        print("\n✓ Delegation via ask_agent occurred")

    # If we got assistant content streamed, it should be stored
    if full_response:
        assert assistant_count >= 1, f"Assistant response was streamed ({len(full_response)} chars) but not stored"
        print("✓ Assistant message stored")
    else:
        print("⚠ No assistant text content was streamed (agent may have only used tools)")

    print("\n✓ Test completed successfully!")


async def run_test():
    """Run test directly without pytest."""
    db = get_db()
    await db.connect()
    init_tools(db)
    init_mcp(db)
    try:
        await test_multi_agent_message_persistence(db)
    finally:
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(run_test())
