"""Event handlers for streaming.

Handles:
- Child agent events (from ask_agent tool via event sink)
- Tool call events (from agent execution)
- Action events (from action() tool)

TOOL ARGUMENT EXTRACTION
------------------------
Pydantic-ai uses ArgsDict objects to hold tool arguments. The args may be:
- None (no arguments)
- ArgsDict with .args_dict attribute (most common)
- Plain dict
- JSON string

The extract_tool_args function handles all these formats.

IMPORTANT: At PartStartEvent, args may be None or incomplete because pydantic-ai
streams arguments incrementally. Full args are only available at PartEndEvent.

CHILD EVENT PERSISTENCE
-----------------------
Child agent tool calls are saved to the database during streaming.
This ensures session history captures the full multi-agent conversation:

1. child_tool_start -> Saves tool message with arguments
2. child_content -> Marks state.child_content_streamed = True
3. child_tool_result -> Emits completion event (action or tool)

The content itself is saved by the parent's post-stream persistence.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from loguru import logger

from remlight.agentic.streaming.events import (
    ActionEvent,
    ToolCallEvent,
)
from remlight.agentic.streaming.formatters import format_content_chunk, format_sse_event
from remlight.agentic.streaming.state import StreamingState


async def process_child_event(
    child_event: dict,
    state: StreamingState,
    session_id: str | None = None,
    user_id: str | None = None,
    model: str | None = None,
    agent_schema: str | None = None,
) -> AsyncGenerator[str, None]:
    """Process a child agent event and yield SSE chunks.

    Child events come from ask_agent tool via the event sink.
    Types:
    - child_tool_start: Child is calling a tool (saves to DB)
    - child_content: Child is streaming text
    - child_tool_result: Child tool completed

    Args:
        child_event: Event dict from child agent
        state: Streaming state (updated in place)
        session_id: Session ID for persistence (optional)
        user_id: User ID for persistence (optional)
        model: Model name for metadata (optional)
        agent_schema: Agent schema name for metadata (optional)

    Yields:
        SSE-formatted strings
    """
    event_type = child_event.get("type", "")
    child_agent = child_event.get("agent_name", "child")

    if event_type == "child_tool_start":
        tool_name = f"{child_agent}:{child_event.get('tool_name', 'tool')}"
        # Use pydantic-ai's tool_call_id if available, otherwise generate one
        tool_id = child_event.get("tool_call_id") or f"call_{uuid.uuid4().hex[:8]}"
        arguments = child_event.get("arguments")

        # Normalize arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        elif not isinstance(arguments, dict):
            arguments = None

        # Emit SSE event
        yield format_sse_event(
            ToolCallEvent(
                tool_name=tool_name,
                tool_id=tool_id,
                status="started",
                arguments=arguments,
            )
        )

        # Save child tool call to database for session persistence
        if session_id:
            try:
                from remlight.services.session import SessionMessageStore

                store = SessionMessageStore(user_id=user_id or "anonymous")
                # Use child agent name from event, fall back to agent_schema
                effective_agent = child_agent if child_agent != "child" else agent_schema
                tool_msg = {
                    "role": "tool",
                    # Content is tool args as JSON - parsed on session reload
                    "content": json.dumps(arguments) if arguments else "",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "agent_schema": effective_agent,
                    "model": model,
                }
                await store.store_session_messages(
                    session_id=session_id,
                    messages=[tool_msg],
                    user_id=user_id,
                    compress=False,
                )
            except Exception as e:
                logger.warning(f"Failed to save child tool call: {e}")

    elif event_type == "child_content":
        content = child_event.get("content", "")
        if content:
            state.mark_child_content(child_agent)
            yield format_content_chunk(content, state)

    elif event_type == "child_tool_result":
        result = child_event.get("result")
        tool_name = child_event.get("tool_name", "tool")
        tool_call_id = child_event.get("tool_call_id") or f"call_{uuid.uuid4().hex[:8]}"

        # Check for action event from child
        if isinstance(result, dict) and result.get("_action_event"):
            yield format_sse_event(ActionEvent(
                action_type=result.get("action_type", "observation"),
                payload=result.get("payload"),
            ))
        else:
            yield format_sse_event(
                ToolCallEvent(
                    tool_name=f"{child_agent}:{tool_name}",
                    tool_id=tool_call_id,
                    status="completed",
                    result=result,
                )
            )


def extract_tool_args(part_or_args: Any) -> dict | None:
    """Extract tool arguments from a ToolCallPart or raw args.

    Handles various formats from pydantic-ai:
    - ToolCallPart object with .args attribute
    - ArgsDict object with .args_dict attribute
    - Plain dict
    - JSON string
    """
    # If it's a ToolCallPart, get the .args attribute
    args = getattr(part_or_args, "args", part_or_args)

    if args is None:
        return None

    # ArgsDict object from pydantic-ai (the most common case)
    if hasattr(args, "args_dict"):
        return args.args_dict

    # Plain dict
    if isinstance(args, dict):
        return args

    # JSON string
    if isinstance(args, str):
        if not args.strip():
            return {}
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return {"raw": args}

    return None


async def stream_with_child_events(
    tools_stream,
    child_event_sink: asyncio.Queue,
) -> AsyncGenerator[tuple[str, Any], None]:
    """Multiplex tool events with child events.

    Concurrently reads from:
    - tools_stream: Tool execution events from pydantic-ai
    - child_event_sink: Child agent events from ask_agent

    Yields:
        Tuples of (source, event) where source is "tool" or "child"
    """
    tool_iter = tools_stream.__aiter__()
    pending_tool: asyncio.Task | None = None
    pending_child: asyncio.Task | None = None

    try:
        pending_tool = asyncio.create_task(tool_iter.__anext__())
    except StopAsyncIteration:
        # No tool events, drain child queue
        while not child_event_sink.empty():
            try:
                child_event = child_event_sink.get_nowait()
                yield ("child", child_event)
            except asyncio.QueueEmpty:
                break
        return

    pending_child = asyncio.create_task(
        _get_child_event_with_timeout(child_event_sink, timeout=0.05)
    )

    try:
        while True:
            tasks = {t for t in [pending_tool, pending_child] if t is not None}
            if not tasks:
                break

            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            for task in done:
                try:
                    result = task.result()
                except asyncio.TimeoutError:
                    # Timeout on child queue - restart wait
                    if task is pending_child:
                        pending_child = asyncio.create_task(
                            _get_child_event_with_timeout(child_event_sink, timeout=0.05)
                        )
                    continue
                except StopAsyncIteration:
                    # Tool stream exhausted
                    if task is pending_tool:
                        pending_tool = None
                        # Cancel child wait and drain queue
                        if pending_child:
                            pending_child.cancel()
                            try:
                                await pending_child
                            except asyncio.CancelledError:
                                pass
                        while not child_event_sink.empty():
                            try:
                                child_event = child_event_sink.get_nowait()
                                yield ("child", child_event)
                            except asyncio.QueueEmpty:
                                break
                        return
                    continue

                # Handle successful results
                if task is pending_child and result is not None:
                    yield ("child", result)
                    pending_child = asyncio.create_task(
                        _get_child_event_with_timeout(child_event_sink, timeout=0.05)
                    )
                elif task is pending_tool:
                    yield ("tool", result)
                    try:
                        pending_tool = asyncio.create_task(tool_iter.__anext__())
                    except StopAsyncIteration:
                        pending_tool = None
                elif task is pending_child and result is None:
                    # Timeout - restart wait
                    pending_child = asyncio.create_task(
                        _get_child_event_with_timeout(child_event_sink, timeout=0.05)
                    )
    finally:
        # Clean up any pending tasks
        for task in [pending_tool, pending_child]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


async def _get_child_event_with_timeout(
    queue: asyncio.Queue,
    timeout: float = 0.05,
) -> dict | None:
    """Get an event from the queue with a timeout.

    Returns None on timeout, allowing the caller to check for tool events.
    """
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None
