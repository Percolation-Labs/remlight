"""Event handlers for streaming.

Handles:
- Child agent events (from ask_agent tool via event sink)
- Tool call events (from agent execution)
- Action events (from action() tool)
"""

import asyncio
import json
import uuid
from typing import Any, AsyncGenerator

from remlight.agentic.streaming.events import (
    ActionEvent,
    ToolCallEvent,
)
from remlight.agentic.streaming.formatters import format_content_chunk, format_sse_event
from remlight.agentic.streaming.state import StreamingState


async def process_child_event(
    child_event: dict,
    state: StreamingState,
) -> AsyncGenerator[str, None]:
    """Process a child agent event and yield SSE chunks.

    Child events come from ask_agent tool via the event sink.
    Types:
    - child_tool_start: Child is calling a tool
    - child_content: Child is streaming text
    - child_tool_result: Child tool completed

    Args:
        child_event: Event dict from child agent
        state: Streaming state (updated in place)

    Yields:
        SSE-formatted strings
    """
    event_type = child_event.get("type", "")
    child_agent = child_event.get("agent_name", "child")

    if event_type == "child_tool_start":
        tool_name = f"{child_agent}:{child_event.get('tool_name', 'tool')}"
        tool_id = f"call_{uuid.uuid4().hex[:8]}"

        yield format_sse_event(
            ToolCallEvent(
                tool_name=tool_name,
                tool_id=tool_id,
                status="started",
                arguments=child_event.get("arguments"),
            )
        )

    elif event_type == "child_content":
        content = child_event.get("content", "")
        if content:
            state.mark_child_content(child_agent)
            yield format_content_chunk(content, state)

    elif event_type == "child_tool_result":
        result = child_event.get("result")

        # Check for action event from child
        if isinstance(result, dict) and result.get("_action_event"):
            yield format_sse_event(ActionEvent(
                action_type=result.get("action_type", "observation"),
                payload=result.get("payload"),
            ))
        else:
            yield format_sse_event(
                ToolCallEvent(
                    tool_name=f"{child_agent}:tool",
                    tool_id=f"call_{uuid.uuid4().hex[:8]}",
                    status="completed",
                    result=str(result)[:200] if result else None,
                )
            )


def extract_tool_args(raw_args: Any) -> dict:
    """Extract tool arguments from various formats."""
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError:
            return {"raw": raw_args}
    return {}


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
