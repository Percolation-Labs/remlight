"""Core streaming generator for agent responses.

Provides two modes:
- SSE streaming: OpenAI-compatible Server-Sent Events (for API)
- Plain streaming: Raw text chunks (for CLI)
"""

import asyncio
import uuid
from typing import Any, AsyncGenerator

from loguru import logger

from remlight.agentic.streaming.events import (
    DoneEvent,
    ErrorEvent,
    ProgressEvent,
    ToolCallEvent,
)
from remlight.agentic.streaming.formatters import (
    format_content_chunk,
    format_done,
    format_sse_event,
)
from remlight.agentic.streaming.handlers import (
    extract_tool_args,
    process_child_event,
    stream_with_child_events,
)
from remlight.agentic.streaming.state import StreamingState


async def stream_sse(
    agent,
    prompt: str,
    *,
    model: str | None = None,
    request_id: str | None = None,
    message_id: str | None = None,
    session_id: str | None = None,
    agent_schema: str | None = None,
    context=None,
    message_history: list | None = None,
    tool_calls_out: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream agent response as OpenAI-compatible SSE events.

    This is the core streaming generator that handles:
    - Text content streaming
    - Tool call events
    - Child agent event proxying (via event sink)
    - Metadata events (from register_metadata tool)
    - Progress events

    Args:
        agent: Pydantic AI Agent instance
        prompt: User prompt
        model: Model name for metadata
        request_id: Optional request ID
        message_id: Optional pre-generated message ID
        session_id: Session ID for metadata
        agent_schema: Agent schema name
        context: Optional AgentContext for multi-agent support
        message_history: Optional pydantic-ai message history
        tool_calls_out: Optional mutable list to capture tool calls

    Yields:
        SSE-formatted strings
    """
    from remlight.agentic.context import (
        get_current_context,
        set_current_context,
        set_event_sink,
    )

    state = StreamingState.create(
        model=model or "unknown",
        request_id=request_id,
        message_id=message_id,
    )

    # Set up context for multi-agent propagation
    previous_context = None
    if context is not None:
        previous_context = get_current_context()
        set_current_context(context)

    # Set up event sink for child agent event proxying
    child_event_sink: asyncio.Queue = asyncio.Queue()
    set_event_sink(child_event_sink)

    try:
        # Emit initial progress event
        yield format_sse_event(ProgressEvent(
            step=1,
            total_steps=state.total_steps,
            label="Processing request",
        ))

        # Use agent.iter() for complete execution with tool calls
        iter_kwargs = {"message_history": message_history} if message_history else {}

        async with agent.iter(prompt, **iter_kwargs) as agent_run:
            async for node in agent_run:
                from pydantic_ai.agent import Agent

                if Agent.is_model_request_node(node):
                    async for chunk in _handle_model_request_node(
                        node, agent_run, state
                    ):
                        yield chunk

                elif Agent.is_call_tools_node(node):
                    async for chunk in _handle_call_tools_node(
                        node,
                        agent_run,
                        state,
                        child_event_sink,
                        message_id,
                        session_id,
                        agent_schema,
                        model,
                        tool_calls_out,
                    ):
                        yield chunk

        # Final chunk with finish_reason
        yield format_content_chunk("", state, finish_reason="stop")

        # Emit done event
        yield format_sse_event(DoneEvent(reason="stop"))
        yield format_done()

    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield format_sse_event(ErrorEvent(
            code="stream_error",
            message=str(e),
            recoverable=True,
        ))
        yield format_sse_event(DoneEvent(reason="error"))
        yield format_done()

    finally:
        set_event_sink(None)
        if context is not None:
            set_current_context(previous_context)


async def stream_plain(
    agent,
    prompt: str,
    *,
    message_history: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Simple streaming for CLI - yields plain text chunks.

    No SSE formatting, no event tracking. Just raw text content.

    Args:
        agent: Pydantic AI Agent instance
        prompt: User prompt
        message_history: Optional pydantic-ai message history

    Yields:
        Plain text strings
    """
    try:
        iter_kwargs = {"message_history": message_history} if message_history else {}

        async with agent.iter(prompt, **iter_kwargs) as stream:
            async for node in stream:
                from pydantic_ai.agent import Agent

                if Agent.is_model_request_node(node):
                    async with node.stream(stream.ctx) as request_stream:
                        async for event in request_stream:
                            if hasattr(event, "delta") and hasattr(event.delta, "content_delta"):
                                content = event.delta.content_delta
                                if content:
                                    yield content
    except Exception as e:
        logger.error(f"CLI streaming error: {e}")
        yield f"\nError: {e}"


# =============================================================================
# Internal handlers for different node types
# =============================================================================


async def _handle_model_request_node(
    node,
    agent_run,
    state: StreamingState,
) -> AsyncGenerator[str, None]:
    """Handle model request node - text content and tool call starts."""
    async with node.stream(agent_run.ctx) as request_stream:
        async for event in request_stream:
            event_type = type(event).__name__

            # Handle text content - skip if child already streamed
            if event_type == "PartDeltaEvent":
                if state.child_content_streamed:
                    continue
                if hasattr(event, "delta") and hasattr(event.delta, "content_delta"):
                    content = event.delta.content_delta
                    if content:
                        yield format_content_chunk(content, state)

            # Handle text/tool start
            elif event_type == "PartStartEvent":
                if hasattr(event, "part"):
                    part_type = type(event.part).__name__

                    if part_type == "TextPart":
                        if state.child_content_streamed:
                            continue
                        if event.part.content:
                            yield format_content_chunk(event.part.content, state)

                    elif part_type == "ToolCallPart":
                        tool_name = event.part.tool_name
                        if tool_name == "final_result":
                            continue

                        tool_id = f"call_{uuid.uuid4().hex[:8]}"
                        args_dict = extract_tool_args(
                            getattr(event.part, "args", None)
                        )

                        state.register_tool_call(tool_name, tool_id, event.index, args_dict)

                        yield format_sse_event(
                            ToolCallEvent(
                                tool_name=tool_name,
                                tool_id=tool_id,
                                status="started",
                                arguments=args_dict,
                            )
                        )

                        # Update progress
                        state.current_step = 2
                        yield format_sse_event(ProgressEvent(
                            step=state.current_step,
                            total_steps=state.total_steps,
                            label=f"Calling {tool_name}",
                        ))


async def _handle_call_tools_node(
    node,
    agent_run,
    state: StreamingState,
    child_event_sink: asyncio.Queue,
    message_id: str | None,
    session_id: str | None,
    agent_schema: str | None,
    model: str | None,
    tool_calls_out: list | None,
) -> AsyncGenerator[str, None]:
    """Handle call tools node - tool results and child events."""
    async with node.stream(agent_run.ctx) as tools_stream:
        async for event_source, event_data in stream_with_child_events(
            tools_stream=tools_stream,
            child_event_sink=child_event_sink,
        ):
            if event_source == "child":
                async for chunk in process_child_event(event_data, state):
                    yield chunk
                continue

            # Handle tool events
            tool_event = event_data
            event_type = type(tool_event).__name__

            if event_type == "FunctionToolResultEvent":
                result_content = (
                    tool_event.result.content
                    if hasattr(tool_event.result, "content")
                    else tool_event.result
                )

                # Complete the tool call and get data
                tool_data = state.complete_tool_call(result_content)
                if tool_data is None:
                    tool_data = {
                        "tool_name": "tool",
                        "tool_id": f"call_{uuid.uuid4().hex[:8]}",
                        "result": result_content,
                    }

                is_action_event = False

                # Check for action event from action() tool
                if isinstance(result_content, dict) and result_content.get("_action_event"):
                    is_action_event = True
                    tool_data["is_action"] = True
                    action_type = result_content.get("action_type", "observation")
                    payload = result_content.get("payload", {})

                    # Emit ActionEvent for all action types
                    from remlight.agentic.streaming.events import ActionEvent
                    yield format_sse_event(ActionEvent(
                        action_type=action_type,
                        payload=payload,
                    ))

                # Capture tool call for persistence
                if tool_calls_out is not None:
                    tool_calls_out.append(tool_data)

                if not is_action_event:
                    yield format_sse_event(
                        ToolCallEvent(
                            tool_name=tool_data.get("tool_name", "tool"),
                            tool_id=tool_data.get("tool_id", "unknown"),
                            status="completed",
                            result=str(result_content)[:200] if result_content else None,
                        )
                    )

                # Update progress
                state.current_step = 3
                yield format_sse_event(ProgressEvent(
                    step=state.current_step,
                    total_steps=state.total_steps,
                    label="Generating response",
                ))


# =============================================================================
# Persistence wrappers
# =============================================================================


async def stream_sse_with_save(
    agent,
    prompt: str,
    *,
    model: str | None = None,
    request_id: str | None = None,
    agent_schema: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    context=None,
    message_history: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Wrapper around stream_sse that saves messages after streaming.

    Accumulates text content during streaming and saves to the database
    after the stream completes.

    NOTE: Call save_user_message() BEFORE this function to save the user's message.
    This function only saves tool calls and assistant responses.

    Args:
        agent: Pydantic AI agent instance
        prompt: User prompt
        model: Model name
        request_id: Optional request ID
        agent_schema: Agent schema name
        session_id: Session ID for message storage
        user_id: User ID for message storage
        context: Agent context for multi-agent propagation
        message_history: Optional pydantic-ai message history

    Yields:
        SSE-formatted strings
    """
    import json

    # Pre-generate message_id for frontend consistency
    message_id = str(uuid.uuid4())

    # Accumulate content during streaming
    accumulated_content: list[str] = []

    # Capture tool calls for persistence
    tool_calls: list[dict] = []

    async for chunk in stream_sse(
        agent=agent,
        prompt=prompt,
        model=model,
        request_id=request_id,
        message_id=message_id,
        session_id=session_id,
        agent_schema=agent_schema,
        context=context,
        message_history=message_history,
        tool_calls_out=tool_calls,
    ):
        yield chunk

        # Extract text content from OpenAI-format chunks
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            try:
                data_str = chunk[6:].strip()
                if data_str:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            accumulated_content.append(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    # After streaming completes, save messages to database
    if session_id:
        from datetime import datetime, timezone

        from remlight.services.session import SessionMessageStore

        timestamp = datetime.now(timezone.utc).isoformat()
        messages_to_store = []

        # Store tool call messages
        for tool_call in tool_calls:
            if not tool_call:
                continue
            tool_message = {
                "role": "tool",
                "content": json.dumps(tool_call.get("result", {}), default=str),
                "timestamp": timestamp,
                "tool_call_id": tool_call.get("tool_id"),
                "tool_name": tool_call.get("tool_name"),
                "tool_arguments": tool_call.get("arguments"),
            }
            messages_to_store.append(tool_message)

        # Store assistant text response
        full_content = None
        if accumulated_content:
            full_content = "".join(accumulated_content)
        else:
            # Fallback to text_response from tool results
            for tool_call in tool_calls:
                if not tool_call:
                    continue
                result = tool_call.get("result")
                if isinstance(result, dict) and result.get("text_response"):
                    text_response = result["text_response"]
                    if text_response and str(text_response).strip():
                        full_content = str(text_response)
                        break

        if full_content:
            assistant_message = {
                "id": message_id,
                "role": "assistant",
                "content": full_content,
                "timestamp": timestamp,
            }
            messages_to_store.append(assistant_message)

        if messages_to_store:
            try:
                store = SessionMessageStore(user_id=user_id or "anonymous")
                await store.store_session_messages(
                    session_id=session_id,
                    messages=messages_to_store,
                    user_id=user_id,
                    compress=False,
                )
            except Exception:
                pass  # Persistence failures shouldn't break streaming


async def save_user_message(
    session_id: str,
    user_id: str | None,
    content: str,
) -> None:
    """
    Save user message to database before streaming.

    Shared utility for consistent user message storage.

    Args:
        session_id: Session ID
        user_id: User ID (optional)
        content: Message content
    """
    if not session_id:
        return

    from datetime import datetime, timezone

    from remlight.services.session import SessionMessageStore

    user_msg = {
        "role": "user",
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    try:
        store = SessionMessageStore(user_id=user_id or "anonymous")
        await store.store_session_messages(
            session_id=session_id,
            messages=[user_msg],
            user_id=user_id,
            compress=False,
        )
    except Exception:
        pass
