"""Streaming module for OpenAI-compatible SSE responses with child agent support.

Architecture:
```
User Request → stream_agent_response → agent.iter() → SSE Events → Client
                     │
                     ├── Parent agent events (text, tool calls)
                     │
                     └── Child agent events (via ask_agent tool)
                              │
                              ▼
                         Event Sink (asyncio.Queue)
                              │
                              ▼
                         _process_child_event() → SSE + accumulation

stream_agent_response_with_save: Wrapper that persists messages after streaming
```

Key Design Decision (DUPLICATION FIX):
When child_content is streamed, state.child_content_streamed is set True.
Parent TextPartDelta events are SKIPPED when this flag is True,
preventing content from being emitted twice.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from pydantic import BaseModel


class ToolCallEvent(BaseModel):
    """SSE event for tool call start/end."""

    type: str = "tool_call"
    tool_name: str
    tool_id: str
    status: str = "started"
    arguments: dict[str, Any] | None = None
    result: Any = None


class MetadataEvent(BaseModel):
    """SSE event for response metadata (from register_metadata tool).

    Fields match the typed register_metadata tool for consistency.
    """

    type: str = "metadata"
    message_id: str | None = None
    in_reply_to: str | None = None
    session_id: str | None = None
    agent_schema: str | None = None
    responding_agent: str | None = None
    session_name: str | None = None
    confidence: float | None = None
    sources: list[str] | None = None
    model_version: str | None = None
    latency_ms: int | None = None
    token_count: int | None = None
    # Risk assessment fields
    risk_level: str | None = None
    risk_score: int | None = None
    risk_reasoning: str | None = None
    recommended_action: str | None = None
    # Generic extension
    extra: dict[str, Any] | None = None


class ProgressEvent(BaseModel):
    """SSE event for progress updates."""

    type: str = "progress"
    step: int = 1
    total_steps: int = 3
    label: str = "Processing"
    status: str = "in_progress"


class DoneEvent(BaseModel):
    """SSE event for stream completion."""

    type: str = "done"
    reason: str = "stop"


class ErrorEvent(BaseModel):
    """SSE event for errors."""

    type: str = "error"
    code: str = "stream_error"
    message: str
    details: dict[str, Any] | None = None
    recoverable: bool = True


@dataclass
class StreamingState:
    """Tracks state during streaming."""

    request_id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    created_at: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    start_time: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    model: str = "unknown"
    current_text: str = ""
    token_count: int = 0
    is_first_chunk: bool = True

    # Tool call tracking
    tool_calls: list[dict] = field(default_factory=list)
    active_tool_calls: dict[int, tuple[str, str]] = field(default_factory=dict)
    pending_tool_completions: list[tuple[str, str]] = field(default_factory=list)
    pending_tool_data: dict[str, dict] = field(default_factory=dict)
    current_tool_id: str | None = None

    # Metadata tracking
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_registered: bool = False

    # Progress tracking
    current_step: int = 1
    total_steps: int = 3

    # Message tracking
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Child agent tracking
    child_content_streamed: bool = False
    responding_agent: str | None = None

    def latency_ms(self) -> int:
        """Calculate latency in milliseconds."""
        return int((datetime.now(timezone.utc).timestamp() - self.start_time) * 1000)

    @classmethod
    def create(cls, model: str = "unknown", request_id: str | None = None) -> "StreamingState":
        """Create a new streaming state."""
        state = cls(model=model)
        if request_id:
            state.request_id = request_id
        return state


def format_sse_event(event: BaseModel) -> str:
    """Format a Pydantic event model as SSE."""
    event_type = event.type if hasattr(event, "type") else "message"
    data = event.model_dump(exclude_none=True)
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def format_content_chunk(
    content: str,
    state: StreamingState,
    finish_reason: str | None = None,
) -> str:
    """Format content chunk in OpenAI SSE format."""
    state.current_text += content
    state.token_count += len(content.split())

    chunk = {
        "id": state.request_id,
        "object": "chat.completion.chunk",
        "created": state.created_at,
        "model": state.model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant" if state.is_first_chunk else None,
                    "content": content,
                } if content or state.is_first_chunk else {},
                "finish_reason": finish_reason,
            }
        ],
    }

    # Clean up None values from delta
    if chunk["choices"][0]["delta"].get("role") is None:
        chunk["choices"][0]["delta"].pop("role", None)
    if not chunk["choices"][0]["delta"].get("content"):
        chunk["choices"][0]["delta"].pop("content", None)

    state.is_first_chunk = False
    return f"data: {json.dumps(chunk)}\n\n"


async def stream_agent_response(
    agent,
    prompt: str,
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
    Stream agent response as OpenAI-compatible SSE events with child agent support.

    Emits SSE events:
    - progress: Step indicators
    - tool_call: When agent calls a tool (started/completed)
    - metadata: From register_metadata tool calls
    - content: Text content chunks (OpenAI format)
    - done: Stream completion
    - error: On failure

    Args:
        agent: Pydantic AI Agent instance
        prompt: User prompt
        model: Model name for metadata
        request_id: Optional request ID
        message_id: Optional pre-generated message ID
        session_id: Session ID for metadata
        agent_schema: Agent schema name
        context: Optional AgentContext for multi-agent support
        message_history: Optional pydantic-ai message history for multi-turn
        tool_calls_out: Optional mutable list to capture tool calls for persistence

    Yields:
        SSE-formatted strings
    """
    from remlight.agentic.context import set_current_context, set_event_sink, get_current_context

    state = StreamingState.create(model=model or "unknown", request_id=request_id)
    if message_id:
        state.message_id = message_id

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

                            # Handle text start
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
                                        state.current_tool_id = tool_id
                                        state.active_tool_calls[event.index] = (tool_name, tool_id)
                                        state.pending_tool_completions.append((tool_name, tool_id))

                                        # Extract arguments
                                        args_dict = {}
                                        if hasattr(event.part, "args"):
                                            raw_args = event.part.args
                                            if isinstance(raw_args, str):
                                                try:
                                                    args_dict = json.loads(raw_args)
                                                except json.JSONDecodeError:
                                                    args_dict = {"raw": raw_args}
                                            elif isinstance(raw_args, dict):
                                                args_dict = raw_args

                                        yield format_sse_event(
                                            ToolCallEvent(
                                                tool_name=tool_name,
                                                tool_id=tool_id,
                                                status="started",
                                                arguments=args_dict,
                                            )
                                        )

                                        # Track for persistence
                                        state.pending_tool_data[tool_id] = {
                                            "tool_name": tool_name,
                                            "tool_id": tool_id,
                                            "arguments": args_dict,
                                        }

                                        # Update progress
                                        state.current_step = 2
                                        yield format_sse_event(ProgressEvent(
                                            step=state.current_step,
                                            total_steps=state.total_steps,
                                            label=f"Calling {tool_name}",
                                        ))

                elif Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as tools_stream:
                        async for event_source, event_data in _stream_with_child_events(
                            tools_stream=tools_stream,
                            child_event_sink=child_event_sink,
                        ):
                            if event_source == "child":
                                async for chunk in _process_child_event(event_data, state):
                                    yield chunk
                                continue

                            # Handle tool events
                            tool_event = event_data
                            event_type = type(tool_event).__name__

                            if event_type == "FunctionToolResultEvent":
                                # Get tool name/id from pending queue
                                if state.pending_tool_completions:
                                    tool_name, tool_id = state.pending_tool_completions.pop(0)
                                else:
                                    tool_name = "tool"
                                    tool_id = f"call_{uuid.uuid4().hex[:8]}"

                                result_content = tool_event.result.content if hasattr(tool_event.result, "content") else tool_event.result
                                is_metadata_event = False

                                # Check for metadata event from register_metadata tool
                                if isinstance(result_content, dict) and result_content.get("_metadata_event"):
                                    is_metadata_event = True
                                    state.metadata_registered = True
                                    state.metadata.update(result_content)

                                    if not state.responding_agent and result_content.get("agent_schema"):
                                        state.responding_agent = result_content["agent_schema"]

                                    # Emit metadata event with all typed fields
                                    yield format_sse_event(MetadataEvent(
                                        message_id=message_id,
                                        session_id=session_id,
                                        agent_schema=agent_schema,
                                        responding_agent=state.responding_agent,
                                        session_name=result_content.get("session_name"),
                                        confidence=result_content.get("confidence"),
                                        sources=result_content.get("sources"),
                                        model_version=model,
                                        risk_level=result_content.get("risk_level"),
                                        risk_score=result_content.get("risk_score"),
                                        risk_reasoning=result_content.get("risk_reasoning"),
                                        recommended_action=result_content.get("recommended_action"),
                                        extra=result_content.get("extra"),
                                    ))

                                # Capture tool call for persistence
                                if tool_calls_out is not None and tool_id in state.pending_tool_data:
                                    tool_data = state.pending_tool_data[tool_id]
                                    tool_data["result"] = result_content
                                    tool_data["is_metadata"] = is_metadata_event
                                    tool_calls_out.append(tool_data)
                                    del state.pending_tool_data[tool_id]

                                if not is_metadata_event:
                                    yield format_sse_event(
                                        ToolCallEvent(
                                            tool_name=tool_name,
                                            tool_id=tool_id,
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

        # Final chunk with finish_reason
        yield format_content_chunk("", state, finish_reason="stop")

        # Emit metadata if not already registered via tool
        if not state.metadata_registered:
            yield format_sse_event(MetadataEvent(
                message_id=message_id,
                session_id=session_id,
                agent_schema=agent_schema,
                responding_agent=state.responding_agent,
                confidence=1.0,
                model_version=model,
                latency_ms=state.latency_ms(),
                token_count=state.token_count,
            ))

        # Emit done event
        yield format_sse_event(DoneEvent(reason="stop"))
        yield "data: [DONE]\n\n"

    except Exception as e:
        yield format_sse_event(ErrorEvent(
            code="stream_error",
            message=str(e),
            recoverable=True,
        ))
        yield format_sse_event(DoneEvent(reason="error"))
        yield "data: [DONE]\n\n"

    finally:
        set_event_sink(None)
        if context is not None:
            set_current_context(previous_context)


async def stream_agent_response_with_save(
    agent,
    prompt: str,
    model: str | None = None,
    request_id: str | None = None,
    agent_schema: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    context=None,
    message_history: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Wrapper around stream_agent_response that saves messages after streaming.

    This accumulates all text content during streaming and saves it to the database
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
    # Pre-generate message_id for frontend consistency
    message_id = str(uuid.uuid4())

    # Accumulate content during streaming
    accumulated_content: list[str] = []

    # Capture tool calls for persistence
    tool_calls: list[dict] = []

    async for chunk in stream_agent_response(
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
        from remlight.services.session import SessionMessageStore
        from datetime import datetime, timezone

        timestamp = datetime.now(timezone.utc).isoformat()
        messages_to_store = []

        # Store tool call messages (message_type: "tool")
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

        # Store assistant text response (if any)
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
    """
    if not session_id:
        return

    from remlight.services.session import SessionMessageStore
    from datetime import datetime, timezone

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


# ============================================================================
# Helper functions for child agent event handling
# ============================================================================


async def _stream_with_child_events(
    tools_stream,
    child_event_sink: asyncio.Queue,
) -> AsyncGenerator[tuple[str, Any], None]:
    """
    Multiplex tool events with child events.

    Yields tuples of (event_type, event_data) where event_type is
    "tool" or "child".
    """
    tool_iter = tools_stream.__aiter__()
    pending_tool: asyncio.Task | None = None
    pending_child: asyncio.Task | None = None

    try:
        pending_tool = asyncio.create_task(tool_iter.__anext__())
    except StopAsyncIteration:
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
                    if task is pending_child:
                        pending_child = asyncio.create_task(
                            _get_child_event_with_timeout(child_event_sink, timeout=0.05)
                        )
                    continue
                except StopAsyncIteration:
                    if task is pending_tool:
                        pending_tool = None
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
                    pending_child = asyncio.create_task(
                        _get_child_event_with_timeout(child_event_sink, timeout=0.05)
                    )
    finally:
        for task in [pending_tool, pending_child]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


async def _get_child_event_with_timeout(
    queue: asyncio.Queue, timeout: float = 0.05
) -> dict | None:
    """Get an event from the queue with a timeout."""
    try:
        return await asyncio.wait_for(queue.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


async def _process_child_event(
    child_event: dict,
    state: StreamingState,
) -> AsyncGenerator[str, None]:
    """Process a child agent event and yield SSE chunks."""
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
            state.child_content_streamed = True
            state.responding_agent = child_agent
            yield format_content_chunk(content, state)

    elif event_type == "child_tool_result":
        result = child_event.get("result")

        # Check for metadata from child
        if isinstance(result, dict) and result.get("_metadata_event"):
            state.metadata.update(result)
            state.metadata_registered = True
            if result.get("agent_schema"):
                state.responding_agent = result["agent_schema"]

            yield format_sse_event(MetadataEvent(
                message_id=state.message_id,
                agent_schema=result.get("agent_schema"),
                responding_agent=state.responding_agent,
                session_name=result.get("session_name"),
                confidence=result.get("confidence"),
                sources=result.get("sources"),
                risk_level=result.get("risk_level"),
                risk_score=result.get("risk_score"),
                extra=result.get("extra"),
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


async def stream_simple(
    agent,
    prompt: str,
) -> AsyncGenerator[str, None]:
    """Simple streaming for CLI - yields plain text chunks."""
    try:
        async with agent.iter(prompt) as stream:
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
        yield f"\nError: {e}"
