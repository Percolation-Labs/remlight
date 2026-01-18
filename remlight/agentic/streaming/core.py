"""
Core Streaming Generator for Agent Responses
=============================================

This module implements REAL-TIME STREAMING of agent responses. It's what makes
the agent feel responsive - users see text appearing as it's generated, not
waiting for the entire response to complete. This is mostly an adapter between
the framework (PydanticAI) and the SSE events we want to send.

PYDANTIC-AI EVENT TYPES
-----------------------

We handle these pydantic-ai streaming events:

| Event Type      | Part Type      | When Emitted                | What We Do                          |
|-----------------|----------------|-----------------------------|------------------------------------|
| PartStartEvent  | TextPart       | LLM starts text output      | Emit content chunk                 |
| PartStartEvent  | ToolCallPart   | LLM wants to call a tool    | Emit tool_call(started), no args   |
| PartDeltaEvent  | TextPartDelta  | Text content streaming      | Emit content chunk                 |
| PartEndEvent    | ToolCallPart   | Tool call args complete     | Emit tool_call(executing) + args   |
| FunctionToolResultEvent | -      | Tool execution completed    | Emit tool_call(completed) + result |

WHY PartEndEvent MATTERS FOR TOOL ARGS
--------------------------------------
Tool call arguments in pydantic-ai are streamed incrementally. At PartStartEvent,
args may be None or incomplete. The full arguments are only available at PartEndEvent.

Tool call lifecycle:
1. PartStartEvent(ToolCallPart) → args=None/partial → emit tool_call(started)
2. PartEndEvent(ToolCallPart)   → args=complete    → emit tool_call(executing)
3. FunctionToolResultEvent      → result available → emit tool_call(completed)

REFERENCE IMPLEMENTATION
------------------------
This follows the pattern from remstack/rem/src/rem/api/routers/chat/streaming.py

STREAMING ARCHITECTURE
----------------------

    User Request
         │
         ▼
    ┌────────────────────────────────────────────────────────────────┐
    │                    stream_sse() Generator                       │
    │                                                                │
    │  1. Set up context (set_current_context)                       │
    │  2. Set up event sink for child agents                         │
    │  3. agent.iter() loop:                                         │
    │       │                                                        │
    │       ├── ModelRequestNode (LLM generating)                    │
    │       │   ├── TextPart → yield content chunk                   │
    │       │   └── ToolCallPart → yield tool_call event             │
    │       │                                                        │
    │       └── CallToolsNode (tool executing)                       │
    │           ├── Tool result → yield tool completion              │
    │           └── Child events (from ask_agent) → yield as-is     │
    │                                                                │
    │  4. yield done event                                           │
    │  5. Clean up context                                           │
    └────────────────────────────────────────────────────────────────┘
         │
         ▼ (SSE format)
    data: {"choices":[{"delta":{"content":"Hello"}}]}
    data: {"choices":[{"delta":{"content":" world"}}]}
    ...
    data: [DONE]


TWO STREAMING MODES
-------------------

1. **stream_sse()** - OpenAI-compatible SSE (for API)
   - Formats as Server-Sent Events
   - Includes tool call events
   - Supports child agent event proxying
   - Used by: /api/v1/chat/completions endpoint

2. **stream_plain()** - Raw text (for CLI)
   - Just yields text strings
   - Logs tool calls via loguru
   - Simpler, no SSE formatting
   - Used by: `rem ask` CLI command


CHILD AGENT EVENT PROXYING
--------------------------

When parent agent calls ask_agent(), the child's output should stream
to the user, not buffer. This is achieved via the event sink pattern:

    Parent streaming loop
         │
         ├── Creates asyncio.Queue
         │
         ├── set_event_sink(queue)
         │
         ├── Tool calls ask_agent()
         │       │
         │       └── Child agent streams
         │               │
         │               └── push_event() ──► queue
         │
         ├── Merges: agent events + queue events
         │
         └── Yields to client


CHILD CONTENT DEDUPLICATION
---------------------------

When a child streams content, the parent should NOT also stream its own
content (which would just be repeating the child). The `child_content_streamed`
flag in StreamingState tracks this:

    if state.child_content_streamed:
        # Skip parent's TextPart - child already streamed this
        continue


SSE EVENT TYPES
---------------

- Content chunks: Regular text output (OpenAI format)
- ToolCallEvent: Tool invocation start/completion
- ActionEvent: Agent action emissions (observation, elicit, etc.)
- ProgressEvent: Progress indicators
- DoneEvent: Stream completion
- ErrorEvent: Error conditions
"""

import asyncio
import os
import uuid
from typing import Any, AsyncGenerator

from loguru import logger

# Max characters to show in debug logging for tool results
# Configurable via environment for debugging long tool outputs
MAX_TOOL_RESULT_CHARS = int(os.environ.get("MAX_TOOL_RESULT_CHARS", "200"))

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
    user_id: str | None = None,
    agent_schema: str | None = None,
    context=None,
    message_history: list | None = None,
    tool_calls_out: list | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream agent response as OpenAI-compatible Server-Sent Events.

    This is the PRIMARY streaming function for API responses. It produces
    SSE-formatted output compatible with OpenAI's chat completions API.

    THE STREAMING LOOP
    -----------------
    The function uses pydantic-ai's agent.iter() which yields nodes:

    1. ModelRequestNode: LLM is generating response
       - TextPart: Content chunks → yield as SSE content
       - ToolCallPart: Tool invocation → yield tool_call event

    2. CallToolsNode: Tools are being executed
       - FunctionToolResultEvent: Tool completed → yield tool result
       - Child events (from ask_agent) → yield via event sink

    CONTEXT MANAGEMENT
    -----------------
    Sets up two ContextVars before execution:

    1. _current_agent_context: Enables tools to access parent context
       (used by ask_agent to inherit user_id, session_id, etc.)

    2. _parent_event_sink: Enables child agents to push events
       (used by ask_agent to stream child output through parent)

    Both are cleaned up in the finally block.

    SSE OUTPUT FORMAT
    ----------------
    OpenAI-compatible format:
        data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
        data: {"id":"...","choices":[{"delta":{"content":" world"}}]}
        data: {"event":"tool_call","data":{...}}
        data: [DONE]

    Args:
        agent: Pydantic AI Agent instance (from create_agent)
        prompt: User's message to the agent
        model: Model name for response metadata
        request_id: Correlation ID for tracing
        message_id: Pre-generated message ID for frontend consistency
        session_id: Session ID for metadata/logging
        agent_schema: Agent schema name for metadata
        context: AgentContext for multi-agent propagation
        message_history: Previous conversation (pydantic-ai format)
        tool_calls_out: Mutable list to capture tool calls for persistence

    Yields:
        SSE-formatted strings (each ending with \\n\\n)
    """
    from remlight.agentic.context import (
        get_current_context,
        set_current_context,
        set_event_sink,
    )

    # Initialize streaming state (tracks IDs, tool calls, child content flag)
    state = StreamingState.create(
        model=model or "unknown",
        request_id=request_id,
        message_id=message_id,
    )

    # =========================================================================
    # CONTEXT SETUP
    # =========================================================================
    # Set context for multi-agent propagation. This enables:
    # - ask_agent tool to inherit parent's user_id, session_id
    # - Child events to stream through parent's response
    # =========================================================================
    previous_context = None
    if context is not None:
        previous_context = get_current_context()
        set_current_context(context)

    # Set up event sink for child agent event proxying
    # ask_agent will push child events to this queue
    child_event_sink: asyncio.Queue = asyncio.Queue()
    set_event_sink(child_event_sink)

    try:
        # Emit initial progress event (UI can show "Processing...")
        yield format_sse_event(ProgressEvent(
            step=1,
            total_steps=state.total_steps,
            label="Processing request",
        ))

        # =====================================================================
        # MAIN STREAMING LOOP
        # =====================================================================
        # agent.iter() provides async iteration over agent execution.
        # Each iteration yields a "node" representing a phase of execution.
        # =====================================================================
        iter_kwargs = {"message_history": message_history} if message_history else {}

        async with agent.iter(prompt, **iter_kwargs) as agent_run:
            async for node in agent_run:
                from pydantic_ai.agent import Agent

                # ModelRequestNode: LLM is generating (text or tool calls)
                if Agent.is_model_request_node(node):
                    async for chunk in _handle_model_request_node(
                        node, agent_run, state, session_id, user_id, model, agent_schema
                    ):
                        yield chunk

                # CallToolsNode: Tools are being executed
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
                        user_id,
                    ):
                        yield chunk

        # Final chunk with finish_reason (signals completion)
        yield format_content_chunk("", state, finish_reason="stop")

        # Emit done event (custom REMLight event) and SSE terminator
        yield format_sse_event(DoneEvent(reason="stop"))
        yield format_done()

    except Exception as e:
        # Error handling: emit error event, don't crash
        logger.error(f"Streaming error: {e}")
        yield format_sse_event(ErrorEvent(
            code="stream_error",
            message=str(e),
            recoverable=True,
        ))
        yield format_sse_event(DoneEvent(reason="error"))
        yield format_done()

    finally:
        # =====================================================================
        # CLEANUP
        # =====================================================================
        # Always clean up context, even if exception occurred.
        # This prevents context leaking between requests.
        # =====================================================================
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

    This is the CLI-optimized streaming mode. Unlike stream_sse():
    - No SSE formatting (just raw text)
    - No event sink setup (CLI is single-agent)
    - Tool calls logged via loguru (not yielded as events)

    CLI OUTPUT EXAMPLE
    -----------------
    $ rem ask "What is machine learning?"

    Machine learning is a subset of artificial intelligence...
    [DEBUG] Tool call: search({'query': 'machine learning'})
    [DEBUG] Tool result: {'results': [...]}

    The user sees clean text output. Debug logging (via LOGURU_LEVEL=DEBUG)
    shows tool activity for troubleshooting.

    Args:
        agent: Pydantic AI Agent instance
        prompt: User's question/command
        message_history: Optional conversation history (pydantic-ai format)

    Yields:
        Plain text strings (no SSE formatting)
    """
    try:
        iter_kwargs = {"message_history": message_history} if message_history else {}

        async with agent.iter(prompt, **iter_kwargs) as stream:
            async for node in stream:
                from pydantic_ai.agent import Agent

                # ModelRequestNode: LLM is generating
                if Agent.is_model_request_node(node):
                    async with node.stream(stream.ctx) as request_stream:
                        async for event in request_stream:
                            event_type = type(event).__name__

                            # Log tool call starts (debug visibility)
                            if event_type == "PartStartEvent" and hasattr(event, "part"):
                                part_type = type(event.part).__name__
                                if part_type == "ToolCallPart":
                                    tool_name = event.part.tool_name
                                    args = getattr(event.part, "args", None)
                                    logger.debug(f"Tool call: {tool_name}({args})")

                            # Stream text content to user
                            if hasattr(event, "delta") and hasattr(event.delta, "content_delta"):
                                content = event.delta.content_delta
                                if content:
                                    yield content

                # CallToolsNode: Tools are being executed
                elif Agent.is_call_tools_node(node):
                    # Log tool results (debug visibility, not yielded)
                    async with node.stream(stream.ctx) as tools_stream:
                        async for tool_event in tools_stream:
                            event_type = type(tool_event).__name__
                            if event_type == "FunctionToolResultEvent":
                                result = (
                                    tool_event.result.content
                                    if hasattr(tool_event.result, "content")
                                    else tool_event.result
                                )
                                # Truncate long results for readable logging
                                result_str = str(result)[:MAX_TOOL_RESULT_CHARS]
                                if len(str(result)) > MAX_TOOL_RESULT_CHARS:
                                    result_str += "..."
                                logger.debug(f"Tool result: {result_str}")

    except Exception as e:
        logger.error(f"CLI streaming error: {e}")
        yield f"\nError: {e}"


# =============================================================================
# INTERNAL HANDLERS FOR PYDANTIC-AI NODE TYPES
# =============================================================================
#
# pydantic-ai's agent.iter() yields "nodes" representing execution phases.
# These handlers process each node type and yield appropriate SSE output.
#
# Node types:
# - ModelRequestNode: LLM is generating (text or tool calls)
# - CallToolsNode: Tools are being executed
# =============================================================================


async def _handle_model_request_node(
    node,
    agent_run,
    state: StreamingState,
    session_id: str | None = None,
    user_id: str | None = None,
    model: str | None = None,
    agent_schema: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Handle ModelRequestNode - LLM is generating response.

    This node type fires when the LLM is producing output. The output can be:
    - TextPart/PartDeltaEvent: Text content chunks
    - ToolCallPart: The LLM wants to call a tool

    CHILD CONTENT DEDUPLICATION
    ---------------------------
    When ask_agent runs a child, the child's content is already streamed
    via the event sink. We set state.child_content_streamed = True.

    When the parent continues, it may try to emit its own TextPart with
    the same content (because the child result is in its context). We
    skip parent text when child_content_streamed is True to prevent
    duplicate output.

    TOOL CALL TRACKING AND PERSISTENCE
    ----------------------------------
    When ToolCallPart is detected:
    1. Generate unique tool_id
    2. Register in state (for matching with result later)
    3. Save to DB immediately (so child tool calls come AFTER parent)
    4. Emit ToolCallEvent with status="started"
    5. Update progress

    Args:
        node: ModelRequestNode from agent.iter()
        agent_run: The agent run context
        state: StreamingState for tracking
        session_id: Session ID for persistence
        user_id: User ID for persistence

    Yields:
        SSE-formatted strings for content and tool calls
    """
    async with node.stream(agent_run.ctx) as request_stream:
        async for event in request_stream:
            event_type = type(event).__name__

            # PartDeltaEvent: Incremental text content
            if event_type == "PartDeltaEvent":
                # Skip if child already streamed content (prevents duplication)
                if state.child_content_streamed:
                    continue
                if hasattr(event, "delta") and hasattr(event.delta, "content_delta"):
                    content = event.delta.content_delta
                    if content:
                        yield format_content_chunk(content, state)

            # PartStartEvent: Beginning of a new part (text or tool call)
            elif event_type == "PartStartEvent":
                if hasattr(event, "part"):
                    part_type = type(event.part).__name__

                    # TextPart: Start of text content
                    if part_type == "TextPart":
                        if state.child_content_streamed:
                            continue
                        if event.part.content:
                            yield format_content_chunk(event.part.content, state)

                    # ThinkingPart: Some models emit thinking content (treat as text)
                    elif part_type == "ThinkingPart":
                        if state.child_content_streamed:
                            continue
                        if event.part.content:
                            yield format_content_chunk(event.part.content, state)

                    # ToolCallPart: LLM wants to call a tool
                    elif part_type == "ToolCallPart":
                        tool_name = event.part.tool_name

                        # Skip pydantic-ai's internal final_result tool
                        if tool_name == "final_result":
                            continue

                        # Generate unique ID and extract arguments
                        # Pass the part directly - extract_tool_args handles ArgsDict
                        tool_id = f"call_{uuid.uuid4().hex[:8]}"
                        args_dict = extract_tool_args(event.part)

                        # Register for later matching with result
                        state.register_tool_call(tool_name, tool_id, event.index, args_dict)

                        # NOTE: DB save moved to PartEndEvent where args are complete

                        # Emit tool call start event (UI can show "Calling search...")
                        yield format_sse_event(
                            ToolCallEvent(
                                tool_name=tool_name,
                                tool_id=tool_id,
                                status="started",
                                arguments=args_dict,
                            )
                        )

                        # Update progress indicator
                        state.current_step = 2
                        yield format_sse_event(ProgressEvent(
                            step=state.current_step,
                            total_steps=state.total_steps,
                            label=f"Calling {tool_name}",
                        ))

            # PartEndEvent: Tool call arguments are now complete
            # Args may be streamed incrementally and only fully available at PartEndEvent
            elif event_type == "PartEndEvent":
                if hasattr(event, "part"):
                    part_type = type(event.part).__name__
                    if part_type == "ToolCallPart":
                        tool_name = event.part.tool_name
                        if tool_name == "final_result":
                            continue

                        # Re-extract args now that streaming is complete
                        args_dict = extract_tool_args(event.part)

                        # Update the registered tool call with complete args
                        if hasattr(event, "index") and event.index in state.active_tool_indices:
                            tool_id = state.active_tool_indices[event.index]
                            state.update_tool_args(tool_id, args_dict)

                            # Save parent tool call to DB with complete args
                            # This ensures parent tool calls appear BEFORE child tool calls
                            if session_id:
                                try:
                                    from datetime import datetime, timezone
                                    import json
                                    from remlight.services.session import SessionMessageStore

                                    store = SessionMessageStore(user_id=user_id or "anonymous")
                                    tool_msg = {
                                        "role": "tool",
                                        "content": json.dumps(args_dict) if args_dict else "{}",
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "tool_call_id": tool_id,
                                        "tool_name": tool_name,
                                        "agent_schema": agent_schema,
                                        "model": model,
                                    }
                                    await store.store_session_messages(
                                        session_id=session_id,
                                        messages=[tool_msg],
                                        user_id=user_id,
                                        compress=False,
                                    )
                                except Exception as e:
                                    logger.warning(f"Failed to save parent tool call: {e}")

                            # Emit updated tool_call event with complete args
                            # This lets the frontend update the tool call display
                            yield format_sse_event(
                                ToolCallEvent(
                                    tool_name=tool_name,
                                    tool_id=tool_id,
                                    status="executing",
                                    arguments=args_dict,
                                )
                            )

                            # Remove from active indices
                            del state.active_tool_indices[event.index]
                            if event.index in state.active_tool_calls:
                                del state.active_tool_calls[event.index]


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
    user_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Handle CallToolsNode - Tools are being executed.

    This node type fires when pydantic-ai is executing tool calls. We:
    1. Merge tool events with child events (from ask_agent via event sink)
    2. Process tool results (emit completion events)
    3. Handle action tool specially (emit ActionEvent)
    4. Capture tool data for persistence

    MERGING TOOL AND CHILD EVENTS
    ----------------------------
    The stream_with_child_events() helper merges two async sources:
    - tools_stream: Events from tool execution
    - child_event_sink: Events pushed by child agents (via push_event)

    This enables real-time streaming of child agent output.

    ACTION TOOL PATTERN
    ------------------
    The action() tool returns results with _action_event=True marker.
    When detected, we emit ActionEvent instead of ToolCallEvent.
    This enables structured metadata (confidence, sources, etc.)
    to be streamed separately from content.

    TOOL DATA CAPTURE
    ----------------
    tool_calls_out is a mutable list provided by the caller.
    We append tool data (name, id, args, result) for later persistence.
    This enables storing tool calls in the session history.

    Args:
        node: CallToolsNode from agent.iter()
        agent_run: The agent run context
        state: StreamingState for tracking
        child_event_sink: Queue for child agent events
        message_id: Message ID for metadata
        session_id: Session ID for metadata/persistence
        agent_schema: Agent schema name for metadata
        model: Model name for metadata
        tool_calls_out: Mutable list to capture tool calls
        user_id: User ID for session persistence

    Yields:
        SSE-formatted strings for tool completions and child events
    """
    async with node.stream(agent_run.ctx) as tools_stream:
        # Merge tool stream with child event sink
        async for event_source, event_data in stream_with_child_events(
            tools_stream=tools_stream,
            child_event_sink=child_event_sink,
        ):
            # Child event from ask_agent
            if event_source == "child":
                async for chunk in process_child_event(
                    event_data, state, session_id=session_id, user_id=user_id,
                    model=model, agent_schema=agent_schema
                ):
                    yield chunk
                continue

            # Tool event from pydantic-ai
            tool_event = event_data
            event_type = type(tool_event).__name__

            # FunctionToolResultEvent: Tool has completed
            if event_type == "FunctionToolResultEvent":
                # Extract result content
                result_content = (
                    tool_event.result.content
                    if hasattr(tool_event.result, "content")
                    else tool_event.result
                )

                # Match with registered tool call to get full data
                tool_data = state.complete_tool_call(result_content)
                if tool_data is None:
                    # Fallback if no registered call (shouldn't happen)
                    tool_data = {
                        "tool_name": "tool",
                        "tool_id": f"call_{uuid.uuid4().hex[:8]}",
                        "result": result_content,
                    }

                is_action_event = False

                # =============================================================
                # ACTION TOOL SPECIAL HANDLING
                # =============================================================
                # The action() tool returns {_action_event: True, ...}
                # We emit ActionEvent instead of ToolCallEvent for these.
                # This enables structured metadata streaming.
                # =============================================================
                if isinstance(result_content, dict) and result_content.get("_action_event"):
                    is_action_event = True
                    tool_data["is_action"] = True
                    action_type = result_content.get("action_type", "observation")
                    payload = result_content.get("payload", {})

                    # Emit ActionEvent (distinct from ToolCallEvent)
                    from remlight.agentic.streaming.events import ActionEvent
                    yield format_sse_event(ActionEvent(
                        action_type=action_type,
                        payload=payload,
                    ))

                    # Handle session_name updates from observation payloads
                    if action_type == "observation" and payload.get("session_name"):
                        session_name = payload["session_name"]
                        # Update session in state for metadata event
                        state.metadata["session_name"] = session_name
                        # Update session name in database (async, best-effort)
                        if session_id:
                            try:
                                from remlight.services.session import SessionMessageStore
                                store = SessionMessageStore(user_id=user_id or "anonymous")
                                await store.update_session_name(
                                    session_id=session_id,
                                    name=session_name,
                                )
                            except Exception as e:
                                logger.warning(f"Failed to update session name: {e}")

                # Capture tool call data for persistence
                if tool_calls_out is not None:
                    tool_calls_out.append(tool_data)

                # Emit tool completion event (unless it was an action)
                if not is_action_event:
                    yield format_sse_event(
                        ToolCallEvent(
                            tool_name=tool_data.get("tool_name", "tool"),
                            tool_id=tool_data.get("tool_id", "unknown"),
                            status="completed",
                            arguments=tool_data.get("arguments"),
                            result=result_content,
                        )
                    )

                # Update progress (tool done, generating response)
                state.current_step = 3
                yield format_sse_event(ProgressEvent(
                    step=state.current_step,
                    total_steps=state.total_steps,
                    label="Generating response",
                ))


# =============================================================================
# PERSISTENCE WRAPPERS
# =============================================================================
#
# These functions wrap the core streaming functions with message persistence.
# They enable multi-turn conversations by saving messages to the database.
#
# MESSAGE STORAGE ORDER:
# 1. save_user_message() - BEFORE agent execution
# 2. stream_sse_with_save() - streams response, saves AFTER completion
#
# This ensures the user's message is always stored, even if the agent fails.
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
    Wrapper around stream_sse that persists messages after streaming.

    This is the PRODUCTION streaming function used by API endpoints.
    It wraps stream_sse() to add automatic message persistence.

    PERSISTENCE FLOW
    ---------------
    1. User message saved BEFORE (via save_user_message)
    2. Stream executes, content and tool calls captured
    3. After stream completes, tool messages and assistant response saved

    WHY SAVE AFTER STREAMING?
    ------------------------
    We can't know the full assistant response until streaming completes.
    We accumulate content chunks during streaming, then persist the
    complete message at the end.

    For tool calls, we capture data during streaming via tool_calls_out,
    then persist all tool messages together.

    FAILURE HANDLING
    ---------------
    Persistence is best-effort. If database operations fail:
    - Errors are silently caught
    - Streaming continues uninterrupted
    - User still gets their response

    This ensures streaming reliability even with database issues.

    MESSAGE ID CONSISTENCY
    ---------------------
    We pre-generate message_id before streaming. This ID is:
    - Included in SSE events (for frontend tracking)
    - Used as the database record ID
    - Enables optimistic UI updates

    Args:
        agent: Pydantic AI agent instance
        prompt: User's message
        model: Model name for metadata
        request_id: Correlation ID for tracing
        agent_schema: Agent schema name for metadata
        session_id: Session ID for message storage (required for persistence)
        user_id: User ID for message storage
        context: AgentContext for multi-agent propagation
        message_history: Previous conversation (pydantic-ai format)

    Yields:
        SSE-formatted strings (passthrough from stream_sse)
    """
    import json

    # Pre-generate message_id for frontend consistency
    # This ID will be in SSE events AND used as database record ID
    message_id = str(uuid.uuid4())

    # Accumulate content during streaming for persistence
    accumulated_content: list[str] = []

    # Capture tool calls for persistence (mutable list passed to stream_sse)
    tool_calls: list[dict] = []

    # Stream with content accumulation
    async for chunk in stream_sse(
        agent=agent,
        prompt=prompt,
        model=model,
        request_id=request_id,
        message_id=message_id,
        session_id=session_id,
        user_id=user_id,
        agent_schema=agent_schema,
        context=context,
        message_history=message_history,
        tool_calls_out=tool_calls,
    ):
        yield chunk

        # Extract text content from OpenAI-format chunks for accumulation
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

    # =========================================================================
    # POST-STREAMING PERSISTENCE
    # =========================================================================
    # After streaming completes, save tool calls and assistant response.
    # This happens synchronously but shouldn't block (DB ops are fast).
    # =========================================================================
    if session_id:
        from datetime import datetime, timezone

        from remlight.services.session import SessionMessageStore

        timestamp = datetime.now(timezone.utc).isoformat()
        messages_to_store = []

        # NOTE: Parent tool calls are now saved during streaming at PartStartEvent
        # to ensure correct chronological ordering (parent before child).
        # We skip saving them again here to avoid duplicates.
        # Child tool calls are saved during streaming via process_child_event.
        # Only action tool calls with special payloads are saved here.

        # Store assistant text response (role: "assistant")
        full_content = None
        if accumulated_content:
            full_content = "".join(accumulated_content)
        else:
            # Fallback: check tool results for text_response
            # (some agents return content via tools)
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
                "id": message_id,  # Same ID as in SSE events
                "role": "assistant",
                "content": full_content,
                "timestamp": timestamp,
            }
            messages_to_store.append(assistant_message)

        # Persist messages (best-effort, don't fail streaming)
        if messages_to_store:
            try:
                store = SessionMessageStore(user_id=user_id or "anonymous")
                await store.store_session_messages(
                    session_id=session_id,
                    messages=messages_to_store,
                    user_id=user_id,
                    compress=False,  # Store uncompressed (compress on load)
                )
                logger.debug(f"Stored {len(messages_to_store)} post-stream messages")
            except Exception as e:
                logger.error(f"Post-stream persistence failed: {e}")


async def save_user_message(
    session_id: str,
    user_id: str | None,
    content: str,
) -> None:
    """
    Save user message to database BEFORE streaming.

    This should be called BEFORE stream_sse_with_save() to ensure
    the user's message is persisted even if the agent fails.

    ORDER MATTERS
    ------------
    1. save_user_message() - Persist user input
    2. stream_sse_with_save() - Execute and persist response

    If we only saved after streaming and the agent crashed, we'd lose
    the user's message. Saving first ensures the conversation is preserved.

    Args:
        session_id: Session ID (required, returns early if empty)
        user_id: User ID for data isolation
        content: The user's message text
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
        pass  # Best-effort, don't fail the request
