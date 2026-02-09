"""
Agent Adapter - The canonical way to run agents with OpenAI-compatible SSE streaming.

This module replaces the legacy provider.py, runner.py, streaming/, and context.py
with a minimal, clean implementation.

Usage:
    from remlight.agentic import AgentAdapter, AgentSchema

    schema = AgentSchema.load("query-agent")
    adapter = AgentAdapter(schema)

    async with adapter.run_stream(prompt, message_history=messages) as result:
        async for event in result.stream_openai_sse():
            print_sse(event)  # CLI helper
            yield event       # API streaming
        messages = result.to_messages(session_id)

Features:
    - OpenAI-compatible SSE streaming (chat.completion.chunk format)
    - Automatic tool call formatting
    - Multi-agent delegation via ask_agent tool
    - Structured output capture from delegate agents
    - Message conversion for database persistence
"""

import asyncio
import json
import time
import uuid
from contextvars import ContextVar
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from pydantic_ai import Agent

from remlight.models.entities import Message


# Context variable for sub-agent event streaming
# When set, child agents will forward their SSE events to this queue
_child_event_sink: ContextVar[asyncio.Queue | None] = ContextVar("child_event_sink", default=None)


def get_child_event_sink() -> asyncio.Queue | None:
    """Get the current child event sink if set."""
    return _child_event_sink.get()


def set_child_event_sink(queue: asyncio.Queue | None) -> None:
    """Set the child event sink for sub-agent streaming."""
    _child_event_sink.set(queue)


def format_child_agent_event(agent_name: str, event: str) -> str:
    """Transform an SSE event to indicate it came from a child agent.

    Emits events in the format the UI expects:
    - child_content: text content from child agent
    - child_tool_start: tool call started
    - child_tool_result: tool call completed with result
    """
    # Skip [DONE] and done events - parent handles its own termination
    if "data: [DONE]" in event or 'event: done' in event:
        return ""

    # For tool_call events, convert to child_tool_start/child_tool_result
    if event.startswith("event: tool_call"):
        lines = event.strip().split("\n")
        if len(lines) > 1 and lines[1].startswith("data:"):
            try:
                data = json.loads(lines[1][5:].strip())
                status = data.get("status", "")
                tool_name = data.get("tool_name", "unknown")
                tool_id = data.get("tool_id") or data.get("tool_call_id")

                if status == "started":
                    child_event = {
                        "type": "child_tool_start",
                        "agent_name": agent_name,
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "arguments": data.get("arguments"),
                    }
                    return f"event: child_tool_start\ndata: {json.dumps(child_event)}\n\n"
                elif status == "completed":
                    child_event = {
                        "type": "child_tool_result",
                        "agent_name": agent_name,
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                        "result": data.get("result"),
                    }
                    return f"event: child_tool_result\ndata: {json.dumps(child_event)}\n\n"
            except (json.JSONDecodeError, IndexError):
                pass
        return ""

    # For other custom events (action), add agent_name
    if event.startswith("event:"):
        lines = event.strip().split("\n")
        event_type = lines[0].replace("event:", "").strip()
        if event_type in ("action",) and len(lines) > 1 and lines[1].startswith("data:"):
            try:
                data = json.loads(lines[1][5:].strip())
                data["agent_name"] = agent_name
                return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            except (json.JSONDecodeError, IndexError):
                pass
        return ""

    # For data events (OpenAI chunks), extract content and wrap as child_content
    if event.startswith("data:"):
        try:
            data = json.loads(event[5:].strip())
            if "choices" in data and data["choices"]:
                delta = data["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    child_event = {
                        "type": "child_content",
                        "agent_name": agent_name,
                        "content": content,
                    }
                    return f"event: child_content\ndata: {json.dumps(child_event)}\n\n"
        except (json.JSONDecodeError, KeyError):
            pass

    return ""


def extract_sse_content(event: str) -> str | None:
    """Extract text content from an SSE event.

    Returns the content string if present, None otherwise.
    Handles OpenAI-format chat completion chunks.
    """
    # Skip non-data events (tool_call, action, done, etc.)
    if not event.startswith("data: "):
        return None
    if event.startswith("data: [DONE]"):
        return None

    try:
        data = json.loads(event[6:].strip())
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("delta", {}).get("content")
    except (json.JSONDecodeError, KeyError, IndexError):
        pass
    return None


def print_sse(event: str) -> None:
    """Print content from SSE event to stdout."""
    content = extract_sse_content(event)
    if content:
        print(content, end="", flush=True)


async def collect_sse_text(sse_stream) -> str:
    """Collect all text content from an SSE stream.

    Args:
        sse_stream: Async generator yielding SSE event strings

    Returns:
        Concatenated text content from all events
    """
    text = ""
    async for event in sse_stream:
        content = extract_sse_content(event)
        if content:
            text += content
    return text


# Tools that are loaded as delegates (not from MCP)
DELEGATE_TOOL_NAMES = {"ask_agent"}


class StreamResult:
    """Result handle from AgentAdapter.run_stream().

    Provides:
    - stream_openai_sse(): Async generator of SSE events
    - to_messages(): Convert to Message entities for persistence
    - all_messages(): Raw pydantic-ai messages
    """

    def __init__(self, agent_run, ctx):
        self._agent_run = agent_run
        self._ctx = ctx
        self._request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        self._created_at = int(time.time())
        self._model = "unknown"
        self._is_first_chunk = True
        self._tool_index = 0
        self._pending_tool_calls = {}  # tool_call_id -> {name, args}
        self._part_index_to_tool_id: dict[int, str] = {}  # part index -> tool_call_id
        self._child_event_queue: asyncio.Queue | None = None

    async def stream_openai_sse(self) -> AsyncGenerator[str, None]:
        """Yield OpenAI-compatible SSE events with custom tool_call events.

        Also yields events from child agents (via ask_agent) with child_agent event types.
        """
        # Set up child event queue for sub-agent streaming
        self._child_event_queue = asyncio.Queue()
        previous_sink = _child_event_sink.get()
        _child_event_sink.set(self._child_event_queue)

        try:
            async for node in self._agent_run:
                # Drain any child events that arrived during tool execution
                while not self._child_event_queue.empty():
                    try:
                        child_event = self._child_event_queue.get_nowait()
                        yield child_event
                    except asyncio.QueueEmpty:
                        break

                if Agent.is_model_request_node(node):
                    async with node.stream(self._ctx) as request_stream:
                        async for event in request_stream:
                            event_type = type(event).__name__

                            if event_type == "PartDeltaEvent":
                                if hasattr(event, "delta"):
                                    # Text content delta
                                    if hasattr(event.delta, "content_delta"):
                                        content = event.delta.content_delta
                                        if content:
                                            yield self._format_chunk(content)
                                    # Tool args delta - accumulate args for the specific tool call
                                    if hasattr(event.delta, "args_delta"):
                                        args_delta = event.delta.args_delta
                                        if args_delta:
                                            # Find the tool call this delta belongs to via part index
                                            part_index = event.index
                                            tool_id = self._part_index_to_tool_id.get(part_index)
                                            if tool_id and tool_id in self._pending_tool_calls:
                                                pending = self._pending_tool_calls[tool_id]
                                                if pending.get("_args_buffer") is None:
                                                    pending["_args_buffer"] = ""
                                                pending["_args_buffer"] += args_delta

                            elif event_type == "PartStartEvent" and hasattr(event, "part"):
                                part_type = type(event.part).__name__
                                if part_type == "TextPart" and event.part.content:
                                    yield self._format_chunk(event.part.content)
                                elif part_type == "ToolCallPart":
                                    # Emit custom tool_call event (started)
                                    yield self._format_tool_call_event(event.part, "started")
                                    # Store for later completion
                                    part_index = event.index
                                    tool_id = getattr(event.part, "tool_call_id", None) or f"call_{uuid.uuid4().hex[:8]}"
                                    # Map part index to tool_id for correlating args_delta
                                    self._part_index_to_tool_id[part_index] = tool_id
                                    # Extract args - may be JSON string or dict
                                    raw_args = getattr(event.part, "args", None)
                                    if isinstance(raw_args, str) and raw_args:
                                        try:
                                            parsed_args = json.loads(raw_args)
                                        except json.JSONDecodeError:
                                            parsed_args = {"raw": raw_args}
                                    elif isinstance(raw_args, dict):
                                        parsed_args = raw_args
                                    else:
                                        parsed_args = {}
                                    self._pending_tool_calls[tool_id] = {
                                        "name": event.part.tool_name,
                                        "args": parsed_args,
                                        "tool_id": tool_id,
                                    }

                # NOTE Some examples of custom UI formatting here but could be any protocol
                elif Agent.is_call_tools_node(node):
                    # Tool execution - emit executing/completed events
                    # Use concurrent approach to stream child events DURING tool execution
                    async with node.stream(self._ctx) as tools_stream:
                        async for tool_event in self._stream_with_child_events(tools_stream):
                            # Check if this is a child event (string) vs tool event
                            if isinstance(tool_event, str):
                                yield tool_event
                                continue

                            # Process tool result
                            if hasattr(tool_event, "result"):
                                tool_call_id = getattr(tool_event, "tool_call_id", None)
                                if tool_call_id and tool_call_id in self._pending_tool_calls:
                                    pending = self._pending_tool_calls[tool_call_id]

                                    # Extract serializable result
                                    result = self._extract_serializable_result(tool_event.result)

                                    # Check if this is an action event
                                    if pending["name"] == "action" and isinstance(result, dict):
                                        if result.get("_action_event"):
                                            yield self._format_action_event(result)

                                    # Get final args
                                    final_args = pending["args"]
                                    if not final_args and pending.get("_args_buffer"):
                                        try:
                                            final_args = json.loads(pending["_args_buffer"])
                                        except json.JSONDecodeError:
                                            final_args = {"raw": pending["_args_buffer"]}

                                    # Emit completed event
                                    yield self._format_tool_call_completed(
                                        pending["name"],
                                        pending["tool_id"],
                                        final_args,
                                        result
                                    )
                                    del self._pending_tool_calls[tool_call_id]

            # Drain any remaining child events
            while not self._child_event_queue.empty():
                try:
                    child_event = self._child_event_queue.get_nowait()
                    yield child_event
                except asyncio.QueueEmpty:
                    break

            yield self._format_chunk("", finish_reason="stop")
            yield self._format_done_event()
            yield "data: [DONE]\n\n"

        finally:
            # Restore previous sink
            _child_event_sink.set(previous_sink)

    def all_messages(self) -> list:
        """Get raw pydantic-ai messages after streaming completes."""
        return self._agent_run.result.all_messages()

    def to_messages(self, session_id: str | None = None) -> list[Message]:
        """Convert pydantic-ai messages to Message entities.

        Captures:
        - User prompts (UserPromptPart)
        - Tool calls (ToolCallPart) as 'tool' role
        - Tool results from delegates (ToolReturnPart for ask_agent with structured output)
        - Assistant text responses (TextPart)
        
        Does not capture tool responses e.g. DB lookups (but could be added optionally)
        """
        msgs = []
        for m in self.all_messages():
            msg_type = type(m).__name__

            if msg_type == "ModelRequest":
                parts = getattr(m, "parts", [])
                for part in parts:
                    part_type = type(part).__name__

                    # User message
                    if part_type == "UserPromptPart":
                        content = getattr(part, "content", "")
                        msgs.append(Message(role="user", content=str(content), session_id=session_id))

                    # Tool return - capture delegate results with structured output
                    elif part_type == "ToolReturnPart":
                        tool_name = getattr(part, "tool_name", "")
                        content = getattr(part, "content", None)

                        # Only capture delegate tool results with structured output
                        # NOTE These are saved in the database and hence the context/memory for the conversation
                        if tool_name in DELEGATE_TOOL_NAMES and isinstance(content, dict):
                            if content.get("is_structured_output") and "output" in content:
                                # Save structured output as tool_result message
                                output = content["output"]
                                output_str = json.dumps(output) if isinstance(output, dict) else str(output)
                                msgs.append(Message(
                                    role="tool_result",
                                    content=output_str,
                                    session_id=session_id,
                                ))

            # Assistant message or tool call (ModelResponse)
            elif msg_type == "ModelResponse":
                parts = getattr(m, "parts", [])
                text_parts = []
                for part in parts:
                    part_type = type(part).__name__
                    if part_type == "TextPart":
                        text_parts.append(getattr(part, "content", ""))
                    elif part_type == "ToolCallPart":
                        # Tool call - store as tool message
                        tool_name = getattr(part, "tool_name", "")
                        args = getattr(part, "args", {})
                        msgs.append(Message(role="tool", content=f"{tool_name}({args})", session_id=session_id))
                if text_parts:
                    msgs.append(Message(role="assistant", content="".join(text_parts), session_id=session_id))

        return msgs

    def _format_chunk(self, content: str, finish_reason: str | None = None) -> str:
        """Format as OpenAI SSE chunk."""
        delta: dict[str, Any] = {}
        if self._is_first_chunk:
            delta["role"] = "assistant"
            self._is_first_chunk = False
        if content:
            delta["content"] = content

        chunk = {
            "id": self._request_id,
            "object": "chat.completion.chunk",
            "created": self._created_at,
            "model": self._model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(chunk)}\n\n"

    async def _stream_with_child_events(self, tools_stream) -> AsyncGenerator[Any, None]:
        """Merge tool stream with child events, yielding both concurrently.

        This allows child agent SSE events to be yielded DURING tool execution,
        not just after the tool completes.

        Uses asyncio.wait with FIRST_COMPLETED to truly interleave events
        from both sources without polling delays.
        """
        tools_iter = tools_stream.__aiter__()
        tool_stream_done = False

        # Persistent task for getting child events - recreated after each yield
        child_task: asyncio.Task | None = None
        tool_task: asyncio.Task | None = None

        try:
            while not tool_stream_done:
                # Create tasks for both streams if needed
                if tool_task is None:
                    tool_task = asyncio.create_task(
                        self._get_next_tool_event(tools_iter),
                        name="tool_event"
                    )

                if child_task is None:
                    child_task = asyncio.create_task(
                        self._child_event_queue.get(),
                        name="child_event"
                    )

                # Wait for whichever completes first
                pending = {tool_task, child_task}
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED
                )

                for completed_task in done:
                    if completed_task is child_task:
                        # Child event arrived - yield it immediately
                        try:
                            child_event = completed_task.result()
                            yield child_event
                        except asyncio.CancelledError:
                            pass
                        # Reset child task to get next event
                        child_task = None

                    elif completed_task is tool_task:
                        # Tool event arrived
                        result = completed_task.result()
                        if result is None:
                            # Stream ended (StopAsyncIteration)
                            tool_stream_done = True
                        else:
                            yield result
                        # Reset tool task for next event
                        tool_task = None

        finally:
            # Cancel any pending tasks
            if child_task and not child_task.done():
                child_task.cancel()
                try:
                    await child_task
                except asyncio.CancelledError:
                    pass
            if tool_task and not tool_task.done():
                tool_task.cancel()
                try:
                    await tool_task
                except asyncio.CancelledError:
                    pass

        # Drain remaining child events after tool stream ends
        while not self._child_event_queue.empty():
            try:
                child_event = self._child_event_queue.get_nowait()
                yield child_event
            except asyncio.QueueEmpty:
                break

    async def _get_next_tool_event(self, tools_iter):
        """Get next tool event, returning None on stream end."""
        try:
            return await tools_iter.__anext__()
        except StopAsyncIteration:
            return None

    def _extract_serializable_result(self, raw_result: Any) -> Any:
        """Extract a JSON-serializable result from pydantic-ai tool result."""
        # If it's already a basic type, use it
        if raw_result is None or isinstance(raw_result, (str, int, float, bool)):
            return raw_result

        # If it's a dict or list, check if serializable
        if isinstance(raw_result, (dict, list)):
            try:
                json.dumps(raw_result)
                return raw_result
            except (TypeError, ValueError):
                pass

        # Check for content attribute (ToolReturnPart)
        if hasattr(raw_result, "content"):
            content = raw_result.content
            # Recursively extract from content
            return self._extract_serializable_result(content)

        # Check for model_dump (Pydantic model)
        if hasattr(raw_result, "model_dump"):
            return raw_result.model_dump()

        # Check for __dict__
        if hasattr(raw_result, "__dict__"):
            try:
                result = {k: v for k, v in raw_result.__dict__.items() if not k.startswith("_")}
                json.dumps(result)
                return result
            except (TypeError, ValueError):
                pass

        # Last resort: string representation
        return str(raw_result)

    def _format_tool_call_event(self, part, status: str) -> str:
        """Format custom tool_call SSE event."""
        tool_id = getattr(part, "tool_call_id", None) or f"call_{uuid.uuid4().hex[:8]}"
        args = getattr(part, "args", None)

        event_data = {
            "type": "tool_call",
            "tool_name": part.tool_name,
            "tool_id": tool_id,
            "status": status,
        }
        if args and status != "started":
            event_data["arguments"] = args

        return f"event: tool_call\ndata: {json.dumps(event_data)}\n\n"

    def _format_tool_call_completed(self, tool_name: str, tool_id: str, args: Any, result: Any) -> str:
        """Format tool_call completed event with result."""
        # Ensure result is JSON serializable
        try:
            json.dumps(result)
            serializable_result = result
        except (TypeError, ValueError):
            serializable_result = str(result)

        event_data = {
            "type": "tool_call",
            "tool_name": tool_name,
            "tool_id": tool_id,
            "status": "completed",
            "arguments": args,
            "result": serializable_result,
        }
        return f"event: tool_call\ndata: {json.dumps(event_data)}\n\n"

    def _format_action_event(self, result: dict) -> str:
        """Format action event from action tool result."""
        event_data = {
            "type": "action",
            "action_type": result.get("action_type", "unknown"),
            "payload": result.get("payload", {}),
            "_action_event": True,
        }
        return f"event: action\ndata: {json.dumps(event_data)}\n\n"

    def _format_done_event(self) -> str:
        """Format done event."""
        return f"event: done\ndata: {{\"type\": \"done\", \"reason\": \"stop\"}}\n\n"

    def get_output(self) -> Any:
        """Get the agent's structured output after streaming completes.

        Returns the Pydantic model for structured_output agents,
        or text string for text agents.
        """
        try:
            return self._agent_run.result.output
        except Exception:
            return None


def _get_delegate_tools(schema) -> list:
    """Get delegate tools (like ask_agent) declared in schema."""
    tools = []
    tool_names = {t.name for t in schema.tools} if schema.tools else set()

    if "ask_agent" in tool_names:
        from remlight.api.routers.tools import ask_agent
        tools.append(ask_agent)

    return tools


def _filter_mcp_tools(schema):
    """Return schema with delegate tools removed (to avoid MCP conflict)."""
    if not schema.tools:
        return schema

    filtered = [t for t in schema.tools if t.name not in DELEGATE_TOOL_NAMES]

    if len(filtered) == len(schema.tools):
        return schema

    return schema.model_copy(update={
        "json_schema_extra": schema.json_schema_extra.model_copy(update={"tools": filtered})
    })


class AgentAdapter:
    """
    The canonical adapter for running agents with streaming.

    Wraps AgentSchema to create a pydantic-ai Agent and yield OpenAI-compatible
    SSE events. Replaces the legacy create_agent/streaming architecture.

    Usage:
        schema = AgentSchema.load("query-agent")
        adapter = AgentAdapter(schema)

        async with adapter.run_stream(prompt) as result:
            async for event in result.stream_openai_sse():
                yield event
            messages = result.to_messages(session_id)
    """

    def __init__(self, schema, **input_options):
        """Initialize adapter with schema and optional overrides.

        Args:
            schema: AgentSchema instance
            **input_options: Override model, temperature, etc.
        """
        self._schema = schema
        self._input_options = input_options
        self._agent = None

    async def _ensure_agent(self):
        """Lazily create the agent with toolsets and resource tools."""
        if self._agent is not None:
            return

        from remlight.agentic.tool_resolver import resolve_tools_from_schema

        options = self._schema.get_options(**self._input_options)

        # Filter out delegate tools from MCP to avoid name conflict
        mcp_schema = _filter_mcp_tools(self._schema)

        # Resolve both toolsets (MCP tools) and resource tools
        toolsets, resource_tools = await resolve_tools_from_schema(mcp_schema)

        # Get delegate tools (ask_agent etc) as standalone tools
        delegate_tools = _get_delegate_tools(self._schema)

        # Combine all standalone tools (delegates + resources)
        all_tools = []
        if delegate_tools:
            all_tools.extend(delegate_tools)
        if resource_tools:
            all_tools.extend(resource_tools)

        # Build agent
        agent_kwargs = {
            "system_prompt": self._schema.get_system_prompt(),
            "output_type": self._schema.to_output_schema(),
            **options
        }
        if toolsets:
            agent_kwargs["toolsets"] = toolsets
        if all_tools:
            agent_kwargs["tools"] = all_tools

        self._agent = Agent(**agent_kwargs)

        # Debug: dump payload to example-payload.yaml
        await self._dump_agent_payload(agent_kwargs, toolsets, all_tools)

    async def _dump_agent_payload(self, agent_kwargs: dict, toolsets: list | None, tools: list | None):
        """Dump agent configuration to example-payload.yaml for debugging."""
        import os
        import yaml
        from pathlib import Path

        if not os.getenv("DEBUG_AGENT_PAYLOAD"):
            return

        payload = {
            "agent_name": self._schema.name,
            "model": agent_kwargs.get("model"),
            "model_settings": agent_kwargs.get("model_settings"),
            "system_prompt": agent_kwargs.get("system_prompt"),
            "output_type": str(agent_kwargs.get("output_type")),
            "tools": [],
            "toolsets": [],
        }

        # Extract tool info from standalone tools
        if tools:
            for tool in tools:
                payload["tools"].append({
                    "name": getattr(tool, "__name__", str(tool)),
                    "doc": getattr(tool, "__doc__", None),
                    "annotations": {k: str(v) for k, v in getattr(tool, "__annotations__", {}).items()},
                })

        # Extract tool info from toolsets (MCP tools)
        if toolsets:
            for ts in toolsets:
                try:
                    # Get tool definitions - try different methods
                    if hasattr(ts, "list_tool_defs"):
                        tool_defs = await ts.list_tool_defs(None)  # ctx=None
                    elif hasattr(ts, "wrapped") and hasattr(ts.wrapped, "list_tool_defs"):
                        tool_defs = await ts.wrapped.list_tool_defs(None)
                    else:
                        payload["toolsets"].append({"type": type(ts).__name__, "info": "could not list tools"})
                        continue

                    for td in tool_defs:
                        payload["toolsets"].append({
                            "name": td.name,
                            "description": td.description,
                            "parameters": td.parameters_json_schema if hasattr(td, "parameters_json_schema") else None,
                        })
                except Exception as e:
                    payload["toolsets"].append({"error": str(e)})

        # Write to file
        output_path = Path(__file__).parent / "minimal" / "example-payload.yaml"
        with open(output_path, "w") as f:
            yaml.dump(payload, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    @asynccontextmanager
    async def run_stream(
        self,
        prompt: str,
        *,
        message_history: list | None = None,
    ):
        """Run agent with streaming.

        Args:
            prompt: User message
            message_history: Optional previous messages for context

        Yields:
            StreamResult with stream_openai_sse() and to_messages() methods
        """
        await self._ensure_agent()

        async with self._agent.iter(prompt, message_history=message_history) as agent_run:
            yield StreamResult(agent_run, agent_run.ctx)
