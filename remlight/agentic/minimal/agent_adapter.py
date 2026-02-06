"""
Agent Adapter - Wraps AgentSchema to yield OpenAI-compatible SSE events.

Usage:
    adapter = AgentAdapter(schema)
    async with adapter.run_stream(prompt, message_history=messages) as result:
        async for event in result.stream_openai_sse():
            print_sse(event)  # CLI helper
            yield event       # API use
        messages = result.to_messages(session_id)
"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from pydantic_ai import Agent

from remlight.models.entities import Message


def print_sse(event: str) -> None:
    """Print content from SSE event to stdout."""
    if event.startswith("data: ") and not event.startswith("data: [DONE]"):
        try:
            data = json.loads(event[6:].strip())
            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("delta", {}).get("content")
                if content:
                    print(content, end="", flush=True)
        except (json.JSONDecodeError, KeyError):
            pass


class StreamResult:
    """Result handle from AgentAdapter.run_stream()."""

    def __init__(self, agent_run, ctx):
        self._agent_run = agent_run
        self._ctx = ctx
        self._request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        self._created_at = int(time.time())
        self._model = "unknown"
        self._is_first_chunk = True
        self._tool_index = 0

    async def stream_openai_sse(self) -> AsyncGenerator[str, None]:
        """Yield OpenAI-compatible SSE events."""
        async for node in self._agent_run:
            if Agent.is_model_request_node(node):
                async with node.stream(self._ctx) as request_stream:
                    async for event in request_stream:
                        event_type = type(event).__name__

                        if event_type == "PartDeltaEvent":
                            if hasattr(event, "delta") and hasattr(event.delta, "content_delta"):
                                content = event.delta.content_delta
                                if content:
                                    yield self._format_chunk(content)

                        elif event_type == "PartStartEvent" and hasattr(event, "part"):
                            part_type = type(event.part).__name__
                            if part_type == "TextPart" and event.part.content:
                                yield self._format_chunk(event.part.content)
                            elif part_type == "ToolCallPart":
                                yield self._format_tool_call(event.part)

            elif Agent.is_call_tools_node(node):
                async with node.stream(self._ctx) as tools_stream:
                    async for _ in tools_stream:
                        pass

        yield self._format_chunk("", finish_reason="stop")
        yield "data: [DONE]\n\n"

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

    def _format_tool_call(self, part) -> str:
        """Format tool call as OpenAI SSE chunk."""
        args = getattr(part, "args", None)
        args_str = args if isinstance(args, str) else json.dumps(args) if args else "{}"
        tool_call = {
            "index": self._tool_index,
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": part.tool_name, "arguments": args_str},
        }
        self._tool_index += 1
        chunk = {
            "id": self._request_id,
            "object": "chat.completion.chunk",
            "created": self._created_at,
            "model": self._model,
            "choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}],
        }
        return f"data: {json.dumps(chunk)}\n\n"


DELEGATE_TOOL_NAMES = {"ask_agent"}  # Tools we load as delegates, not from MCP


def get_delegate_tools(schema) -> list:
    """
    Get delegate tools (like ask_agent) declared in schema.

    If schema declares ask_agent in tools, we load the production version
    which has full streaming and structured output support.
    """
    tools = []
    tool_names = {t.name for t in schema.tools} if schema.tools else set()

    if "ask_agent" in tool_names:
        from remlight.api.routers.tools import ask_agent
        tools.append(ask_agent)

    return tools


def filter_mcp_tools(schema):
    """Return schema with delegate tools removed (to avoid MCP conflict)."""
    if not schema.tools:
        return schema

    # Filter out delegate tools from MCP loading
    filtered = [t for t in schema.tools if t.name not in DELEGATE_TOOL_NAMES]

    if len(filtered) == len(schema.tools):
        return schema  # No changes needed

    # Create new schema with filtered tools
    return schema.model_copy(update={
        "json_schema_extra": schema.json_schema_extra.model_copy(update={"tools": filtered})
    })


class AgentAdapter:
    """
    Wraps AgentSchema to create agent and yield OpenAI-compatible SSE events.

    Usage:
        adapter = AgentAdapter(schema)
        async with adapter.run_stream(prompt) as result:
            async for event in result.stream_openai_sse():
                print_sse(event)
            messages = result.to_messages(session_id)

    All agents get a `delegate(agent_name, prompt)` tool for multi-agent orchestration.
    """

    def __init__(self, schema, **input_options):
        self._schema = schema
        self._input_options = input_options
        self._agent = None

    async def _ensure_agent(self):
        """Lazily create the agent with toolsets."""
        if self._agent is not None:
            return

        from remlight.agentic.tool_resolver import resolve_tools_from_schema

        options = self._schema.get_options(**self._input_options)

        # Filter out delegate tools from MCP to avoid name conflict
        mcp_schema = filter_mcp_tools(self._schema)
        toolsets = await resolve_tools_from_schema(mcp_schema)

        # Get delegate tools (ask_agent etc) as standalone tools
        delegate_tools = get_delegate_tools(self._schema)

        # Build agent with tools
        agent_kwargs = {
            "system_prompt": self._schema.get_system_prompt(),
            "output_type": self._schema.to_output_schema(),
            **options
        }
        if toolsets:
            agent_kwargs["toolsets"] = toolsets
        if delegate_tools:
            agent_kwargs["tools"] = delegate_tools

        self._agent = Agent(**agent_kwargs)

    @asynccontextmanager
    async def run_stream(
        self,
        prompt: str,
        *,
        message_history: list | None = None,
    ):
        """Run agent with streaming."""
        await self._ensure_agent()

        # Context manager ensures agent_run is properly closed when caller exits.
        # agent.iter() returns an async context manager that manages the run lifecycle.
        async with self._agent.iter(prompt, message_history=message_history) as agent_run:
            yield StreamResult(agent_run, agent_run.ctx)
