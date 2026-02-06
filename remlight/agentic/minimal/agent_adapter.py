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
        Filters to user, assistant, and tool call messages only (not tool responses).
        """
        msgs = []
        for m in self.all_messages():
            msg_type = type(m).__name__

            # User message (ModelRequest with user content)
            if msg_type == "ModelRequest":
                parts = getattr(m, "parts", [])
                for part in parts:
                    part_type = type(part).__name__
                    if part_type == "UserPromptPart":
                        content = getattr(part, "content", "")
                        msgs.append(Message(role="user", content=str(content), session_id=session_id))

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


class AgentAdapter:
    """
    Wraps AgentSchema to create agent and yield OpenAI-compatible SSE events.

    Usage:
        adapter = AgentAdapter(schema)
        async with adapter.run_stream(prompt) as result:
            async for event in result.stream_openai_sse():
                print_sse(event)
            messages = result.to_messages(session_id)
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

        #options loads from the schema or defaults to settings allowing overrides at runtime
        options = self._schema.get_options(**self._input_options)
        
        #PydanticAI has tooling to build mcp tools from local or remote mcp servers
        #see https://ai.pydantic.dev/web/#builtin-tool-support
        toolsets = await resolve_tools_from_schema(self._schema)

        self._agent = Agent(
            #typically comes from the object description/docstring with some formatting
            system_prompt=self._schema.get_system_prompt(),
            #the schema when structure_output is true only (we remove docstring/description)
            output_type=self._schema.to_output_schema(),
            toolsets=toolsets,
            #options like model, temperature, usage_limits just passed through
            **options
        )

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
