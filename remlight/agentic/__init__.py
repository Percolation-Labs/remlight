"""REMLight agentic module - declarative agents with YAML schemas.

The canonical way to run agents:

    from remlight.agentic import AgentAdapter, AgentSchema

    schema = AgentSchema.load("query-agent")
    adapter = AgentAdapter(schema)

    async with adapter.run_stream(prompt) as result:
        async for event in result.stream_openai_sse():
            yield event
        messages = result.to_messages(session_id)
"""

from remlight.agentic.adapter import AgentAdapter, StreamResult, print_sse
from remlight.agentic.agent_schema import AgentSchema
from remlight.agentic.tool_resolver import resolve_tools_from_schema

__all__ = [
    "AgentAdapter",
    "StreamResult",
    "print_sse",
    "AgentSchema",
    "resolve_tools_from_schema",
]
