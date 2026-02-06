"""
Minimal Pydantic Agent Construction
===================================

The Flow:
    1. schema = AgentSchema.load(name)
    2. messages = repository.find({"session_id": session_id})
    3. adapter = AgentAdapter(schema, **input_options)
    4. async with adapter.run_stream(prompt) as result:
           async for event in result.stream_openai_sse():
               print_sse(event)  # CLI
               yield event       # API
           messages = result.to_messages(session_id)
    5. repository.upsert(messages)

Usage:
    python -m remlight.agentic.minimal.example
    python -m remlight.agentic.minimal.example "What is REM?"
"""

from remlight.agentic.minimal.agent_adapter import AgentAdapter, print_sse

__all__ = ["AgentAdapter", "print_sse"]
