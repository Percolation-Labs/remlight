"""
Minimal Pydantic Agent Construction
===================================

This module demonstrates the core agent execution flow used by the API.
In production, POST /chat/completions/{session_id} wraps this same logic.

Starting the API:
    # Start API server on port 8000 (default)
    uvicorn remlight.api.main:app --port 8000


API Usage:
    # OpenAI request/response models used as a standard

    POST http://localhost:8000/api/v1/chat/completions/{session_id}

    Headers:
        X-User-Id: user-123              # User identifier (or from JWT)
        X-Agent-Schema: query-agent      # Agent to use

    The API extracts these via AgentContext.from_request(request):

        context = AgentContext.from_request(request)
        # context.user_id      -> from JWT or X-User-Id header
        # context.session_id   -> from URL path or X-Session-Id header
        # context.agent_schema -> from X-Agent-Schema header

The Flow (this example below would be wrapped by api):
    1. Load schema from YAML (from db, file or cache)
    2. Ensure session exists, load message history (implement some sort of filtering for context size)
    3. Create AgentAdapter (builds Pydantic.AI agent with mcp tools from schema)
    4. Stream OpenAI-compatible SSE events
    5. Convert and save messages to database

CLI Usage:
    # Requires API server running for MCP tools
    # LOCAL only mcp can easily be used but this shows the general case
    uvicorn remlight.api.main:app --port 8000 &

    # Run example
    python -m remlight.agentic.minimal.example
    python -m remlight.agentic.minimal.example "What is REM?"

Debug: View Actual LLM Payload:
    To see the exact JSON payload sent to OpenAI (system prompt, tools, messages):

    import logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('openai._base_client').setLevel(logging.DEBUG)

    This logs the full request including:
    - model, temperature, messages
    - tools array (only tools declared in YAML schema are sent)
    - tool_choice settings

    See example-payload.yaml for a sample of what gets sent.

See also:
    - example-payload.yaml: Actual payload sent to OpenAI API
    - example.sse.txt: Sample SSE events from stream_openai_sse()
    - example.yaml: Converted messages (what gets saved to DB)
"""

import asyncio
from uuid import UUID, uuid4

from remlight.agentic.agent_schema import AgentSchema
from remlight.agentic.adapter import AgentAdapter, print_sse
from remlight.models.entities import Message, Session
from remlight.services.repository import Repository

 
####
### utility to save example for illustration
####
def save_example_outputs(prompt: str, session_id: UUID, sse_events: list, messages: list):
    """Save SSE events and message dump to example files (for documentation)."""
    import pathlib
    import yaml
    here = pathlib.Path(__file__).parent

    with open(here / "example.sse.txt", "w") as f:
        f.write("# SSE events from stream_openai_sse()\n")
        for event in sse_events:
            f.write(event)

    output = {
        "prompt": prompt,
        "session_id": str(session_id),
        "converted_messages": [{"role": m.role, "content": m.content} for m in messages],
    }
    with open(here / "example.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False, allow_unicode=True)

####
### Full flow example
####

async def main(prompt: str, session_id: UUID, save_examples: bool = False, **input_options):
    # 1. Load schema: this loads from file or database and should be cached
    schema = AgentSchema.load("orchestrator-agent")

    # 2. Ensure session and load messages - in prod you might build session in other ways (client controls session id)
    await Repository(Session).upsert(Session(id=session_id))
    
    # a repository should abstract in schema agnostic way and manage db connections
    message_repo = Repository(Message)
    # use generic filtering 
    # TODO: make sure to use context aware loading e.g. max messages or max context size
    # IF tiktoken or something use to estimate tokens/possibly saved, DB function can do this
    messages = await message_repo.find({"session_id": session_id})

    # 3. Create adapter - should use caching internally for API use cases
    adapter = AgentAdapter(schema, **input_options)

    # 4. Stream
    sse_events = []  #save for output example
    async with adapter.run_stream(prompt, message_history=messages or None) as result:
        #stream first and collet after
        async for event in result.stream_openai_sse():
            sse_events.append(event)
            print_sse(event)
        #convert pydantic ai collection messages to our format in the adapter.
        messages = result.to_messages(session_id)

    # 5. Save
    await message_repo.upsert(messages)

    if save_examples:
        save_example_outputs(prompt, session_id, sse_events, messages)


if __name__ == "__main__":
    import sys
    save = "--save" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--save"]
    prompt = args[0] if args else "What is REM?"
    asyncio.run(main(prompt, uuid4(), save_examples=save))
