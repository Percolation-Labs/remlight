# Minimal Pydantic Agent Construction

The simplest path to construct a Pydantic AI agent from `AgentSchema`.

## The Flow

```python
repository = Repository(Message)

# 1. Load schema
schema = AgentSchema.load("query-agent")

# 2. Load messages
messages = await repository.find({"session_id": session_id}) if session_id else []

# 3. Create adapter
adapter = AgentAdapter(schema, **input_options)

# 4. Stream
async with adapter.run_stream(prompt, message_history=messages) as result:
    async for event in result.stream_openai_sse():
        print_sse(event)  # CLI
        yield event       # API
    messages = result.to_messages(session_id)

# 5. Save
await repository.upsert(messages)
```

## Run

```bash
python -m remlight.agentic.minimal.example
python -m remlight.agentic.minimal.example "What is REM?"
```

## Files

- `example.py` - Runnable demo
- `agent_adapter.py` - AgentAdapter, StreamResult, print_sse
