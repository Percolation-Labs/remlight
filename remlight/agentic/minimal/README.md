# Minimal Pydantic Agent Construction

The simplest path to construct a Pydantic AI agent from `AgentSchema`.

## The Flow

```python
# 1. Load schema
schema = AgentSchema.load("orchestrator-agent")

# 2. Ensure session and load messages
await Repository(Session).upsert(Session(id=session_id))
messages = await Repository(Message).find({"session_id": session_id})

# 3. Create adapter
adapter = AgentAdapter(schema, **input_options)

# 4. Stream
async with adapter.run_stream(prompt, message_history=messages) as result:
    async for event in result.stream_openai_sse():
        print_sse(event)  # CLI
        yield event       # API
    messages = result.to_messages(session_id)

# 5. Save
await message_repo.upsert(messages)
```

## Run

```bash
# Start API server (required for MCP tools)
uvicorn remlight.api.main:app --port 8000 &

# Run example
python -m remlight.agentic.minimal.example
python -m remlight.agentic.minimal.example "What is REM?"

# Regenerate example output files
python -m remlight.agentic.minimal.example --save
```

## Files

| File | Purpose |
|------|---------|
| `example.py` | Runnable demo with full flow |
| `agent_adapter.py` | AgentAdapter, StreamResult, print_sse |
| `example.sse.txt` | Sample SSE events (includes tool calls) |
| `example.yaml` | Converted messages saved to DB |
