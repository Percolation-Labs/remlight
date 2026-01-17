# Examples

This folder contains example outputs from REMLight demonstrating the message storage format and SSE streaming protocol.

## Files

| File | Description |
|------|-------------|
| `session.yaml` | Exported session showing how messages are stored in the database |
| `sse.txt` | Raw SSE stream output showing real-time streaming events |

## Running the Examples

### CLI

```bash
# Ask a question (creates a session)
rem ask "What is RemLight and what are its main features?" --schema orchestrator-agent

# Multi-turn conversation (use same session UUID)
rem ask "Tell me more about the streaming" --session f2c5db60-fc72-429a-9768-bf7d83733126
```

### API

```bash
# Stream a chat completion
curl -X POST http://localhost:8080/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: f2c5db60-fc72-429a-9768-bf7d83733126" \
  -d '{
    "messages": [{"role": "user", "content": "What is RemLight?"}],
    "model": "openai:gpt-4.1",
    "stream": true,
    "agent_schema": "orchestrator-agent"
  }'
```

### Chat Client

For an interactive experience, use the React chat client:

```bash
cd app
npm install
VITE_API_BASE_URL=http://localhost:8080/api npm run dev
```

Open http://localhost:3000 to test agents with real-time streaming visualization.

---

## Message Storage (`session.yaml`)

The database stores three types of messages:

### 1. User Messages (`role: user`)

```yaml
- id: 926be78f-2c51-46d2-a764-f8bdb223076d
  role: user
  content: What is RemLight and what are its main features?
  status: completed
```

User input is stored as-is with a timestamp.

### 2. Tool Call Messages (`role: tool`)

```yaml
- id: 8b8787ed-3f7c-431e-ab67-0050b74b0888
  role: tool
  content: '{"query": "LOOKUP remlight"}'
  metadata:
    tool_name: query-agent:search
    tool_call_id: call_1161e795
    agent_schema: query-agent
```

Tool calls store the **invocation arguments** (not the result). Key fields:
- `content`: JSON with the tool arguments
- `tool_name`: Which tool was called (format: `agent:tool` for child agents)
- `tool_call_id`: Unique ID to match with SSE events
- `agent_schema`: Which agent made the call

**Note**: Tool *results* are streamed via SSE but not stored separately. The database captures the agent's reasoning (what tools it decided to call) rather than external data.

### 3. Assistant Messages (`role: assistant`)

```yaml
- id: d4a53bb8-2fe1-4e0c-b0e0-7bd30d240f36
  role: assistant
  content: "RemLight is a lightweight agentic framework..."
  status: completed
```

The final text response from the agent.

---

## SSE Streaming (`sse.txt`)

The API streams events using Server-Sent Events (SSE) with OpenAI-compatible format plus custom events for tool calls.

### Event Types

| Event | Purpose |
|-------|---------|
| `data: {...}` | OpenAI-format content chunks |
| `event: tool_call` | Tool invocation start/complete |
| `event: action` | Agent action metadata (confidence, sources) |
| `event: progress` | Progress indicators for UI |
| `event: done` | Stream completion |

### Content Streaming (OpenAI Format)

```
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {"content": "RemLight"}}]}
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {"content": " is"}}]}
data: {"id": "chatcmpl-xxx", "choices": [{"delta": {"content": " a"}}]}
```

Text content arrives as incremental chunks. The UI appends each `content` delta to build the full response.

### Tool Call Events

```
event: tool_call
data: {"type": "tool_call", "tool_name": "search", "tool_id": "call_xxx", "status": "started"}

event: tool_call
data: {"type": "tool_call", "tool_name": "search", "tool_id": "call_xxx", "status": "completed", "result": "..."}
```

Tool calls have three statuses:
- `started`: Tool invocation begins (UI shows spinner)
- `executing`: Arguments are complete (UI shows args)
- `completed`: Result is available (UI shows result)

### Child Agent Tool Calls

When the orchestrator calls `ask_agent`, child tool calls are prefixed with the agent name:

```
event: tool_call
data: {"tool_name": "query-agent:search", "tool_id": "call_xxx", "status": "started"}
```

The format `agent-name:tool-name` lets the UI show nested tool calls under their parent agent.

### Action Events

```
event: action
data: {"type": "action", "action_type": "observation", "payload": {"confidence": 0.95, "sources": ["remlight"]}}
```

Actions emit structured metadata about the agent's reasoning:
- `confidence`: How confident the agent is (0.0-1.0)
- `sources`: Entity keys referenced
- `session_name`: Suggested name for the session

---

## UI Consumption

The React chat client (`app/`) consumes these streams to show:

1. **Content**: Text appears character-by-character as it streams
2. **Tool Calls**: Expandable cards showing tool name, arguments, and results
3. **Nested Agents**: Child agent calls indented under their parent `ask_agent`
4. **Progress**: "Calling search...", "Generating response..." indicators
5. **Metadata**: Confidence scores and source references

See [app/README.md](../app/README.md) for chat client documentation.
