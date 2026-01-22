---
entity_key: sse-events
title: SSE Event Protocol
tags: [reference, sse, streaming, protocols]
---

# SSE Event Protocol

REM streams agent responses using Server-Sent Events (SSE), compatible with OpenAI's streaming format.

## Event Types

| Event | Purpose | When Emitted |
|-------|---------|--------------|
| `progress` | Step indicators | Before major operations |
| `content` | Text chunks (OpenAI format) | During text generation |
| `tool_call` | Tool invocation lifecycle | Tool start/complete |
| `action` | Agent actions (elicit, delegate) | When agent performs actions |
| `metadata` | Observation metadata | When agent records observations |
| `error` | Error events | On recoverable errors |
| `done` | Stream completion | When response finishes |

## Wire Format

SSE events use two formats:

**Custom events** (progress, tool_call, action, metadata, error, done):
```text
event: {type}
data: {json}

```

**Content chunks** (OpenAI-compatible):
```text
data: {json}

```

**Stream terminator**:
```text
data: [DONE]

```

---

## Event Samples

### Progress Event

Indicates current step in multi-step processing:

```text
event: progress
data: {"type":"progress","step":1,"total_steps":3,"label":"Searching knowledge base","status":"in_progress"}

```

### Content Chunk (OpenAI Format)

Streaming text content:

```text
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706140800,"model":"openai:gpt-4.1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706140800,"model":"openai:gpt-4.1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706140800,"model":"openai:gpt-4.1","choices":[{"index":0,"delta":{"content":" world!"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1706140800,"model":"openai:gpt-4.1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

```

### Tool Call Events

Full lifecycle: `started` → `executing` → `completed`

**Started** (tool invoked):
```text
event: tool_call
data: {"type":"tool_call","tool_name":"search","tool_id":"call_abc123","status":"started","arguments":null}

```

**Executing** (with arguments):
```text
event: tool_call
data: {"type":"tool_call","tool_name":"search","tool_id":"call_abc123","status":"executing","arguments":{"query":"LOOKUP neural-networks","limit":10}}

```

**Completed** (with result):
```text
event: tool_call
data: {"type":"tool_call","tool_name":"search","tool_id":"call_abc123","status":"completed","arguments":{"query":"LOOKUP neural-networks","limit":10},"result":{"status":"success","results":[{"key":"doc-1","title":"Neural Networks Guide","score":0.95}],"total":1}}

```

### Action Events

Agent actions other than tool calls:

**Elicit** (request user input):
```text
event: action
data: {"type":"action","action_type":"elicit","payload":{"question":"Would you like more details?","options":["Yes","No"],"timeout_seconds":30}}

```

**Delegate** (multi-agent):
```text
event: action
data: {"type":"action","action_type":"delegate","payload":{"target_agent":"sentiment-analyzer","task":"Analyze the sentiment","context":{"document_id":"doc-123"}}}

```

**Patch Schema** (agent-builder):
```text
event: action
data: {"type":"action","action_type":"patch_schema","payload":{"patches":[{"op":"replace","path":"/description","value":"Updated description"}]}}

```

### Metadata Event

Observation data from agents:

```text
event: metadata
data: {"type":"metadata","message_id":"msg-456","session_id":"sess-123","agent_schema":"query-agent","confidence":0.92,"sources":["doc-1","doc-2"],"session_name":"Research Session"}

```

### Error Event

Recoverable errors:

```text
event: error
data: {"type":"error","code":"rate_limit_exceeded","message":"Rate limit exceeded, retrying...","details":{"retry_after_ms":1000},"recoverable":true}

```

### Done Event

Stream completion:

```text
event: done
data: {"type":"done","reason":"stop"}

data: [DONE]

```

---

## Complete Stream Example

A typical response stream:

```text
event: progress
data: {"type":"progress","step":1,"total_steps":3,"label":"Processing query","status":"in_progress"}

event: tool_call
data: {"type":"tool_call","tool_name":"search","tool_id":"call_001","status":"started"}

event: tool_call
data: {"type":"tool_call","tool_name":"search","tool_id":"call_001","status":"completed","arguments":{"query":"SEARCH ML"},"result":{"results":[{"title":"ML Guide"}]}}

event: progress
data: {"type":"progress","step":2,"total_steps":3,"label":"Generating response","status":"in_progress"}

data: {"id":"req-001","object":"chat.completion.chunk","choices":[{"delta":{"role":"assistant"}}]}

data: {"id":"req-001","object":"chat.completion.chunk","choices":[{"delta":{"content":"Based on my search, "}}]}

data: {"id":"req-001","object":"chat.completion.chunk","choices":[{"delta":{"content":"machine learning is..."}}]}

event: metadata
data: {"type":"metadata","confidence":0.88,"sources":["ml-guide"]}

data: {"id":"req-001","object":"chat.completion.chunk","choices":[{"delta":{},"finish_reason":"stop"}]}

event: progress
data: {"type":"progress","step":3,"total_steps":3,"label":"Complete","status":"completed"}

event: done
data: {"type":"done","reason":"stop"}

data: [DONE]

```

---

## Testing with rem-simulator

Use the `rem-simulator` agent to generate test events:

```bash
# Show help
rem ask "help" --schema rem-simulator

# Test all event types
rem ask "test all" --schema rem-simulator

# Test specific types
rem ask "test tools" --schema rem-simulator
rem ask "test actions" --schema rem-simulator
rem ask "test text" --schema rem-simulator
```

## See also

- `REM LOOKUP headers` - HTTP header protocol
- `REM LOOKUP messages` - Message persistence
- `REM LOOKUP multi-agent` - Multi-agent streaming
