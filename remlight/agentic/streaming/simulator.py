"""
REM Simulator - Synthetic SSE Event Generator for UI Testing
=============================================================

Generates sample SSE events demonstrating all supported content types.
Used for testing UI clients without requiring an actual LLM.

Supported test modes (based on prompt):
- "test all" - Full demonstration of all event types
- "test text" - Markdown text streaming
- "test tools" - Tool call lifecycle events
- "test actions" - Action events (observation, elicit, patch_schema)
- "test structured" - Structured output response
- "test error" - Recoverable error handling
- "help" - Show available test modes and examples

Events generated:
- progress: Step progress indicators
- content: OpenAI-format text chunks (streaming markdown)
- tool_call: Tool invocation lifecycle (started → executing → completed)
- action: Action events (observation, elicit, delegate, patch_schema)
- error: Recoverable error events
- done: Stream completion
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from remlight.agentic.streaming.events import (
    ActionEvent,
    DoneEvent,
    ErrorEvent,
    ProgressEvent,
    ToolCallEvent,
)
from remlight.agentic.streaming.formatters import (
    format_content_chunk,
    format_done,
    format_sse_event,
)
from remlight.agentic.streaming.state import StreamingState


# Sample markdown content for text streaming
SAMPLE_MARKDOWN = """# REM Simulator Response

This is a **simulated response** demonstrating markdown rendering.

## Features Tested

1. **Headers** - H1, H2, H3 levels
2. **Text formatting** - Bold, *italic*, `inline code`
3. **Lists** - Ordered and unordered
4. **Code blocks** - With syntax highlighting

## Code Example

```python
def hello_world():
    '''A simple function demonstrating code blocks.'''
    print("Hello from REM Simulator!")
    return {"status": "success", "message": "Test complete"}
```

## Table Example

| Event Type | Description |
|------------|-------------|
| progress | Step indicators |
| tool_call | Tool invocation |
| action | Agent actions |
| content | Text chunks |

---

*Simulation complete. All markdown elements rendered successfully.*
"""

# Structured output sample
SAMPLE_STRUCTURED_OUTPUT = {
    "analysis": {
        "summary": "This is a simulated structured output response",
        "sentiment": "positive",
        "confidence": 0.95,
        "key_points": [
            "Demonstrates Pydantic model output",
            "Shows nested object structure",
            "Includes typed fields",
        ],
    },
    "metadata": {
        "model": "rem-simulator",
        "timestamp": None,  # Will be set at runtime
        "tokens_used": 42,
    },
}

# Help text showing available modes
HELP_TEXT = """# REM Simulator - Help

The REM Simulator generates synthetic SSE events for testing UI clients without requiring an actual LLM.

## Available Test Modes

| Command | Description | Events Generated |
|---------|-------------|------------------|
| `help` | Show this help message | content |
| `test all` | Full demonstration (default) | All event types |
| `test text` | Markdown streaming | content, progress |
| `test tools` | Tool call lifecycle | tool_call (started→executing→completed) |
| `test actions` | Action events | action (observation, elicit, delegate, patch_schema) |
| `test structured` | Pydantic output | tool_call with structured result |
| `test error` | Error handling | error (recoverable) |

## Usage Examples

### CLI
```bash
# Show help
rem ask "help" --schema rem-simulator

# Test all event types
rem ask "test all" --schema rem-simulator

# Test specific event type
rem ask "test tools" --schema rem-simulator

# Combine multiple modes
rem ask "test tools test actions" --schema rem-simulator
```

### API
```bash
# Test markdown streaming
curl -N -X POST http://localhost:8000/api/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "X-Agent-Schema: rem-simulator" \\
  -d '{"messages": [{"role": "user", "content": "test text"}], "stream": true}'

# Test tool calls
curl -N -X POST http://localhost:8000/api/v1/chat/completions \\
  -H "X-Agent-Schema: rem-simulator" \\
  -d '{"messages": [{"role": "user", "content": "test tools"}], "stream": true}'
```

## Event Types Reference

| Event | Format | Purpose |
|-------|--------|---------|
| `progress` | `{step, total_steps, label}` | Step indicators |
| `content` | OpenAI delta format | Streaming text |
| `tool_call` | `{tool_name, tool_id, status, arguments?, result?}` | Tool invocation |
| `action` | `{action_type, payload}` | Agent actions |
| `error` | `{code, message, recoverable}` | Error handling |
| `done` | `{reason}` | Stream completion |

---
*Use this simulator to verify your UI correctly handles all event types.*
"""


def parse_test_mode(prompt: str) -> set[str]:
    """Parse prompt to determine which test modes to run."""
    prompt_lower = prompt.lower().strip()

    modes = set()

    # Check for help first
    if prompt_lower == "help" or prompt_lower.startswith("help ") or " help" in prompt_lower:
        return {"help"}

    if "test all" in prompt_lower or not any(
        x in prompt_lower for x in ["test text", "test tools", "test actions", "test structured", "test error"]
    ):
        # Default to all modes
        modes = {"text", "tools", "actions", "structured"}
    else:
        if "test text" in prompt_lower:
            modes.add("text")
        if "test tools" in prompt_lower:
            modes.add("tools")
        if "test actions" in prompt_lower:
            modes.add("actions")
        if "test structured" in prompt_lower:
            modes.add("structured")
        if "test error" in prompt_lower:
            modes.add("error")

    return modes


async def stream_simulator_sse(
    prompt: str,
    *,
    model: str = "rem-simulator",
    request_id: str | None = None,
    message_id: str | None = None,
    delay_ms: int = 50,
) -> AsyncGenerator[str, None]:
    """
    Generate synthetic SSE events for UI testing.

    Args:
        prompt: User prompt (controls test mode)
        model: Model name for metadata
        request_id: Correlation ID
        message_id: Pre-generated message ID
        delay_ms: Delay between chunks (simulates streaming)

    Yields:
        SSE-formatted strings
    """
    # Initialize state
    state = StreamingState.create(
        model=model,
        request_id=request_id or f"sim_{uuid.uuid4().hex[:8]}",
        message_id=message_id,
    )

    delay = delay_ms / 1000.0  # Convert to seconds
    modes = parse_test_mode(prompt)

    # =========================================================================
    # HELP MODE - Show available test options
    # =========================================================================
    if "help" in modes:
        yield format_sse_event(ProgressEvent(
            step=1,
            total_steps=1,
            label="Displaying help",
        ))

        # Stream help text
        chunk_size = 15
        for i in range(0, len(HELP_TEXT), chunk_size):
            chunk = HELP_TEXT[i:i + chunk_size]
            yield format_content_chunk(chunk, state)
            await asyncio.sleep(delay * 0.3)

        yield format_content_chunk("", state, finish_reason="stop")
        yield format_sse_event(DoneEvent(reason="stop"))
        yield format_done()
        return

    # =========================================================================
    # 1. PROGRESS EVENT - Start
    # =========================================================================
    yield format_sse_event(ProgressEvent(
        step=1,
        total_steps=4,
        label="Starting simulation",
    ))
    await asyncio.sleep(delay)

    # =========================================================================
    # 2. TEXT CONTENT - Streaming markdown
    # =========================================================================
    if "text" in modes:
        yield format_sse_event(ProgressEvent(
            step=2,
            total_steps=4,
            label="Generating text content",
        ))

        # Stream markdown content character by character (or in chunks)
        chunk_size = 10  # Characters per chunk
        for i in range(0, len(SAMPLE_MARKDOWN), chunk_size):
            chunk = SAMPLE_MARKDOWN[i:i + chunk_size]
            yield format_content_chunk(chunk, state)
            await asyncio.sleep(delay * 0.5)  # Faster for text

    # =========================================================================
    # 3. TOOL CALL EVENTS - Full lifecycle
    # =========================================================================
    if "tools" in modes:
        yield format_sse_event(ProgressEvent(
            step=2,
            total_steps=4,
            label="Demonstrating tool calls",
        ))
        await asyncio.sleep(delay)

        # Tool 1: search tool
        tool_id_1 = f"call_{uuid.uuid4().hex[:8]}"

        # Started
        yield format_sse_event(ToolCallEvent(
            tool_name="search",
            tool_id=tool_id_1,
            status="started",
            arguments=None,
        ))
        await asyncio.sleep(delay * 2)

        # Executing (with arguments)
        yield format_sse_event(ToolCallEvent(
            tool_name="search",
            tool_id=tool_id_1,
            status="executing",
            arguments={"query": "LOOKUP simulation-test", "limit": 10},
        ))
        await asyncio.sleep(delay * 3)

        # Completed (with result)
        yield format_sse_event(ToolCallEvent(
            tool_name="search",
            tool_id=tool_id_1,
            status="completed",
            arguments={"query": "LOOKUP simulation-test", "limit": 10},
            result={
                "status": "success",
                "results": [
                    {"key": "doc-1", "title": "Simulation Guide", "score": 0.95},
                    {"key": "doc-2", "title": "Testing Procedures", "score": 0.87},
                ],
                "total": 2,
            },
        ))
        await asyncio.sleep(delay)

        # Tool 2: action tool (demonstrates action event integration)
        tool_id_2 = f"call_{uuid.uuid4().hex[:8]}"

        yield format_sse_event(ToolCallEvent(
            tool_name="action",
            tool_id=tool_id_2,
            status="started",
            arguments=None,
        ))
        await asyncio.sleep(delay)

        yield format_sse_event(ToolCallEvent(
            tool_name="action",
            tool_id=tool_id_2,
            status="completed",
            arguments={"type": "observation", "payload": {"confidence": 0.9}},
            result={"_action_event": True, "action_type": "observation", "payload": {"confidence": 0.9}},
        ))
        await asyncio.sleep(delay)

    # =========================================================================
    # 4. ACTION EVENTS - Various action types
    # =========================================================================
    if "actions" in modes:
        yield format_sse_event(ProgressEvent(
            step=3,
            total_steps=4,
            label="Demonstrating action events",
        ))
        await asyncio.sleep(delay)

        # Observation action
        yield format_sse_event(ActionEvent(
            action_type="observation",
            payload={
                "confidence": 0.92,
                "sources": ["doc-1", "doc-2"],
                "session_name": "Simulator Test Session",
            },
        ))
        await asyncio.sleep(delay)

        # Elicit action (request user input)
        yield format_sse_event(ActionEvent(
            action_type="elicit",
            payload={
                "question": "Would you like to see more test events?",
                "options": ["Yes, show all", "No, that's enough"],
                "timeout_seconds": 30,
            },
        ))
        await asyncio.sleep(delay)

        # Delegate action (multi-agent)
        yield format_sse_event(ActionEvent(
            action_type="delegate",
            payload={
                "target_agent": "worker-agent",
                "task": "Process the simulation results",
                "context": {"simulation_id": "sim-001"},
            },
        ))
        await asyncio.sleep(delay)

        # Patch schema action (agent-builder style)
        yield format_sse_event(ActionEvent(
            action_type="patch_schema",
            payload={
                "patches": [
                    {"op": "replace", "path": "/description", "value": "Updated via simulation"},
                    {"op": "add", "path": "/metadata/tools/-", "value": {"name": "new_tool"}},
                ],
            },
        ))
        await asyncio.sleep(delay)

    # =========================================================================
    # 5. STRUCTURED OUTPUT - Pydantic model response
    # =========================================================================
    if "structured" in modes:
        yield format_sse_event(ProgressEvent(
            step=3,
            total_steps=4,
            label="Generating structured output",
        ))
        await asyncio.sleep(delay)

        # Emit structured output as a special tool_call event
        structured_tool_id = f"call_{uuid.uuid4().hex[:8]}"

        # Update timestamp
        output = SAMPLE_STRUCTURED_OUTPUT.copy()
        output["metadata"] = output["metadata"].copy()
        output["metadata"]["timestamp"] = time.time()

        yield format_sse_event(ToolCallEvent(
            tool_name="structured_output",
            tool_id=structured_tool_id,
            status="completed",
            arguments={},
            result=output,
        ))
        await asyncio.sleep(delay)

        # Also stream a text representation
        yield format_content_chunk(
            "\n\n## Structured Output Result\n\n```json\n" +
            json.dumps(output, indent=2) +
            "\n```\n",
            state,
        )
        await asyncio.sleep(delay)

    # =========================================================================
    # 6. ERROR EVENT - Recoverable error demonstration
    # =========================================================================
    if "error" in modes:
        yield format_sse_event(ProgressEvent(
            step=3,
            total_steps=4,
            label="Demonstrating error handling",
        ))
        await asyncio.sleep(delay)

        yield format_sse_event(ErrorEvent(
            code="simulated_error",
            message="This is a simulated recoverable error for testing",
            details={
                "error_type": "SimulationError",
                "retry_after_ms": 1000,
                "suggestion": "This error is intentional for testing purposes",
            },
            recoverable=True,
        ))
        await asyncio.sleep(delay)

        # Show recovery
        yield format_content_chunk(
            "\n\n> **Note:** The error above is simulated. The system recovered successfully.\n",
            state,
        )
        await asyncio.sleep(delay)

    # =========================================================================
    # 7. COMPLETION
    # =========================================================================
    yield format_sse_event(ProgressEvent(
        step=4,
        total_steps=4,
        label="Simulation complete",
        status="completed",
    ))

    # Final content chunk with finish_reason
    yield format_content_chunk("", state, finish_reason="stop")

    # Done event
    yield format_sse_event(DoneEvent(reason="stop"))

    # SSE terminator
    yield format_done()


async def stream_simulator_plain(
    prompt: str,
    *,
    delay_ms: int = 20,
) -> AsyncGenerator[str, None]:
    """
    Stream simulator output as plain text (for CLI).

    Args:
        prompt: User prompt (controls test mode)
        delay_ms: Delay between chunks

    Yields:
        Plain text strings
    """
    delay = delay_ms / 1000.0
    modes = parse_test_mode(prompt)

    # Help mode
    if "help" in modes:
        yield HELP_TEXT
        return

    yield "=== REM Simulator ===\n\n"

    if "text" in modes:
        # Stream markdown
        chunk_size = 20
        for i in range(0, len(SAMPLE_MARKDOWN), chunk_size):
            yield SAMPLE_MARKDOWN[i:i + chunk_size]
            await asyncio.sleep(delay)

    if "tools" in modes:
        yield "\n\n[Tool: search] Searching for 'simulation-test'...\n"
        await asyncio.sleep(delay * 5)
        yield "[Tool: search] Found 2 results\n"

    if "actions" in modes:
        yield "\n[Action: observation] confidence=0.92\n"
        yield "[Action: elicit] Requesting user input\n"
        yield "[Action: delegate] Delegating to worker-agent\n"
        yield "[Action: patch_schema] Applying 2 patches\n"

    if "structured" in modes:
        yield "\n\n## Structured Output:\n"
        yield json.dumps(SAMPLE_STRUCTURED_OUTPUT, indent=2)
        yield "\n"

    if "error" in modes:
        yield "\n[Error] Simulated recoverable error\n"
        yield "[Recovered] Continuing execution...\n"

    yield "\n\n=== Simulation Complete ===\n"


def is_simulator_agent(agent_schema: str | None) -> bool:
    """Check if the agent schema is the simulator."""
    return agent_schema == "rem-simulator"
