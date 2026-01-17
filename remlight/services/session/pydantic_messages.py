"""
Convert Stored Messages to Pydantic-AI Native Format
=====================================================

This module is the BRIDGE between our simplified storage format and pydantic-ai's
native message types. It enables multi-turn conversations by reconstructing
the message history that pydantic-ai expects.

THE MESSAGE FORMAT MISMATCH PROBLEM
-----------------------------------

We store messages in a SIMPLIFIED format (easy to store, easy to read):

    {"role": "user", "content": "Find documents about AI"}
    {"role": "assistant", "content": "I found several documents..."}
    {"role": "tool", "content": '{"results": [...]}', "tool_name": "search", ...}

But pydantic-ai (and LLM APIs) expect a SPECIFIC format:

    ModelRequest(parts=[UserPromptPart(content="Find documents about AI")])
    ModelResponse(parts=[
        ToolCallPart(tool_name="search", args={...}, tool_call_id="call_123"),
        TextPart(content="I found several documents...")
    ])
    ModelRequest(parts=[
        ToolReturnPart(tool_name="search", content={...}, tool_call_id="call_123")
    ])

KEY INSIGHT: SYNTHESIZING TOOL CALLS
------------------------------------

When we store a tool result, we only store the result (ToolReturnPart equivalent).
But LLM APIs require a MATCHING ToolCallPart for each ToolReturnPart.

Example - what we DON'T have stored:
    {"role": "assistant_tool_call", "tool_name": "search", "args": {...}}

Example - what we DO have stored:
    {"role": "tool", "tool_name": "search", "tool_call_id": "...", "tool_arguments": {...}}

Solution: We SYNTHESIZE the ToolCallPart from the tool message metadata:

    # From stored tool message:
    tool_msg = {"tool_name": "search", "tool_call_id": "call_123", "tool_arguments": {"query": "AI"}}

    # We synthesize:
    ToolCallPart(tool_name="search", args={"query": "AI"}, tool_call_id="call_123")

This enables proper conversation replay.


THE CONVERSION PIPELINE
-----------------------

    SessionMessageStore.load_session_messages()
                    │
                    ▼
    [{"role": "user", "content": "..."},
     {"role": "assistant", "content": "..."},
     {"role": "tool", "content": "...", "tool_name": "...", ...}]
                    │
                    │ session_to_pydantic_messages()
                    ▼
    [ModelRequest(parts=[SystemPromptPart(...)]),
     ModelRequest(parts=[UserPromptPart(...)]),
     ModelResponse(parts=[ToolCallPart(...), TextPart(...)]),
     ModelRequest(parts=[ToolReturnPart(...)])]
                    │
                    ▼
    agent.run(prompt, message_history=pydantic_messages)


SYSTEM PROMPT INJECTION
-----------------------

IMPORTANT: pydantic-ai only auto-adds system prompts when message_history is EMPTY.

When passing message_history, you MUST include the system prompt via this function:

    pydantic_messages = session_to_pydantic_messages(
        session_history=raw_history,
        system_prompt=agent.system_prompt  # REQUIRED for multi-turn
    )

The system prompt is injected as the FIRST message (SystemPromptPart).

This is important because:
1. Each turn may use a different agent (multi-agent sessions)
2. The system prompt defines the agent's behavior
3. Without it, the agent doesn't know its role
"""

import json
import re
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


def _sanitize_tool_name(tool_name: str) -> str:
    """
    Sanitize tool name for OpenAI API compatibility.

    OpenAI's function calling API has strict requirements on tool names.
    They must match the pattern: ^[a-zA-Z0-9_-]+$

    Invalid characters (spaces, dots, special chars) are replaced with underscores.

    Examples:
        "search.documents" → "search_documents"
        "my tool!" → "my_tool_"
        "get-user-profile" → "get-user-profile" (unchanged, dashes OK)

    Args:
        tool_name: Original tool name (may contain invalid chars)

    Returns:
        Sanitized tool name safe for OpenAI API
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)


def session_to_pydantic_messages(
    session_history: list[dict[str, Any]],
    system_prompt: str | None = None,
) -> list[ModelMessage]:
    """
    Convert stored session messages to pydantic-ai's native ModelMessage format.

    This is the CORE CONVERSION function that bridges our storage format
    to pydantic-ai's expected format. It handles:

    1. System prompt injection (for multi-turn conversations)
    2. User message conversion
    3. Assistant message with tool call synthesis
    4. Tool result conversion

    SYSTEM PROMPT INJECTION
    ----------------------
    CRITICAL: pydantic-ai only auto-adds system prompts when message_history is EMPTY.

    When message_history is provided, pydantic-ai assumes it's a complete conversation
    and doesn't add the system prompt. So we MUST inject it here.

    This is especially important for multi-agent sessions where different agents
    may handle different turns - each agent needs its own system prompt.

    TOOL CALL SYNTHESIS
    ------------------
    LLM APIs require ToolCallPart for each ToolReturnPart. We synthesize
    ToolCallPart from stored metadata:

        Stored: {"role": "tool", "tool_name": "search", "tool_call_id": "call_123",
                 "tool_arguments": {"query": "AI"}, "content": '{"results": [...]}'}

        Synthesized: ToolCallPart(tool_name="search", args={"query": "AI"},
                                  tool_call_id="call_123")

        Converted: ToolReturnPart(tool_name="search", content={"results": [...]},
                                  tool_call_id="call_123")

    MESSAGE GROUPING
    ---------------
    Tool messages following an assistant message are grouped together:

        assistant + tool + tool → ModelResponse(ToolCallPart, ToolCallPart, TextPart)
                                  + ModelRequest(ToolReturnPart, ToolReturnPart)

    This matches how LLM APIs expect the conversation flow.

    Args:
        session_history: Message dicts from SessionMessageStore.load_session_messages()
            Format: [{"role": "...", "content": "...", ...}, ...]
        system_prompt: Agent's system prompt (from schema.description).
            REQUIRED for multi-turn - without it, agent loses its identity.

    Returns:
        List of ModelMessage ready for agent.run(message_history=...).
        Types: ModelRequest (user input, tool returns) and ModelResponse (assistant output)

    Example:
        raw_history = await store.load_session_messages(session_id)
        pydantic_history = session_to_pydantic_messages(
            session_history=raw_history,
            system_prompt="You are a helpful assistant..."
        )
        result = await agent.run(prompt, message_history=pydantic_history)
    """
    messages: list[ModelMessage] = []

    # ==========================================================================
    # SYSTEM PROMPT INJECTION
    # ==========================================================================
    # Prepend system prompt as first message. This is REQUIRED for multi-turn
    # conversations because pydantic-ai won't auto-add it when history exists.
    # ==========================================================================
    if system_prompt:
        messages.append(ModelRequest(parts=[SystemPromptPart(content=system_prompt)]))

    i = 0
    while i < len(session_history):
        msg = session_history[i]
        role = msg.get("role", "")
        content = msg.get("content") or ""

        # ==========================================================================
        # USER MESSAGE
        # ==========================================================================
        # Simple conversion: user content → ModelRequest with UserPromptPart
        # ==========================================================================
        if role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))

        # ==========================================================================
        # ASSISTANT MESSAGE
        # ==========================================================================
        # Assistant messages are complex because they may have associated tool calls.
        # We need to:
        # 1. Look ahead for tool messages that follow this assistant message
        # 2. Synthesize ToolCallPart for each tool (what the model requested)
        # 3. Create ToolReturnPart for each tool (what the tool returned)
        # 4. Group all parts into proper ModelResponse and ModelRequest
        #
        # LLM API expectation:
        #   ModelResponse: [ToolCallPart, ToolCallPart, TextPart]  ← Model output
        #   ModelRequest:  [ToolReturnPart, ToolReturnPart]        ← Tool results
        # ==========================================================================
        elif role == "assistant":
            # Collect tool calls and returns by looking ahead
            tool_calls = []
            tool_returns = []

            # Look ahead for consecutive tool messages following this assistant
            j = i + 1
            while j < len(session_history) and session_history[j].get("role") == "tool":
                tool_msg = session_history[j]
                tool_name = tool_msg.get("tool_name", "unknown_tool")
                tool_call_id = tool_msg.get("tool_call_id", f"call_{j}")
                tool_content = tool_msg.get("content") or "{}"

                # =============================================================
                # TOOL ARGUMENTS RECOVERY
                # =============================================================
                # We need the arguments that were passed TO the tool.
                # Source priority:
                # 1. Explicit tool_arguments field (stored for parent calls)
                # 2. Parsed from content JSON (fallback for child calls)
                # =============================================================
                tool_arguments = tool_msg.get("tool_arguments")
                if tool_arguments is None and isinstance(tool_content, str):
                    try:
                        tool_arguments = json.loads(tool_content)
                    except json.JSONDecodeError:
                        tool_arguments = {}

                # Parse tool content as JSON for the result
                if isinstance(tool_content, str):
                    try:
                        tool_result = json.loads(tool_content)
                    except json.JSONDecodeError:
                        tool_result = {"raw": tool_content}
                else:
                    tool_result = tool_content

                # Sanitize tool name for OpenAI API compatibility
                safe_tool_name = _sanitize_tool_name(tool_name)

                # =============================================================
                # SYNTHESIZE TOOL CALL
                # =============================================================
                # This is the key insight: we create a ToolCallPart that matches
                # the ToolReturnPart. LLM APIs require this pairing.
                # =============================================================
                tool_calls.append(
                    ToolCallPart(
                        tool_name=safe_tool_name,
                        args=tool_arguments if tool_arguments else {},
                        tool_call_id=tool_call_id,
                    )
                )

                # Create corresponding ToolReturnPart (the actual result)
                tool_returns.append(
                    ToolReturnPart(
                        tool_name=safe_tool_name,
                        content=tool_result,
                        tool_call_id=tool_call_id,
                    )
                )

                j += 1

            # =============================================================
            # BUILD MODEL RESPONSE
            # =============================================================
            # ModelResponse represents what the model produced:
            # - ToolCallPart(s): Tools the model wanted to call
            # - TextPart: Any text content from the model
            # =============================================================
            response_parts = []
            response_parts.extend(tool_calls)  # Tool calls first
            if content:
                response_parts.append(TextPart(content=content))  # Text last

            if response_parts:
                messages.append(
                    ModelResponse(
                        parts=response_parts,
                        model_name="recovered",  # Marker that this is reconstructed
                    )
                )

            # =============================================================
            # BUILD TOOL RETURNS REQUEST
            # =============================================================
            # ModelRequest with ToolReturnPart(s): The tool execution results
            # This follows the ModelResponse in the conversation flow
            # =============================================================
            if tool_returns:
                messages.append(ModelRequest(parts=tool_returns))

            # Skip the tool messages we just processed
            # (j is now pointing past the last tool message)
            i = j - 1

        # ==========================================================================
        # ORPHAN TOOL MESSAGE
        # ==========================================================================
        # Tool messages without a preceding assistant message.
        # This can happen in edge cases or data migration scenarios.
        #
        # We handle it by synthesizing BOTH the ToolCallPart AND ToolReturnPart.
        # This ensures the conversation is valid for LLM replay.
        # ==========================================================================
        elif role == "tool":
            tool_name = msg.get("tool_name", "unknown_tool")
            tool_call_id = msg.get("tool_call_id", f"call_{i}")
            tool_content = msg.get("content") or "{}"

            # Recover tool arguments
            tool_arguments = msg.get("tool_arguments")
            if tool_arguments is None and isinstance(tool_content, str):
                try:
                    tool_arguments = json.loads(tool_content)
                except json.JSONDecodeError:
                    tool_arguments = {}

            # Parse tool result
            if isinstance(tool_content, str):
                try:
                    tool_result = json.loads(tool_content)
                except json.JSONDecodeError:
                    tool_result = {"raw": tool_content}
            else:
                tool_result = tool_content

            safe_tool_name = _sanitize_tool_name(tool_name)

            # Synthesize ModelResponse with ToolCallPart (what model requested)
            messages.append(
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=safe_tool_name,
                            args=tool_arguments if tool_arguments else {},
                            tool_call_id=tool_call_id,
                        )
                    ],
                    model_name="recovered",
                )
            )

            # Add ModelRequest with ToolReturnPart (what tool returned)
            messages.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=safe_tool_name,
                            content=tool_result,
                            tool_call_id=tool_call_id,
                        )
                    ]
                )
            )

        elif role == "system":
            # Skip system messages - we inject via system_prompt parameter
            # Old system messages in history are ignored
            pass

        i += 1

    return messages


def audit_session_history(
    session_id: str,
    agent_name: str,
    prompt: str,
    raw_session_history: list[dict[str, Any]],
    pydantic_messages_count: int,
) -> None:
    """
    Dump session history to a YAML file for debugging multi-turn issues.

    DEBUG TOOL
    ----------
    This function is for debugging session/context issues. When enabled:
    - Each agent invocation appends an entry to a YAML file
    - File contains the raw session history before conversion
    - Helps debug "why did the agent not see previous messages?"

    ENABLING
    --------
    Set environment variable: DEBUG_AUDIT_SESSION=1

    Optionally set: DEBUG_AUDIT_DIR=/path/to/dir (default: /tmp)

    OUTPUT FORMAT
    ------------
    /tmp/{session_id}.yaml:
        session_id: "sess-abc123"
        invocations:
          - timestamp: "2024-01-15T10:30:00Z"
            agent_name: "query-agent"
            prompt: "What is AI?"
            raw_history_count: 0
            pydantic_messages_count: 1
            raw_session_history: []

          - timestamp: "2024-01-15T10:31:00Z"
            agent_name: "query-agent"
            prompt: "Tell me more"
            raw_history_count: 2
            pydantic_messages_count: 3
            raw_session_history:
              - role: user
                content: "What is AI?"
              - role: assistant
                content: "AI is..."

    SILENT FAILURES
    --------------
    All errors are silently caught. This is a debug tool - it should
    never break production flow.

    Args:
        session_id: Session being debugged
        agent_name: Agent handling this invocation
        prompt: Current user prompt
        raw_session_history: What was loaded from DB
        pydantic_messages_count: How many pydantic messages were created
    """
    import os

    if not os.environ.get("DEBUG_AUDIT_SESSION"):
        return

    try:
        import yaml
        from pathlib import Path
        from datetime import datetime, timezone

        audit_dir = Path(os.environ.get("DEBUG_AUDIT_DIR", "/tmp"))
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / f"{session_id}.yaml"

        # Build audit entry
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_name": agent_name,
            "prompt": prompt,
            "raw_history_count": len(raw_session_history),
            "pydantic_messages_count": pydantic_messages_count,
            "raw_session_history": raw_session_history,
        }

        # Load existing audit data (if file exists)
        existing_data: dict[str, Any] = {"session_id": session_id, "invocations": []}
        if audit_file.exists():
            with open(audit_file) as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    existing_data = {
                        "session_id": loaded.get("session_id", session_id),
                        "invocations": loaded.get("invocations", []),
                    }

        # Append new entry
        existing_data["invocations"].append(entry)

        # Write back
        with open(audit_file, "w") as f:
            yaml.dump(existing_data, f, default_flow_style=False, allow_unicode=True)
    except Exception:
        pass  # Debug tool - never fail
