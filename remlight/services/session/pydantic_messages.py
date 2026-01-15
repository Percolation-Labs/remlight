"""Convert stored session messages to pydantic-ai native message format.

This module enables proper conversation history replay by converting our simplified
storage format into pydantic-ai's native ModelRequest/ModelResponse types.

Key insight: When we store tool results, we only store the result (ToolReturnPart).
But LLM APIs require matching ToolCallPart for each ToolReturnPart. So we synthesize
the ToolCallPart from stored metadata (tool_name, tool_call_id) and arguments.

Storage format (our simplified format):
    {"role": "user", "content": "..."}
    {"role": "assistant", "content": "..."}
    {"role": "tool", "content": "{...}", "tool_name": "...", "tool_call_id": "...", "tool_arguments": {...}}

Pydantic-ai format (what the LLM expects):
    ModelRequest(parts=[UserPromptPart(content="...")])
    ModelResponse(parts=[TextPart(content="..."), ToolCallPart(...)])  # Call
    ModelRequest(parts=[ToolReturnPart(...)])  # Result
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
    """Sanitize tool name for OpenAI API compatibility.

    OpenAI requires tool names to match pattern: ^[a-zA-Z0-9_-]+$
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name)


def session_to_pydantic_messages(
    session_history: list[dict[str, Any]],
    system_prompt: str | None = None,
) -> list[ModelMessage]:
    """Convert stored session messages to pydantic-ai ModelMessage format.

    IMPORTANT: pydantic-ai only auto-adds system prompts when message_history is empty.
    When passing message_history to agent.run(), you MUST include the system prompt
    via the system_prompt parameter here.

    Args:
        session_history: List of message dicts from SessionMessageStore.load_session_messages()
        system_prompt: The agent's system prompt (from schema description). Required
            for proper agent behavior on subsequent turns.

    Returns:
        List of ModelMessage (ModelRequest | ModelResponse) ready for agent.run(message_history=...)
    """
    messages: list[ModelMessage] = []

    # Prepend agent's system prompt if provided
    if system_prompt:
        messages.append(ModelRequest(parts=[SystemPromptPart(content=system_prompt)]))

    i = 0
    while i < len(session_history):
        msg = session_history[i]
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))

        elif role == "assistant":
            # Check if there are following tool messages that should be grouped
            tool_calls = []
            tool_returns = []

            # Look ahead for tool messages that follow this assistant message
            j = i + 1
            while j < len(session_history) and session_history[j].get("role") == "tool":
                tool_msg = session_history[j]
                tool_name = tool_msg.get("tool_name", "unknown_tool")
                tool_call_id = tool_msg.get("tool_call_id", f"call_{j}")
                tool_content = tool_msg.get("content") or "{}"

                # tool_arguments: prefer explicit field, fallback to parsing content
                tool_arguments = tool_msg.get("tool_arguments")
                if tool_arguments is None and isinstance(tool_content, str):
                    try:
                        tool_arguments = json.loads(tool_content)
                    except json.JSONDecodeError:
                        tool_arguments = {}

                # Parse tool content for result
                if isinstance(tool_content, str):
                    try:
                        tool_result = json.loads(tool_content)
                    except json.JSONDecodeError:
                        tool_result = {"raw": tool_content}
                else:
                    tool_result = tool_content

                safe_tool_name = _sanitize_tool_name(tool_name)

                # Synthesize ToolCallPart (what the model "called")
                tool_calls.append(
                    ToolCallPart(
                        tool_name=safe_tool_name,
                        args=tool_arguments if tool_arguments else {},
                        tool_call_id=tool_call_id,
                    )
                )

                # Create ToolReturnPart (the actual result)
                tool_returns.append(
                    ToolReturnPart(
                        tool_name=safe_tool_name,
                        content=tool_result,
                        tool_call_id=tool_call_id,
                    )
                )

                j += 1

            # Build the assistant's ModelResponse
            response_parts = []
            response_parts.extend(tool_calls)
            if content:
                response_parts.append(TextPart(content=content))

            if response_parts:
                messages.append(
                    ModelResponse(
                        parts=response_parts,
                        model_name="recovered",
                    )
                )

            # Add tool returns as ModelRequest
            if tool_returns:
                messages.append(ModelRequest(parts=tool_returns))

            # Skip the tool messages we just processed
            i = j - 1

        elif role == "tool":
            # Orphan tool message (no preceding assistant) - synthesize both parts
            tool_name = msg.get("tool_name", "unknown_tool")
            tool_call_id = msg.get("tool_call_id", f"call_{i}")
            tool_content = msg.get("content") or "{}"

            tool_arguments = msg.get("tool_arguments")
            if tool_arguments is None and isinstance(tool_content, str):
                try:
                    tool_arguments = json.loads(tool_content)
                except json.JSONDecodeError:
                    tool_arguments = {}

            if isinstance(tool_content, str):
                try:
                    tool_result = json.loads(tool_content)
                except json.JSONDecodeError:
                    tool_result = {"raw": tool_content}
            else:
                tool_result = tool_content

            safe_tool_name = _sanitize_tool_name(tool_name)

            # Synthesize the tool call (ModelResponse with ToolCallPart)
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

            # Add the tool return (ModelRequest with ToolReturnPart)
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
            # Skip system messages - handled by Agent.system_prompt
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
    Dump session history to a YAML file for debugging.

    Only runs when DEBUG_AUDIT_SESSION env var is set.
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

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent_name": agent_name,
            "prompt": prompt,
            "raw_history_count": len(raw_session_history),
            "pydantic_messages_count": pydantic_messages_count,
            "raw_session_history": raw_session_history,
        }

        existing_data: dict[str, Any] = {"session_id": session_id, "invocations": []}
        if audit_file.exists():
            with open(audit_file) as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    existing_data = {
                        "session_id": loaded.get("session_id", session_id),
                        "invocations": loaded.get("invocations", []),
                    }

        existing_data["invocations"].append(entry)

        with open(audit_file, "w") as f:
            yaml.dump(existing_data, f, default_flow_style=False, allow_unicode=True)
    except Exception:
        pass
