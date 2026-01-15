"""Unified agent runner for API and CLI.

Provides a single entry point for running agents with:
- Session history loading
- Context management
- Message persistence
- Streaming (SSE or plain text)

Both API and CLI use these functions to ensure consistent behavior.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from loguru import logger

from remlight.agentic.context import AgentContext
from remlight.agentic.streaming import stream_plain, stream_sse
from remlight.settings import settings


async def run_streaming(
    agent,
    prompt: str,
    *,
    context: AgentContext | None = None,
    model: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    agent_schema: dict | None = None,
    agent_schema_name: str | None = None,
    request_id: str | None = None,
    persist_messages: bool = True,
    output_format: str = "sse",  # "sse" or "plain"
) -> AsyncGenerator[str, None]:
    """
    Run agent with streaming output.

    This is the unified entry point for both API and CLI streaming.
    Handles:
    1. Loading session history (if session_id provided)
    2. Saving user message (before streaming)
    3. Streaming agent response
    4. Saving assistant response (after streaming)

    Args:
        agent: Pydantic AI Agent instance
        prompt: User prompt
        context: Optional AgentContext
        model: Model name for metadata
        session_id: Session ID for persistence and history loading
        user_id: User ID for persistence
        agent_schema: Agent schema name for metadata
        request_id: Optional request ID
        persist_messages: Whether to persist messages (default: True)
        output_format: "sse" for API or "plain" for CLI

    Yields:
        Formatted strings (SSE or plain text depending on output_format)
    """
    # Get session_id from context if not provided
    if context and not session_id:
        session_id = context.session_id
    if context and not user_id:
        user_id = context.user_id

    # Load session history if session_id provided
    message_history = None
    if session_id and settings.postgres.enabled:
        message_history = await _load_session_history(
            session_id=session_id,
            user_id=user_id,
            agent=agent,
            agent_schema=agent_schema,
        )

    # Save user message BEFORE streaming
    if persist_messages and session_id and settings.postgres.enabled:
        await _save_user_message(
            session_id=session_id,
            user_id=user_id,
            content=prompt,
        )

    # Stream based on format
    if output_format == "plain":
        async for chunk in stream_plain(
            agent=agent,
            prompt=prompt,
            message_history=message_history,
        ):
            yield chunk
        return

    # SSE streaming with persistence
    message_id = str(uuid.uuid4())
    accumulated_content: list[str] = []
    tool_calls: list[dict] = []

    async for chunk in stream_sse(
        agent=agent,
        prompt=prompt,
        model=model,
        request_id=request_id,
        message_id=message_id,
        session_id=session_id,
        agent_schema=agent_schema,
        context=context,
        message_history=message_history,
        tool_calls_out=tool_calls,
    ):
        yield chunk

        # Extract content from SSE chunks for persistence
        if chunk.startswith("data: ") and not chunk.startswith("data: [DONE]"):
            try:
                data_str = chunk[6:].strip()
                if data_str:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            accumulated_content.append(content)
            except (json.JSONDecodeError, KeyError, IndexError):
                pass

    # Save messages AFTER streaming completes
    if persist_messages and session_id and settings.postgres.enabled:
        await _save_assistant_response(
            session_id=session_id,
            user_id=user_id,
            message_id=message_id,
            accumulated_content=accumulated_content,
            tool_calls=tool_calls,
        )


async def run_sync(
    agent,
    prompt: str,
    *,
    context: AgentContext | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    agent_schema: dict | None = None,
    persist_messages: bool = True,
) -> Any:
    """
    Run agent synchronously (non-streaming).

    Unified entry point for non-streaming execution.
    Handles session history and message persistence.

    Args:
        agent: Pydantic AI Agent instance
        prompt: User prompt
        context: Optional AgentContext
        session_id: Session ID for persistence
        user_id: User ID for persistence
        agent_schema: Agent schema dict for system prompt extraction
        persist_messages: Whether to persist messages

    Returns:
        Agent result (pydantic-ai RunResult)
    """
    # Get session_id from context if not provided
    if context and not session_id:
        session_id = context.session_id
    if context and not user_id:
        user_id = context.user_id

    # Load session history
    message_history = None
    if session_id and settings.postgres.enabled:
        message_history = await _load_session_history(
            session_id=session_id,
            user_id=user_id,
            agent=agent,
            agent_schema=agent_schema,
        )

    # Save user message
    if persist_messages and session_id and settings.postgres.enabled:
        await _save_user_message(
            session_id=session_id,
            user_id=user_id,
            content=prompt,
        )

    # Run agent
    run_kwargs = {}
    if message_history:
        run_kwargs["message_history"] = message_history

    result = await agent.run(prompt, **run_kwargs)

    # Save assistant response
    if persist_messages and session_id and settings.postgres.enabled:
        output_str = str(result.output) if hasattr(result, "output") else str(result)
        await _save_assistant_response(
            session_id=session_id,
            user_id=user_id,
            message_id=str(uuid.uuid4()),
            accumulated_content=[output_str],
            tool_calls=[],
        )

    return result


# =============================================================================
# Internal helpers
# =============================================================================


async def _load_session_history(
    session_id: str,
    user_id: str | None,
    agent,
    agent_schema: dict | None = None,
) -> list | None:
    """
    Load and convert session history to pydantic-ai format.

    Multi-Agent System Prompt Handling:
    - Sessions CAN be shared between agents in multi-agent orchestration
    - System prompt is loaded from CURRENT agent's schema (not stored messages)
    - This enables each agent to use its own prompt with shared history
    - System prompt injection happens at reconstruction time, not storage time
    """
    try:
        from remlight.services.session import (
            SessionMessageStore,
            session_to_pydantic_messages,
        )

        store = SessionMessageStore(user_id=user_id or "anonymous")
        raw_history = await store.load_session_messages(
            session_id=session_id,
            user_id=user_id,
            compress_on_load=True,
        )

        if not raw_history:
            return None

        # Get system prompt from schema (preferred) or fall back to agent
        from remlight.agentic.schema import get_system_prompt as extract_system_prompt

        system_prompt = None
        if agent_schema:
            system_prompt = extract_system_prompt(agent_schema)
        elif hasattr(agent, "_system_prompts") and agent._system_prompts:
            system_prompt = agent._system_prompts[0]

        message_history = session_to_pydantic_messages(
            raw_history,
            system_prompt=system_prompt,
        )

        logger.debug(
            f"Loaded {len(raw_history)} messages -> {len(message_history)} pydantic messages"
        )
        return message_history

    except Exception as e:
        logger.warning(f"Failed to load session history: {e}")
        return None


async def _save_user_message(
    session_id: str,
    user_id: str | None,
    content: str,
) -> None:
    """
    Save user message to database.

    Also injects a context message with current date and user key
    if this is the first message in a session (for profile lookup).
    """
    try:
        from remlight.services.session import SessionMessageStore

        timestamp = datetime.now(timezone.utc)
        messages_to_store = []

        # Build user context hint (date + user key for profile lookup)
        context_parts = [f"Date: {timestamp.strftime('%Y-%m-%d')}"]
        if user_id:
            context_parts.append(f"User: {user_id}")
            context_parts.append("Hint: Use MCP resource user://profile for user context.")

        # Add context as system message metadata (not a separate message)
        user_msg = {
            "role": "user",
            "content": content,
            "timestamp": timestamp.isoformat(),
            "metadata": {
                "context_hint": "\n".join(context_parts),
            },
        }
        messages_to_store.append(user_msg)

        store = SessionMessageStore(user_id=user_id or "anonymous")
        await store.store_session_messages(
            session_id=session_id,
            messages=messages_to_store,
            user_id=user_id,
            compress=False,
        )
        logger.debug(f"Saved user message for session {session_id}")

    except Exception as e:
        logger.warning(f"Failed to save user message: {e}")


async def _save_assistant_response(
    session_id: str,
    user_id: str | None,
    message_id: str,
    accumulated_content: list[str],
    tool_calls: list[dict],
) -> None:
    """
    Save assistant response and tool calls to database.

    Stores:
    - tool_call: The tool invocation (name + arguments) - NOT the result
    - assistant: The agent's text response (includes note-taking/decisions)

    Tool responses are NOT stored - the agent captures relevant insights
    in its assistant response via a note-taking pattern.
    """
    try:
        from remlight.services.session import SessionMessageStore

        timestamp = datetime.now(timezone.utc).isoformat()
        messages_to_store = []

        # Store tool_call messages (NOT tool_response)
        # Only store the invocation (name + arguments), not the result
        for tool_call in tool_calls:
            if not tool_call:
                continue
            tool_call_message = {
                "role": "tool_call",  # Changed from "tool" to "tool_call"
                "content": json.dumps({
                    "tool_name": tool_call.get("tool_name"),
                    "arguments": tool_call.get("arguments"),
                }, default=str),
                "timestamp": timestamp,
                "tool_call_id": tool_call.get("tool_id"),
                "tool_name": tool_call.get("tool_name"),
                # Note: NOT storing tool_call.get("result") - agent uses note-taking pattern
            }
            messages_to_store.append(tool_call_message)

        # Store assistant text response (includes agent's notes/decisions)
        full_content = "".join(accumulated_content) if accumulated_content else None

        # Fallback to text_response from tool results (for backward compat)
        if not full_content:
            for tool_call in tool_calls:
                if not tool_call:
                    continue
                result = tool_call.get("result")
                if isinstance(result, dict) and result.get("text_response"):
                    text_response = result["text_response"]
                    if text_response and str(text_response).strip():
                        full_content = str(text_response)
                        break

        if full_content:
            assistant_message = {
                "id": message_id,
                "role": "assistant",
                "content": full_content,
                "timestamp": timestamp,
            }
            messages_to_store.append(assistant_message)

        if messages_to_store:
            store = SessionMessageStore(user_id=user_id or "anonymous")
            await store.store_session_messages(
                session_id=session_id,
                messages=messages_to_store,
                user_id=user_id,
                compress=False,
            )
            logger.debug(
                f"Saved {len(messages_to_store)} messages for session {session_id}"
            )

    except Exception as e:
        logger.warning(f"Failed to save assistant response: {e}")
