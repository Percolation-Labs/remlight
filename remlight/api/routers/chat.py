"""Chat completions router - OpenAI-compatible chat endpoint.

Provides:
- POST /chat/completions/{session_id} - OpenAI-compatible chat completions with session (preferred)
- POST /chat/completions - OpenAI-compatible chat completions (session via X-Session-Id header)
- Multi-agent support via X-Agent-Schema header

Uses the canonical AgentAdapter for all agent execution.
"""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from remlight.agentic.adapter import AgentAdapter, print_sse
from remlight.agentic.agent_schema import AgentSchema
from remlight.models.entities import Message, Session
from remlight.services.repository import Repository

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    """Single chat message."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None


def _get_default_schema() -> dict[str, Any]:
    """Default agent schema for chat."""
    return {
        "type": "object",
        "description": """You are a helpful assistant with access to a knowledge base.
Use the search tool to find relevant information before answering.
Use action(type='observation', payload={confidence, sources}) to record metadata.""",
        "properties": {
            "answer": {"type": "string", "description": "Your response"},
        },
        "required": ["answer"],
        "json_schema_extra": {
            "kind": "agent",
            "name": "default-agent",
            "version": "1.0.0",
            "structured_output": False,
            "tools": [
                {"name": "search"},
                {"name": "action"},
            ],
        },
    }


def _is_simulator_agent(schema_uri: str | None) -> bool:
    """Check if this is the special simulator agent."""
    return schema_uri == "rem-simulator"


async def _stream_simulator_sse(prompt: str, model: str):
    """Stream simulator SSE events (bypasses LLM)."""
    import json
    import time
    import uuid

    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Simulate a response
    content = f"[Simulator] Received: {prompt}"

    # First chunk with role
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': content}, 'finish_reason': None}]})}\n\n"

    # Final chunk
    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/completions/{session_id}")
@router.post("/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
    session_id: str | None = None,
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports streaming SSE responses with session persistence.

    Session ID can be provided in two ways (URL parameter takes precedence):
    - URL path: POST /chat/completions/{session_id} (preferred)
    - Header: X-Session-Id

    Other headers:
    - X-User-Id: User identifier
    - X-Agent-Schema: Agent schema name or path

    The endpoint automatically:
    1. Loads session history from the database (if session_id provided)
    2. Saves user message before processing
    3. Saves assistant response after streaming completes
    4. Persists tool calls for context reconstruction
    """
    # Extract context from headers
    user_id = req.headers.get("x-user-id", "anonymous")
    effective_session_id = session_id or req.headers.get("x-session-id")
    schema_uri = req.headers.get("x-agent-schema")

    # Handle simulator agent specially - bypass LLM entirely
    if _is_simulator_agent(schema_uri):
        user_messages = [m for m in request.messages if m.role == "user"]
        prompt = user_messages[-1].content if user_messages else "test all"

        if request.stream:
            return StreamingResponse(
                _stream_simulator_sse(prompt=prompt, model="rem-simulator"),
                media_type="text/event-stream",
            )
        else:
            return {
                "id": "chatcmpl-simulator",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Simulator response."}, "finish_reason": "stop"}],
            }

    # Load agent schema
    if schema_uri:
        try:
            schema = AgentSchema.load(schema_uri)
        except Exception:
            schema = AgentSchema.model_validate(_get_default_schema())
    else:
        schema = AgentSchema.model_validate(_get_default_schema())

    # Build prompt from most recent user message
    user_messages = [m for m in request.messages if m.role == "user"]
    prompt = user_messages[-1].content if user_messages else (request.messages[-1].content if request.messages else "")

    # Load message history and save user message
    message_history = None
    session_uuid = None

    if effective_session_id:
        try:
            session_uuid = UUID(effective_session_id)

            # Ensure session exists
            await Repository(Session).upsert(Session(id=session_uuid))

            # Load history
            message_repo = Repository(Message)
            existing_messages = await message_repo.find({"session_id": session_uuid})
            if existing_messages:
                message_history = existing_messages

            # Save user message before streaming
            user_msg = Message(role="user", content=prompt, session_id=session_uuid)
            await message_repo.upsert(user_msg)
        except Exception:
            pass

    # Build adapter options
    adapter_options = {}
    if request.model:
        adapter_options["model"] = request.model
    if request.temperature is not None:
        adapter_options["temperature"] = request.temperature

    adapter = AgentAdapter(schema, **adapter_options)

    if request.stream:
        async def stream_with_save():
            """Stream SSE and save messages after completion."""
            messages_to_save = []

            async with adapter.run_stream(prompt, message_history=message_history) as result:
                async for event in result.stream_openai_sse():
                    yield event

                # Get messages (excludes user message since we already saved it)
                all_msgs = result.to_messages(session_uuid)
                # Filter out user message (already saved)
                messages_to_save = [m for m in all_msgs if m.role != "user"]

            # Save after streaming completes
            if session_uuid and messages_to_save:
                try:
                    await Repository(Message).upsert(messages_to_save)
                except Exception:
                    pass

        return StreamingResponse(
            stream_with_save(),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming: run and return
        async with adapter.run_stream(prompt, message_history=message_history) as result:
            # Consume stream to get final result
            async for _ in result.stream_openai_sse():
                pass

            messages = result.to_messages(session_uuid)

            # Save messages (excluding user which was already saved)
            if session_uuid:
                try:
                    msgs_to_save = [m for m in messages if m.role != "user"]
                    if msgs_to_save:
                        await Repository(Message).upsert(msgs_to_save)
                except Exception:
                    pass

            # Get assistant content
            assistant_msgs = [m for m in messages if m.role == "assistant"]
            content = assistant_msgs[-1].content if assistant_msgs else ""

            return {
                "id": "chatcmpl-remlight",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            }
