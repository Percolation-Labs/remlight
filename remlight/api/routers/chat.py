"""Chat completions router - OpenAI-compatible chat endpoint.

Provides:
- POST /chat/completions - OpenAI-compatible chat completions with streaming
- Session persistence with X-Session-Id header
- Multi-agent support via X-Agent-Schema header
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from remlight.agentic import AgentContext, create_agent, schema_from_yaml
from remlight.agentic.streaming import save_user_message, stream_sse_with_save

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


def get_default_agent_schema() -> dict[str, Any]:
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
            "tools": [
                {"name": "search"},
                {"name": "action"},
            ],
        },
    }


@router.post("/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    req: Request,
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports streaming SSE responses with session persistence.

    Headers:
    - X-User-Id: User identifier
    - X-Session-Id: Session identifier (enables multi-turn context)
    - X-Agent-Schema: Agent schema name or path

    The endpoint automatically:
    1. Loads session history from the database (if X-Session-Id provided)
    2. Saves user message before processing
    3. Saves assistant response after streaming completes
    4. Persists tool calls for context reconstruction
    """
    from remlight.api.mcp_main import get_mcp_tools

    # Get context from headers with user profile hint
    context = await AgentContext.from_headers_with_profile(dict(req.headers))

    # Get agent schema from header or use default
    # schema_uri can be an agent name (e.g., "orchestrator-agent") or a path
    schema_uri = req.headers.get("x-agent-schema")
    if schema_uri:
        # First try to look up by name from the registry
        from remlight.api.routers.tools import get_agent_schema
        schema = get_agent_schema(schema_uri)
        if schema is None:
            # Fall back to trying to parse as YAML content (legacy support)
            try:
                schema = schema_from_yaml(schema_uri)
            except Exception:
                schema = get_default_agent_schema()
    else:
        schema = get_default_agent_schema()

    # Get MCP tools
    tools = await get_mcp_tools()

    # Create agent with context
    agent_runtime = await create_agent(
        schema=schema,
        model_name=request.model,
        tools=tools,
        context=context,
    )

    # Build prompt from most recent user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if user_messages:
        prompt = user_messages[-1].content
    else:
        prompt = request.messages[-1].content if request.messages else ""

    # Load session history if session_id is provided
    message_history = None
    session_id = context.session_id
    user_id = context.user_id

    if session_id:
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

            if raw_history:
                system_prompt = None
                if hasattr(schema, "description"):
                    system_prompt = schema.description
                elif isinstance(schema, dict):
                    system_prompt = schema.get("description")

                message_history = session_to_pydantic_messages(
                    raw_history,
                    system_prompt=system_prompt,
                )
        except Exception:
            pass

        # Save user message BEFORE streaming
        await save_user_message(
            session_id=session_id,
            user_id=user_id,
            content=prompt,
        )

    agent_schema_name = None
    if hasattr(schema, "json_schema_extra") and hasattr(schema.json_schema_extra, "name"):
        agent_schema_name = schema.json_schema_extra.name
    elif isinstance(schema, dict) and "json_schema_extra" in schema:
        agent_schema_name = schema["json_schema_extra"].get("name")

    if request.stream:
        return StreamingResponse(
            stream_sse_with_save(
                agent=agent_runtime.agent,
                prompt=prompt,
                model=request.model,
                agent_schema=agent_schema_name,
                session_id=session_id,
                user_id=user_id,
                context=context,
                message_history=message_history,
            ),
            media_type="text/event-stream",
        )
    else:
        # Non-streaming response
        run_kwargs = {}
        if message_history:
            run_kwargs["message_history"] = message_history

        result = await agent_runtime.agent.run(prompt, **run_kwargs)

        # Save assistant response
        if session_id:
            from remlight.services.session import SessionMessageStore
            from datetime import datetime, timezone

            output_str = str(result.output) if hasattr(result, "output") else str(result)

            store = SessionMessageStore(user_id=user_id or "anonymous")
            await store.store_session_messages(
                session_id=session_id,
                messages=[{
                    "role": "assistant",
                    "content": output_str,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }],
                user_id=user_id,
            )

        return {
            "id": "chatcmpl-remlight",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(result.output) if hasattr(result, "output") else str(result),
                    },
                    "finish_reason": "stop",
                }
            ],
        }
