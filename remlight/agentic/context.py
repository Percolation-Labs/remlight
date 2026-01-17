"""
Agent Execution Context and Multi-Agent Context Propagation
============================================================

This module manages the WHO (user, tenant) and WHAT (session, model, schema)
for agent execution. It's the glue between HTTP requests and agent invocations.

ARCHITECTURE OVERVIEW
--------------------

                    HTTP Request
                         │
                         ▼
            ┌─────────────────────────┐
            │    AgentContext.from_   │
            │    request(request)     │
            │                         │
            │  Extracts:              │
            │  - user_id from JWT     │
            │  - session_id           │
            │  - model, schema, etc.  │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │    set_current_context  │◄──── ContextVar stores it
            │    (context)            │
            └────────────┬────────────┘
                         │
                         ▼
            ┌─────────────────────────┐
            │    Agent Execution      │
            │                         │
            │    Tools can call:      │
            │    get_current_context()│
            │    to access context    │
            └────────────┬────────────┘
                         │
                         ▼  (if ask_agent tool called)
            ┌─────────────────────────┐
            │    parent.child_context │
            │    (agent_schema_uri)   │
            │                         │
            │    Creates child with:  │
            │    - Same user_id       │
            │    - Same session_id    │
            │    - Different agent    │
            └─────────────────────────┘


THREE WAYS TO CREATE CONTEXT
----------------------------

1. **from_request(request)** - PREFERRED for API endpoints
   Extracts user_id from validated JWT token in request.state (secure).
   Falls back to X-User-Id header only if JWT not present.

2. **from_headers(headers)** - For testing/CLI
   Constructs from raw HTTP headers dict. Less secure (trusts headers).

3. **Direct instantiation** - For unit tests
   AgentContext(user_id="test", session_id="abc")


CONTEXTVAR FOR MULTI-AGENT PROPAGATION
--------------------------------------

When agents call other agents (via ask_agent tool), context needs to flow:

    Parent Agent
        │
        └── calls ask_agent("child-agent", "do something")
                │
                ├── get_current_context()  ← Gets parent's context
                │
                ├── parent.child_context() ← Creates child context
                │
                └── Child Agent executes with child context

The _current_agent_context ContextVar makes this possible:
- Set before agent execution starts
- Tools can retrieve via get_current_context()
- Automatically cleaned up after execution

Similarly, _parent_event_sink ContextVar enables child agent streaming:
- Parent sets up an event queue
- Child pushes events to the queue
- Parent's streaming loop yields child events to client


HEADER MAPPINGS
--------------
    X-User-Id        → user_id (fallback to JWT)
    X-Tenant-Id      → tenant_id (default: "default")
    X-Session-Id     → session_id
    X-Agent-Schema   → agent_schema_uri
    X-Model-Name     → default_model
    X-Is-Eval        → is_eval
    X-Client-Id      → client_id


DESIGN PRINCIPLES
-----------------

1. **Context is data, not behavior**
   AgentContext is a Pydantic BaseModel - pure data that describes
   who is making the request and what resources they want.

2. **Passed, not stored**
   Context is passed to functions that need it. Agents don't store
   context - they receive it as a parameter.

3. **Immutable by convention**
   Create new contexts (via child_context) rather than mutating.

4. **Secure defaults**
   JWT extraction preferred over header trust. Anonymous access
   returns None user_id (not fake IDs).
"""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator, TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, Field

from remlight.settings import settings

if TYPE_CHECKING:
    from starlette.requests import Request


# =============================================================================
# CONTEXTVAR DECLARATIONS
# =============================================================================
# ContextVars provide async-safe thread-local storage. They're propagated
# through async call chains, making them perfect for multi-agent context.
#
# Without ContextVars, we'd need to pass context through every function call,
# including third-party libraries. ContextVars enable "ambient" context.
# =============================================================================

# Thread-local context for current agent execution
# This is the PRIMARY mechanism for context propagation in multi-agent scenarios.
#
# Set by: streaming layer (stream_sse, stream_plain) before agent execution
# Read by: MCP tools (ask_agent, action) to access parent context
# Cleared by: streaming layer after agent execution completes
_current_agent_context: ContextVar["AgentContext | None"] = ContextVar(
    "current_agent_context", default=None
)

# Event sink for streaming child agent events to parent
# This enables REAL-TIME STREAMING of child agent output through parent.
#
# When a parent agent calls ask_agent:
# 1. Parent's streaming loop creates an asyncio.Queue
# 2. Queue is set via set_event_sink()
# 3. ask_agent's inner streaming pushes events to this queue
# 4. Parent's streaming loop consumes from queue and yields to client
#
# This prevents buffering - child content streams immediately to the user.
_parent_event_sink: ContextVar["asyncio.Queue | None"] = ContextVar(
    "parent_event_sink", default=None
)


def get_current_context() -> "AgentContext | None":
    """
    Get the current agent context from ContextVar.

    THE AMBIENT CONTEXT PATTERN
    ---------------------------
    This function enables tools to access context without explicit parameters.
    When an agent executes, the streaming layer sets the context via
    set_current_context(). Any tool the agent calls can then retrieve it.

    MULTI-AGENT USE CASE
    --------------------
    When ask_agent tool is called, it needs the parent's context to:
    - Know who the user is (user_id)
    - Maintain conversation continuity (session_id)
    - Apply proper data isolation (tenant_id)

    Without ContextVar, we'd need to thread context through every function,
    including into pydantic-ai internals. ContextVar makes this transparent.

    Returns:
        Current AgentContext if set, None if not in an agent execution context

    Example:
        # In an MCP tool (e.g., ask_agent)
        parent_context = get_current_context()
        if parent_context:
            # Create child context inheriting from parent
            child_context = parent_context.child_context(
                agent_schema_uri="sentiment-analyzer"
            )
            # Child now has same user_id, session_id, tenant_id
    """
    return _current_agent_context.get()


def set_current_context(ctx: "AgentContext | None") -> None:
    """
    Set the current agent context in ContextVar.

    LIFECYCLE
    ---------
    1. Streaming layer calls set_current_context(context) BEFORE agent.run()
    2. Agent executes, tools can access via get_current_context()
    3. Streaming layer calls set_current_context(None) AFTER completion

    Passing None clears the context, which is important to avoid leaking
    context between unrelated requests.

    Args:
        ctx: AgentContext to set, or None to clear
    """
    _current_agent_context.set(ctx)


@contextmanager
def agent_context_scope(ctx: "AgentContext") -> Generator["AgentContext", None, None]:
    """
    Context manager for scoped context setting with automatic cleanup.

    SAFE NESTING
    ------------
    In multi-agent scenarios, contexts nest:

        with agent_context_scope(parent_ctx):
            # Parent agent runs
            with agent_context_scope(child_ctx):  # ask_agent
                # Child agent runs, sees child_ctx
            # Back to parent_ctx
        # Context cleared

    This context manager saves the previous context before setting the new one,
    and restores it in the finally block. This ensures proper cleanup even if
    the agent raises an exception.

    Args:
        ctx: AgentContext to set for this scope

    Yields:
        The same context (for convenience)

    Example:
        context = AgentContext(user_id="user-123", session_id="sess-456")
        with agent_context_scope(context):
            # get_current_context() returns context
            result = await agent.run("Hello")
        # get_current_context() returns previous value (or None)
    """
    previous = _current_agent_context.get()
    _current_agent_context.set(ctx)
    try:
        yield ctx
    finally:
        # Always restore, even if exception occurred
        _current_agent_context.set(previous)


# =============================================================================
# EVENT SINK FOR STREAMING MULTI-AGENT DELEGATION
# =============================================================================
#
# When a parent agent calls ask_agent, we want the child's output to stream
# to the user in REAL-TIME, not buffer until the child completes.
#
# The event sink pattern enables this:
#
#     Parent's streaming loop
#           │
#           ├── Creates asyncio.Queue
#           │
#           ├── set_event_sink(queue)
#           │
#           ├── Tool execution (ask_agent called)
#           │       │
#           │       └── Child agent streams
#           │               │
#           │               └── push_event(content) ─────► queue
#           │
#           ├── queue.get() ─────────────────────────────► yield to client
#           │
#           └── set_event_sink(None)
#
# =============================================================================


def get_event_sink() -> "asyncio.Queue | None":
    """
    Get the parent's event sink queue for streaming child events.

    Called by ask_agent to get the queue where it should push child events.
    Returns None if not in a streaming context (e.g., sync execution).

    EVENT TYPES PUSHED
    -----------------
    - Content chunks: Child agent's text output
    - ToolCallEvent: Child's tool invocations
    - ActionEvent: Child's action emissions
    - MetadataEvent: Child's metadata

    All pushed events are yielded by the parent's streaming loop.

    Returns:
        asyncio.Queue for pushing events, or None if not streaming
    """
    return _parent_event_sink.get()


def set_event_sink(sink: "asyncio.Queue | None") -> None:
    """
    Set the event sink queue for child agents.

    Called by the streaming layer (stream_sse) to establish the channel
    for child event proxying.

    Args:
        sink: Queue for child events, or None to clear
    """
    _parent_event_sink.set(sink)


@contextmanager
def event_sink_scope(sink: "asyncio.Queue") -> Generator["asyncio.Queue", None, None]:
    """
    Context manager for scoped event sink setting with automatic cleanup.

    STREAMING ARCHITECTURE
    ---------------------
    The parent's streaming loop uses this pattern:

        async def stream_sse(...):
            child_event_sink = asyncio.Queue()

            with event_sink_scope(child_event_sink):
                async with agent.iter(prompt) as stream:
                    # Merge agent stream with child events
                    async for source, event in stream_with_child_events(
                        stream, child_event_sink
                    ):
                        if source == "child":
                            yield format_child_event(event)
                        else:
                            yield format_agent_event(event)

    The scope ensures:
    1. Sink is available during tool execution
    2. Cleanup happens even if exception occurs
    3. Nested scopes work correctly

    Args:
        sink: asyncio.Queue for child events

    Yields:
        The same queue (for convenience)
    """
    previous = _parent_event_sink.get()
    _parent_event_sink.set(sink)
    try:
        yield sink
    finally:
        _parent_event_sink.set(previous)


async def push_event(event: Any) -> bool:
    """
    Push an event to the parent's event sink queue.

    Called by ask_agent's internal streaming to proxy child events.
    This is how child agent output reaches the parent's response stream.

    NON-BLOCKING
    -----------
    Uses queue.put() which is async but fast (just adds to queue).
    The parent's streaming loop consumes at its own pace.

    GRACEFUL DEGRADATION
    -------------------
    Returns False if no sink is available. This allows the same code
    to work in both streaming (sink set) and sync (no sink) contexts.

    Args:
        event: Any streaming event to push:
               - ("child_content", "text") for content chunks
               - ToolCallEvent for tool invocations
               - ActionEvent for actions
               - etc.

    Returns:
        True if pushed successfully, False if no sink available
    """
    sink = _parent_event_sink.get()
    if sink is not None:
        await sink.put(event)
        return True
    return False


class AgentContext(BaseModel):
    """
    Session and configuration context for agent execution.

    AgentContext is the answer to "who is running this agent and how should
    it be configured?" It bundles identity (user, tenant), session state,
    and runtime configuration into a single, immutable data structure.

    WHY PYDANTIC BASEMODEL (NOT DATACLASS)?
    ---------------------------------------
    We use Pydantic BaseModel instead of dataclass for:

    1. **Validation**: Fields are validated on construction
    2. **Serialization**: Easy JSON/dict conversion for logging, storage
    3. **Field defaults with factories**: default_factory for settings
    4. **FastAPI integration**: Works seamlessly with FastAPI dependency injection

    FIELD BREAKDOWN
    ---------------

    IDENTITY FIELDS (who):
    - user_id: UUID hash of user's email (from JWT or header)
    - tenant_id: Organization/workspace isolation
    - client_id: Which client app (web, mobile, cli)

    SESSION FIELDS (what):
    - session_id: Conversation identifier for multi-turn continuity
    - agent_schema_uri: Which agent schema to use

    CONFIGURATION FIELDS (how):
    - default_model: LLM model to use
    - is_eval: Whether this is an evaluation run (affects logging/tracing)
    - user_profile_hint: Pre-loaded user context for personalization

    IMMUTABILITY PATTERN
    -------------------
    Context should not be mutated after creation. Instead of:

        context.session_id = "new-session"  # DON'T DO THIS

    Create a new context:

        new_context = context.model_copy(update={"session_id": "new-session"})

    Or use child_context() for multi-agent scenarios.

    Example:
        # From API request (preferred)
        context = AgentContext.from_request(request)

        # From headers (for testing/CLI)
        context = AgentContext.from_headers(headers)

        # Direct construction (for unit tests)
        context = AgentContext(
            user_id="user-123",
            tenant_id="acme-corp",
            session_id="sess-456",
        )
    """

    # =========================================================================
    # IDENTITY FIELDS - Who is making this request?
    # =========================================================================

    user_id: str | None = Field(
        default=None,
        description="User identifier for tracking and personalization. "
                    "UUID5 hash of user's email address. None = anonymous/shared data.",
    )

    tenant_id: str = Field(
        default="default",
        description="Tenant identifier for multi-tenancy data isolation. "
                    "All database queries filter by tenant_id.",
    )

    client_id: str | None = Field(
        default=None,
        description="Client application identifier (e.g., 'web', 'mobile', 'cli'). "
                    "Useful for analytics and client-specific behavior.",
    )

    # =========================================================================
    # SESSION FIELDS - What context is this request in?
    # =========================================================================

    session_id: str | None = Field(
        default=None,
        description="Session/conversation identifier for multi-turn continuity. "
                    "When set, message history is loaded from the session.",
    )

    agent_schema_uri: str | None = Field(
        default=None,
        description="URI of the agent schema to invoke. "
                    "Allows dynamic agent selection via X-Agent-Schema header.",
    )

    # =========================================================================
    # CONFIGURATION FIELDS - How should the agent execute?
    # =========================================================================

    default_model: str = Field(
        default_factory=lambda: settings.llm.default_model,
        description="LLM model identifier (e.g., 'openai:gpt-4.1'). "
                    "Can be overridden per-request via X-Model-Name header.",
    )

    is_eval: bool = Field(
        default=False,
        description="Whether this is an evaluation session. "
                    "Affects tracing/logging - eval sessions may be stored separately.",
    )

    user_profile_hint: str | None = Field(
        default=None,
        description="Pre-loaded user profile text for agent context. "
                    "Enables personalization without the agent needing to look up user info.",
    )

    model_config = {"populate_by_name": True}

    def child_context(
        self,
        agent_schema_uri: str | None = None,
        model_override: str | None = None,
    ) -> "AgentContext":
        """
        Create a child context for nested agent calls (multi-agent pattern).

        MULTI-AGENT ORCHESTRATION
        -------------------------
        When a parent agent calls ask_agent("child-agent", ...), the child
        needs its own context that:

        1. INHERITS identity (same user, tenant, session)
        2. CHANGES agent schema (different agent type)
        3. OPTIONALLY changes model (e.g., cheaper model for subtasks)

        This method creates that child context cleanly.

        INHERITANCE RULES
        ----------------
        Inherited (same as parent):
        - user_id: Same user making the request
        - tenant_id: Same tenant for data isolation
        - session_id: Same session for conversation continuity
        - is_eval: Same evaluation status
        - client_id: Same client application
        - user_profile_hint: Same user context

        Overridable:
        - agent_schema_uri: Which agent schema the child uses
        - default_model: LLM model for the child

        WHY NOT JUST COPY?
        -----------------
        We could use model_copy(), but child_context() is explicit about:
        - Which fields are safe to change
        - What the intended use case is
        - Documenting the multi-agent pattern

        Args:
            agent_schema_uri: Agent schema URI for the child
                             (defaults to parent's if not specified)
            model_override: Optional different model for child
                           (e.g., faster/cheaper for subtasks)

        Returns:
            New AgentContext for the child agent execution

        Example:
            parent_context = get_current_context()
            child_context = parent_context.child_context(
                agent_schema_uri="sentiment-analyzer",
                model_override="openai:gpt-4.1-mini",  # Cheaper for simple task
            )
            runtime = await create_agent(schema, context=child_context)
        """
        return AgentContext(
            # Identity inherited from parent
            user_id=self.user_id,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            # Session inherited from parent (same conversation)
            session_id=self.session_id,
            # Configuration: can be overridden
            default_model=model_override or self.default_model,
            agent_schema_uri=agent_schema_uri or self.agent_schema_uri,
            # Flags inherited
            is_eval=self.is_eval,
            user_profile_hint=self.user_profile_hint,
        )

    @staticmethod
    def get_user_id_or_default(
        user_id: str | None,
        source: str = "context",
        default: str | None = None,
    ) -> str | None:
        """
        Get user_id with safe handling of None (anonymous access).

        USER ID CONVENTIONS IN REMLIGHT
        -------------------------------
        1. user_id is a UUID5 hash of the user's EMAIL, not the JWT `sub` claim
        2. The auth middleware hashes the email: uuid5(NAMESPACE_DNS, email)
        3. This ensures deterministic, privacy-preserving user identification

        ANONYMOUS ACCESS PATTERN
        -----------------------
        When user_id is None (unauthenticated request):
        - Database queries use WHERE user_id IS NULL
        - This returns SHARED/PUBLIC data only
        - User cannot see other users' private data

        We explicitly DON'T generate fake user IDs because:
        - Fake IDs would create "ghost" user data
        - Anonymous users should see shared data, not isolated data
        - It's clearer to handle None explicitly in queries

        Args:
            user_id: The user identifier (may be None for anonymous)
            source: Where this call originated (for logging/debugging)
            default: Explicit default for testing (NOT auto-generated)

        Returns:
            user_id if provided, explicit default if provided, None otherwise

        Example:
            # In an MCP tool
            user_id = AgentContext.get_user_id_or_default(
                context.user_id,
                source="search_tool"
            )
            # Use in query
            results = await repo.find({"user_id": user_id})
            # If user_id is None → returns shared records
            # If user_id is set → returns user's + shared records
        """
        if user_id is not None:
            return user_id
        if default is not None:
            logger.debug(f"Using explicit default user_id '{default}' from {source}")
            return default
        # No fake user IDs - return None for anonymous/unauthenticated
        logger.debug(f"No user_id from {source}, using None (anonymous/shared data)")
        return None

    @classmethod
    def from_request(cls, request: "Request") -> "AgentContext":
        """
        Construct AgentContext from a FastAPI Request object.

        This is the PREFERRED method for API endpoints. It extracts user_id
        from the authenticated user in request.state (set by auth middleware
        from JWT token), which is more secure than trusting X-User-Id header.

        Priority for user_id:
        1. request.state.user.id - From validated JWT token (SECURE)
        2. X-User-Id header - Fallback for backwards compatibility

        Args:
            request: FastAPI Request object

        Returns:
            AgentContext with user from JWT and other values from headers

        Example:
            @app.post("/api/v1/chat/completions")
            async def chat(request: Request, body: ChatRequest):
                context = AgentContext.from_request(request)
                # context.user_id is from JWT, not header
        """
        # Get headers dict
        headers = dict(request.headers)
        normalized = {k.lower(): v for k, v in headers.items()}

        # Extract user_id from authenticated user (JWT) - this is the source of truth
        user_id = None
        tenant_id = "default"

        if hasattr(request, "state"):
            user = getattr(request.state, "user", None)
            if user and isinstance(user, dict):
                user_id = user.get("id")
                # Also get tenant_id from authenticated user if available
                if user.get("tenant_id"):
                    tenant_id = user.get("tenant_id")
                if user_id:
                    logger.debug(f"User ID from JWT: {user_id}")

        # Fallback to X-User-Id header if no authenticated user
        if not user_id:
            user_id = normalized.get("x-user-id")
            if user_id:
                logger.debug(f"User ID from X-User-Id header (fallback): {user_id}")

        # Override tenant_id from header if provided
        header_tenant = normalized.get("x-tenant-id")
        if header_tenant:
            tenant_id = header_tenant

        # Parse X-Is-Eval header
        is_eval_str = normalized.get("x-is-eval", "").lower()
        is_eval = is_eval_str in ("true", "1", "yes")

        return cls(
            user_id=user_id,
            tenant_id=tenant_id,
            session_id=normalized.get("x-session-id"),
            default_model=normalized.get("x-model-name") or settings.llm.default_model,
            agent_schema_uri=normalized.get("x-agent-schema"),
            is_eval=is_eval,
            client_id=normalized.get("x-client-id"),
        )

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> "AgentContext":
        """
        Construct AgentContext from HTTP headers dict.

        NOTE: Prefer from_request() for API endpoints as it extracts user_id
        from the validated JWT token in request.state, which is more secure.

        Reads standard headers:
        - X-User-Id: User identifier (fallback - prefer JWT)
        - X-Tenant-Id: Tenant identifier
        - X-Session-Id: Session identifier
        - X-Model-Name: Model override
        - X-Agent-Schema: Agent schema URI
        - X-Is-Eval: Whether this is an evaluation session (true/false)
        - X-Client-Id: Client identifier (e.g., "web", "mobile", "cli")

        Args:
            headers: Dictionary of HTTP headers (case-insensitive)

        Returns:
            AgentContext with values from headers

        Example:
            headers = {
                "X-User-Id": "user123",
                "X-Tenant-Id": "acme-corp",
                "X-Session-Id": "sess-456",
                "X-Model-Name": "anthropic:claude-sonnet-4-5-20250929",
                "X-Is-Eval": "true",
                "X-Client-Id": "web"
            }
            context = AgentContext.from_headers(headers)
        """
        # Normalize header keys to lowercase for case-insensitive lookup
        normalized = {k.lower(): v for k, v in headers.items()}

        # Parse X-Is-Eval header (accepts "true", "1", "yes" as truthy)
        is_eval_str = normalized.get("x-is-eval", "").lower()
        is_eval = is_eval_str in ("true", "1", "yes")

        return cls(
            user_id=normalized.get("x-user-id"),
            tenant_id=normalized.get("x-tenant-id", "default"),
            session_id=normalized.get("x-session-id"),
            default_model=normalized.get("x-model-name") or settings.llm.default_model,
            agent_schema_uri=normalized.get("x-agent-schema"),
            is_eval=is_eval,
            client_id=normalized.get("x-client-id"),
        )

    @classmethod
    async def from_headers_with_profile(cls, headers: dict[str, str]) -> "AgentContext":
        """
        Construct AgentContext from HTTP headers and load user profile hint.

        Always injects current date/time into the profile hint for agent
        self-awareness. Also loads user profile if user_id is available.

        Args:
            headers: Dictionary of HTTP headers (case-insensitive)

        Returns:
            AgentContext with user profile hint loaded (always includes date/time)
        """
        context = cls.from_headers(headers)

        # Always load profile hint (includes date/time, and user profile if available)
        try:
            from remlight.api.routers.tools import get_user_profile_hint
            context = context.model_copy(
                update={"user_profile_hint": await get_user_profile_hint(context.user_id)}
            )
        except Exception as e:
            logger.debug(f"Failed to load user profile hint: {e}")

        return context
