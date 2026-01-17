"""
Session Message Storage - Persistence for Multi-Turn Conversations
===================================================================

This module handles storing and retrieving conversation messages. It enables
multi-turn conversations by persisting messages to PostgreSQL.

THE "STORE UNCOMPRESSED, COMPRESS ON LOAD" PATTERN
--------------------------------------------------

This is the KEY DESIGN DECISION of the session system:

    STORE (uncompressed)         LOAD (compressed)
          │                            │
          ▼                            ▼
    ┌─────────────┐              ┌─────────────┐
    │  Database   │              │  Database   │
    │             │              │             │
    │ Full        │              │ Full        │
    │ messages    │──────────────│ messages    │
    │ stored      │              │ stored      │
    └─────────────┘              └─────────────┘
          │                            │
          │                            │ ← Compress on read
          ▼                            ▼
    Full audit trail             Context-efficient
    for analytics                for LLM

WHY THIS PATTERN?

1. **Full Audit Trail**: Every message is stored in full. This enables:
   - Analytics on conversation patterns
   - Debugging issues
   - Compliance/auditing requirements
   - No data loss

2. **Efficient LLM Context**: When loading for agent execution:
   - Long assistant messages are truncated
   - REM LOOKUP hints inserted for recovery
   - Keeps context window manageable

3. **Flexibility**: Different loads can have different compression:
   - LLM context: aggressive compression
   - Audit export: no compression
   - Analytics: selective compression


MESSAGE COMPRESSION RULES
-------------------------

| Role      | Compression on Load        |
|-----------|---------------------------|
| user      | NEVER - user input sacred  |
| tool      | NEVER - structured metadata|
| assistant | IF length > 400 chars      |

Tool messages are NEVER compressed because they contain:
- tool_call_id (needed for message reconstruction)
- tool_name (needed for pydantic-ai format)
- tool_arguments (needed for conversation replay)
- Structured result data

REM LOOKUP PATTERN
------------------

When a long assistant message is compressed, we insert a hint:

    {first 200 chars}

    ... [Message truncated - REM LOOKUP session-{id}-msg-{idx} to recover full content] ...

    {last 200 chars}

This enables the agent to:
1. See the beginning and end for context
2. Retrieve full content via REM LOOKUP if needed
3. Know exactly how to recover the content

The entity_key format is deterministic, enabling O(1) lookup.
"""

import hashlib
import json
from typing import Any
from uuid import uuid4

from loguru import logger

from remlight.models.entities import Message, Session
from remlight.services.repository import Repository
from remlight.settings import settings


# PostgreSQL has limits on VARCHAR/TEXT index sizes
# Entity keys are used in indexes, so we cap their length
MAX_ENTITY_KEY_LENGTH = 255


def truncate_key(key: str, max_length: int = MAX_ENTITY_KEY_LENGTH) -> str:
    """
    Truncate an entity key to fit database constraints.

    Entity keys (like "session-{uuid}-msg-{idx}") can get long.
    PostgreSQL indexes have size limits, so we truncate if needed.

    To maintain uniqueness after truncation, we append an MD5 hash suffix.
    This ensures different long keys don't collide after truncation.

    Args:
        key: The entity key to (possibly) truncate
        max_length: Maximum allowed length

    Returns:
        Original key if short enough, or truncated key with hash suffix
    """
    if len(key) <= max_length:
        return key
    # Append hash to maintain uniqueness after truncation
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    truncated = key[: max_length - 9] + "-" + hash_suffix
    logger.warning(f"Truncated key from {len(key)} to {len(truncated)} chars: {key[:50]}...")
    return truncated


class MessageCompressor:
    """
    Compress and decompress session messages with REM LOOKUP recovery.

    This class handles the compression logic for long assistant messages.
    It's used ONLY during message loading (not storage).

    COMPRESSION STRATEGY
    -------------------
    Keep first N and last N characters, replace middle with LOOKUP hint:

        {first 200 chars}

        ... [Message truncated - REM LOOKUP {key} to recover full content] ...

        {last 200 chars}

    This preserves:
    - Opening context (what the assistant started saying)
    - Closing context (how the assistant concluded)
    - Recovery path (the exact LOOKUP key)

    WHY 200 CHARACTERS?
    ------------------
    - Long enough to preserve meaningful context
    - Short enough to save significant tokens
    - 400 char threshold = 2x200 = minimum for compression to make sense
    """

    def __init__(self, truncate_length: int = 200):
        """
        Initialize message compressor.

        Args:
            truncate_length: Characters to keep from start AND end (default: 200)
                            Total preserved = 2 * truncate_length = 400 chars minimum
        """
        self.truncate_length = truncate_length
        # Only compress if longer than 2x truncate_length (400 chars by default)
        self.min_length_for_compression = truncate_length * 2

    def compress_message(
        self, message: dict[str, Any], entity_key: str | None = None
    ) -> dict[str, Any]:
        """
        Compress a message by truncating content and adding REM LOOKUP hint.

        COMPRESSION PROCESS
        ------------------
        1. Check if message qualifies (long enough, right role)
        2. Extract first N and last N characters
        3. Insert REM LOOKUP hint in the middle
        4. Add metadata flags for tracking

        OUTPUT FORMAT
        ------------
        Original (1500 chars):
            "Machine learning is a subset of AI that... [1500 chars of explanation]"

        Compressed (450 chars):
            "Machine learning is a subset of AI that... [200 chars]

            ... [Message truncated - REM LOOKUP session-abc-msg-3 to recover full content] ...

            [last 200 chars] ...applications in many fields."

        METADATA FLAGS
        -------------
        Compressed messages include:
        - _compressed: True (marker for is_compressed())
        - _original_length: Original content length
        - _entity_key: REM LOOKUP key for recovery

        These flags enable:
        - Detecting compressed messages
        - Showing "(truncated)" in UI
        - Recovering full content on demand

        Args:
            message: Message dict with role and content
            entity_key: REM LOOKUP key for recovery (e.g., "session-abc-msg-3")

        Returns:
            Compressed message dict (or copy of original if not compressible)
        """
        content = message.get("content") or ""

        # Don't compress short messages or system messages
        # System messages are rare and usually configuration, not conversation
        if (
            len(content) <= self.min_length_for_compression
            or message.get("role") == "system"
        ):
            return message.copy()

        # Compress: keep first N and last N chars
        n = self.truncate_length
        start = content[:n]
        end = content[-n:]

        # Create compressed content with REM LOOKUP hint
        if entity_key:
            # Include recoverable key for agent to use
            compressed_content = f"{start}\n\n... [Message truncated - REM LOOKUP {entity_key} to recover full content] ...\n\n{end}"
        else:
            # No key = no recovery path, just show omission
            compressed_content = f"{start}\n\n... [Message truncated - {len(content) - 2*n} characters omitted] ...\n\n{end}"

        # Build compressed message with metadata
        compressed_message = message.copy()
        compressed_message["content"] = compressed_content
        compressed_message["_compressed"] = True
        compressed_message["_original_length"] = len(content)
        if entity_key:
            compressed_message["_entity_key"] = entity_key

        logger.debug(
            f"Compressed message from {len(content)} to {len(compressed_content)} chars (key={entity_key})"
        )

        return compressed_message

    def decompress_message(
        self, message: dict[str, Any], full_content: str
    ) -> dict[str, Any]:
        """
        Decompress a message by restoring full content.

        Args:
            message: Compressed message dict
            full_content: Full content to restore

        Returns:
            Decompressed message dict
        """
        decompressed = message.copy()
        decompressed["content"] = full_content
        decompressed.pop("_compressed", None)
        decompressed.pop("_original_length", None)
        decompressed.pop("_entity_key", None)

        return decompressed

    def is_compressed(self, message: dict[str, Any]) -> bool:
        """Check if a message is compressed."""
        return message.get("_compressed", False)

    def get_entity_key(self, message: dict[str, Any]) -> str | None:
        """Get REM lookup key from compressed message."""
        return message.get("_entity_key")


class SessionMessageStore:
    """
    Store and retrieve session messages with compression support.

    This is the PRIMARY interface for message persistence. It handles:
    - Storing messages to PostgreSQL (uncompressed for full audit trail)
    - Loading messages (optionally compressed for LLM context efficiency)
    - Session creation/management
    - REM LOOKUP key generation for message recovery

    USAGE PATTERN IN STREAMING
    -------------------------
    The streaming layer uses this store in two phases:

        # BEFORE agent execution
        await save_user_message(session_id, user_id, prompt)

        # AFTER agent execution (in stream_sse_with_save)
        store = SessionMessageStore(user_id)
        await store.store_session_messages(session_id, messages)

    USAGE PATTERN IN AGENT LOADING
    -----------------------------
    Before running an agent with history:

        store = SessionMessageStore(user_id)
        raw_history = await store.load_session_messages(
            session_id,
            compress_on_load=True  # Compress for LLM efficiency
        )
        pydantic_history = session_to_pydantic_messages(raw_history, system_prompt)
        await agent.run(prompt, message_history=pydantic_history)

    DATA ISOLATION
    --------------
    The user_id parameter provides data isolation:
    - Messages are stored with user_id
    - Loading filters by user_id
    - Users can only see their own messages

    REPOSITORY PATTERN
    -----------------
    Uses the generic Repository for all database operations.
    No raw SQL in this class - all queries go through Repository.
    """

    def __init__(self, user_id: str, compressor: MessageCompressor | None = None):
        """
        Initialize session message store.

        Args:
            user_id: User identifier for data isolation.
                    All stored messages will have this user_id.
                    Loading will filter by this user_id.
            compressor: Optional custom MessageCompressor.
                       Creates default (200 char truncation) if None.
        """
        self.user_id = user_id
        self.compressor = compressor or MessageCompressor()
        # Use Repository pattern for all database operations
        self._message_repo = Repository(Message)
        self._session_repo = Repository(Session)

    async def _ensure_session_exists(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> None:
        """
        Ensure session exists in database, creating if necessary.

        AUTO-CREATION PATTERN
        --------------------
        Sessions are auto-created on first message storage. This enables:
        - Simple client API (just pass session_id, no explicit create)
        - Idempotent operations (safe to call multiple times)
        - Lazy creation (sessions only exist if messages exist)

        The session_id comes from the client (X-Session-Id header).
        It's typically a UUID generated by the frontend on conversation start.

        BEST-EFFORT CREATION
        -------------------
        Session creation is best-effort. If it fails:
        - We log a warning
        - We continue with message storage
        - Foreign key constraints may fail, but that's caught elsewhere

        This ensures session issues don't block message storage.

        Args:
            session_id: UUID from client (X-Session-Id header)
            user_id: Optional user for session ownership
        """
        if not settings.postgres.enabled:
            logger.debug("Postgres disabled, skipping session check")
            return

        try:
            # Check if session already exists
            existing = await self._session_repo.get_by_id(session_id)
            if existing:
                return  # Session already exists, nothing to do

            # Create new session with client-provided UUID as ID
            session = Session(
                id=session_id,
                name=session_id,  # Default name to UUID (can be updated later)
                user_id=user_id or self.user_id,
                tenant_id=self.user_id,
                status="active",
            )
            await self._session_repo.upsert(session, generate_embeddings=False)
            logger.info(f"Created session {session_id} for user {user_id or self.user_id}")

        except Exception as e:
            # Best-effort: log but don't fail
            # Message storage may still work if session exists from elsewhere
            logger.warning(f"Failed to ensure session exists: {e}")

    async def update_session_name(
        self,
        session_id: str,
        name: str,
    ) -> None:
        """
        Update the session name in the database.

        Called when an agent provides a session_name via action(type="observation").

        Args:
            session_id: Session ID to update
            name: New name for the session
        """
        if not settings.postgres.enabled:
            logger.debug("Postgres disabled, skipping session name update")
            return

        try:
            # Get existing session
            session = await self._session_repo.get_by_id(session_id)
            if session:
                session.name = name
                await self._session_repo.upsert(session, generate_embeddings=False)
                logger.info(f"Updated session {session_id} name to: {name}")
            else:
                logger.warning(f"Session {session_id} not found for name update")
        except Exception as e:
            logger.warning(f"Failed to update session name: {e}")

    async def store_message(
        self,
        session_id: str,
        message: dict[str, Any],
        message_index: int,
        user_id: str | None = None,
    ) -> str:
        """
        Store a long assistant message as a Message entity for REM lookup.

        Args:
            session_id: Parent session identifier
            message: Message dict to store
            message_index: Index of message in conversation

        Returns:
            Entity key for REM lookup (message ID)
        """
        if not settings.postgres.enabled:
            logger.debug("Postgres disabled, skipping message storage")
            return f"msg-{message_index}"

        # Create entity key for REM LOOKUP
        entity_key = truncate_key(f"session-{session_id}-msg-{message_index}")

        # Use pre-generated id from message dict if available
        msg_id = message.get("id") or str(uuid4())

        msg = Message(
            id=msg_id,
            content=message.get("content") or "",
            role=message.get("role", "assistant"),  # DB column is 'role', not 'message_type'
            session_id=session_id,
            tenant_id=self.user_id,
            user_id=user_id or self.user_id,
            metadata={
                "message_index": message_index,
                "entity_key": entity_key,
                "timestamp": message.get("timestamp"),
            },
            trace_id=message.get("trace_id"),
            span_id=message.get("span_id"),
        )

        await self._message_repo.upsert(msg, generate_embeddings=False)
        logger.debug(f"Stored assistant response: {entity_key} (id={msg_id})")
        return entity_key

    async def retrieve_message(self, entity_key: str) -> str | None:
        """
        Retrieve full message content by REM lookup key.

        Uses LOOKUP query pattern: finds message by entity_key in metadata.

        Args:
            entity_key: REM lookup key (session-{id}-msg-{index})

        Returns:
            Full message content or None if not found
        """
        if not settings.postgres.enabled:
            logger.debug("Postgres disabled, cannot retrieve message")
            return None

        try:
            # LOOKUP pattern: find message by entity_key in metadata
            query = """
                SELECT * FROM messages
                WHERE metadata->>'entity_key' = $1
                  AND tenant_id = $2
                  AND deleted_at IS NULL
                LIMIT 1
            """
            row = await self._message_repo.db.fetchrow(query, entity_key, self.user_id)

            if row:
                logger.debug(f"Retrieved message via LOOKUP: {entity_key}")
                return row.get("content")

            logger.warning(f"Message not found via LOOKUP: {entity_key}")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve message {entity_key}: {e}")
            return None

    async def store_session_messages(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        user_id: str | None = None,
        compress: bool = False,  # Compression happens on reload, not store
    ) -> list[dict[str, Any]]:
        """
        Store all session messages to the database.

        Args:
            session_id: Session UUID
            messages: List of messages to store
            user_id: Optional user identifier
            compress: Whether to return compressed versions (default: False)

        Returns:
            List of stored messages (optionally compressed)
        """
        if not settings.postgres.enabled:
            logger.debug("Postgres disabled, returning messages uncompressed")
            return messages

        # Ensure session exists before storing messages
        await self._ensure_session_exists(session_id, user_id)

        stored_messages = []

        for idx, message in enumerate(messages):
            content = message.get("content") or ""
            role = message.get("role", "assistant")

            # Build metadata dict
            msg_metadata: dict[str, Any] = {
                "message_index": idx,
                "timestamp": message.get("timestamp"),
            }

            # For tool messages, include tool call details
            if role == "tool":
                if message.get("tool_call_id"):
                    msg_metadata["tool_call_id"] = message["tool_call_id"]
                if message.get("tool_name"):
                    msg_metadata["tool_name"] = message["tool_name"]
                if message.get("tool_arguments"):
                    msg_metadata["tool_arguments"] = message["tool_arguments"]
                # Include agent and model metadata for all tool messages
                if message.get("agent_schema"):
                    msg_metadata["agent_schema"] = message["agent_schema"]
                if message.get("model"):
                    msg_metadata["model"] = message["model"]

            # Use pre-generated id if provided
            msg_id = message.get("id") or str(uuid4())

            # Extract tool_calls for assistant messages (stored in JSONB column)
            # Must be dict or None, not a list (Message model constraint)
            tool_calls_data = message.get("tool_calls")
            if isinstance(tool_calls_data, list):
                # Convert list to dict or None
                tool_calls_data = None if not tool_calls_data else {"items": tool_calls_data}
            elif not tool_calls_data:
                tool_calls_data = None

            msg = Message(
                id=msg_id,
                content=content,
                role=role,  # DB column is 'role', not 'message_type'
                session_id=session_id,
                tenant_id=self.user_id,
                user_id=user_id or self.user_id,
                metadata=msg_metadata,
                tool_calls=tool_calls_data,  # Store tool calls in JSONB column
                trace_id=message.get("trace_id"),
                span_id=message.get("span_id"),
            )

            try:
                await self._message_repo.upsert(msg, generate_embeddings=False)
            except Exception as e:
                logger.warning(f"Failed to store message {idx}: {e}")

            # Optionally compress for return
            if compress and role == "assistant" and len(content) > self.compressor.min_length_for_compression:
                entity_key = truncate_key(f"session-{session_id}-msg-{idx}")
                stored_messages.append(self.compressor.compress_message(message, entity_key))
            else:
                stored_messages.append(message.copy())

        logger.debug(f"Stored {len(messages)} messages for session {session_id}")
        return stored_messages

    async def load_session_messages(
        self,
        session_id: str,
        user_id: str | None = None,
        compress_on_load: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Load session messages from database for conversation replay.

        This is the CORE METHOD for context reconstruction. It loads stored
        messages and optionally compresses them for LLM context efficiency.

        THE LOADING PIPELINE
        -------------------
        1. Query database for session messages (via Repository)
        2. Filter by tenant_id for data isolation
        3. Parse metadata from JSONB column
        4. Reconstruct tool call info for tool messages
        5. Optionally compress long assistant messages
        6. Return in chronological order

        COMPRESSION RULES
        ----------------
        | Role      | compress_on_load=True     | compress_on_load=False |
        |-----------|---------------------------|------------------------|
        | user      | Never compressed          | Never compressed       |
        | tool      | Never compressed          | Never compressed       |
        | assistant | Compressed if >400 chars  | Never compressed       |

        Tool messages are NEVER compressed because they contain structured
        metadata (tool_call_id, tool_name, tool_arguments) that's needed
        for pydantic-ai message reconstruction.

        METADATA RECONSTRUCTION
        ----------------------
        Tool messages have metadata stored as JSONB:
            {"tool_call_id": "call_abc", "tool_name": "search", "tool_arguments": {...}}

        This metadata is extracted and added to the message dict for
        session_to_pydantic_messages() to use.

        Args:
            session_id: Session identifier (UUID)
            user_id: Optional user filter (uses self.user_id if not provided)
            compress_on_load: Whether to compress long assistant messages
                             True = efficient LLM context
                             False = full content (for debugging/export)

        Returns:
            List of message dicts in chronological order:
            [
                {"role": "user", "content": "Hello", "timestamp": "..."},
                {"role": "assistant", "content": "Hi there", "timestamp": "..."},
                {"role": "tool", "content": "{...}", "tool_name": "search", ...},
            ]
        """
        if not settings.postgres.enabled:
            logger.debug("Postgres disabled, returning empty message list")
            return []

        try:
            # Load messages via Repository (ordered by created_at ASC)
            messages = await self._message_repo.get_by_session(session_id)

            # Filter by tenant_id for data isolation
            # This ensures users only see their own messages
            messages = [m for m in messages if m.tenant_id == self.user_id]

            message_dicts = []
            for idx, msg in enumerate(messages):
                role = msg.role or "assistant"
                content = msg.content or ""

                # Parse metadata from model (already a dict from Pydantic)
                metadata = msg.metadata
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                metadata = metadata or {}

                # Build base message dict
                msg_dict: dict[str, Any] = {
                    "role": role,
                    "content": content,
                    "timestamp": msg.created_at.isoformat() if msg.created_at else None,
                }

                # For tool messages, reconstruct tool call metadata
                # This is critical for session_to_pydantic_messages()
                if role == "tool" and metadata:
                    if metadata.get("tool_call_id"):
                        msg_dict["tool_call_id"] = metadata["tool_call_id"]
                    if metadata.get("tool_name"):
                        msg_dict["tool_name"] = metadata["tool_name"]
                    if metadata.get("tool_arguments"):
                        msg_dict["tool_arguments"] = metadata["tool_arguments"]

                # =============================================================
                # CONDITIONAL COMPRESSION
                # =============================================================
                # Only compress:
                # - When compress_on_load=True
                # - For assistant messages (user and tool never compressed)
                # - For messages exceeding min_length_for_compression (400 chars)
                # =============================================================
                if (
                    compress_on_load
                    and role == "assistant"
                    and len(content) > self.compressor.min_length_for_compression
                ):
                    # Generate deterministic entity key for REM LOOKUP recovery
                    entity_key = truncate_key(f"session-{session_id}-msg-{idx}")
                    msg_dict = self.compressor.compress_message(msg_dict, entity_key)

                message_dicts.append(msg_dict)

            logger.debug(
                f"Loaded {len(message_dicts)} messages for session {session_id} "
                f"(compress_on_load={compress_on_load})"
            )
            return message_dicts

        except Exception as e:
            logger.error(f"Failed to load session messages: {e}")
            # Return empty list on error - don't crash the agent
            return []

    async def retrieve_full_message(self, session_id: str, message_index: int) -> str | None:
        """
        Retrieve full message content by session and message index (for REM LOOKUP).

        Args:
            session_id: Session identifier
            message_index: Index of message in session

        Returns:
            Full message content or None if not found
        """
        entity_key = truncate_key(f"session-{session_id}-msg-{message_index}")
        return await self.retrieve_message(entity_key)


async def reload_session(
    session_id: str,
    user_id: str,
    compress_on_load: bool = True,
) -> list[dict]:
    """
    Reload all messages for a session from the database.

    Convenience function for loading session history.

    Args:
        session_id: Session/conversation identifier
        user_id: User identifier for data isolation
        compress_on_load: Whether to compress long assistant messages

    Returns:
        List of message dicts in chronological order
    """
    if not session_id:
        return []

    store = SessionMessageStore(user_id=user_id)
    return await store.load_session_messages(
        session_id=session_id, user_id=user_id, compress_on_load=compress_on_load
    )
