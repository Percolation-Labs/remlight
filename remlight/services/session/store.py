"""Session message storage for conversation history.

Design Pattern:
- All messages stored UNCOMPRESSED in the database for full audit/analysis
- Compression happens only on RELOAD when reconstructing context for the LLM
- Tool messages (role: "tool") are NEVER compressed - contain structured metadata

Message Types:
- user: User messages - stored and reloaded as-is
- tool: Tool call messages (e.g., register_metadata) - NEVER compressed
- assistant: May be compressed on reload if long (>400 chars) with REM LOOKUP hints
"""

import hashlib
import json
from typing import Any
from uuid import uuid4

from loguru import logger

from remlight.models.entities import Message, Session
from remlight.services.repository import Repository
from remlight.settings import settings


# Max length for entity keys
MAX_ENTITY_KEY_LENGTH = 255


def truncate_key(key: str, max_length: int = MAX_ENTITY_KEY_LENGTH) -> str:
    """Truncate a key to max length, preserving useful suffix if possible."""
    if len(key) <= max_length:
        return key
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    truncated = key[: max_length - 9] + "-" + hash_suffix
    logger.warning(f"Truncated key from {len(key)} to {len(truncated)} chars: {key[:50]}...")
    return truncated


class MessageCompressor:
    """Compress and decompress session messages with REM lookup keys."""

    def __init__(self, truncate_length: int = 200):
        """
        Initialize message compressor.

        Args:
            truncate_length: Number of characters to keep from start/end (default: 200)
        """
        self.truncate_length = truncate_length
        self.min_length_for_compression = truncate_length * 2

    def compress_message(
        self, message: dict[str, Any], entity_key: str | None = None
    ) -> dict[str, Any]:
        """
        Compress a message by truncating long content and adding REM lookup key.

        Args:
            message: Message dict with role and content
            entity_key: Optional REM lookup key for full message recovery

        Returns:
            Compressed message dict
        """
        content = message.get("content") or ""

        # Don't compress short messages or system messages
        if (
            len(content) <= self.min_length_for_compression
            or message.get("role") == "system"
        ):
            return message.copy()

        # Compress long messages
        n = self.truncate_length
        start = content[:n]
        end = content[-n:]

        # Create compressed content with REM lookup hint
        if entity_key:
            compressed_content = f"{start}\n\n... [Message truncated - REM LOOKUP {entity_key} to recover full content] ...\n\n{end}"
        else:
            compressed_content = f"{start}\n\n... [Message truncated - {len(content) - 2*n} characters omitted] ...\n\n{end}"

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
    """Store and retrieve session messages with compression.

    Usage:
        store = SessionMessageStore(user_id="user-123")

        # Store messages (uncompressed in DB)
        await store.store_session_messages(session_id, messages)

        # Load messages (optionally compressed for LLM context)
        history = await store.load_session_messages(session_id, compress_on_load=True)
    """

    def __init__(self, user_id: str, compressor: MessageCompressor | None = None):
        """
        Initialize session message store.

        Args:
            user_id: User identifier for data isolation
            compressor: Optional message compressor (creates default if None)
        """
        self.user_id = user_id
        self.compressor = compressor or MessageCompressor()
        self._message_repo = Repository(Message)
        self._session_repo = Repository(Session)

    async def _ensure_session_exists(
        self,
        session_id: str,
        user_id: str | None = None,
    ) -> None:
        """
        Ensure session exists, creating it if necessary.

        Args:
            session_id: Session UUID from X-Session-Id header
            user_id: Optional user identifier
        """
        if not settings.postgres.enabled:
            logger.debug("Postgres disabled, skipping session check")
            return

        try:
            # Check if session already exists by UUID
            existing = await self._session_repo.get(session_id)
            if existing:
                return  # Session already exists

            # Create new session with the provided UUID as id
            session_data = {
                "id": session_id,
                "name": session_id,  # Default name to UUID
                "user_id": user_id or self.user_id,
                "tenant_id": self.user_id,
                "status": "active",
            }
            await self._session_repo.create(session_data)
            logger.info(f"Created session {session_id} for user {user_id or self.user_id}")

        except Exception as e:
            # Log but don't fail - session creation is best-effort
            logger.warning(f"Failed to ensure session exists: {e}")

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

        msg_data = {
            "id": msg_id,
            "content": message.get("content") or "",
            "role": message.get("role", "assistant"),  # DB column is 'role', not 'message_type'
            "session_id": session_id,
            "tenant_id": self.user_id,
            "user_id": user_id or self.user_id,
            "metadata": json.dumps({
                "message_index": message_index,
                "entity_key": entity_key,
                "timestamp": message.get("timestamp"),
            }),
        }

        if message.get("trace_id"):
            msg_data["trace_id"] = message["trace_id"]
        if message.get("span_id"):
            msg_data["span_id"] = message["span_id"]

        await self._message_repo.create(msg_data)
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

            # Use pre-generated id if provided
            msg_id = message.get("id") or str(uuid4())

            msg_data = {
                "id": msg_id,
                "content": content,
                "role": role,  # DB column is 'role', not 'message_type'
                "session_id": session_id,
                "tenant_id": self.user_id,
                "user_id": user_id or self.user_id,
                "metadata": json.dumps(msg_metadata),
            }

            if message.get("trace_id"):
                msg_data["trace_id"] = message["trace_id"]
            if message.get("span_id"):
                msg_data["span_id"] = message["span_id"]

            try:
                await self._message_repo.create(msg_data)
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
        Load session messages from database.

        Compression on Load:
        - Tool messages (role: "tool") are NEVER compressed
        - User messages are returned as-is
        - Assistant messages MAY be compressed if long with REM LOOKUP hints

        Args:
            session_id: Session identifier
            user_id: Optional user identifier for filtering
            compress_on_load: Whether to compress long assistant messages (default: True)

        Returns:
            List of session messages in chronological order
        """
        if not settings.postgres.enabled:
            logger.debug("Postgres disabled, returning empty message list")
            return []

        try:
            # Load messages using repository
            rows = await self._message_repo.get_by_session(session_id)

            # Filter by tenant_id (user_id for isolation)
            rows = [r for r in rows if r.get("tenant_id") == self.user_id]

            message_dicts = []
            for idx, row in enumerate(rows):
                role = row.get("role") or "assistant"  # DB column is 'role'
                content = row.get("content") or ""

                # Parse metadata
                metadata = row.get("metadata")
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                metadata = metadata or {}

                msg_dict: dict[str, Any] = {
                    "role": role,
                    "content": content,
                    "timestamp": row["created_at"].isoformat() if row.get("created_at") else None,
                }

                # For tool messages, reconstruct tool call metadata
                if role == "tool" and metadata:
                    if metadata.get("tool_call_id"):
                        msg_dict["tool_call_id"] = metadata["tool_call_id"]
                    if metadata.get("tool_name"):
                        msg_dict["tool_name"] = metadata["tool_name"]
                    if metadata.get("tool_arguments"):
                        msg_dict["tool_arguments"] = metadata["tool_arguments"]

                # Compress long ASSISTANT messages on load (NEVER tool messages)
                if (
                    compress_on_load
                    and role == "assistant"
                    and len(content) > self.compressor.min_length_for_compression
                ):
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
