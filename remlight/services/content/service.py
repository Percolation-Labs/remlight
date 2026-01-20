"""
ContentService for file processing.

Pipeline:
1. Read file from URI (local, S3, or HTTP) via FileSystemService
2. Extract content via provider plugins
3. Save File entity to database with parsed_output
4. Optionally chunk and embed into resources (future: ingest_resources)
"""

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from remlight.models.entities import File
from remlight.services.fs import FileSystemService, get_fs_service

from .providers import (
    ContentProvider,
    DocProvider,
    TextProvider,
    DOC_EXTENSIONS,
    TEXT_EXTENSIONS,
)


class ContentService:
    """
    Service for processing files: read → extract → save.

    Supports (via FileSystemService):
    - Local file paths (file:// or plain paths)
    - S3 URIs (s3://bucket/key)
    - HTTP/HTTPS URLs
    - Pluggable content providers
    """

    def __init__(self, fs: FileSystemService | None = None):
        """
        Initialize ContentService.

        Args:
            fs: FileSystemService instance (uses singleton if not provided)
        """
        self.fs = fs or get_fs_service()
        self.providers: dict[str, ContentProvider] = {}
        self._register_default_providers()

    def _register_default_providers(self):
        """Register default content providers."""
        # Text provider for plain text, code, data files
        text_provider = TextProvider()
        for ext in TEXT_EXTENSIONS:
            self.providers[ext.lower()] = text_provider

        # Doc provider for PDFs, Office docs, images (via Kreuzberg)
        doc_provider = DocProvider()
        for ext in DOC_EXTENSIONS:
            self.providers[ext.lower()] = doc_provider

        logger.debug(f"Registered {len(self.providers)} file extensions")

    async def process_uri(self, uri: str, allow_local: bool = True) -> dict[str, Any]:
        """
        Process a file URI and extract content.

        Args:
            uri: File URI (s3://bucket/key, file:///path, http://, or plain path)
            allow_local: Whether to allow local file paths

        Returns:
            dict with:
                - uri: Original URI
                - content: Extracted text content
                - metadata: File metadata (size, type, etc.)
                - provider: Provider used for extraction

        Raises:
            ValueError: If URI format is invalid
            FileNotFoundError: If file doesn't exist
            RuntimeError: If no provider available for file type
        """
        logger.info(f"Processing URI: {uri}")

        # Read file content via FileSystemService
        content_bytes, file_name, source_type = await self.fs.read_uri(
            uri, allow_local=allow_local
        )

        # Get file extension
        file_path = Path(file_name)
        suffix = file_path.suffix

        # Build metadata
        metadata = {
            "size": len(content_bytes),
            "source_type": source_type,
        }

        # Extract content using provider
        provider = self._get_provider(suffix)
        extracted_content = provider.extract(content_bytes, metadata)

        # Resolve URI for local files
        resolved_uri = uri
        if source_type == "local" and not uri.startswith(("file://", "s3://", "http")):
            resolved_uri = str(Path(uri).absolute())

        return {
            "uri": resolved_uri,
            "content": extracted_content["text"],
            "metadata": {**metadata, **extracted_content.get("metadata", {})},
            "provider": provider.name,
        }

    def _get_provider(self, suffix: str) -> ContentProvider:
        """Get content provider for file extension."""
        suffix_lower = suffix.lower()

        if suffix_lower not in self.providers:
            raise RuntimeError(
                f"No provider available for file type: {suffix}. "
                f"Supported: {', '.join(sorted(set(self.providers.keys())))}"
            )

        return self.providers[suffix_lower]

    def register_provider(self, extensions: list[str], provider: ContentProvider):
        """Register a custom content provider."""
        for ext in extensions:
            ext_lower = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            self.providers[ext_lower] = provider
            logger.debug(f"Registered provider '{provider.name}' for {ext_lower}")

    @staticmethod
    def uri_hash(uri: str) -> str:
        """Generate SHA256 hash of URI for deduplication."""
        return hashlib.sha256(uri.encode()).hexdigest()

    async def parse_file(
        self,
        uri: str,
        user_id: str | None = None,
        save_to_db: bool = True,
        file_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Parse a file and optionally save to database.

        Files are stored globally by default. Avoid setting user_id unless you
        specifically need per-user file isolation (this prevents file sharing).

        Args:
            uri: File URI (local path, s3://, http://)
            user_id: Optional user ID for scoping (rarely needed - makes file
                     user-specific and not visible to other users)
            save_to_db: Whether to save File entity to database
            file_id: Optional custom file ID (defaults to generated UUID).
                     Useful for deterministic IDs or external system integration.

        Returns:
            dict with:
                - file_id: UUID of File entity (if saved)
                - uri: Original URI
                - uri_hash: SHA256 hash of URI
                - content: Extracted text content
                - parsed_output: Full parsing result
                - status: 'completed' or 'failed'
        """
        uri_hash = self.uri_hash(uri)

        try:
            # Process file
            result = await self.process_uri(uri)

            # Build parsed_output
            parsed_output = {
                "text": result["content"],
                "metadata": result["metadata"],
                "provider": result["provider"],
            }

            # Get file info
            file_path = Path(urlparse(uri).path or uri)
            file_name = file_path.name

            # Detect MIME type
            mime_type = result["metadata"].get("content_type")
            if not mime_type:
                import mimetypes
                mime_type, _ = mimetypes.guess_type(file_name)

            # Create File entity (use custom file_id if provided)
            file_entity = File(
                name=file_name,
                uri=uri,
                uri_hash=uri_hash,
                content=result["content"],
                mime_type=mime_type,
                size_bytes=result["metadata"].get("size"),
                processing_status="completed",
                parsed_output=parsed_output,
                user_id=user_id,
                tenant_id=user_id,
            )
            if file_id:
                file_entity.id = file_id

            actual_file_id = str(file_entity.id)

            # Save to database if requested
            if save_to_db:
                from remlight.services.database import get_db
                from remlight.services.repository import Repository

                db = get_db()
                # Only connect if not already connected (allows reusing test fixtures)
                was_connected = db.pool is not None
                if not was_connected:
                    await db.connect()
                try:
                    repo = Repository(File, table_name="files")
                    await repo.upsert(file_entity, conflict_field="uri_hash")
                    logger.info(f"Saved File: {file_name} (id={actual_file_id})")
                finally:
                    # Only disconnect if we initiated the connection
                    if not was_connected:
                        await db.disconnect()

            return {
                "file_id": actual_file_id,
                "uri": uri,
                "uri_hash": uri_hash,
                "name": file_name,
                "content": result["content"],
                "parsed_output": parsed_output,
                "mime_type": mime_type,
                "size_bytes": result["metadata"].get("size"),
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return {
                "uri": uri,
                "uri_hash": uri_hash,
                "status": "failed",
                "error": str(e),
            }

    async def ingest_resources(
        self,
        file_id: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Chunk and embed file content into resources table.

        NOT IMPLEMENTED YET - placeholder for future chunking/embedding.

        Args:
            file_id: UUID of File entity to process
            user_id: User ID for scoping

        Returns:
            dict with resource creation results
        """
        # TODO: Implement chunking and embedding into resources table
        # 1. Load File by ID
        # 2. Chunk content using semantic chunking
        # 3. Create Resource entities for each chunk
        # 4. Generate embeddings
        raise NotImplementedError(
            "ingest_resources not implemented yet. "
            "Use parse_file() for content extraction."
        )


# Singleton instance
_content_service: ContentService | None = None


def get_content_service() -> ContentService:
    """Get or create ContentService singleton."""
    global _content_service
    if _content_service is None:
        _content_service = ContentService()
    return _content_service
