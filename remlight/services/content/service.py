"""
ContentService for file processing.

Pipeline:
1. Read file from URI (local or S3)
2. Extract content via provider plugins
3. Save File entity to database with parsed_output
4. Optionally chunk and embed into resources (future: ingest_resources)
"""

import hashlib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError
from loguru import logger

from remlight.models.entities import File
from remlight.settings import settings

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

    Supports:
    - Local file paths (file:// or plain paths)
    - S3 URIs (s3://bucket/key)
    - Pluggable content providers
    """

    def __init__(self):
        self.providers: dict[str, ContentProvider] = {}
        self._register_default_providers()
        self._s3_client = None

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

    @property
    def s3_client(self):
        """Lazy-load S3 client."""
        if self._s3_client is None:
            self._s3_client = self._create_s3_client()
        return self._s3_client

    def _create_s3_client(self):
        """Create S3 client with environment credentials."""
        import os

        s3_config: dict[str, Any] = {
            "region_name": os.getenv("AWS_REGION", "us-east-1"),
        }

        # Custom endpoint for MinIO/LocalStack
        endpoint_url = os.getenv("S3__ENDPOINT_URL")
        if endpoint_url:
            s3_config["endpoint_url"] = endpoint_url

        # Access keys (not needed with IRSA in EKS)
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        if access_key and secret_key:
            s3_config["aws_access_key_id"] = access_key
            s3_config["aws_secret_access_key"] = secret_key

        return boto3.client("s3", **s3_config)

    def process_uri(self, uri: str) -> dict[str, Any]:
        """
        Process a file URI and extract content.

        Args:
            uri: File URI (s3://bucket/key, file:///path, or plain path)

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

        # Determine if S3 or local file
        if uri.startswith("s3://"):
            return self._process_s3_uri(uri)
        else:
            return self._process_local_file(uri)

    def _process_s3_uri(self, uri: str) -> dict[str, Any]:
        """Process S3 URI."""
        parsed = urlparse(uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")

        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: {uri}")

        logger.debug(f"Downloading s3://{bucket}/{key}")

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content_bytes = response["Body"].read()

            metadata = {
                "size": response["ContentLength"],
                "content_type": response.get("ContentType", ""),
                "last_modified": response["LastModified"].isoformat(),
                "etag": response.get("ETag", "").strip('"'),
            }

            # Extract content using provider
            file_path = Path(key)
            provider = self._get_provider(file_path.suffix)

            extracted_content = provider.extract(content_bytes, metadata)

            return {
                "uri": uri,
                "content": extracted_content["text"],
                "metadata": {**metadata, **extracted_content.get("metadata", {})},
                "provider": provider.name,
            }

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"S3 object not found: {uri}") from e
            elif error_code == "NoSuchBucket":
                raise FileNotFoundError(f"S3 bucket not found: {bucket}") from e
            else:
                raise RuntimeError(f"S3 error: {e}") from e

    def _process_local_file(self, path: str) -> dict[str, Any]:
        """Process local file path."""
        # Handle file:// URI scheme
        if path.startswith("file://"):
            path = path.replace("file://", "")

        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {path}")

        logger.debug(f"Reading local file: {file_path}")

        # Read file content
        content_bytes = file_path.read_bytes()

        # Get metadata
        stat = file_path.stat()
        metadata = {
            "size": stat.st_size,
            "modified": stat.st_mtime,
        }

        # Extract content using provider
        provider = self._get_provider(file_path.suffix)
        extracted_content = provider.extract(content_bytes, metadata)

        return {
            "uri": str(file_path.absolute()),
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

        Args:
            uri: File URI (local path, s3://)
            user_id: User ID for scoping
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
            result = self.process_uri(uri)

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
