"""
Filesystem Abstraction Layer for REMLight.

Provides a unified async interface for reading files from different sources:
- Local filesystem paths
- S3 URIs (s3://bucket/key)
- HTTP/HTTPS URLs

Usage:
    from remlight.services.fs import FileSystemService

    fs = FileSystemService()
    content, filename, source_type = await fs.read_uri("s3://bucket/file.pdf")
    content, filename, source_type = await fs.read_uri("/local/path/file.pdf")
    content, filename, source_type = await fs.read_uri("https://example.com/file.pdf")
"""

import os
from pathlib import Path
from urllib.parse import urlparse

from loguru import logger


class FileSystemService:
    """
    Async service for reading files from various sources (local, S3, HTTP).
    """

    def __init__(
        self,
        s3_endpoint_url: str | None = None,
        s3_access_key_id: str | None = None,
        s3_secret_access_key: str | None = None,
        s3_region: str | None = None,
    ):
        """
        Initialize FileSystemService.

        Args:
            s3_endpoint_url: Custom S3 endpoint (for MinIO/LocalStack)
            s3_access_key_id: AWS access key (or use env AWS_ACCESS_KEY_ID)
            s3_secret_access_key: AWS secret key (or use env AWS_SECRET_ACCESS_KEY)
            s3_region: AWS region (or use env AWS_REGION, default: us-east-1)
        """
        self.s3_endpoint_url = s3_endpoint_url or os.getenv("S3__ENDPOINT_URL")
        self.s3_access_key_id = s3_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.s3_secret_access_key = s3_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.s3_region = s3_region or os.getenv("AWS_REGION", "us-east-1")

    async def read_uri(
        self,
        file_uri: str,
        allow_local: bool = True,
    ) -> tuple[bytes, str, str]:
        """
        Read content from a given URI.

        Args:
            file_uri: The URI of the file to read (local path, s3://, http://)
            allow_local: Whether to allow local file paths (security setting)

        Returns:
            Tuple of (content_bytes, filename, source_type)
            - source_type is one of: "local", "s3", "url"

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If local files not allowed
            ValueError: If URI scheme is unsupported
            RuntimeError: If download fails
        """
        parsed = urlparse(file_uri)
        scheme = parsed.scheme

        if scheme in ("http", "https"):
            source_type = "url"
            file_name = Path(parsed.path).name or "downloaded_file"
            content = await self._read_from_url(file_uri)

        elif scheme == "s3":
            source_type = "s3"
            s3_bucket = parsed.netloc
            s3_key = parsed.path.lstrip("/")
            file_name = Path(s3_key).name
            content = await self._read_from_s3(s3_bucket, s3_key)

        elif scheme == "" or scheme == "file":
            if not allow_local:
                raise PermissionError(
                    "Local file paths are not allowed in this context."
                )
            source_type = "local"
            file_path = Path(file_uri.replace("file://", ""))
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_uri}")
            file_name = file_path.name
            content = await self._read_from_local(file_path)

        else:
            raise ValueError(f"Unsupported URI scheme: {scheme}")

        return content, file_name, source_type

    async def _read_from_url(self, url: str) -> bytes:
        """Read content from HTTP/HTTPS URL."""
        try:
            import aiohttp
        except ImportError:
            # Fallback to sync httpx if aiohttp not available
            import httpx
            logger.debug(f"Reading from URL (httpx): {url}")
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                response.raise_for_status()
                return response.content

        logger.debug(f"Reading from URL (aiohttp): {url}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                return await response.read()

    async def _read_from_s3(self, bucket: str, key: str) -> bytes:
        """Read content from S3."""
        try:
            import aioboto3
        except ImportError:
            # Fallback to sync boto3
            return await self._read_from_s3_sync(bucket, key)

        logger.debug(f"Reading from S3 (async): s3://{bucket}/{key}")
        session = aioboto3.Session()

        client_config = {
            "region_name": self.s3_region,
        }
        if self.s3_endpoint_url:
            client_config["endpoint_url"] = self.s3_endpoint_url
        if self.s3_access_key_id and self.s3_secret_access_key:
            client_config["aws_access_key_id"] = self.s3_access_key_id
            client_config["aws_secret_access_key"] = self.s3_secret_access_key

        async with session.client("s3", **client_config) as s3_client:
            try:
                response = await s3_client.get_object(Bucket=bucket, Key=key)
                return await response["Body"].read()
            except Exception as e:
                logger.error(f"S3 download failed: {e}")
                raise RuntimeError(f"S3 download failed: {e}") from e

    async def _read_from_s3_sync(self, bucket: str, key: str) -> bytes:
        """Fallback sync S3 read using boto3."""
        import boto3
        from botocore.exceptions import ClientError

        logger.debug(f"Reading from S3 (sync fallback): s3://{bucket}/{key}")

        client_config = {
            "region_name": self.s3_region,
        }
        if self.s3_endpoint_url:
            client_config["endpoint_url"] = self.s3_endpoint_url
        if self.s3_access_key_id and self.s3_secret_access_key:
            client_config["aws_access_key_id"] = self.s3_access_key_id
            client_config["aws_secret_access_key"] = self.s3_secret_access_key

        s3_client = boto3.client("s3", **client_config)

        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"S3 object not found: s3://{bucket}/{key}") from e
            elif error_code == "NoSuchBucket":
                raise FileNotFoundError(f"S3 bucket not found: {bucket}") from e
            else:
                raise RuntimeError(f"S3 error: {e}") from e

    async def _read_from_local(self, path: Path) -> bytes:
        """Read content from a local file."""
        logger.debug(f"Reading from local path: {path}")
        return path.read_bytes()


# Singleton instance
_fs_service: FileSystemService | None = None


def get_fs_service() -> FileSystemService:
    """Get or create FileSystemService singleton."""
    global _fs_service
    if _fs_service is None:
        _fs_service = FileSystemService()
    return _fs_service
