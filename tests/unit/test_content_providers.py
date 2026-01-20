"""Unit tests for content providers (no database required)."""

import pytest


class TestTextProvider:
    """Test TextProvider for plain text extraction."""

    def test_extract_markdown(self):
        """Test markdown extraction with heading detection."""
        from remlight.services.content.providers import TextProvider

        provider = TextProvider()
        content = b"# Title\n\nSome text\n\n## Section\n\nMore text"
        result = provider.extract(content, {})

        assert "text" in result
        assert "# Title" in result["text"]
        assert result["metadata"]["line_count"] == 7  # 7 lines including empty
        assert result["metadata"]["heading_count"] == 2
        assert result["metadata"]["encoding"] == "utf-8"

    def test_extract_json(self):
        """Test JSON file extraction."""
        from remlight.services.content.providers import TextProvider

        provider = TextProvider()
        content = b'{"key": "value", "nested": {"inner": 123}}'
        result = provider.extract(content, {})

        assert "text" in result
        assert '"key": "value"' in result["text"]

    def test_extract_latin1_fallback(self):
        """Test fallback to latin-1 for non-UTF8 content."""
        from remlight.services.content.providers import TextProvider

        provider = TextProvider()
        # Latin-1 encoded content (e.g., '\xe9' is 'é' in latin-1)
        content = b"Caf\xe9 au lait"
        result = provider.extract(content, {})

        assert "text" in result
        assert "Caf" in result["text"]

    def test_provider_name(self):
        """Test provider name property."""
        from remlight.services.content.providers import TextProvider

        provider = TextProvider()
        assert provider.name == "text"


class TestProviderMapping:
    """Test extension to provider mapping."""

    def test_text_extensions(self):
        """Test that text extensions return TextProvider."""
        from remlight.services.content.providers import get_provider_for_extension

        for ext in [".md", ".txt", ".json", ".yaml", ".py", ".js"]:
            provider = get_provider_for_extension(ext)
            assert provider is not None
            assert provider.name == "text"

    def test_doc_extensions(self):
        """Test that doc extensions return DocProvider."""
        from remlight.services.content.providers import get_provider_for_extension

        for ext in [".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg"]:
            provider = get_provider_for_extension(ext)
            assert provider is not None
            assert provider.name == "doc"

    def test_unknown_extension(self):
        """Test that unknown extensions return None."""
        from remlight.services.content.providers import get_provider_for_extension

        provider = get_provider_for_extension(".xyz")
        assert provider is None

    def test_case_insensitivity(self):
        """Test that extension matching is case-insensitive."""
        from remlight.services.content.providers import get_provider_for_extension

        provider = get_provider_for_extension(".MD")
        assert provider is not None
        assert provider.name == "text"


class TestContentService:
    """Test ContentService without database."""

    def test_uri_hash(self):
        """Test URI hash generation is deterministic."""
        from remlight.services.content.service import ContentService

        uri = "s3://bucket/path/to/file.pdf"
        hash1 = ContentService.uri_hash(uri)
        hash2 = ContentService.uri_hash(uri)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_uri_hash_different_uris(self):
        """Test different URIs produce different hashes."""
        from remlight.services.content.service import ContentService

        hash1 = ContentService.uri_hash("s3://bucket/file1.pdf")
        hash2 = ContentService.uri_hash("s3://bucket/file2.pdf")

        assert hash1 != hash2

    @pytest.mark.asyncio
    async def test_process_local_file_not_found(self):
        """Test FileNotFoundError for missing local file."""
        from remlight.services.content.service import ContentService

        service = ContentService()

        with pytest.raises(FileNotFoundError):
            await service.process_uri("/nonexistent/file.txt")

    @pytest.mark.asyncio
    async def test_process_local_markdown(self, tmp_path):
        """Test processing a local markdown file."""
        from remlight.services.content.service import ContentService

        # Create test file
        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\nHello world")

        service = ContentService()
        result = await service.process_uri(str(test_file))

        assert result["content"] == "# Test\n\nHello world"
        assert result["provider"] == "text"
        assert result["metadata"]["size"] > 0

    @pytest.mark.asyncio
    async def test_process_local_json(self, tmp_path):
        """Test processing a local JSON file."""
        from remlight.services.content.service import ContentService

        # Create test file
        test_file = tmp_path / "test.json"
        test_file.write_text('{"name": "test"}')

        service = ContentService()
        result = await service.process_uri(str(test_file))

        assert '{"name": "test"}' in result["content"]
        assert result["provider"] == "text"

    @pytest.mark.asyncio
    async def test_process_file_uri_scheme(self, tmp_path):
        """Test processing with file:// URI scheme."""
        from remlight.services.content.service import ContentService

        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        service = ContentService()
        result = await service.process_uri(f"file://{test_file}")

        assert result["content"] == "content"

    @pytest.mark.asyncio
    async def test_unsupported_extension(self, tmp_path):
        """Test RuntimeError for unsupported file extension."""
        from remlight.services.content.service import ContentService

        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")

        service = ContentService()

        with pytest.raises(RuntimeError, match="No provider available"):
            await service.process_uri(str(test_file))
