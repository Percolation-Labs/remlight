"""Integration tests for file parsing with database."""

from pathlib import Path

import pytest

# Path to test data
TEST_DATA_DIR = Path(__file__).parent.parent / "data"
TEST_PDF_PATH = TEST_DATA_DIR / "test_document.pdf"


@pytest.fixture
async def clean_files(db):
    """Clean up test files after each test."""
    yield
    await db.execute("DELETE FROM files WHERE user_id = 'test-user'")


@pytest.mark.asyncio
async def test_parse_markdown_saves_to_db(db, tmp_path, clean_files):
    """Test that parse_file saves File entity to database for markdown."""
    from remlight.services.content import ContentService
    from remlight.services.repository import Repository
    from remlight.models.entities import File

    # Create test file
    test_file = tmp_path / "test.md"
    test_file.write_text("# Test Document\n\nThis is a test document.")

    # Parse and save
    service = ContentService()
    result = await service.parse_file(
        uri=str(test_file),
        user_id="test-user",
        save_to_db=True,
    )

    assert result["status"] == "completed"
    assert result["name"] == "test.md"
    assert result["uri_hash"] is not None
    assert result["file_id"] is not None

    # Verify file was saved to database
    repo = Repository(File, table_name="files")
    saved_file = await repo.get_by_name("test.md")

    assert saved_file is not None
    assert saved_file.name == "test.md"
    assert saved_file.processing_status == "completed"
    assert "# Test Document" in saved_file.content
    assert saved_file.parsed_output["provider"] == "text"


@pytest.mark.asyncio
async def test_parse_file_upsert_by_uri_hash(db, tmp_path, clean_files):
    """Test that parsing the same file updates existing record (no duplicate)."""
    from remlight.services.content import ContentService

    # Create test file
    test_file = tmp_path / "upsert_test.md"
    test_file.write_text("Initial content")

    service = ContentService()

    # First parse
    result1 = await service.parse_file(
        uri=str(test_file),
        user_id="test-user",
        save_to_db=True,
    )

    uri_hash = result1["uri_hash"]

    # Update file content
    test_file.write_text("Updated content")

    # Second parse (same URI, should update not create duplicate)
    result2 = await service.parse_file(
        uri=str(test_file),
        user_id="test-user",
        save_to_db=True,
    )

    # Should have same uri_hash
    assert result2["uri_hash"] == uri_hash

    # Verify only one record exists with that uri_hash
    count = await db.fetchval(
        "SELECT COUNT(*) FROM files WHERE uri_hash = $1",
        uri_hash
    )
    assert count == 1, f"Expected 1 file record, got {count} (duplicate created!)"


@pytest.mark.asyncio
async def test_parse_file_no_save(db, tmp_path):
    """Test parse_file with save_to_db=False doesn't write to database."""
    from remlight.services.content import ContentService

    # Create test file
    test_file = tmp_path / "nosave.md"
    test_file.write_text("No save content")

    service = ContentService()

    result = await service.parse_file(
        uri=str(test_file),
        user_id="test-user",
        save_to_db=False,
    )

    assert result["status"] == "completed"
    assert result["content"] == "No save content"

    # Verify file was NOT saved to database
    count = await db.fetchval(
        "SELECT COUNT(*) FROM files WHERE name = $1",
        "nosave.md"
    )
    assert count == 0


@pytest.mark.asyncio
async def test_parse_file_error_handling(tmp_path):
    """Test parse_file returns error for missing file."""
    from remlight.services.content import ContentService

    service = ContentService()

    result = await service.parse_file(
        uri="/nonexistent/file.md",
        user_id="test-user",
        save_to_db=False,
    )

    assert result["status"] == "failed"
    assert "error" in result


@pytest.mark.asyncio
async def test_parse_file_mime_type_detection(db, tmp_path, clean_files):
    """Test MIME type detection for parsed files."""
    from remlight.services.content import ContentService

    # Test markdown
    md_file = tmp_path / "test.md"
    md_file.write_text("# Markdown")

    service = ContentService()
    result = await service.parse_file(
        uri=str(md_file),
        user_id="test-user",
        save_to_db=True,
    )

    assert result["mime_type"] == "text/markdown"

    # Test JSON
    json_file = tmp_path / "test.json"
    json_file.write_text('{"key": "value"}')

    result = await service.parse_file(
        uri=str(json_file),
        user_id="test-user",
        save_to_db=True,
    )

    assert result["mime_type"] == "application/json"


@pytest.mark.asyncio
async def test_parse_file_tool(db, tmp_path, clean_files):
    """Test parse_file MCP tool function."""
    from remlight.api.routers.tools import parse_file

    # Create test file
    test_file = tmp_path / "tool_test.md"
    test_file.write_text("# Tool Test\n\nContent")

    result = await parse_file(
        uri=str(test_file),
        user_id="test-user",
        save_to_db=True,
    )

    assert result["status"] == "completed"
    assert result["name"] == "tool_test.md"
    assert "# Tool Test" in result["content"]


@pytest.mark.asyncio
async def test_files_table_exists(db):
    """Test that files table was created by install.sql."""
    # Check table exists
    exists = await db.fetchval("""
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'files'
        )
    """)
    assert exists is True

    # Check columns
    columns = await db.fetch("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'files'
    """)
    column_names = [c["column_name"] for c in columns]

    assert "id" in column_names
    assert "name" in column_names
    assert "uri" in column_names
    assert "uri_hash" in column_names
    assert "content" in column_names
    assert "parsed_output" in column_names
    assert "processing_status" in column_names
    assert "mime_type" in column_names


@pytest.mark.asyncio
async def test_parse_pdf_with_kreuzberg(db, clean_files):
    """Test parsing a real PDF file with Kreuzberg.

    Requires: pip install kreuzberg>=4.0
    """
    pytest.importorskip("kreuzberg", reason="kreuzberg not installed")

    from remlight.services.content import ContentService
    from remlight.services.repository import Repository
    from remlight.models.entities import File

    # Use test PDF
    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found: {TEST_PDF_PATH}")

    service = ContentService()
    result = await service.parse_file(
        uri=str(TEST_PDF_PATH),
        user_id="test-user",
        save_to_db=True,
    )

    assert result["status"] == "completed"
    assert result["name"] == "test_document.pdf"
    assert result["mime_type"] == "application/pdf"

    # Check content was extracted
    assert result["content"] is not None
    assert len(result["content"]) > 0

    # Check parsed_output metadata
    assert result["parsed_output"]["provider"] == "doc"
    assert "kreuzberg" in result["parsed_output"]["metadata"]["parser"]
    assert result["parsed_output"]["metadata"]["version"] == "4.x"

    # Verify file was saved to database
    repo = Repository(File, table_name="files")
    saved_file = await repo.get_by_name("test_document.pdf")

    assert saved_file is not None
    assert saved_file.processing_status == "completed"
    assert saved_file.mime_type == "application/pdf"


@pytest.mark.asyncio
async def test_parse_pdf_reparse_no_duplicate(db, clean_files):
    """Test that re-parsing same PDF doesn't create duplicate File entry."""
    pytest.importorskip("kreuzberg", reason="kreuzberg not installed")

    from remlight.services.content import ContentService

    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found: {TEST_PDF_PATH}")

    service = ContentService()

    # First parse
    result1 = await service.parse_file(
        uri=str(TEST_PDF_PATH),
        user_id="test-user",
        save_to_db=True,
    )

    uri_hash = result1["uri_hash"]
    file_id_1 = result1["file_id"]

    # Second parse (same file)
    result2 = await service.parse_file(
        uri=str(TEST_PDF_PATH),
        user_id="test-user",
        save_to_db=True,
    )

    # Same uri_hash
    assert result2["uri_hash"] == uri_hash

    # Verify only one record exists
    count = await db.fetchval(
        "SELECT COUNT(*) FROM files WHERE uri_hash = $1",
        uri_hash
    )
    assert count == 1, f"Expected 1 file, got {count} (duplicate created on re-parse!)"


@pytest.mark.asyncio
async def test_parse_pdf_content_extraction(db, clean_files):
    """Test that PDF text content is properly extracted."""
    pytest.importorskip("kreuzberg", reason="kreuzberg not installed")

    from remlight.services.content import ContentService

    if not TEST_PDF_PATH.exists():
        pytest.skip(f"Test PDF not found: {TEST_PDF_PATH}")

    service = ContentService()
    result = await service.parse_file(
        uri=str(TEST_PDF_PATH),
        user_id="test-user",
        save_to_db=False,
    )

    assert result["status"] == "completed"
    content = result["content"]

    # PDF should contain these text strings
    assert "REMLight" in content or "Test" in content or "Document" in content, \
        f"Expected PDF content, got: {content[:200]}"
