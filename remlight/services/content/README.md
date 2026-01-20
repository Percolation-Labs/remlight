# Content Service

Document parsing and ingestion service using [Kreuzberg](https://github.com/kreuzberg-dev/kreuzberg) v4.0+ (Rust core, no torch dependencies).

## Supported Formats

| Provider | Extensions | Description |
|----------|------------|-------------|
| `text` | `.md`, `.txt`, `.json`, `.yaml`, `.py`, `.js`, `.ts`, etc. | UTF-8 text extraction |
| `doc` | `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.png`, `.jpg` | Kreuzberg extraction with OCR fallback |

## Quick Start

```python
from remlight.services.content import ContentService

service = ContentService()

# Parse and save to files table (default)
result = await service.parse_file(uri="/path/to/document.pdf")

# Parse from S3
result = await service.parse_file(uri="s3://bucket/report.docx")

# Parse from URL
result = await service.parse_file(uri="https://example.com/paper.pdf")

# Parse without saving to DB
result = await service.parse_file(uri="/path/to/document.pdf", save_to_db=False)
```

**Note**: Files are stored globally by default. Avoid setting `user_id` unless you specifically need per-user file isolation (this prevents file sharing).

## Where Data Gets Saved

### Files Table (`files`)

| Column | Type | Description |
|--------|------|-------------|
| `id` | UUID | Primary key |
| `name` | VARCHAR | Original filename |
| `uri` | VARCHAR | Source path (local or s3://) |
| `uri_hash` | VARCHAR(64) | SHA256 of URI for deduplication |
| `content` | TEXT | Extracted text content |
| `mime_type` | VARCHAR | MIME type (application/pdf, etc.) |
| `size_bytes` | BIGINT | File size |
| `processing_status` | VARCHAR | pending, completed, failed |
| `parsed_output` | JSONB | Full parse result with metadata |

## Parse Result JSON

### PDF Example

```python
result = await service.parse_file(uri="tests/data/test_document.pdf")
```

**Returns:**

```json
{
  "file_id": "5ca89624-bfe8-428a-9dd0-ab81e1930127",
  "uri": "/Users/sirsh/code/mr_saoirse/remlight/tests/data/test_document.pdf",
  "uri_hash": "17c2622d61ac489dfbdb89c4a0e0369188347044676d596dfd53685ec996b9af",
  "name": "test_document.pdf",
  "content": "REMLight Test Document\r\nThis is a test PDF for integration testing.\r\nIt contains simple text content.\r\nCreated for file parsing tests.",
  "parsed_output": {
    "text": "REMLight Test Document\r\nThis is a test PDF for integration testing...",
    "metadata": {
      "size": 763,
      "parser": "kreuzberg",
      "version": "4.x",
      "file_extension": ".pdf"
    },
    "provider": "doc"
  },
  "mime_type": "application/pdf",
  "size_bytes": 763,
  "status": "completed"
}
```

### Database Record

After saving, query the `files` table:

```sql
SELECT id, name, uri_hash, mime_type, processing_status, parsed_output
FROM files
WHERE name = 'test_document.pdf';
```

The `parsed_output` JSONB column contains:

```json
{
  "text": "REMLight Test Document\r\nThis is a test PDF for integration testing...",
  "metadata": {
    "size": 763,
    "parser": "kreuzberg",
    "version": "4.x",
    "file_extension": ".pdf"
  },
  "provider": "doc"
}
```

## Deduplication

Files are deduplicated by `uri_hash` (SHA256 of the URI). Re-parsing the same file updates the existing record instead of creating a duplicate:

```python
# First parse - creates new record
result1 = await service.parse_file(uri="/path/to/file.pdf", save_to_db=True)

# Second parse - updates existing record (same uri_hash)
result2 = await service.parse_file(uri="/path/to/file.pdf", save_to_db=True)

assert result1["uri_hash"] == result2["uri_hash"]  # Same hash
```

## CLI Usage

```bash
# Parse and save to database
rem parse document.pdf

# Parse from S3
rem parse s3://bucket/report.docx

# Parse from URL
rem parse https://example.com/paper.pdf

# Parse without saving
rem parse document.pdf --no-save

# Output as plain text
rem parse document.pdf --output text
```

## MCP Tool

The `parse_file` tool is available via MCP:

```python
from remlight.api.routers.tools import parse_file

result = await parse_file(uri="/path/to/document.pdf")
result = await parse_file(uri="s3://bucket/report.docx")
result = await parse_file(uri="https://example.com/paper.pdf", save_to_db=False)
```

## S3 and HTTP Support

```python
# S3
result = await service.parse_file(uri="s3://my-bucket/documents/report.pdf")

# HTTP/HTTPS
result = await service.parse_file(uri="https://example.com/paper.pdf")
```

Requires `boto3` for S3 and AWS credentials configured.

## Installation

```bash
# Install with content dependencies
pip install remlight[content]

# Or install kreuzberg separately
pip install kreuzberg>=4.0
```

## Future: ingest_resources

The `ingest_resources()` method is a placeholder for chunking and embedding file content into the `resources` table:

```python
# NOT IMPLEMENTED YET
await service.ingest_resources(file_id="...", user_id="...")
```

This will:
1. Load the File by ID
2. Chunk content using semantic chunking
3. Create Resource entities for each chunk
4. Generate embeddings for vector search
