"""Content provider plugins for file parsing.

Providers extract text and metadata from different file types.
Uses Kreuzberg for document parsing (PDF, DOCX, PPTX, XLSX, images).
"""

import json
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger


class ContentProvider(ABC):
    """Base class for content extraction providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging/debugging."""
        pass

    @abstractmethod
    def extract(self, content: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Extract text content from file bytes.

        Args:
            content: Raw file bytes
            metadata: File metadata (size, type, etc.)

        Returns:
            dict with:
                - text: Extracted text content
                - metadata: Additional metadata from extraction (optional)
        """
        pass


class TextProvider(ContentProvider):
    """
    Text content provider for plain text formats.

    Supports:
    - Markdown (.md, .markdown) - With heading detection
    - JSON (.json) - Pretty-printed text extraction
    - YAML (.yaml, .yml) - Text extraction
    - Plain text (.txt) - Direct UTF-8 extraction
    - Code files (.py, .js, .ts, etc.) - Source code as text
    """

    @property
    def name(self) -> str:
        return "text"

    def extract(self, content: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract text content from plain text files."""
        # Decode UTF-8 (with fallback to latin-1)
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            logger.debug("UTF-8 decode failed, falling back to latin-1")
            text = content.decode("latin-1")

        # Basic text analysis
        lines = text.split("\n")
        headings = [line for line in lines if line.strip().startswith("#")]

        extraction_metadata = {
            "line_count": len(lines),
            "heading_count": len(headings) if headings else None,
            "char_count": len(text),
            "encoding": "utf-8",
        }

        return {
            "text": text,
            "metadata": extraction_metadata,
        }


class DocProvider(ContentProvider):
    """
    Document content provider using Kreuzberg v4.0+ (Rust core).

    Supports multiple document formats via Kreuzberg:
    - PDF (.pdf) - Text extraction with OCR fallback
    - Word (.docx) - Native format support
    - PowerPoint (.pptx) - Slide content extraction
    - Excel (.xlsx) - Spreadsheet data extraction
    - Images (.png, .jpg) - OCR text extraction

    Requires: pip install kreuzberg>=4.0
    No torch/pytorch dependencies - uses Rust core.
    """

    @property
    def name(self) -> str:
        return "doc"

    def _get_extension(self, content_type: str, default: str = ".pdf") -> str:
        """Map MIME type to file extension."""
        mime_map = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
        }
        return mime_map.get(content_type, default)

    def _is_daemon_process(self) -> bool:
        """Check if running in a daemon process."""
        import multiprocessing
        try:
            return multiprocessing.current_process().daemon
        except Exception:
            return False

    def _parse_in_subprocess(self, file_path: Path) -> dict:
        """Run kreuzberg in a separate subprocess to bypass daemon restrictions."""
        script = """
import json
import sys
from pathlib import Path
from kreuzberg import ExtractionConfig, extract_file_sync

# Parse document with kreuzberg 4.x
config = ExtractionConfig()
result = extract_file_sync(Path(sys.argv[1]), config=config)

output = {
    'content': result.content,
    'tables': [],
    'metadata': {}
}
print(json.dumps(output))
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(file_path)],
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Subprocess parsing failed: {result.stderr}")

        return json.loads(result.stdout)

    def extract(self, content: bytes, metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract document content using Kreuzberg v4.0+."""
        import tempfile

        content_type = metadata.get("content_type", "")
        suffix = self._get_extension(content_type, default=".pdf")

        # Write bytes to temp file for kreuzberg
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Check if running in daemon process (e.g., uvicorn workers)
            if self._is_daemon_process():
                logger.info("Daemon process detected - using subprocess workaround")
                try:
                    result_dict = self._parse_in_subprocess(tmp_path)
                    text = result_dict["content"]
                    extraction_metadata = {
                        "table_count": len(result_dict.get("tables", [])),
                        "parser": "kreuzberg_subprocess",
                        "version": "4.x",
                        "file_extension": suffix,
                    }
                except Exception as e:
                    logger.error(f"Subprocess parsing failed: {e}. Falling back to direct.")
                    # Fallback to direct call
                    from kreuzberg import ExtractionConfig, extract_file_sync
                    config = ExtractionConfig()
                    result = extract_file_sync(tmp_path, config=config)
                    text = result.content
                    extraction_metadata = {
                        "parser": "kreuzberg_fallback",
                        "version": "4.x",
                        "file_extension": suffix,
                    }
            else:
                # Normal execution - use sync API directly
                try:
                    from kreuzberg import ExtractionConfig, extract_file_sync

                    config = ExtractionConfig()
                    result = extract_file_sync(tmp_path, config=config)
                    text = result.content
                    extraction_metadata = {
                        "parser": "kreuzberg",
                        "version": "4.x",
                        "file_extension": suffix,
                    }
                except ImportError:
                    raise RuntimeError(
                        "Kreuzberg not installed. Install with: pip install kreuzberg>=4.0"
                    )

            return {
                "text": text,
                "metadata": extraction_metadata,
            }
        finally:
            # Clean up temp file
            try:
                tmp_path.unlink()
            except Exception:
                pass


# Extension to provider mapping
TEXT_EXTENSIONS = [
    ".md", ".markdown", ".txt", ".json", ".yaml", ".yml",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb",
    ".php", ".swift", ".kt", ".scala", ".sh", ".bash",
    ".zsh", ".fish", ".ps1", ".bat", ".cmd", ".sql",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".xml", ".toml", ".ini", ".cfg", ".conf", ".env",
]

DOC_EXTENSIONS = [
    ".pdf", ".docx", ".doc", ".pptx", ".ppt",
    ".xlsx", ".xls", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".bmp", ".tiff",
]


def get_provider_for_extension(extension: str) -> ContentProvider | None:
    """Get the appropriate provider for a file extension."""
    ext_lower = extension.lower()
    if ext_lower in TEXT_EXTENSIONS:
        return TextProvider()
    elif ext_lower in DOC_EXTENSIONS:
        return DocProvider()
    return None
