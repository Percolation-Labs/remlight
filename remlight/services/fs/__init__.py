"""Filesystem abstraction layer for REMLight."""

from remlight.services.fs.service import (
    FileSystemService,
    get_fs_service,
)
from remlight.services.fs.files import (
    temp_file_from_bytes,
    temp_file_empty,
    temp_directory,
    ensure_parent_exists,
    safe_delete,
)

__all__ = [
    "FileSystemService",
    "get_fs_service",
    "temp_file_from_bytes",
    "temp_file_empty",
    "temp_directory",
    "ensure_parent_exists",
    "safe_delete",
]
