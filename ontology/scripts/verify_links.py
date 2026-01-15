#!/usr/bin/env python3
"""
Verify links in the ontology markdown files.

This script:
1. Parses all markdown files for entity keys (from frontmatter)
2. Checks that all [[entity-key|...]] references resolve to existing entities
3. Verifies parent/children relationships are bidirectional
4. Reports broken links, orphaned entities, and relationship inconsistencies

Usage:
    python ontology/scripts/verify_links.py
    python ontology/scripts/verify_links.py /path/to/ontology
"""

import re
import sys
from pathlib import Path
from typing import NamedTuple

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)


class Entity(NamedTuple):
    """Represents an entity parsed from a markdown file."""

    key: str
    title: str
    file_path: Path
    parent: str | None
    children: list[str]
    related: list[str]


def parse_frontmatter(content: str) -> dict | None:
    """Extract YAML frontmatter from markdown content."""
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return None


def extract_entity_links(content: str) -> list[str]:
    """
    Extract all [[entity-key|...]] references from content.

    Link format: [[entity-key|Display Text]]
    Entity keys must be lowercase letters, numbers, and hyphens.
    """
    pattern = r"\[\[([a-z0-9-]+)\|[^\]]+\]\]"
    return re.findall(pattern, content)


def load_entities(ontology_dir: Path) -> dict[str, Entity]:
    """Load all entities from markdown files."""
    entities = {}

    for md_file in ontology_dir.rglob("*.md"):
        # Skip special files
        if md_file.name.startswith("_"):
            continue
        if "scripts" in md_file.parts:
            continue
        if md_file.name == "README.md":
            continue  # Skip README files

        content = md_file.read_text()
        frontmatter = parse_frontmatter(content)

        if not frontmatter:
            continue

        entity_key = frontmatter.get("entity_key")
        if not entity_key:
            continue

        entity = Entity(
            key=entity_key,
            title=frontmatter.get("title", entity_key),
            file_path=md_file,
            parent=frontmatter.get("parent"),
            children=frontmatter.get("children", []) or [],
            related=frontmatter.get("related", []) or [],
        )

        if entity_key in entities:
            print(f"WARNING: Duplicate entity key '{entity_key}'")
            print(f"  - {entities[entity_key].file_path}")
            print(f"  - {md_file}")

        entities[entity_key] = entity

    return entities


def verify_links(ontology_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """
    Verify all links in the ontology.

    Returns:
        Tuple of (errors, warnings, info messages)
    """
    errors = []
    warnings = []
    info = []

    entities = load_entities(ontology_dir)
    info.append(f"Found {len(entities)} entities")

    # Root entities that don't need parents
    root_keys = {"ai-root"}

    # Check all markdown files for entity link references
    for md_file in ontology_dir.rglob("*.md"):
        if "scripts" in md_file.parts:
            continue
        if md_file.name == "README.md":
            continue

        content = md_file.read_text()
        referenced_keys = extract_entity_links(content)

        for ref_key in referenced_keys:
            if ref_key not in entities and ref_key not in root_keys:
                errors.append(
                    f"Broken link: [[{ref_key}|...]] in {md_file.name} "
                    f"(entity '{ref_key}' not found)"
                )

    # Verify parent/child relationships
    for key, entity in entities.items():
        # Check parent exists (allow special root keys)
        if entity.parent and entity.parent not in entities:
            if entity.parent not in root_keys:
                errors.append(
                    f"Invalid parent: '{entity.parent}' for entity '{key}' "
                    f"(parent not found)"
                )

        # Check children exist
        for child_key in entity.children:
            if child_key not in entities:
                warnings.append(
                    f"Missing child: '{child_key}' listed in '{key}' "
                    f"(child entity not found)"
                )

        # Check related entities exist
        for related_key in entity.related:
            if related_key not in entities and related_key not in root_keys:
                warnings.append(
                    f"Missing related: '{related_key}' listed in '{key}' "
                    f"(related entity not found)"
                )

        # Verify bidirectional parent-child relationships
        if entity.parent and entity.parent in entities:
            parent_entity = entities[entity.parent]
            if key not in parent_entity.children:
                warnings.append(
                    f"Non-bidirectional: '{key}' has parent '{entity.parent}' "
                    f"but parent doesn't list '{key}' as child"
                )

        for child_key in entity.children:
            if child_key in entities:
                child_entity = entities[child_key]
                if child_entity.parent != key:
                    warnings.append(
                        f"Non-bidirectional: '{key}' lists '{child_key}' as child "
                        f"but child's parent is '{child_entity.parent}'"
                    )

    # Find orphaned entities (no parent, not root)
    for key, entity in entities.items():
        if key not in root_keys and not entity.parent:
            warnings.append(f"Orphaned entity: '{key}' has no parent")

    return errors, warnings, info


def main():
    """Run link verification."""
    # Get ontology directory from args or find relative to script
    if len(sys.argv) > 1:
        ontology_dir = Path(sys.argv[1])
    else:
        script_dir = Path(__file__).parent
        ontology_dir = script_dir.parent

    if not ontology_dir.exists():
        print(f"ERROR: Ontology directory not found: {ontology_dir}")
        sys.exit(1)

    print(f"Verifying links in: {ontology_dir}\n")

    errors, warnings, info = verify_links(ontology_dir)

    # Print results
    for msg in info:
        print(f"INFO: {msg}")
    print()

    if warnings:
        print("WARNINGS:")
        for msg in warnings:
            print(f"  - {msg}")
        print()

    if errors:
        print("ERRORS:")
        for msg in errors:
            print(f"  - {msg}")
        print()
        print(f"Verification FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        sys.exit(1)
    else:
        print(f"Verification PASSED: 0 errors, {len(warnings)} warning(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()
