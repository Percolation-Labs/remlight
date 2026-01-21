"""SQL query builder for Pydantic models.

Generates INSERT, UPDATE, SELECT queries from Pydantic model instances.
Handles serialization and parameter binding automatically.
"""

import hashlib
import json
import uuid
from typing import Any, Type

from pydantic import BaseModel


def get_natural_key(model: BaseModel) -> str | None:
    """
    Get natural key from model following precedence: uri -> key -> name.

    Used for generating deterministic IDs from business keys.
    Does NOT include 'id' since that's what we're trying to generate.

    Args:
        model: Pydantic model instance

    Returns:
        Natural key string or None
    """
    for field in ["uri", "key", "name"]:
        if hasattr(model, field):
            value = getattr(model, field)
            if value:
                return str(value)
    return None


def get_entity_key(model: BaseModel) -> str:
    """
    Get entity key for KV store following precedence: name -> key -> uri -> id.

    For KV store lookups, we prefer human-readable identifiers first (name/key),
    then URIs, with id as the fallback. This allows users to lookup entities
    by their natural names like "panic-disorder" instead of UUIDs.

    Args:
        model: Pydantic model instance

    Returns:
        Entity key string (guaranteed to exist)
    """
    for field in ["name", "key", "uri", "id"]:
        if hasattr(model, field):
            value = getattr(model, field)
            if value:
                return str(value)
    # Should never reach here since id always exists in CoreModel
    raise ValueError(f"Model {type(model)} has no name, key, uri, or id field")


def generate_deterministic_id(user_id: str | None, entity_key: str) -> uuid.UUID:
    """
    Generate deterministic UUID from hash of (user_id, entity_key).

    Args:
        user_id: User identifier (optional)
        entity_key: Entity key field value

    Returns:
        Deterministic UUID
    """
    # Combine user_id and key for hashing
    combined = f"{user_id or 'system'}:{entity_key}"
    hash_bytes = hashlib.sha256(combined.encode()).digest()
    # Use first 16 bytes for UUID
    return uuid.UUID(bytes=hash_bytes[:16])


def model_to_dict(model: BaseModel, exclude_none: bool = True) -> dict[str, Any]:
    """
    Convert Pydantic model to dict suitable for SQL insertion.

    Generates deterministic ID if not present, based on hash(user_id, key).
    Serializes JSONB fields (list[dict], dict) to JSON strings for asyncpg.

    Args:
        model: Pydantic model instance
        exclude_none: Exclude None values (default: True)

    Returns:
        Dict of field_name -> value with JSONB fields as JSON strings
    """
    # Use python mode to preserve datetime objects
    data = model.model_dump(exclude_none=exclude_none, mode="python")

    # Generate deterministic ID if not present
    if not data.get("id"):
        natural_key = get_natural_key(model)
        if natural_key:
            user_id = data.get("user_id")
            data["id"] = generate_deterministic_id(user_id, natural_key)
        else:
            # Fallback to random UUID if no natural key (uri/key/name)
            data["id"] = uuid.uuid4()

    # Serialize JSONB fields for asyncpg
    # PostgreSQL TEXT[] array fields should remain as Python lists
    # JSONB fields (dicts, list[dict]) should be JSON-serialized
    pg_array_fields = {"tags", "interests", "preferred_topics"}  # TEXT[] columns

    for key, value in data.items():
        if key in pg_array_fields:
            # Keep as Python list for PostgreSQL TEXT[] columns
            continue
        if isinstance(value, (dict, list)) and key not in ("id",):
            data[key] = json.dumps(value)

    return data


def build_insert(
    model: BaseModel, table_name: str, return_id: bool = True
) -> tuple[str, list[Any]]:
    """
    Build INSERT query from Pydantic model.

    Args:
        model: Pydantic model instance
        table_name: Target table name
        return_id: Return the inserted ID (default: True)

    Returns:
        Tuple of (sql_query, parameters)
    """
    data = model_to_dict(model)

    fields = list(data.keys())
    placeholders = [f"${i+1}" for i in range(len(fields))]
    values = [data[field] for field in fields]

    sql = f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"

    if return_id:
        sql += " RETURNING id"

    return sql, values


def build_upsert(
    model: BaseModel,
    table_name: str,
    conflict_field: str = "id",
    return_id: bool = True,
) -> tuple[str, list[Any]]:
    """
    Build INSERT ... ON CONFLICT DO UPDATE (upsert) query from Pydantic model.

    Args:
        model: Pydantic model instance
        table_name: Target table name
        conflict_field: Field to check for conflicts (default: "id")
        return_id: Return the inserted/updated ID (default: True)

    Returns:
        Tuple of (sql_query, parameters)
    """
    data = model_to_dict(model)

    fields = list(data.keys())
    placeholders = [f"${i+1}" for i in range(len(fields))]
    values = [data[field] for field in fields]

    # Build update clause (exclude conflict field)
    update_fields = [f for f in fields if f != conflict_field]
    update_clauses = [f"{field} = EXCLUDED.{field}" for field in update_fields]
    # Always clear deleted_at on upsert to "un-delete" soft-deleted records
    if "deleted_at" not in update_fields:
        update_clauses.append("deleted_at = NULL")

    # Single-line format for easier SQL surgery in repository (embedding injection)
    sql = f"INSERT INTO {table_name} ({', '.join(fields)}) VALUES ({', '.join(placeholders)}) ON CONFLICT ({conflict_field}) DO UPDATE SET {', '.join(update_clauses)}"

    if return_id:
        sql += " RETURNING id"

    return sql, values


def build_select(
    model_class: Type[BaseModel],
    table_name: str,
    filters: dict[str, Any],
    order_by: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> tuple[str, list[Any]]:
    """
    Build SELECT query with filters.

    Args:
        model_class: Pydantic model class (for field validation)
        table_name: Source table name
        filters: Dict of field -> value filters (AND-ed together)
        order_by: Optional ORDER BY clause
        limit: Optional LIMIT
        offset: Optional OFFSET

    Returns:
        Tuple of (sql_query, parameters)
    """
    where_clauses = ["deleted_at IS NULL"]  # Soft delete filter
    params = []
    param_idx = 1

    for field, value in filters.items():
        where_clauses.append(f"{field} = ${param_idx}")
        params.append(value)
        param_idx += 1

    sql = f"SELECT * FROM {table_name} WHERE {' AND '.join(where_clauses)}"

    if order_by:
        sql += f" ORDER BY {order_by}"

    if limit is not None:
        sql += f" LIMIT ${param_idx}"
        params.append(limit)
        param_idx += 1

    if offset is not None:
        sql += f" OFFSET ${param_idx}"
        params.append(offset)

    return sql, params


def build_delete(
    table_name: str, id_value: str, tenant_id: str | None = None
) -> tuple[str, list[Any]]:
    """
    Build soft DELETE query (sets deleted_at).

    Args:
        table_name: Target table name
        id_value: ID of record to delete
        tenant_id: Optional tenant ID for isolation

    Returns:
        Tuple of (sql_query, parameters)
    """
    if tenant_id:
        sql = f"""
            UPDATE {table_name}
            SET deleted_at = NOW(), updated_at = NOW()
            WHERE id = $1 AND tenant_id = $2 AND deleted_at IS NULL
            RETURNING id
        """
        return sql.strip(), [id_value, tenant_id]
    else:
        sql = f"""
            UPDATE {table_name}
            SET deleted_at = NOW(), updated_at = NOW()
            WHERE id = $1 AND deleted_at IS NULL
            RETURNING id
        """
        return sql.strip(), [id_value]


def build_count(
    table_name: str, filters: dict[str, Any]
) -> tuple[str, list[Any]]:
    """
    Build COUNT query with filters.

    Args:
        table_name: Source table name
        filters: Dict of field -> value filters (AND-ed together)

    Returns:
        Tuple of (sql_query, parameters)
    """
    where_clauses = ["deleted_at IS NULL"]
    params = []
    param_idx = 1

    for field, value in filters.items():
        where_clauses.append(f"{field} = ${param_idx}")
        params.append(value)
        param_idx += 1

    sql = f"SELECT COUNT(*) as count FROM {table_name} WHERE {' AND '.join(where_clauses)}"

    return sql, params
