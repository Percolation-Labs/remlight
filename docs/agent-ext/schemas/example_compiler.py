"""
Compiler: Derive execution topology from schema.

Walks the Pydantic model and extracts:
- Routing table from field triggers
- Tool assignments from field metadata
- Prompts from model/field descriptions
"""

from typing import Any, get_args, get_origin
from pydantic import BaseModel


def get_inner_type(annotation) -> type | None:
    """Get the inner type from list[T] or other generic."""
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        return args[0] if args else None
    return None


def compile_schema(model: type[BaseModel]) -> dict[str, Any]:
    """
    Compile a Pydantic model into execution topology.

    Returns:
        {
            "orchestrator": {
                "name": "project",
                "prompt": "...",  # from model docstring
                "fields": {...},  # top-level view
            },
            "fragments": {
                "team": {
                    "type": TeamMember,
                    "prompt": "...",  # from TeamMember docstring
                    "tools": [...],
                    "structured_output": False,
                },
                ...
            },
            "routing": [
                {"trigger": "team", "fragment": "team"},
                ...
            ],
        }
    """
    result = {
        "orchestrator": {
            "name": model.__name__.lower(),
            "prompt": model.__doc__ or "",
            "fields": {},
        },
        "fragments": {},
        "routing": [],
        "_root_type": model,
    }

    for field_name, field_info in model.model_fields.items():
        extra = field_info.json_schema_extra or {}
        if isinstance(extra, dict):
            triggers = extra.get("triggers", [])
            tools = extra.get("tools", [])
            structured = extra.get("structured_output", False)
        else:
            triggers, tools, structured = [], [], False

        # Record in orchestrator view
        result["orchestrator"]["fields"][field_name] = {
            "description": field_info.description,
            "has_triggers": bool(triggers),
        }

        # If field has triggers, it becomes a fragment
        if triggers:
            inner_type = get_inner_type(field_info.annotation)
            fragment_type = inner_type or field_info.annotation

            # Get tools and settings from the fragment type's model_config
            type_extra = {}
            if hasattr(fragment_type, "model_config"):
                type_extra = fragment_type.model_config.get("json_schema_extra", {})

            result["fragments"][field_name] = {
                "type": fragment_type,
                "prompt": fragment_type.__doc__ if hasattr(fragment_type, "__doc__") else "",
                "schema": fragment_type.model_json_schema() if hasattr(fragment_type, "model_json_schema") else {},
                "tools": type_extra.get("tools", []),
                "structured_output": type_extra.get("structured_output", False),
            }

            # Add routing entries
            for trigger in triggers:
                result["routing"].append({
                    "trigger": trigger,
                    "fragment": field_name,
                })

    return result


def generate_routing_table(compiled: dict) -> str:
    """Generate routing table as markdown."""
    lines = [
        f"# Routing: {compiled['orchestrator']['name']}",
        "",
        "| Trigger | Fragment |",
        "|---------|----------|",
    ]
    for entry in compiled["routing"]:
        lines.append(f"| {entry['trigger']} | {entry['fragment']} |")
    return "\n".join(lines)


def generate_yaml(compiled: dict, fragment_name: str) -> str:
    """Generate YAML definition for a fragment."""
    frag = compiled["fragments"][fragment_name]

    # Format tools with name and description
    tools_lines = []
    for t in frag["tools"]:
        if isinstance(t, dict):
            tools_lines.append(f"    - name: {t['name']}")
            if t.get("description"):
                tools_lines.append(f"      description: {t['description']}")
        else:
            tools_lines.append(f"    - name: {t}")
    tools_yaml = "\n".join(tools_lines)

    return f"""type: object

description: |
{_indent(frag['prompt'], 2)}

properties:
  # Schema derived from {frag['type'].__name__}

json_schema_extra:
  kind: agent
  name: {fragment_name}-handler
  version: "1.0.0"
  structured_output: {str(frag['structured_output']).lower()}
  tools:
{tools_yaml}
"""


def _indent(text: str, spaces: int) -> str:
    """Indent text by N spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in (text or "").strip().split("\n"))


def generate_orchestrator_yaml(compiled: dict) -> str:
    """Generate YAML definition for the orchestrator."""
    orch = compiled["orchestrator"]

    # Get tools from root model if available
    tools_lines = []
    root_type = compiled.get("_root_type")
    if root_type and hasattr(root_type, "model_config"):
        root_extra = root_type.model_config.get("json_schema_extra", {})
        for t in root_extra.get("tools", []):
            if isinstance(t, dict):
                tools_lines.append(f"    - name: {t['name']}")
                if t.get("description"):
                    tools_lines.append(f"      description: {t['description']}")

    # Build delegates list
    delegates = "\n".join(f"  - {name}" for name in compiled["fragments"].keys())

    return f"""type: object

description: |
{_indent(orch['prompt'], 2)}

  ## Delegates
{delegates}

properties:
  status:
    type: string
    enum: [success, partial, failed]
  results:
    type: object
    description: Aggregated results from delegates

json_schema_extra:
  kind: agent
  name: {orch['name']}-orchestrator
  version: "1.0.0"
  structured_output: false
  tools:
{chr(10).join(tools_lines) if tools_lines else '    []'}
"""


def generate_full_yaml(compiled: dict) -> dict[str, str]:
    """Generate YAML for orchestrator and all fragments."""
    result = {
        f"{compiled['orchestrator']['name']}-orchestrator": generate_orchestrator_yaml(compiled),
    }

    for fragment_name in compiled["fragments"]:
        result[f"{fragment_name}-handler"] = generate_yaml(compiled, fragment_name)

    return result


if __name__ == "__main__":
    from example_model import Project

    compiled = compile_schema(Project)

    print("=== Orchestrator ===")
    print(f"Name: {compiled['orchestrator']['name']}")
    print(f"Fields: {list(compiled['orchestrator']['fields'].keys())}")

    print("\n=== Fragments ===")
    for name, frag in compiled["fragments"].items():
        print(f"  {name}: {frag['type'].__name__}, tools={frag['tools']}")

    print("\n=== Routing Table ===")
    print(generate_routing_table(compiled))

    print("\n" + "=" * 60)
    print("GENERATED YAML FILES")
    print("=" * 60)

    for name, yaml_content in generate_full_yaml(compiled).items():
        print(f"\n--- {name}.yaml ---")
        print(yaml_content)
