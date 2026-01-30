"""
Example: Schema-as-Architecture

The schema is a pure domain model. Model docstrings are prompts.
Field descriptions guide extraction. model_config.json_schema_extra
defines tools and metadata.
"""

from pydantic import BaseModel, Field


class TeamMember(BaseModel):
    """
    You extract and manage team member information.

    ## Algorithm
    1. Identify team members mentioned in the input
    2. Extract name, role, and capacity if stated
    3. Use lookup_user to verify against known users
    4. Use check_availability to get current capacity

    ## Rules
    - capacity_hours defaults to 40 if not specified
    - Flag if capacity < 10 hours (limited availability)
    - Role should match standard titles when possible
    """

    name: str = Field(description="Full name of the team member")
    role: str = Field(description="Job title or role (e.g., 'Engineer', 'Designer')")
    capacity_hours: float = Field(
        default=40.0,
        description="Available hours per week. Default 40. Flag if < 10.",
    )

    model_config = {
        "json_schema_extra": {
            "tools": [
                {"name": "lookup_user", "description": "Find user by name or email"},
                {"name": "check_availability", "description": "Get available hours for date range"},
            ],
        }
    }


class Task(BaseModel):
    """
    You extract tasks from input and return structured JSON.

    ## Algorithm
    1. Identify distinct tasks or work items in the input
    2. Generate a unique task_id for each (T-001, T-002, ...)
    3. Extract title, assignee, status, and estimated_hours
    4. Call create_task to persist each task

    ## Rules
    - task_id format: T-NNN (e.g., T-001)
    - status must be one of: pending, in_progress, done, blocked
    - If no status mentioned, default to pending
    - estimated_hours: extract if stated, otherwise leave null
    - Check assignee availability before confirming assignment
    """

    task_id: str = Field(description="Unique identifier. Format: T-NNN")
    title: str = Field(description="Brief task title describing the work")
    assignee: str | None = Field(
        default=None,
        description="Team member assigned. Verify availability first.",
    )
    status: str = Field(
        default="pending",
        description="pending | in_progress | done | blocked",
    )
    estimated_hours: float | None = Field(
        default=None,
        description="Estimated hours to complete. Null if unknown.",
    )

    model_config = {
        "json_schema_extra": {
            "structured_output": True,
            "override_temperature": 0.0,
            "tools": [
                {"name": "create_task", "description": "Persist task to project store"},
                {"name": "update_status", "description": "Change task status"},
            ],
        }
    }


class Milestone(BaseModel):
    """
    You plan and manage project milestones.

    ## Algorithm
    1. Identify milestones, releases, or deadlines in the input
    2. Generate milestone_id (M-001, M-002, ...)
    3. Extract target_date in YYYY-MM-DD format
    4. Link relevant task_ids to the milestone
    5. Use set_deadline to persist

    ## Rules
    - Convert date formats to YYYY-MM-DD
    - Verify linked tasks exist before associating
    - Flag if task estimates exceed time to milestone
    - One milestone per major deliverable or release
    """

    milestone_id: str = Field(description="Unique identifier. Format: M-NNN")
    name: str = Field(description="Milestone name (e.g., 'Beta Release', 'v1.0')")
    target_date: str = Field(description="Target date. Format: YYYY-MM-DD")
    task_ids: list[str] = Field(
        default_factory=list,
        description="Task IDs that must complete for this milestone",
    )

    model_config = {
        "json_schema_extra": {
            "tools": [
                {"name": "set_deadline", "description": "Set or update milestone date"},
                {"name": "link_tasks", "description": "Associate tasks with milestone"},
            ],
        }
    }


class Project(BaseModel):
    """
    You coordinate project management by routing to specialist handlers.

    ## Algorithm
    1. Analyze the user request
    2. Match keywords to determine which fragment(s) are involved:
       - team/members/assign/capacity → delegate to team handler
       - task/todo/work/backlog → delegate to tasks handler
       - milestone/deadline/release → delegate to milestones handler
    3. Use ask_agent to delegate with relevant context
    4. Aggregate results and respond

    ## Rules
    - Use read_resource to get user context before routing
    - Multiple fragments may be involved (e.g., "assign task" → tasks + team)
    - Always delegate extraction work; do not extract directly
    - Report aggregated status: success, partial, or failed
    """

    name: str = Field(description="Project name")
    description: str = Field(description="Project description and goals")

    team: list[TeamMember] = Field(
        default_factory=list,
        description="Team members. Route team queries here.",
        json_schema_extra={
            "triggers": ["team", "members", "assign", "capacity", "who"],
        },
    )

    tasks: list[Task] = Field(
        default_factory=list,
        description="Project tasks. Route task queries here.",
        json_schema_extra={
            "triggers": ["task", "todo", "work", "backlog", "status"],
        },
    )

    milestones: list[Milestone] = Field(
        default_factory=list,
        description="Milestones. Route planning queries here.",
        json_schema_extra={
            "triggers": ["milestone", "deadline", "release", "schedule", "when"],
        },
    )

    model_config = {
        "json_schema_extra": {
            "tools": [
                {"name": "ask_agent", "description": "Delegate to specialist handler"},
                {"name": "read_resource", "description": "Read user://profile or project://config"},
            ],
        }
    }
