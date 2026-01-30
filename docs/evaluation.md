# REMLight Evaluation System

REMLight includes a built-in evaluation system that enables continuous quality assessment of agent responses through user feedback, automated evaluators, and structured test collections.

## Architecture Overview

The evaluation system consists of three decoupled components:

> **Note: Scenarios vs Collections**
>
> These concepts are currently separate but conceptually close:
> - **Scenarios**: 1:1 with sessions, used during development to label specific conversations with detailed descriptions for precise retrieval and replay
> - **Collections**: Many sessions to one collection, descriptions are more general/vague, used for batch evaluation across groups of sessions
>
> A future refactor could merge these concepts - e.g., allowing Scenarios to optionally contain multiple sessions, or treating a Collection as a "meta-scenario" with child sessions. For now they remain separate to support distinct workflows.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Evaluation System                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Feedback   │    │  Scenarios   │    │    Collections       │  │
│  │  (ratings)   │    │ (test cases) │    │  (session groups)    │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                      │                │
│         │                   │                      │                │
│         ▼                   ▼                      ▼                │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                        Sessions                               │  │
│  │                  (conversation history)                       │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Feedback (User Ratings)

Feedback captures quality ratings from end users or automated evaluators:

- **Source**: `user` (human), `evaluator` (agent), `automated` (rules)
- **Score**: Numeric rating 0.0 to 1.0
- **Label**: Categorical (thumbs_up, thumbs_down, helpful, not_helpful)
- **Comment**: Free text explanation

Feedback is stored locally AND optionally sent to Phoenix for observability.

### Scenarios (Test Cases)

Scenarios are admin-created test cases that label sessions for:

- Regression testing
- Edge case documentation
- Training data curation

Scenarios support semantic search via embeddings on description.

### Collections (Session Groups)

Collections group multiple sessions for batch operations:

- Evaluation runs across many sessions
- Test suites with known expected behaviors
- Analysis of agent performance over time

## Workflow: Building an Evaluation Pipeline

### Step 1: Capture Sessions

Sessions are automatically captured during normal agent operation:

```bash
# Run agent and interact
rem ask "What is machine learning?" --schema my-agent
```

### Step 2: Find Interesting Sessions

Search for sessions matching criteria:

```bash
# List recent sessions
curl http://localhost:9001/api/v1/sessions?limit=20

# Search sessions with specific agent
curl http://localhost:9001/api/v1/sessions?agent_name=query-agent
```

### Step 3: Clone Sessions for Testing

Clone interesting sessions to create reproducible test cases:

```bash
# Clone a session
curl -X POST http://localhost:9001/api/v1/sessions/{session_id}/clone \
  -H "Content-Type: application/json" \
  -d '{
    "new_name": "ML explanation test case",
    "include_messages": true,
    "truncate_at_message": 5
  }'
```

Clone options:
- `new_name`: Name for the cloned session
- `include_messages`: Whether to copy messages (default: true)
- `truncate_at_message`: Only copy first N messages
- `add_to_collection`: Automatically add to a collection

### Step 4: Create Collections

Group sessions into collections for batch evaluation. Collections are searchable by name (fuzzy), description (semantic), and tags:

```bash
# Create a collection
curl -X POST http://localhost:9001/api/v1/collections \
  -H "Content-Type: application/json" \
  -d '{
    "name": "ML Explanation Tests",
    "description": "Test cases for ML explanation quality",
    "tags": ["ml", "testing", "quality"]
  }'

# Search collections by name
curl -X POST http://localhost:9001/api/v1/collections/search \
  -H "Content-Type: application/json" \
  -d '{"name_contains": "ML"}'

# Search collections by description (semantic)
curl -X POST http://localhost:9001/api/v1/collections/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning quality assessment"}'

# Search by tags
curl -X POST http://localhost:9001/api/v1/collections/search \
  -H "Content-Type: application/json" \
  -d '{"tags": ["ml", "testing"], "tag_match": "any"}'

# Add sessions manually
curl -X POST http://localhost:9001/api/v1/collections/{collection_id}/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "uuid-of-session",
    "notes": "Good example of clear explanation"
  }'

# Add sessions from query
curl -X POST http://localhost:9001/api/v1/collections/{collection_id}/sessions/from-query \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "query-agent",
    "created_after": "2024-01-01T00:00:00Z",
    "limit": 50
  }'
```

### Step 5: Export Collection for Evaluation

Get all sessions in a collection as JSON:

```bash
curl http://localhost:9001/api/v1/collections/{collection_id}/export?include_messages=true
```

Response format:
```json
{
  "collection": {
    "id": "uuid",
    "name": "ML Explanation Tests",
    "session_count": 10
  },
  "sessions": [
    {
      "session_id": "uuid",
      "name": "Session name",
      "messages": [
        {"role": "user", "content": "What is ML?"},
        {"role": "assistant", "content": "Machine learning is..."}
      ]
    }
  ],
  "exported_at": "2024-01-15T10:30:00Z"
}
```

### Step 6: Run Evaluator Agent

Evaluators are agents that review sessions and provide ratings:

```python
# Example evaluator agent schema
name: quality-evaluator
description: Evaluates agent response quality

system_prompt: |
  You are an evaluation agent. Review conversations and rate quality.

  For each session, assess:
  1. Accuracy of information
  2. Clarity of explanation
  3. Helpfulness

  Use the `action` tool to record your evaluation.

tools:
  - action
```

The evaluator uses the `action` tool to submit feedback:

```python
action(
    type="feedback",
    payload={
        "session_id": "uuid",
        "score": 0.85,
        "label": "helpful",
        "comment": "Clear explanation with good examples",
        "source": "evaluator"
    }
)
```

### Step 7: Submit Feedback

Feedback can be submitted via API:

```bash
# User feedback
curl -X POST http://localhost:9001/api/v1/scenarios/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "uuid",
    "label": "thumbs_up",
    "score": 0.9,
    "comment": "Very helpful response",
    "source": "user"
  }'

# Evaluator feedback
curl -X POST http://localhost:9001/api/v1/scenarios/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "uuid",
    "score": 0.75,
    "label": "helpful",
    "comment": "Good but could be more concise",
    "source": "evaluator"
  }'
```

## Evaluators vs Continuation Agents

Both evaluators and continuation agents work with sessions, but serve different purposes:

| Aspect | Evaluator | Continuation Agent |
|--------|-----------|-------------------|
| Purpose | Review and rate past interactions | Continue conversations |
| Modifies session | No (read-only) | Yes (adds messages) |
| Output | Feedback records | New messages |
| Use case | Quality assessment, testing | Multi-turn conversations |

### Evaluator Pattern

```python
# Evaluator: Reviews but doesn't modify
session = load_session(session_id)
for message in session.messages:
    quality = assess_quality(message)
    submit_feedback(session_id, quality)
```

### Continuation Pattern

```python
# Continuation: Adds to the conversation
session = load_session(session_id)
response = agent.continue_session(session, new_input="Tell me more")
# New messages added to session
```

## Database Schema

### Feedback Table

```sql
CREATE TABLE feedback (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    message_id UUID REFERENCES messages(id),
    trace_id VARCHAR(256),    -- Phoenix trace
    span_id VARCHAR(256),     -- Phoenix span
    name VARCHAR(256),        -- Annotation type
    score FLOAT,              -- 0.0 to 1.0
    label VARCHAR(128),       -- thumbs_up, helpful, etc.
    comment TEXT,             -- Free text
    source VARCHAR(64),       -- user, evaluator, automated
    metadata JSONB,
    user_id VARCHAR(256),
    created_at TIMESTAMP
);
```

### Collections Table

```sql
CREATE TABLE collections (
    id UUID PRIMARY KEY,
    name VARCHAR(512) UNIQUE,
    description TEXT,
    session_count INTEGER,    -- Auto-updated via trigger
    status VARCHAR(64),       -- active, archived, running
    query_filter JSONB,       -- Saved query for auto-population
    metadata JSONB,
    tags TEXT[]
);
```

### Collection Sessions (Junction)

```sql
CREATE TABLE collection_sessions (
    id UUID PRIMARY KEY,
    collection_id UUID REFERENCES collections(id),
    session_id UUID REFERENCES sessions(id),
    ordinal INTEGER,          -- Ordering
    notes TEXT,               -- Why included
    UNIQUE (collection_id, session_id)
);
```

## API Reference

### Feedback

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/scenarios/feedback` | POST | Submit feedback |

### Collections

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/collections` | GET | List collections |
| `/api/v1/collections` | POST | Create collection |
| `/api/v1/collections/search` | POST | Search collections (name, description, tags) |
| `/api/v1/collections/{id}` | GET | Get collection |
| `/api/v1/collections/{id}` | PUT | Update collection |
| `/api/v1/collections/{id}` | DELETE | Delete collection |
| `/api/v1/collections/{id}/sessions` | GET | List sessions in collection |
| `/api/v1/collections/{id}/sessions` | POST | Add session to collection |
| `/api/v1/collections/{id}/sessions/batch` | POST | Add multiple sessions |
| `/api/v1/collections/{id}/sessions/from-query` | POST | Add sessions matching query |
| `/api/v1/collections/{id}/sessions/{session_id}` | DELETE | Remove session |
| `/api/v1/collections/{id}/export` | GET | Export collection as JSON |

### Sessions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sessions` | GET | List sessions |
| `/api/v1/sessions/{id}/messages` | GET | Get session messages |
| `/api/v1/sessions/{id}/clone` | POST | Clone session |
| `/api/v1/sessions/{id}/export` | GET | Export as YAML |

## Phoenix Integration

When OTEL is enabled, feedback is automatically forwarded to Phoenix:

```bash
# Enable OTEL in .env
OTEL__ENABLED=true
OTEL__COLLECTOR_ENDPOINT=http://localhost:6006
```

Feedback appears as span annotations in Phoenix, enabling:
- Visualization of quality over time
- Correlation with latency and errors
- Filtering traces by rating

## Best Practices

1. **Start with user feedback**: Enable thumbs up/down in the UI to collect real ratings
2. **Build collections incrementally**: Add interesting sessions as you discover them
3. **Use evaluators for regression**: Run evaluators on collections after agent changes
4. **Track trends**: Monitor feedback scores over time to detect regressions
5. **Combine sources**: Use both user and evaluator feedback for comprehensive assessment
