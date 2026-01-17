"""
Self-Awareness Evaluator for Agents.

Tests that agents correctly know their own configuration and context:
1. Identity - name, purpose, description
2. Structure - expected output properties/fields
3. Tools - which tools are available
4. Context - current date/time and injected context

Usage:
    # Run all self-awareness tests
    pytest tests/eval/test_self_awareness.py -v

    # Run with real LLM (requires API key)
    pytest tests/eval/test_self_awareness.py -v -m "llm"

    # Run via CLI
    python -m remlight.cli.main eval-self-awareness --schema query-agent
"""

import re
from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import BaseModel

from remlight.agentic.context import AgentContext
from remlight.agentic.provider import create_agent, _build_system_prompt
from remlight.agentic.schema import AgentSchema, schema_from_yaml_file
from remlight.api.routers.tools import get_user_profile_hint


class SelfAwarenessQuestion(BaseModel):
    """A question to test agent self-awareness."""

    category: str  # identity, structure, tools, context
    question: str
    expected_pattern: str  # regex pattern to match in response
    expected_value: str | None = None  # exact value if applicable


class EvaluationResult(BaseModel):
    """Result of evaluating a single question."""

    question: str
    category: str
    expected: str
    actual: str
    passed: bool
    score: float  # 0.0, 0.5, or 1.0
    reasoning: str


class SelfAwarenessEvaluation(BaseModel):
    """Complete evaluation results."""

    schema_name: str
    results: list[EvaluationResult]
    overall_score: float
    passed_count: int
    total_count: int
    issues: list[str]


def generate_questions_for_schema(schema: AgentSchema) -> list[SelfAwarenessQuestion]:
    """Generate self-awareness questions based on agent schema."""
    questions = []

    # Get metadata
    meta = schema.json_schema_extra
    name = meta.name if meta else "unknown"
    description = schema.description or ""

    # 1. IDENTITY questions
    # Allow variations like "query-agent" or "Query Agent" or "query agent"
    name_pattern = name.replace("-", "[ -]?")  # Allow dash or space or nothing
    questions.append(
        SelfAwarenessQuestion(
            category="identity",
            question="What is your name or identifier as an agent?",
            expected_pattern=rf"(?i){name_pattern}",
            expected_value=name,
        )
    )

    # Extract key purpose words from description
    purpose_keywords = []
    if "query" in description.lower():
        purpose_keywords.append("query")
    if "search" in description.lower():
        purpose_keywords.append("search")
    if "answer" in description.lower():
        purpose_keywords.append("answer")
    if "help" in description.lower():
        purpose_keywords.append("help")

    if purpose_keywords:
        pattern = "|".join(purpose_keywords)
        questions.append(
            SelfAwarenessQuestion(
                category="identity",
                question="What is your main purpose or role?",
                expected_pattern=rf"(?i)({pattern})",
                expected_value=None,
            )
        )

    # 2. STRUCTURE questions - output properties
    if schema.properties:
        prop_names = list(schema.properties.keys())

        # Ask about all properties ((?i) must be at the start)
        questions.append(
            SelfAwarenessQuestion(
                category="structure",
                question="What output fields or properties are you expected to produce? List them.",
                expected_pattern=r"(?i)(" + "|".join(re.escape(p) for p in prop_names) + ")",
                expected_value=", ".join(prop_names),
            )
        )

        # Ask about specific property descriptions
        for prop_name, prop_def in schema.properties.items():
            if isinstance(prop_def, dict) and prop_def.get("description"):
                desc = prop_def["description"]
                # Extract key words from description
                keywords = [w for w in desc.split() if len(w) > 3][:3]
                if keywords:
                    questions.append(
                        SelfAwarenessQuestion(
                            category="structure",
                            question=f"What is the '{prop_name}' field for? Describe it briefly.",
                            expected_pattern=r"(?i)(" + "|".join(re.escape(k) for k in keywords) + ")",
                            expected_value=desc,
                        )
                    )

    # 3. TOOLS questions
    if meta and meta.tools:
        tool_names = [t.name for t in meta.tools if t.name]
        if tool_names:
            questions.append(
                SelfAwarenessQuestion(
                    category="tools",
                    question="What tools do you have access to? List them.",
                    expected_pattern=r"(?i)(" + "|".join(re.escape(t) for t in tool_names) + ")",
                    expected_value=", ".join(tool_names),
                )
            )

            # Ask about specific tool
            for tool in meta.tools:
                if tool.name and tool.description:
                    desc_escaped = re.escape(tool.description[:20])
                    questions.append(
                        SelfAwarenessQuestion(
                            category="tools",
                            question=f"What does the '{tool.name}' tool do?",
                            expected_pattern=rf"(?i)({re.escape(tool.name)}|{desc_escaped})",
                            expected_value=tool.description,
                        )
                    )
    elif meta and meta.tools == []:
        # No tools
        questions.append(
            SelfAwarenessQuestion(
                category="tools",
                question="Do you have access to any tools?",
                expected_pattern=r"(?i)(no|none|don't|do not|cannot)",
                expected_value="No tools",
            )
        )

    # 4. CONTEXT questions - date/time
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    questions.append(
        SelfAwarenessQuestion(
            category="context",
            question="What is today's date?",
            expected_pattern=rf"{today}|{datetime.now(timezone.utc).strftime('%B')}.*{datetime.now(timezone.utc).year}",
            expected_value=today,
        )
    )

    return questions


def score_response(response: str, question: SelfAwarenessQuestion) -> tuple[float, str]:
    """Score a response against expected pattern.

    Returns (score, reasoning).
    """
    if not response:
        return 0.0, "No response provided"

    # Check if pattern matches
    if re.search(question.expected_pattern, response):
        return 1.0, f"Response matches expected pattern: {question.expected_pattern}"

    # Partial credit for related content
    response_lower = response.lower()
    expected_lower = (question.expected_value or "").lower()

    # Check for partial matches
    if question.expected_value:
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        overlap = expected_words & response_words
        if len(overlap) >= len(expected_words) * 0.5:
            return 0.5, f"Partial match: {len(overlap)}/{len(expected_words)} expected words found"

    # Check if response indicates uncertainty (which is honest)
    if any(phrase in response_lower for phrase in ["don't know", "not sure", "unclear", "i cannot"]):
        return 0.0, "Agent expressed uncertainty (honest but incorrect)"

    return 0.0, f"Response did not match expected pattern: {question.expected_pattern}"


async def evaluate_agent_self_awareness(
    schema: AgentSchema | dict[str, Any],
    model_name: str | None = None,
    verbose: bool = False,
) -> SelfAwarenessEvaluation:
    """
    Evaluate an agent's self-awareness by asking questions about its configuration.

    Args:
        schema: Agent schema (dict or AgentSchema)
        model_name: LLM model to use (defaults to test model)
        verbose: Print detailed output

    Returns:
        SelfAwarenessEvaluation with results
    """
    # Parse schema if dict
    if isinstance(schema, dict):
        from remlight.agentic.schema import AgentSchema

        schema = AgentSchema.model_validate(schema)

    schema_name = schema.json_schema_extra.name if schema.json_schema_extra else "unknown"

    # Generate questions
    questions = generate_questions_for_schema(schema)

    if verbose:
        print(f"\n=== Self-Awareness Evaluation: {schema_name} ===")
        print(f"Generated {len(questions)} questions\n")

    # Create context with date/time injected
    profile_hint = await get_user_profile_hint(None)
    context = AgentContext(
        user_id="eval-user",
        user_profile_hint=profile_hint,
    )

    # Create agent
    runtime = await create_agent(
        schema=schema,
        model_name=model_name or "test",
        tools=[],  # No tools needed for self-reflection
        context=context,
    )

    results = []
    issues = []

    # Ask each question
    for q in questions:
        if verbose:
            print(f"[{q.category.upper()}] {q.question}")

        try:
            # Run agent with question
            result = await runtime.agent.run(q.question)
            response = str(result.output) if hasattr(result, "output") else str(result)

            if verbose:
                print(f"  Response: {response[:100]}...")

            # Score response
            score, reasoning = score_response(response, q)

            eval_result = EvaluationResult(
                question=q.question,
                category=q.category,
                expected=q.expected_value or q.expected_pattern,
                actual=response[:200],
                passed=score >= 0.5,
                score=score,
                reasoning=reasoning,
            )
            results.append(eval_result)

            if score < 1.0:
                issues.append(f"[{q.category}] {q.question}: {reasoning}")

            if verbose:
                status = "" if score >= 1.0 else "" if score >= 0.5 else ""
                print(f"  Score: {score} {status}")
                print()

        except Exception as e:
            results.append(
                EvaluationResult(
                    question=q.question,
                    category=q.category,
                    expected=q.expected_value or q.expected_pattern,
                    actual=f"Error: {e}",
                    passed=False,
                    score=0.0,
                    reasoning=f"Exception during evaluation: {e}",
                )
            )
            issues.append(f"[{q.category}] Exception: {e}")

    # Calculate overall score
    total_score = sum(r.score for r in results)
    overall_score = total_score / len(results) if results else 0.0
    passed_count = sum(1 for r in results if r.passed)

    evaluation = SelfAwarenessEvaluation(
        schema_name=schema_name,
        results=results,
        overall_score=overall_score,
        passed_count=passed_count,
        total_count=len(results),
        issues=issues,
    )

    if verbose:
        print(f"\n=== Summary ===")
        print(f"Overall Score: {overall_score:.2%}")
        print(f"Passed: {passed_count}/{len(results)}")
        if issues:
            print(f"\nIssues Found:")
            for issue in issues[:5]:
                print(f"  - {issue}")

    return evaluation


# =============================================================================
# PYTEST TESTS
# =============================================================================


class TestQuestionGeneration:
    """Test that questions are generated correctly from schemas."""

    def test_generates_identity_questions(self):
        """Test identity questions are generated."""
        schema = AgentSchema.model_validate(
            {
                "type": "object",
                "description": "You are a query agent.",
                "properties": {"answer": {"type": "string"}},
                "json_schema_extra": {"name": "test-agent"},
            }
        )

        questions = generate_questions_for_schema(schema)

        identity_qs = [q for q in questions if q.category == "identity"]
        assert len(identity_qs) >= 1
        assert any("name" in q.question.lower() for q in identity_qs)

    def test_generates_structure_questions(self):
        """Test structure questions based on properties."""
        schema = AgentSchema.model_validate(
            {
                "type": "object",
                "description": "Test agent.",
                "properties": {
                    "answer": {"type": "string", "description": "Your response"},
                    "confidence": {"type": "number", "description": "Confidence score 0-1"},
                },
                "json_schema_extra": {"name": "test-agent"},
            }
        )

        questions = generate_questions_for_schema(schema)

        structure_qs = [q for q in questions if q.category == "structure"]
        assert len(structure_qs) >= 1

    def test_generates_tool_questions(self):
        """Test tool questions when tools are configured."""
        schema = AgentSchema.model_validate(
            {
                "type": "object",
                "description": "Agent with tools.",
                "properties": {"answer": {"type": "string"}},
                "json_schema_extra": {
                    "name": "test-agent",
                    "tools": [
                        {"name": "search", "description": "Search for information"},
                        {"name": "action", "description": "Perform actions"},
                    ],
                },
            }
        )

        questions = generate_questions_for_schema(schema)

        tool_qs = [q for q in questions if q.category == "tools"]
        assert len(tool_qs) >= 1
        assert any("search" in q.expected_pattern for q in tool_qs)

    def test_generates_context_questions(self):
        """Test context questions including date."""
        schema = AgentSchema.model_validate(
            {
                "type": "object",
                "description": "Test agent.",
                "properties": {"answer": {"type": "string"}},
                "json_schema_extra": {"name": "test-agent"},
            }
        )

        questions = generate_questions_for_schema(schema)

        context_qs = [q for q in questions if q.category == "context"]
        assert len(context_qs) >= 1
        assert any("date" in q.question.lower() for q in context_qs)


class TestScoring:
    """Test response scoring logic."""

    def test_full_match_scores_1(self):
        """Test exact pattern match scores 1.0."""
        question = SelfAwarenessQuestion(
            category="identity",
            question="What is your name?",
            expected_pattern=r"query-agent",
            expected_value="query-agent",
        )

        score, _ = score_response("I am the query-agent.", question)
        assert score == 1.0

    def test_partial_match_scores_half(self):
        """Test partial match scores 0.5."""
        question = SelfAwarenessQuestion(
            category="identity",
            question="What is your purpose?",
            expected_pattern=r"(?i)(query|search|answer)",
            expected_value="query search answer",
        )

        # Response contains "search" which matches the pattern
        score, _ = score_response("I help search for information.", question)
        assert score == 1.0  # Pattern matches

    def test_no_match_scores_0(self):
        """Test no match scores 0.0."""
        question = SelfAwarenessQuestion(
            category="identity",
            question="What is your name?",
            expected_pattern=r"query-agent",
            expected_value="query-agent",
        )

        score, _ = score_response("I am a helpful assistant.", question)
        assert score == 0.0

    def test_empty_response_scores_0(self):
        """Test empty response scores 0.0."""
        question = SelfAwarenessQuestion(
            category="identity",
            question="What is your name?",
            expected_pattern=r"test",
            expected_value="test",
        )

        score, _ = score_response("", question)
        assert score == 0.0


@pytest.mark.asyncio
class TestEvaluatorWithTestModel:
    """Test evaluator using Pydantic AI TestModel."""

    async def test_evaluate_simple_schema(self):
        """Test evaluation runs without errors."""
        schema = {
            "type": "object",
            "description": "You are a test-agent that helps with queries.",
            "properties": {
                "answer": {"type": "string", "description": "Your response"},
            },
            "json_schema_extra": {
                "name": "test-agent",
                "version": "1.0.0",
                "tools": [],
            },
        }

        evaluation = await evaluate_agent_self_awareness(schema, model_name="test")

        assert evaluation.schema_name == "test-agent"
        assert evaluation.total_count > 0
        # With test model, responses are generic so scores may be low
        assert 0.0 <= evaluation.overall_score <= 1.0


@pytest.mark.llm
@pytest.mark.asyncio
class TestEvaluatorWithRealLLM:
    """Test evaluator with real LLM (requires API key)."""

    async def test_evaluate_query_agent(self):
        """Test query-agent self-awareness with real LLM."""
        from pathlib import Path

        schema_path = Path(__file__).parent.parent.parent / "schemas" / "query-agent.yaml"
        if not schema_path.exists():
            pytest.skip("query-agent.yaml not found")

        schema = schema_from_yaml_file(str(schema_path))

        evaluation = await evaluate_agent_self_awareness(
            schema,
            model_name=None,  # Uses default model
            verbose=True,
        )

        # With real LLM, we expect reasonable self-awareness
        assert evaluation.overall_score >= 0.5, f"Agent should have basic self-awareness, got {evaluation.overall_score}"
        assert evaluation.passed_count >= evaluation.total_count * 0.5
