"""
Integration tests that require LLM API calls.

These tests are marked with @pytest.mark.llm and are SKIPPED by default.
They require actual API keys and incur costs.

To run LLM tests:
    pytest tests/integration/test_llm_queries.py -m "llm" -v

To run all integration tests INCLUDING LLM:
    pytest tests/integration/ -v  # (without -m "not llm")
"""

import pytest


@pytest.mark.llm
@pytest.mark.asyncio
async def test_search_with_real_embeddings(db, rem_service, clean_test_data):
    """
    Test SEARCH query with real embedding generation.

    This test requires:
    - OPENAI_API_KEY environment variable
    - Real LLM API call for embedding generation

    Skip in CI/pre-push hooks with: -m "not llm"
    """
    pytest.skip("LLM test - requires OPENAI_API_KEY and incurs API costs")

    # This would be the actual test with real embeddings
    # from remlight.services.embeddings import get_embedding
    #
    # async def real_embed(text):
    #     return await get_embedding(text)
    #
    # Insert test data with embeddings
    # await db.execute(...)
    #
    # rem_with_embeddings = RemService(db, embed_fn=real_embed)
    # result = await rem_with_embeddings.execute('SEARCH "semantic query" IN resources')
    #
    # assert result.count > 0


@pytest.mark.llm
@pytest.mark.asyncio
async def test_agent_query_with_llm(db, clean_test_data):
    """
    Test agent-driven query using LLM for natural language understanding.

    This test requires actual LLM API calls.
    """
    pytest.skip("LLM test - requires API key and incurs costs")

    # Example of what this test would look like:
    # from remlight.agentic import create_agent
    #
    # agent = create_agent("query_agent")
    # response = await agent.run("Find all projects related to machine learning")
    #
    # assert "projects" in response.lower()


@pytest.mark.llm
@pytest.mark.slow
@pytest.mark.asyncio
async def test_multi_turn_conversation_with_llm():
    """
    Test multi-turn agent conversation.

    Marked as both 'llm' and 'slow' since it involves multiple API calls.
    """
    pytest.skip("LLM test - requires API key and incurs costs")

    # Example multi-turn conversation test
    # agent = create_agent("conversational_agent")
    #
    # Turn 1
    # r1 = await agent.run("What entities do we have?")
    #
    # Turn 2 (with context)
    # r2 = await agent.run("Tell me more about the first one")
    #
    # assert r2 contains relevant information
