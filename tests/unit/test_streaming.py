"""
Unit tests for streaming utilities.

Tests the StreamingState and tool argument extraction logic.

TOOL ARGUMENT FORMATS
---------------------
Pydantic-ai uses various formats for tool arguments:
- ArgsDict object with .args_dict attribute (most common)
- Plain dict
- JSON string
- None (no arguments)

The extract_tool_args function handles all these formats.

TOOL CALL LIFECYCLE
-------------------
Tool calls go through three states:
1. started: PartStartEvent - args may be None/partial
2. executing: PartEndEvent - args complete
3. completed: FunctionToolResultEvent - result available
"""

import json
import pytest
from unittest.mock import MagicMock

from remlight.agentic.streaming.handlers import extract_tool_args
from remlight.agentic.streaming.state import StreamingState
from remlight.agentic.streaming.events import ToolCallEvent
from remlight.agentic.streaming.formatters import format_sse_event


class TestExtractToolArgs:
    """Tests for extract_tool_args function."""

    def test_none_args_returns_none(self):
        """None args should return None."""
        result = extract_tool_args(None)
        assert result is None

    def test_plain_dict_returns_dict(self):
        """Plain dict should be returned as-is."""
        args = {"query": "test", "limit": 10}
        result = extract_tool_args(args)
        assert result == args

    def test_empty_dict_returns_empty_dict(self):
        """Empty dict should return empty dict."""
        result = extract_tool_args({})
        assert result == {}

    def test_json_string_parses_to_dict(self):
        """JSON string should be parsed to dict."""
        args_str = '{"query": "search term", "limit": 5}'
        result = extract_tool_args(args_str)
        assert result == {"query": "search term", "limit": 5}

    def test_empty_string_returns_empty_dict(self):
        """Empty string should return empty dict."""
        result = extract_tool_args("")
        assert result == {}

    def test_whitespace_string_returns_empty_dict(self):
        """Whitespace-only string should return empty dict."""
        result = extract_tool_args("   ")
        assert result == {}

    def test_invalid_json_returns_raw_wrapper(self):
        """Invalid JSON string should return wrapped in 'raw' key."""
        result = extract_tool_args("not valid json")
        assert result == {"raw": "not valid json"}

    def test_args_dict_object_extracts_args_dict(self):
        """ArgsDict object with args_dict attribute should extract it."""
        # Create a simple object with args_dict attribute (simulating ArgsDict)
        class MockArgsDict:
            args_dict = {"query": "test", "limit": 10}

        mock_args = MockArgsDict()

        result = extract_tool_args(mock_args)
        assert result == {"query": "test", "limit": 10}

    def test_tool_call_part_extracts_args(self):
        """ToolCallPart object with .args attribute should extract args."""
        mock_part = MagicMock()
        mock_part.args = {"query": "search", "limit": 5}

        result = extract_tool_args(mock_part)
        assert result == {"query": "search", "limit": 5}

    def test_tool_call_part_with_args_dict_attribute(self):
        """ToolCallPart where .args has .args_dict should extract nested."""
        mock_args = MagicMock()
        mock_args.args_dict = {"nested": "value"}

        mock_part = MagicMock()
        mock_part.args = mock_args

        result = extract_tool_args(mock_part)
        assert result == {"nested": "value"}

    def test_tool_call_part_with_none_args(self):
        """ToolCallPart with None args should return None."""
        mock_part = MagicMock()
        mock_part.args = None

        result = extract_tool_args(mock_part)
        assert result is None

    def test_nested_dict_preserved(self):
        """Nested dict structure should be preserved."""
        args = {
            "query": "test",
            "options": {
                "fuzzy": True,
                "threshold": 0.8
            },
            "tags": ["a", "b", "c"]
        }
        result = extract_tool_args(args)
        assert result == args


class TestStreamingState:
    """Tests for StreamingState dataclass."""

    def test_create_initializes_defaults(self):
        """StreamingState.create() should set sensible defaults."""
        state = StreamingState.create(model="test-model")

        assert state.model == "test-model"
        assert state.request_id.startswith("chatcmpl-")
        assert state.is_first_chunk is True
        assert state.token_count == 0
        assert state.child_content_streamed is False
        assert state.responding_agent is None
        assert state.active_tool_calls == {}
        assert state.active_tool_indices == {}
        assert state.pending_tool_completions == []
        assert state.pending_tool_data == {}

    def test_latency_ms_returns_non_negative(self):
        """latency_ms() should return non-negative milliseconds."""
        state = StreamingState.create(model="test")
        latency = state.latency_ms()
        assert latency >= 0

    def test_append_content_accumulates(self):
        """append_content should accumulate text and count tokens."""
        state = StreamingState.create(model="test")

        state.append_content("Hello ")
        assert state.current_text == "Hello "
        assert state.accumulated_chunks == ["Hello "]

        state.append_content("world!")
        assert state.current_text == "Hello world!"
        assert state.accumulated_chunks == ["Hello ", "world!"]

    def test_get_full_content(self):
        """get_full_content should return joined chunks."""
        state = StreamingState.create(model="test")
        state.accumulated_chunks = ["Hello ", "world", "!"]

        assert state.get_full_content() == "Hello world!"

    def test_mark_first_chunk_sent(self):
        """mark_first_chunk_sent should set is_first_chunk to False."""
        state = StreamingState.create(model="test")
        assert state.is_first_chunk is True

        state.mark_first_chunk_sent()
        assert state.is_first_chunk is False


class TestStreamingStateToolTracking:
    """Tests for StreamingState tool call tracking."""

    def test_register_tool_call(self):
        """register_tool_call should track tool call data."""
        state = StreamingState.create(model="test")

        state.register_tool_call(
            tool_name="search",
            tool_id="call_abc123",
            index=0,
            args={"query": "test"}
        )

        assert state.current_tool_id == "call_abc123"
        assert state.active_tool_calls[0] == ("search", "call_abc123")
        assert state.active_tool_indices[0] == "call_abc123"
        assert ("search", "call_abc123") in state.pending_tool_completions
        assert state.pending_tool_data["call_abc123"] == {
            "tool_name": "search",
            "tool_id": "call_abc123",
            "arguments": {"query": "test"},
        }

    def test_register_tool_call_with_none_args(self):
        """register_tool_call should handle None args (PartStartEvent case)."""
        state = StreamingState.create(model="test")

        state.register_tool_call(
            tool_name="search",
            tool_id="call_abc123",
            index=0,
            args=None  # Args not yet available at PartStartEvent
        )

        assert state.pending_tool_data["call_abc123"]["arguments"] is None

    def test_update_tool_args(self):
        """update_tool_args should update registered tool's arguments."""
        state = StreamingState.create(model="test")

        # Register with None args (PartStartEvent)
        state.register_tool_call(
            tool_name="search",
            tool_id="call_abc123",
            index=0,
            args=None
        )

        # Update with complete args (PartEndEvent)
        state.update_tool_args("call_abc123", {"query": "test", "limit": 10})

        assert state.pending_tool_data["call_abc123"]["arguments"] == {
            "query": "test",
            "limit": 10
        }

    def test_update_tool_args_unknown_tool_id(self):
        """update_tool_args should do nothing for unknown tool_id."""
        state = StreamingState.create(model="test")

        # Should not raise
        state.update_tool_args("unknown_id", {"query": "test"})

        assert "unknown_id" not in state.pending_tool_data

    def test_update_tool_args_with_none_does_not_overwrite(self):
        """update_tool_args with None should not overwrite existing args."""
        state = StreamingState.create(model="test")

        state.register_tool_call(
            tool_name="search",
            tool_id="call_abc123",
            index=0,
            args={"query": "original"}
        )

        # Update with None should not change
        state.update_tool_args("call_abc123", None)

        assert state.pending_tool_data["call_abc123"]["arguments"] == {"query": "original"}

    def test_complete_tool_call(self):
        """complete_tool_call should return tool data with result."""
        state = StreamingState.create(model="test")

        state.register_tool_call(
            tool_name="search",
            tool_id="call_abc123",
            index=0,
            args={"query": "test"}
        )

        result_data = {"results": ["item1", "item2"]}
        tool_data = state.complete_tool_call(result_data)

        assert tool_data is not None
        assert tool_data["tool_name"] == "search"
        assert tool_data["tool_id"] == "call_abc123"
        assert tool_data["arguments"] == {"query": "test"}
        assert tool_data["result"] == result_data

    def test_complete_tool_call_removes_from_pending(self):
        """complete_tool_call should remove tool from pending data."""
        state = StreamingState.create(model="test")

        state.register_tool_call(
            tool_name="search",
            tool_id="call_abc123",
            index=0,
            args={"query": "test"}
        )

        state.complete_tool_call({"results": []})

        assert "call_abc123" not in state.pending_tool_data
        assert len(state.pending_tool_completions) == 0

    def test_complete_tool_call_empty_queue(self):
        """complete_tool_call with no pending tools returns None."""
        state = StreamingState.create(model="test")

        result = state.complete_tool_call({"some": "result"})

        assert result is None

    def test_multiple_tool_calls_fifo_order(self):
        """Multiple tool calls should complete in FIFO order."""
        state = StreamingState.create(model="test")

        # Register two tools
        state.register_tool_call("search", "call_1", 0, {"q": "first"})
        state.register_tool_call("action", "call_2", 1, {"type": "obs"})

        # Complete first
        data1 = state.complete_tool_call("result1")
        assert data1["tool_name"] == "search"
        assert data1["tool_id"] == "call_1"

        # Complete second
        data2 = state.complete_tool_call("result2")
        assert data2["tool_name"] == "action"
        assert data2["tool_id"] == "call_2"


class TestStreamingStateMetadata:
    """Tests for StreamingState metadata tracking."""

    def test_register_metadata(self):
        """register_metadata should update metadata dict."""
        state = StreamingState.create(model="test")

        state.register_metadata({"key1": "value1", "key2": 42})

        assert state.metadata == {"key1": "value1", "key2": 42}
        assert state.metadata_registered is True

    def test_register_metadata_merges(self):
        """register_metadata should merge with existing metadata."""
        state = StreamingState.create(model="test")

        state.register_metadata({"key1": "value1"})
        state.register_metadata({"key2": "value2"})

        assert state.metadata == {"key1": "value1", "key2": "value2"}

    def test_register_metadata_sets_responding_agent(self):
        """register_metadata with agent_schema should set responding_agent."""
        state = StreamingState.create(model="test")

        state.register_metadata({"agent_schema": "my-agent"})

        assert state.responding_agent == "my-agent"

    def test_register_metadata_does_not_overwrite_responding_agent(self):
        """register_metadata should not overwrite existing responding_agent."""
        state = StreamingState.create(model="test")
        state.responding_agent = "existing-agent"

        state.register_metadata({"agent_schema": "new-agent"})

        assert state.responding_agent == "existing-agent"


class TestStreamingStateChildContent:
    """Tests for child content tracking."""

    def test_mark_child_content(self):
        """mark_child_content should set flag and agent name."""
        state = StreamingState.create(model="test")

        state.mark_child_content("child_agent")

        assert state.child_content_streamed is True
        assert state.responding_agent == "child_agent"

    def test_child_content_deduplication_logic(self):
        """After child_content_streamed=True, parent text should be skipped.

        This tests the key deduplication logic. When a child agent streams
        content, the parent should NOT also stream its own text (which would
        just repeat the child's response).
        """
        state = StreamingState.create(model="test")

        # Simulate child streaming content
        state.mark_child_content("intake_agent")

        # In the streaming loop, this check prevents duplicate content:
        should_skip_parent_text = state.child_content_streamed
        assert should_skip_parent_text is True


class TestToolCallEvent:
    """Tests for ToolCallEvent model."""

    def test_tool_call_started_event(self):
        """Tool call started event should have correct structure."""
        event = ToolCallEvent(
            tool_name="search",
            tool_id="call_123",
            status="started",
            arguments=None,
        )

        assert event.tool_name == "search"
        assert event.tool_id == "call_123"
        assert event.status == "started"
        assert event.arguments is None
        assert event.result is None

    def test_tool_call_executing_event(self):
        """Tool call executing event should have complete args."""
        event = ToolCallEvent(
            tool_name="search",
            tool_id="call_123",
            status="executing",
            arguments={"query": "test", "limit": 10},
        )

        assert event.status == "executing"
        assert event.arguments == {"query": "test", "limit": 10}

    def test_tool_call_completed_event(self):
        """Tool call completed event should have result."""
        event = ToolCallEvent(
            tool_name="search",
            tool_id="call_123",
            status="completed",
            arguments={"query": "test"},
            result='{"results": ["item1"]}',
        )

        assert event.status == "completed"
        assert event.result == '{"results": ["item1"]}'


class TestToolCallEventFormatting:
    """Tests for tool call event SSE formatting."""

    def test_format_tool_call_started(self):
        """Tool call started should format as SSE."""
        event = ToolCallEvent(
            tool_name="search",
            tool_id="call_123",
            status="started",
        )

        sse = format_sse_event(event)

        assert "event: tool_call" in sse
        assert '"tool_name":"search"' in sse or '"tool_name": "search"' in sse
        assert '"status":"started"' in sse or '"status": "started"' in sse

    def test_format_tool_call_with_args(self):
        """Tool call with args should include arguments in SSE."""
        event = ToolCallEvent(
            tool_name="search",
            tool_id="call_123",
            status="executing",
            arguments={"query": "test"},
        )

        sse = format_sse_event(event)

        assert "event: tool_call" in sse
        assert "query" in sse
        assert "test" in sse


class TestToolCallLifecycle:
    """Integration tests for complete tool call lifecycle."""

    def test_full_lifecycle_started_executing_completed(self):
        """Test complete tool call lifecycle: started -> executing -> completed.

        This simulates what happens in the streaming loop:
        1. PartStartEvent: Tool call detected, args may be None/partial
        2. PartEndEvent: Args complete, emit executing status
        3. FunctionToolResultEvent: Tool finished, emit completed status
        """
        state = StreamingState.create(model="test")

        # STEP 1: PartStartEvent - tool call detected, args not yet complete
        state.register_tool_call(
            tool_name="search",
            tool_id="call_abc",
            index=0,
            args=None,  # Args may be None at PartStartEvent
        )

        event1 = ToolCallEvent(
            tool_name="search",
            tool_id="call_abc",
            status="started",
            arguments=None,
        )
        assert event1.status == "started"
        assert event1.arguments is None

        # STEP 2: PartEndEvent - args now complete
        complete_args = {"query": "machine learning", "limit": 5}
        state.update_tool_args("call_abc", complete_args)

        event2 = ToolCallEvent(
            tool_name="search",
            tool_id="call_abc",
            status="executing",
            arguments=complete_args,
        )
        assert event2.status == "executing"
        assert event2.arguments == complete_args

        # STEP 3: FunctionToolResultEvent - tool completed
        result = {"results": ["doc1", "doc2"]}
        tool_data = state.complete_tool_call(result)

        event3 = ToolCallEvent(
            tool_name=tool_data["tool_name"],
            tool_id=tool_data["tool_id"],
            status="completed",
            arguments=tool_data["arguments"],
            result=str(result)[:200],
        )
        assert event3.status == "completed"
        assert event3.arguments == complete_args
        assert "doc1" in event3.result

    def test_lifecycle_with_immediate_args(self):
        """Test lifecycle when args are available immediately at PartStartEvent.

        Some tool calls have simple/small args that are complete at start.
        """
        state = StreamingState.create(model="test")

        # Args available immediately
        args = {"action_type": "observation", "payload": {"conf": 0.9}}
        state.register_tool_call(
            tool_name="action",
            tool_id="call_xyz",
            index=0,
            args=args,
        )

        # Even with immediate args, we still get PartEndEvent
        state.update_tool_args("call_xyz", args)  # No change

        # Complete
        result = {"_action_event": True, "action_type": "observation"}
        tool_data = state.complete_tool_call(result)

        assert tool_data["arguments"] == args
        assert tool_data["result"]["_action_event"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
