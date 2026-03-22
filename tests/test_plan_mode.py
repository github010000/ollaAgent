from unittest.mock import MagicMock, patch

import pytest
from ollaAgent.plan_mode import PLAN_SYSTEM_PREFIX, _stream_plan, run_plan


def _make_chunks(*texts: str) -> list[dict]:
    """테스트용 스트림 청크 목록을 생성한다."""
    return [
        {"message": {"content": t}, "done": i == len(texts) - 1}
        for i, t in enumerate(texts)
    ]


# ──────────────────────────────────────────
# Unit Tests: _stream_plan
# ──────────────────────────────────────────


class TestStreamPlan:

    def test_accumulates_content(self):
        stream = iter(_make_chunks("Step 1", " Step 2", " Step 3"))
        with patch("ollaAgent.plan_mode.Live"):
            result = _stream_plan(stream)
        assert result == "Step 1 Step 2 Step 3"

    def test_empty_stream_returns_empty(self):
        with patch("ollaAgent.plan_mode.Live"):
            result = _stream_plan(iter([]))
        assert result == ""

    def test_skips_empty_content_chunks(self):
        stream = iter(
            [
                {"message": {"content": ""}, "done": False},
                {"message": {"content": "hello"}, "done": True},
            ]
        )
        with patch("ollaAgent.plan_mode.Live"):
            result = _stream_plan(stream)
        assert result == "hello"


# ──────────────────────────────────────────
# Unit Tests: run_plan
# ──────────────────────────────────────────


class TestRunPlan:

    def _mock_client(self, chunks: list[dict]) -> MagicMock:
        client = MagicMock()
        client.chat.return_value = iter(chunks)
        return client

    def test_calls_chat_with_empty_tools(self):
        client = self._mock_client(_make_chunks("plan result"))
        with patch("ollaAgent.plan_mode.Live"):
            run_plan("do something", client, "test-model")
        _, kwargs = client.chat.call_args
        assert kwargs["tools"] == []

    def test_plan_system_prefix_in_messages(self):
        client = self._mock_client(_make_chunks("plan"))
        with patch("ollaAgent.plan_mode.Live"):
            run_plan("task", client, "model")
        _, kwargs = client.chat.call_args
        system_msg = next(m for m in kwargs["messages"] if m["role"] == "system")
        assert PLAN_SYSTEM_PREFIX in system_msg["content"]

    def test_base_prompt_appended_to_system(self):
        client = self._mock_client(_make_chunks("plan"))
        with patch("ollaAgent.plan_mode.Live"):
            run_plan("task", client, "model", base_prompt="extra context")
        _, kwargs = client.chat.call_args
        system_msg = next(m for m in kwargs["messages"] if m["role"] == "system")
        assert "extra context" in system_msg["content"]

    def test_task_placed_in_user_message(self):
        client = self._mock_client(_make_chunks("plan"))
        with patch("ollaAgent.plan_mode.Live"):
            run_plan("my task here", client, "model")
        _, kwargs = client.chat.call_args
        user_msg = next(m for m in kwargs["messages"] if m["role"] == "user")
        assert user_msg["content"] == "my task here"

    def test_returns_streamed_content(self):
        client = self._mock_client(_make_chunks("step 1", " step 2"))
        with patch("ollaAgent.plan_mode.Live"):
            result = run_plan("task", client, "model")
        assert result == "step 1 step 2"

    def test_no_base_prompt_uses_prefix_only(self):
        client = self._mock_client(_make_chunks("plan"))
        with patch("ollaAgent.plan_mode.Live"):
            run_plan("task", client, "model", base_prompt="")
        _, kwargs = client.chat.call_args
        system_msg = next(m for m in kwargs["messages"] if m["role"] == "system")
        assert system_msg["content"] == PLAN_SYSTEM_PREFIX
