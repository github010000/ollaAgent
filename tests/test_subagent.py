from unittest.mock import MagicMock, patch

import pytest
from ollaAgent.subagent import SubagentTask, _simple_loop, _worker, run_subagents


def _make_task(**kwargs) -> SubagentTask:
    defaults = dict(
        name="test-agent",
        task="hello",
        model="test-model",
        host="http://localhost:11434",
        cf_client_id="id",
        cf_client_secret="secret",
    )
    defaults.update(kwargs)
    return SubagentTask(**defaults)


def _mock_stream(*texts: str):
    """단순 텍스트 청크 스트림을 반환한다."""
    return iter(
        [
            {"message": {"content": t}, "done": i == len(texts) - 1}
            for i, t in enumerate(texts)
        ]
    )


# ──────────────────────────────────────────
# Unit Tests: SubagentTask
# ──────────────────────────────────────────


class TestSubagentTask:

    def test_required_fields(self):
        task = _make_task()
        assert task.name == "test-agent"
        assert task.task == "hello"
        assert task.model == "test-model"

    def test_defaults(self):
        task = _make_task()
        assert task.max_iterations == 5
        assert "expert coder" in task.system_prompt

    def test_custom_values(self):
        task = _make_task(name="a", task="b", max_iterations=3)
        assert task.max_iterations == 3


# ──────────────────────────────────────────
# Unit Tests: _simple_loop
# ──────────────────────────────────────────


class TestSimpleLoop:

    def test_returns_response_content(self):
        client = MagicMock()
        client.chat.return_value = _mock_stream("result text")
        messages = [{"role": "user", "content": "hi"}]
        result = _simple_loop(messages, client, "model", max_iterations=5)
        assert result == "result text"

    def test_appends_assistant_message(self):
        client = MagicMock()
        client.chat.return_value = _mock_stream("response")
        messages: list[dict] = [{"role": "user", "content": "q"}]
        _simple_loop(messages, client, "model", max_iterations=5)
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "response"

    def test_accumulates_multi_chunk_content(self):
        client = MagicMock()
        client.chat.return_value = _mock_stream("chunk1", " chunk2", " chunk3")
        messages: list[dict] = [{"role": "user", "content": "q"}]
        result = _simple_loop(messages, client, "model", max_iterations=5)
        assert result == "chunk1 chunk2 chunk3"

    def test_calls_chat_with_empty_tools(self):
        client = MagicMock()
        client.chat.return_value = _mock_stream("ok")
        _simple_loop([{"role": "user", "content": "q"}], client, "m", 3)
        _, kwargs = client.chat.call_args
        assert kwargs["tools"] == []

    def test_breaks_after_first_iteration(self):
        """tools=[] 이므로 max_iterations가 여러 번이어도 첫 응답에서 break."""
        client = MagicMock()
        client.chat.return_value = _mock_stream("done")
        _simple_loop([{"role": "user", "content": "q"}], client, "m", max_iterations=10)
        assert client.chat.call_count == 1


# ──────────────────────────────────────────
# Unit Tests: _worker
# ──────────────────────────────────────────


class TestWorker:

    def test_returns_name_and_result(self):
        task = _make_task(name="w1", task="what is 1+1?")
        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = _mock_stream("2")
        with patch("ollama.Client", return_value=mock_client_instance):
            name, result = _worker(task)
        assert name == "w1"
        assert result == "2"

    def test_uses_task_host_and_headers(self):
        task = _make_task(
            host="http://myhost", cf_client_id="cid", cf_client_secret="csec"
        )
        mock_client_instance = MagicMock()
        mock_client_instance.chat.return_value = _mock_stream("ok")
        with patch("ollama.Client", return_value=mock_client_instance) as mock_cls:
            _worker(task)
        _, kwargs = mock_cls.call_args
        assert kwargs["host"] == "http://myhost"
        assert kwargs["headers"]["CF-Access-Client-Id"] == "cid"


# ──────────────────────────────────────────
# Unit Tests: run_subagents
# ──────────────────────────────────────────


class TestRunSubagents:

    def test_empty_tasks_returns_empty(self):
        assert run_subagents([]) == []

    def test_delegates_to_pool_map(self):
        tasks = [_make_task(name="a1"), _make_task(name="a2")]
        expected = [("a1", "res1"), ("a2", "res2")]
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map = MagicMock(return_value=expected)
        with patch("ollaAgent.subagent.Pool", return_value=mock_pool):
            results = run_subagents(tasks)
        assert results == expected

    def test_workers_capped_to_task_count(self):
        tasks = [_make_task()]
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        mock_pool.map = MagicMock(return_value=[("test-agent", "")])
        with patch("ollaAgent.subagent.Pool", return_value=mock_pool) as mock_cls:
            run_subagents(tasks, workers=10)
        _, kwargs = mock_cls.call_args
        assert kwargs["processes"] == 1  # min(10, 1) = 1
