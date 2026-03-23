from unittest.mock import MagicMock, patch

import pytest

import ollaAgent.agent as agent
from ollaAgent.agent import (_is_model_available, _parse_subagent_input,
                             _read_user_input, build_dispatch, execute_tool,
                             list_available_models, run_agentic_loop,
                             stream_response, trim_by_tokens, trim_messages)
from ollaAgent.permissions import PermissionConfig, PermissionMode

_TEST_DISPATCH = build_dispatch(PermissionConfig(mode=PermissionMode.AUTO))

# ──────────────────────────────────────────
# Unit Tests: execute_tool
# ──────────────────────────────────────────


class TestExecuteTool:

    def test_run_python_success(self):
        result = execute_tool("run_python", {"code": "print(1 + 1)"}, _TEST_DISPATCH)
        assert result == "2"

    def test_run_python_stdout(self):
        result = execute_tool("run_python", {"code": "print('hello')"}, _TEST_DISPATCH)
        assert result == "hello"

    def test_run_python_no_output(self):
        result = execute_tool("run_python", {"code": "x = 1"}, _TEST_DISPATCH)
        assert result == "(no output)"

    def test_run_python_timeout(self):
        result = execute_tool(
            "run_python", {"code": "while True: pass"}, _TEST_DISPATCH
        )
        assert "Timeout" in result

    def test_run_python_blocked_shutil(self):
        result = execute_tool(
            "run_python", {"code": "import shutil; shutil.rmtree('/')"}, _TEST_DISPATCH
        )
        assert "Blocked" in result
        assert "shutil.rmtree" in result

    def test_run_python_blocked_eval(self):
        result = execute_tool("run_python", {"code": "eval('1+1')"}, _TEST_DISPATCH)
        assert "Blocked" in result

    def test_run_python_blocked_exec(self):
        result = execute_tool(
            "run_python", {"code": "exec('print(1)')"}, _TEST_DISPATCH
        )
        assert "Blocked" in result

    def test_run_python_syntax_error(self):
        result = execute_tool("run_python", {"code": "def foo(:"}, _TEST_DISPATCH)
        assert result != "(no output)"  # stderr 반환

    def test_unknown_tool(self):
        result = execute_tool("unknown_tool", {}, _TEST_DISPATCH)
        assert "Unknown tool" in result


# ──────────────────────────────────────────
# Unit Tests: trim_messages
# ──────────────────────────────────────────


# ──────────────────────────────────────────
# Unit Tests: trim_by_tokens
# ──────────────────────────────────────────


class TestTrimByTokens:

    def _base_messages(self) -> list[dict]:
        return [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user msg"},
            {"role": "assistant", "content": "assistant msg"},
            {"role": "tool", "content": "tool result A"},
            {"role": "tool", "content": "tool result B"},
        ]

    def test_under_threshold_unchanged(self):
        """token_count가 threshold 미만이면 messages를 그대로 반환한다."""
        msgs = self._base_messages()
        result = trim_by_tokens(msgs, token_count=10_000, threshold=80_000)
        assert result == msgs

    def test_none_token_count_unchanged(self):
        """token_count가 None이면 messages를 그대로 반환한다."""
        msgs = self._base_messages()
        result = trim_by_tokens(msgs, token_count=None, threshold=80_000)
        assert result == msgs

    def test_removes_oldest_tool_message_first(self):
        """threshold 초과 시 가장 오래된 tool 메시지를 먼저 제거한다."""
        msgs = self._base_messages()
        result = trim_by_tokens(msgs, token_count=90_000, threshold=80_000)
        tool_msgs = [m for m in result if m["role"] == "tool"]
        # tool result A (첫 번째)가 제거되고 B는 남아야 함
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "tool result B"

    def test_preserves_system_message(self):
        """threshold 초과 시에도 system 메시지는 항상 보존한다."""
        msgs = self._base_messages()
        result = trim_by_tokens(msgs, token_count=90_000, threshold=80_000)
        assert result[0]["role"] == "system"

    def test_removes_assistant_when_no_tool(self):
        """tool 메시지가 없으면 가장 오래된 assistant 메시지를 제거한다."""
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "old assistant"},
            {"role": "assistant", "content": "new assistant"},
        ]
        result = trim_by_tokens(msgs, token_count=90_000, threshold=80_000)
        assistant_msgs = [m for m in result if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "new assistant"

    def test_fallback_sliding_window(self):
        """tool/assistant 없이 threshold 초과 시 슬라이딩 윈도우 폴백을 적용한다."""
        msgs = [{"role": "system", "content": "sys"}]
        for i in range(25):
            msgs.append({"role": "user", "content": f"user {i}"})
        result = trim_by_tokens(msgs, token_count=90_000, threshold=80_000)
        # system 1 + max_turns 10
        assert result[0]["role"] == "system"
        assert len(result) <= 11


class TestTrimMessages:

    def _make_messages(self, n_turns: int) -> list:
        msgs = [{"role": "system", "content": "system prompt"}]
        for i in range(n_turns):
            msgs.append({"role": "user", "content": f"user {i}"})
            msgs.append({"role": "assistant", "content": f"assistant {i}"})
        return msgs

    def test_keeps_system_message(self):
        messages = self._make_messages(5)
        result = trim_messages(messages, max_turns=4)
        assert result[0]["role"] == "system"

    def test_under_limit_keeps_all(self):
        messages = self._make_messages(5)  # 1 system + 10 turns
        result = trim_messages(messages, max_turns=20)
        assert len(result) == len(messages)

    def test_over_limit_trims(self):
        messages = self._make_messages(15)  # 1 system + 30 messages
        result = trim_messages(messages, max_turns=20)
        assert len(result) == 21  # 1 system + 20 recent

    def test_system_always_first(self):
        messages = self._make_messages(25)
        result = trim_messages(messages, max_turns=10)
        assert result[0]["role"] == "system"
        assert len(result) == 11  # 1 system + 10 recent


# ──────────────────────────────────────────
# Integration Tests: run_agentic_loop (Mock)
# ──────────────────────────────────────────


def _make_chunk(
    content="",
    tool_calls=None,
    thinking="",
    done=False,
    prompt_eval_count=0,
    null_tool_calls=False,
    null_message=False,
    null_content=False,
    null_thinking=False,
):
    """테스트용 stream chunk 생성.

    null_* 플래그: 키는 존재하지만 값이 null인 실제 API 응답 재현용.
    """
    if null_message:
        return {"message": None}
    msg = {}
    msg["content"] = None if null_content else (content or None)
    msg["thinking"] = None if null_thinking else (thinking or None)
    if null_tool_calls:
        msg["tool_calls"] = None  # 실제 API: "tool_calls": null
    elif tool_calls:
        msg["tool_calls"] = tool_calls
    chunk = {"message": msg}
    if done:
        chunk["done"] = True
        chunk["prompt_eval_count"] = prompt_eval_count
    return chunk


def _make_stream(*chunks):
    return iter(chunks)


class TestAgenticLoop:

    def test_no_tool_call_breaks_on_first_iteration(self):
        """tool_call 없으면 1 iteration에서 break, 최종 응답 반환"""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_stream(
            _make_chunk(content="파이보나치는 "),
            _make_chunk(content="수열입니다."),
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "피보나치 수열이 뭐야?"},
        ]

        result = run_agentic_loop(messages, mock_client, _TEST_DISPATCH)
        assert "파이보나치는 수열입니다." in result
        assert mock_client.chat.call_count == 1

    def test_single_tool_call_then_final_answer(self):
        """tool 1회 호출 → 결과 반영 → 최종 답변"""
        mock_client = MagicMock()

        # 1st call: tool_call 반환
        first_stream = _make_stream(
            _make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "function": {
                            "name": "run_python",
                            "arguments": {"code": "print(sum(range(1, 11)))"},
                        },
                    }
                ]
            )
        )
        # 2nd call: 최종 답변
        second_stream = _make_stream(_make_chunk(content="합계는 55입니다."))
        mock_client.chat.side_effect = [first_stream, second_stream]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "1부터 10까지 합계를 계산해줘"},
        ]

        result = run_agentic_loop(messages, mock_client, _TEST_DISPATCH)
        assert mock_client.chat.call_count == 2
        assert "55" in result

        # messages에 tool 결과가 추가됐는지 확인
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == "55"

    def test_max_iterations_exceeded(self):
        """MAX_ITERATIONS 초과 시 루프 탈출"""
        from ollaAgent.agent import MAX_ITERATIONS

        mock_client = MagicMock()

        # 매 호출마다 tool_call만 반환 (종료 조건 없음)
        def always_tool_call():
            return _make_stream(
                _make_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "function": {
                                "name": "run_python",
                                "arguments": {"code": "print(1)"},
                            },
                        }
                    ]
                )
            )

        mock_client.chat.side_effect = [
            always_tool_call() for _ in range(MAX_ITERATIONS)
        ]

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "무한 루프 테스트"},
        ]

        run_agentic_loop(messages, mock_client, _TEST_DISPATCH)
        assert mock_client.chat.call_count == MAX_ITERATIONS

    def test_thinking_content_does_not_appear_in_final_content(self):
        """thinking 내용은 final_content에 포함되지 않음"""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_stream(
            _make_chunk(thinking="내부적으로 생각 중..."),
            _make_chunk(content="최종 답변입니다."),
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "생각해봐"},
        ]

        result = run_agentic_loop(messages, mock_client, _TEST_DISPATCH)
        assert "최종 답변입니다." in result
        assert "내부적으로 생각 중..." not in result

    def test_null_tool_calls_in_stream_does_not_raise(self):
        """실제 API가 tool_calls: null 반환 시 TypeError 없이 정상 처리"""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_stream(
            _make_chunk(null_tool_calls=True),
            _make_chunk(content="응답입니다."),
        )

        messages = [{"role": "user", "content": "안녕"}]
        result = run_agentic_loop(messages, mock_client, _TEST_DISPATCH)
        assert "응답입니다." in result

    def test_null_message_in_stream_does_not_raise(self):
        """실제 API가 message: null 반환 시 정상 처리"""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_stream(
            _make_chunk(null_message=True),
            _make_chunk(content="응답입니다."),
        )

        messages = [{"role": "user", "content": "안녕"}]
        result = run_agentic_loop(messages, mock_client, _TEST_DISPATCH)
        assert "응답입니다." in result

    def test_null_content_and_thinking_in_stream_does_not_raise(self):
        """content, thinking이 null인 chunk 정상 처리"""
        mock_client = MagicMock()
        mock_client.chat.return_value = _make_stream(
            _make_chunk(null_content=True, null_thinking=True),
            _make_chunk(content="정상 응답."),
        )

        messages = [{"role": "user", "content": "테스트"}]
        result = run_agentic_loop(messages, mock_client, _TEST_DISPATCH)
        assert "정상 응답." in result


# ──────────────────────────────────────────
# Unit Tests: stream_response
# ──────────────────────────────────────────


class TestStreamResponse:

    def _run(self, chunks: list) -> tuple:
        """Live/console 렌더링을 패치하고 stream_response 반환값만 검증."""
        with patch("ollaAgent.agent.Live"), patch("ollaAgent.agent.console"):
            return stream_response(iter(chunks))

    def test_content_accumulated(self):
        """여러 chunk의 content가 누적되어 반환된다."""
        chunks = [
            {"message": {"content": "안녕"}, "done": False},
            {"message": {"content": "하세요"}, "done": True, "prompt_eval_count": 50},
        ]
        content, thinking, tool_calls, token = self._run(chunks)
        assert content == "안녕하세요"
        assert thinking == ""
        assert tool_calls == {}
        assert token == 50

    def test_thinking_accumulated(self):
        """thinking 필드가 누적되고 content와 분리된다."""
        chunks = [
            {"message": {"thinking": "생각 중...", "content": ""}, "done": False},
            {
                "message": {"content": "결론입니다."},
                "done": True,
                "prompt_eval_count": 0,
            },
        ]
        content, thinking, _, _ = self._run(chunks)
        assert thinking == "생각 중..."
        assert content == "결론입니다."

    def test_empty_stream_returns_defaults(self):
        """빈 스트림은 모두 기본값 반환."""
        content, thinking, tool_calls, token = self._run([])
        assert content == ""
        assert thinking == ""
        assert tool_calls == {}
        assert token == 0

    def test_token_count_from_done_chunk(self):
        """prompt_eval_count는 done=True chunk에서만 캡처된다."""
        chunks = [
            {"message": {"content": "a"}, "done": False, "prompt_eval_count": 999},
            {"message": {"content": "b"}, "done": True, "prompt_eval_count": 123},
        ]
        _, _, _, token = self._run(chunks)
        assert token == 123

    def test_null_prompt_eval_count_defaults_to_zero(self):
        """prompt_eval_count가 null이면 0으로 처리된다."""
        chunks = [
            {"message": {"content": "hi"}, "done": True, "prompt_eval_count": None}
        ]
        _, _, _, token = self._run(chunks)
        assert token == 0

    def test_tool_calls_accumulated(self):
        """tool_calls가 누적되어 반환된다."""
        chunks = [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"name": "bash", "arguments": {"cmd": "ls"}},
                        }
                    ],
                },
                "done": True,
                "prompt_eval_count": 0,
            }
        ]
        _, _, tool_calls, _ = self._run(chunks)
        assert 0 in tool_calls
        assert tool_calls[0]["name"] == "bash"

    def test_null_tool_calls_returns_empty(self):
        """tool_calls: null 이면 빈 dict 반환."""
        chunks = [
            {
                "message": {"content": "ok", "tool_calls": None},
                "done": True,
                "prompt_eval_count": 0,
            }
        ]
        _, _, tool_calls, _ = self._run(chunks)
        assert tool_calls == {}

    def test_null_message_returns_empty_content(self):
        """message: null 이면 content 빈 문자열 반환."""
        chunks = [{"message": None, "done": True, "prompt_eval_count": 0}]
        content, _, _, _ = self._run(chunks)
        assert content == ""


# ──────────────────────────────────────────
# Unit Tests: _read_user_input
# ──────────────────────────────────────────


class TestReadUserInput:

    def test_single_line_input(self):
        """일반 단일 줄 입력 반환"""
        with patch("builtins.input", return_value="안녕하세요"):
            assert _read_user_input() == "안녕하세요"

    def test_empty_input_returns_empty_string(self):
        """빈 입력은 빈 문자열 반환 (호출부에서 skip)"""
        with patch("builtins.input", return_value=""):
            assert _read_user_input() == ""

    def test_whitespace_only_returns_empty_string(self):
        """공백만 입력하면 strip 후 빈 문자열 반환"""
        with patch("builtins.input", return_value="   "):
            assert _read_user_input() == ""

    def test_multiline_with_backslash_continuation(self):
        """줄 끝 \\ 입력 시 다음 줄과 합쳐서 반환"""
        inputs = iter(["첫째 줄\\", "둘째 줄"])
        with patch("builtins.input", side_effect=inputs):
            result = _read_user_input()
        assert result == "첫째 줄\n둘째 줄"

    def test_multiline_three_lines(self):
        """3줄 연속 이어쓰기"""
        inputs = iter(["line1\\", "line2\\", "line3"])
        with patch("builtins.input", side_effect=inputs):
            result = _read_user_input()
        assert result == "line1\nline2\nline3"

    def test_eof_returns_none(self):
        """EOFError(Ctrl+D) 시 None 반환"""
        with patch("builtins.input", side_effect=EOFError):
            assert _read_user_input() is None

    def test_keyboard_interrupt_returns_none(self):
        """KeyboardInterrupt(Ctrl+C) 시 None 반환"""
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert _read_user_input() is None

    def test_backslash_stripped_from_continuation_line(self):
        """이어쓰기 줄의 끝 \\ 는 결과에 포함되지 않음"""
        inputs = iter(["hello\\", "world"])
        with patch("builtins.input", side_effect=inputs):
            result = _read_user_input()
        assert "\\" not in result
        assert result == "hello\nworld"


# ──────────────────────────────────────────
# Unit Tests: write_file
# ──────────────────────────────────────────


class TestWriteFile:

    def test_creates_new_file(self, tmp_path, monkeypatch):
        """새 파일을 생성하고 내용을 기록한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        target = tmp_path / "hello.txt"
        result = execute_tool(
            "write_file", {"path": str(target), "content": "hello"}, _TEST_DISPATCH
        )
        assert "OK" in result
        assert target.read_text() == "hello"

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        """기존 파일을 덮어쓴다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        target = tmp_path / "file.txt"
        target.write_text("old content")
        execute_tool(
            "write_file",
            {"path": str(target), "content": "new content"},
            _TEST_DISPATCH,
        )
        assert target.read_text() == "new content"

    def test_creates_parent_directories(self, tmp_path, monkeypatch):
        """중간 디렉토리가 없어도 자동 생성한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        target = tmp_path / "a" / "b" / "file.txt"
        result = execute_tool(
            "write_file", {"path": str(target), "content": "deep"}, _TEST_DISPATCH
        )
        assert "OK" in result
        assert target.read_text() == "deep"

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        """SAFE_BASE 외부 경로는 차단한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        result = execute_tool(
            "write_file", {"path": "/etc/passwd", "content": "x"}, _TEST_DISPATCH
        )
        assert "Blocked" in result


# ──────────────────────────────────────────
# Unit Tests: edit_file
# ──────────────────────────────────────────


class TestEditFile:

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        """기존 파일을 전체 overwrite한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        target = tmp_path / "edit_me.txt"
        target.write_text("original")
        result = execute_tool(
            "edit_file", {"path": str(target), "content": "updated"}, _TEST_DISPATCH
        )
        assert "OK" in result
        assert target.read_text() == "updated"

    def test_nonexistent_file_returns_error(self, tmp_path, monkeypatch):
        """존재하지 않는 파일은 오류를 반환한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        result = execute_tool(
            "edit_file",
            {"path": str(tmp_path / "ghost.txt"), "content": "x"},
            _TEST_DISPATCH,
        )
        assert "ERROR" in result
        assert "does not exist" in result


# ──────────────────────────────────────────
# Unit Tests: glob
# ──────────────────────────────────────────


class TestGlob:

    def test_finds_matching_files(self, tmp_path, monkeypatch):
        """패턴과 일치하는 파일 경로를 반환한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.py").write_text("")
        (tmp_path / "c.txt").write_text("")
        result = execute_tool(
            "glob", {"pattern": "*.py", "base_path": str(tmp_path)}, _TEST_DISPATCH
        )
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_no_matches(self, tmp_path, monkeypatch):
        """매칭 없으면 (no matches)를 반환한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        result = execute_tool(
            "glob", {"pattern": "*.xyz", "base_path": str(tmp_path)}, _TEST_DISPATCH
        )
        assert result == "(no matches)"

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        """SAFE_BASE 외부 base_path는 차단한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        result = execute_tool(
            "glob", {"pattern": "*", "base_path": "/etc"}, _TEST_DISPATCH
        )
        assert "Blocked" in result


# ──────────────────────────────────────────
# Unit Tests: grep
# ──────────────────────────────────────────


class TestGrep:

    def test_finds_pattern(self, tmp_path, monkeypatch):
        """패턴과 일치하는 라인을 '파일:라인: 내용' 형식으로 반환한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        target = tmp_path / "sample.txt"
        target.write_text("hello world\nfoo bar\nhello again")
        result = execute_tool(
            "grep", {"pattern": "hello", "path": str(tmp_path)}, _TEST_DISPATCH
        )
        assert "sample.txt" in result
        assert "hello world" in result
        assert "hello again" in result
        assert "foo bar" not in result

    def test_no_matches(self, tmp_path, monkeypatch):
        """매칭 없으면 (no matches)를 반환한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        (tmp_path / "file.txt").write_text("nothing here")
        result = execute_tool(
            "grep", {"pattern": "xyz123", "path": str(tmp_path)}, _TEST_DISPATCH
        )
        assert result == "(no matches)"

    def test_invalid_regex(self, tmp_path, monkeypatch):
        """잘못된 정규식은 오류를 반환한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        result = execute_tool(
            "grep", {"pattern": "[invalid", "path": str(tmp_path)}, _TEST_DISPATCH
        )
        assert "Invalid regex" in result

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        """SAFE_BASE 외부 경로는 차단한다."""
        monkeypatch.setattr(agent, "SAFE_BASE", tmp_path)
        result = execute_tool(
            "grep", {"pattern": "root", "path": "/etc"}, _TEST_DISPATCH
        )
        assert "Blocked" in result


# ──────────────────────────────────────────
# Unit Tests: _parse_subagent_input
# ──────────────────────────────────────────


class TestParseSubagentInput:

    def test_plain_tasks_use_default_model(self):
        result = _parse_subagent_input("task1 | task2", "default-model")
        assert result == [("task1", "default-model"), ("task2", "default-model")]

    def test_global_model_flag(self):
        result = _parse_subagent_input("--model llama3 task1 | task2", "default")
        assert result == [("task1", "llama3"), ("task2", "llama3")]

    def test_per_task_model_prefix(self):
        result = _parse_subagent_input("@llama3 task1 | @qwen task2", "default")
        assert result == [("task1", "llama3"), ("task2", "qwen")]

    def test_mixed_per_task_and_default(self):
        result = _parse_subagent_input("@llama3 task1 | task2", "default")
        assert result == [("task1", "llama3"), ("task2", "default")]

    def test_global_flag_overridden_by_per_task(self):
        result = _parse_subagent_input("--model llama3 task1 | @qwen task2", "default")
        assert result == [("task1", "llama3"), ("task2", "qwen")]

    def test_empty_input_returns_empty(self):
        result = _parse_subagent_input("", "default")
        assert result == []

    def test_single_task_no_pipe(self):
        result = _parse_subagent_input("just one task", "default")
        assert result == [("just one task", "default")]

    def test_pipe_only_skipped(self):
        result = _parse_subagent_input("  |  |  ", "default")
        assert result == []

    def test_global_model_flag_strips_correctly(self):
        result = _parse_subagent_input("--model my-model task text here", "default")
        assert result[0] == ("task text here", "my-model")


# ──────────────────────────────────────────
# Unit Tests: list_available_models
# ──────────────────────────────────────────


class TestListAvailableModels:

    def test_returns_model_names(self):
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(
            models=[
                MagicMock(model="llama3:latest"),
                MagicMock(model="qwen3:8b"),
            ]
        )
        result = list_available_models(mock_client)
        assert result == {"llama3:latest", "qwen3:8b"}

    def test_server_error_returns_none(self):
        mock_client = MagicMock()
        mock_client.list.side_effect = Exception("connection refused")
        result = list_available_models(mock_client)
        assert result is None

    def test_empty_server_returns_empty_set(self):
        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[])
        result = list_available_models(mock_client)
        assert result == set()


# ──────────────────────────────────────────
# Unit Tests: _is_model_available
# ──────────────────────────────────────────


class TestIsModelAvailable:

    def test_exact_match_with_tag(self):
        available = {"llama3:latest", "qwen3:8b"}
        assert _is_model_available("llama3:latest", available) is True

    def test_name_without_tag_matches_latest(self):
        available = {"llama3:latest", "qwen3:8b"}
        assert _is_model_available("llama3", available) is True

    def test_specific_tag_not_latest(self):
        available = {"qwen3:8b"}
        assert _is_model_available("qwen3:8b", available) is True
        assert _is_model_available("qwen3", available) is False  # :latest 없음

    def test_unknown_model_returns_false(self):
        available = {"llama3:latest"}
        assert _is_model_available("unknownmodel", available) is False

    def test_empty_available_returns_false(self):
        assert _is_model_available("llama3", set()) is False
