import pytest
from ollaAgent.permissions import PermissionConfig, PermissionMode
from ollaAgent.tool_bash import tool_bash


def _auto_config() -> PermissionConfig:
    """테스트용 auto 모드 config (confirm 불필요)."""
    return PermissionConfig(mode=PermissionMode.AUTO, deny_patterns=[])


def _default_config() -> PermissionConfig:
    """기본 deny_patterns 포함 auto 모드 config."""
    return PermissionConfig(mode=PermissionMode.AUTO)


class TestToolBash:

    def test_executes_simple_command(self):
        result = tool_bash({"command": "echo hello"}, _auto_config())
        assert result == "hello"

    def test_returns_stdout(self):
        result = tool_bash({"command": "printf 'line1\nline2'"}, _auto_config())
        assert "line1" in result
        assert "line2" in result

    def test_returns_stderr_on_error(self):
        result = tool_bash({"command": "ls /nonexistent_path_xyz"}, _auto_config())
        assert result != "(no output)"

    def test_no_output_returns_placeholder(self):
        result = tool_bash({"command": "true"}, _auto_config())
        assert result == "(no output)"

    def test_empty_command_returns_error(self):
        result = tool_bash({"command": ""}, _auto_config())
        assert "ERROR" in result

    def test_missing_command_key_returns_error(self):
        result = tool_bash({}, _auto_config())
        assert "ERROR" in result

    def test_deny_pattern_blocks_rm_rf(self):
        result = tool_bash({"command": "rm -rf /tmp/test"}, _default_config())
        assert "Blocked" in result

    def test_deny_pattern_blocks_sudo(self):
        result = tool_bash({"command": "sudo ls"}, _default_config())
        assert "Blocked" in result

    def test_permission_denied_by_user(self, monkeypatch):
        import ollaAgent.permissions as permissions

        monkeypatch.setattr(permissions.Confirm, "ask", lambda *a, **kw: False)
        config = PermissionConfig(mode=PermissionMode.PROMPT, deny_patterns=[])
        result = tool_bash({"command": "echo hello"}, config)
        assert "Permission denied" in result

    def test_timeout(self):
        import subprocess

        import ollaAgent.tool_bash as tb

        original_run = subprocess.run

        def mock_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=30)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(subprocess, "run", mock_timeout)
        result = tool_bash({"command": "sleep 60"}, _auto_config())
        monkeypatch.undo()
        assert "Timeout" in result

    def test_deny_mode_blocks_all(self):
        config = PermissionConfig(mode=PermissionMode.DENY, deny_patterns=[])
        result = tool_bash({"command": "echo hello"}, config)
        assert "Permission denied" in result
