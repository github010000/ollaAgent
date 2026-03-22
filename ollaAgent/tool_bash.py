import subprocess
from typing import Any

from ollaAgent.permissions import PermissionConfig, is_denied, request_permission


def tool_bash(args: dict[str, Any], config: PermissionConfig) -> str:
    """bash 명령을 실행하고 stdout/stderr를 반환한다.

    실행 순서:
    1. deny_patterns 매칭 여부 확인 → 매칭 시 즉시 차단
    2. mode에 따라 사용자 허가 요청 (prompt) 또는 자동 허가 (auto)
    3. subprocess로 실행 (shell=True 금지, timeout=30)
    """
    command = args.get("command", "")
    if not command:
        return "ERROR: No command provided"

    denied, pattern = is_denied(command, config)
    if denied:
        return f"ERROR: Blocked - command matches deny pattern: '{pattern}'"

    if not request_permission(command, config):
        return "ERROR: Permission denied by user"

    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout or result.stderr
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout (30s)"
    except Exception as exc:
        return f"ERROR: {exc}"
