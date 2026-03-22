import re
from enum import Enum

from pydantic import BaseModel
from rich.prompt import Confirm


class PermissionMode(str, Enum):
    """bash tool 실행 허가 모드."""

    AUTO = "auto"  # deny list 외 모두 자동 허가
    PROMPT = "prompt"  # deny list 외 사용자에게 확인
    DENY = "deny"  # 모든 명령 거부


DEFAULT_DENY_PATTERNS: list[str] = [
    r"rm\s+.*-rf|rm\s+-rf",  # 재귀 삭제
    r"sudo\s+",  # 권한 상승
    r"dd\s+if=",  # 디스크 덮어쓰기
    r"mkfs",  # 파일시스템 포맷
    r":\(\)\{.*\}",  # fork bomb
    r">\s*/dev/",  # 디바이스 직접 쓰기
    r"curl.+\|.+sh|wget.+\|.+sh",  # 원격 스크립트 실행
    r"chmod\s+777",  # 위험한 권한 변경
]


class PermissionConfig(BaseModel):
    """bash tool 실행 허가 설정."""

    mode: PermissionMode = PermissionMode.PROMPT
    deny_patterns: list[str] = DEFAULT_DENY_PATTERNS


def is_denied(command: str, config: PermissionConfig) -> tuple[bool, str]:
    """deny_patterns 중 하나라도 매칭되면 (True, 매칭된 패턴)을 반환한다."""
    for pattern in config.deny_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True, pattern
    return False, ""


def request_permission(command: str, config: PermissionConfig) -> bool:
    """mode에 따라 실행 허가 여부를 반환한다. deny 체크는 포함하지 않는다."""
    if config.mode == PermissionMode.DENY:
        return False
    if config.mode == PermissionMode.AUTO:
        return True
    return Confirm.ask(f"[yellow]Allow command?[/yellow]\n  {command}")
