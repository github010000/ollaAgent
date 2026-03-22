from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from ollaAgent.permissions import DEFAULT_DENY_PATTERNS, PermissionMode

# ──────────────────────────────────────────
# Config 파일 경로 정의
# ──────────────────────────────────────────

GLOBAL_CONFIG_PATH = Path.home() / ".agents" / "config.yaml"
PROJECT_CONFIG_PATH = Path.cwd() / "config.yaml"
LOCAL_CONFIG_PATH = Path.cwd() / ".agents" / "config.yaml"

DEFAULT_AGENTS_MD = "AGENT.md"


# ──────────────────────────────────────────
# AgentConfig Model
# ──────────────────────────────────────────


class AgentConfig(BaseModel):
    """전체 agent 동작을 제어하는 설정 모델.

    계층 우선순위: global < project < local
    각 레벨에서 기재된 키만 상위 값을 오버라이드한다.
    """

    model: str = "qwen3-coder-next:latest"
    permission_mode: PermissionMode = PermissionMode.PROMPT
    token_threshold: int = 80_000
    max_iterations: int = 10
    deny_patterns: list[str] = DEFAULT_DENY_PATTERNS
    agents_md_path: str = DEFAULT_AGENTS_MD
    ollama_host: str = "http://localhost:11434"
    cf_access_client_id: str = ""
    cf_access_client_secret: str = ""


# ──────────────────────────────────────────
# YAML I/O
# ──────────────────────────────────────────


def _load_yaml(path: Path) -> dict[str, Any]:
    """YAML 파일을 읽어 dict로 반환한다. 파일 없거나 파싱 오류 시 빈 dict 반환."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except yaml.YAMLError as exc:
        print(f"[Config] YAML 파싱 오류 ({path}): {exc} — 해당 레벨 skip")
        return {}


def _ensure_global_config(path: Path) -> None:
    """global config 파일이 없으면 기본값으로 자동 생성한다."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    default = AgentConfig()
    data = {
        "model": default.model,
        "permission_mode": default.permission_mode.value,
        "token_threshold": default.token_threshold,
        "max_iterations": default.max_iterations,
        "deny_patterns": default.deny_patterns,
        "agents_md_path": default.agents_md_path,
        "ollama_host": default.ollama_host,
        "cf_access_client_id": default.cf_access_client_id,
        "cf_access_client_secret": default.cf_access_client_secret,
    }
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(data, fh, allow_unicode=True, default_flow_style=False)
    print(f"[Config] global config 생성: {path}")


# ──────────────────────────────────────────
# 계층 머지
# ──────────────────────────────────────────


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """override에 기재된 키만 base를 교체한다."""
    result = dict(base)
    for key, value in override.items():
        if value is not None:
            result[key] = value
    return result


def load_config(
    global_path: Path = GLOBAL_CONFIG_PATH,
    project_path: Path = PROJECT_CONFIG_PATH,
    local_path: Path = LOCAL_CONFIG_PATH,
) -> AgentConfig:
    """global → project → local 순서로 config를 머지하여 AgentConfig를 반환한다."""
    _ensure_global_config(global_path)

    raw = {}
    for path in (global_path, project_path, local_path):
        raw = _merge(raw, _load_yaml(path))

    return AgentConfig(
        **{k: v for k, v in raw.items() if k in AgentConfig.model_fields}
    )


# ──────────────────────────────────────────
# System Prompt Builder
# ──────────────────────────────────────────


def build_system_prompt(agents_md_path: str) -> str:
    """AGENTS.md 내용을 읽어 system prompt 앞에 prepend한다.

    파일이 없으면 기본 system prompt만 반환한다.
    """
    base_prompt = "You are an expert coder. Use tools when needed."
    path = Path(agents_md_path)
    if not path.exists():
        return base_prompt
    content = path.read_text(encoding="utf-8").strip()
    token_estimate = len(content) // 4
    if token_estimate > 5_000:
        print(
            f"[Config] 경고: {agents_md_path} 크기 ~{token_estimate:,} tokens "
            f"— 매 호출마다 context에 포함됨"
        )
    return f"{content}\n\n---\n\n{base_prompt}"
