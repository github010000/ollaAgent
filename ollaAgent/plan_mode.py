from __future__ import annotations

from typing import Any

from ollama import Client
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()

PLAN_SYSTEM_PREFIX = (
    "You are in PLAN MODE. Do NOT call any tools. "
    "Only plan, no execution. "
    "Return a structured step-by-step plan."
)


# ──────────────────────────────────────────
# Public API
# ──────────────────────────────────────────


def run_plan(
    task: str,
    client: Client,
    model: str,
    base_prompt: str = "",
) -> str:
    """plan 모드: tools=[] 로 모델을 호출해 실행 없이 계획만 반환한다.

    Args:
        task: 계획할 작업 설명.
        client: Ollama 클라이언트.
        model: 사용할 모델 이름.
        base_prompt: 추가 컨텍스트 (optional).

    Returns:
        모델이 생성한 계획 텍스트.
    """
    sys_content = (
        f"{PLAN_SYSTEM_PREFIX}\n\n{base_prompt}" if base_prompt else PLAN_SYSTEM_PREFIX
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": task},
    ]
    console.print(Panel("[bold blue]PLAN MODE[/] — tools disabled", style="blue"))
    stream = client.chat(
        model=model,
        messages=messages,
        tools=[],
        stream=True,
    )
    return _stream_plan(stream)


# ──────────────────────────────────────────
# Private Helpers
# ──────────────────────────────────────────


def _stream_plan(stream: Any) -> str:
    """스트림을 소비하며 content를 누적하고 Live 렌더링 후 반환한다."""
    content = ""
    with Live(console=console, refresh_per_second=10) as live:
        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                content += piece
                live.update(Markdown(content))
    return content
