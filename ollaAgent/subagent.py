from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any

from rich.console import Console

console = Console()


# ──────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────


@dataclass
class SubagentTask:
    """단일 서브에이전트 실행 단위. multiprocessing pickle을 위해 기본 타입만 사용한다."""

    name: str
    task: str
    model: str
    host: str
    cf_client_id: str
    cf_client_secret: str
    system_prompt: str = "You are an expert coder. Answer concisely."
    max_iterations: int = 5


# ──────────────────────────────────────────
# Worker (모듈 레벨 — spawn 방식 pickle 조건)
# ──────────────────────────────────────────


def _simple_loop(
    messages: list[dict],
    client: Any,
    model: str,
    max_iterations: int,
) -> str:
    """Rich 없이 스트리밍으로 단일 응답을 수집해 반환한다.

    tools=[] 이므로 tool_call 없이 한 번에 완료된다.
    """
    final = ""
    for _ in range(max_iterations):
        stream = client.chat(
            model=model,
            messages=messages,
            tools=[],
            stream=True,
        )
        content = ""
        for chunk in stream:
            content += chunk.get("message", {}).get("content", "")
        messages.append({"role": "assistant", "content": content})
        final = content
        break  # tools=[] 이므로 첫 응답에서 완료
    return final


def _worker(task: SubagentTask) -> tuple[str, str]:
    """워커 프로세스: 독립 Client로 태스크를 실행하고 (name, result) 튜플을 반환한다.

    Rich Live는 사용하지 않는다 — 멀티프로세싱 환경에서 터미널 출력이 충돌한다.
    Client import를 함수 내부에 위치시켜 worker 프로세스에서 fresh import가 이루어지도록 한다.
    """
    from ollama import Client  # noqa: PLC0415 — subprocess fresh import 의도적

    client = Client(
        host=task.host,
        headers={
            "CF-Access-Client-Id": task.cf_client_id,
            "CF-Access-Client-Secret": task.cf_client_secret,
        },
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": task.task},
    ]
    result = _simple_loop(messages, client, task.model, task.max_iterations)
    return (task.name, result)


# ──────────────────────────────────────────
# Public API
# ──────────────────────────────────────────


def run_subagents(
    tasks: list[SubagentTask],
    workers: int = 4,
) -> list[tuple[str, str]]:
    """멀티프로세싱으로 SubagentTask 목록을 병렬 실행하고 (name, result) 목록을 반환한다.

    Args:
        tasks: 실행할 SubagentTask 목록.
        workers: 최대 워커 프로세스 수 (기본 4, tasks 수가 더 적으면 tasks 수 사용).

    Returns:
        [(name, result), ...] 형태의 결과 목록.
    """
    if not tasks:
        return []
    num_workers = min(workers, len(tasks))
    with Pool(processes=num_workers) as pool:
        return pool.map(_worker, tasks)
