import glob as glob_module
import json
import os
import re
import subprocess
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from dotenv import load_dotenv
from ollama import Client
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from ollaAgent.config_loader import build_system_prompt, load_config
from ollaAgent.memory import SESSION_DIR, SessionMemory, save_session
from ollaAgent.permissions import PermissionConfig
from ollaAgent.plan_mode import run_plan
from ollaAgent.subagent import SubagentTask, run_subagents
from ollaAgent.tool_bash import tool_bash

load_dotenv()

console = Console()

MAX_ITERATIONS = 10
MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB
SAFE_BASE = Path.cwd()
TOKEN_THRESHOLD = 80_000  # 80k 초과 시 trim 트리거 (qwen3-coder-next 128k 기준)

BLOCKED_KEYWORDS = [
    "shutil.rmtree",
    "os.remove",
    "os.unlink",
    "sys.exit",
    "__import__",
    "eval(",
    "exec(",
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or overwrite an existing file with given content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Overwrite an existing file entirely with new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to edit"},
                    "content": {
                        "type": "string",
                        "description": "New content to write",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. **/*.py)",
                    },
                    "base_path": {
                        "type": "string",
                        "description": "Base directory to search (default: cwd)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents using a regex pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Search recursively (default: true)",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return the output. Dangerous commands will be blocked or require user confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


# ──────────────────────────────────────────
# Tool Helpers
# ──────────────────────────────────────────


def _is_safe_path(path: str) -> bool:
    """경로가 SAFE_BASE 하위인지 확인한다."""
    try:
        return Path(path).resolve().is_relative_to(SAFE_BASE)
    except Exception:
        return False


def _tool_run_python(args: dict[str, Any]) -> str:
    """Python 코드를 실행하고 stdout/stderr를 반환한다."""
    code = args.get("code", "")
    for keyword in BLOCKED_KEYWORDS:
        if keyword in code:
            return f"ERROR: Blocked - dangerous operation detected: '{keyword}'"
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout or result.stderr
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Timeout (10s)"
    except Exception as exc:
        return f"ERROR: {exc}"


def _tool_write_file(args: dict[str, Any]) -> str:
    """파일을 생성하거나 전체 내용을 덮어쓴다."""
    path = args.get("path", "")
    content = args.get("content", "")
    if not _is_safe_path(path):
        return f"ERROR: Blocked - path '{path}' is outside working directory"
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            fh.write(content)
        return f"OK: Written {len(content)} chars to '{path}'"
    except Exception as exc:
        return f"ERROR: {exc}"


def _tool_edit_file(args: dict[str, Any]) -> str:
    """기존 파일 전체를 overwrite한다. 파일이 없으면 오류를 반환한다."""
    path = args.get("path", "")
    if not Path(path).exists():
        return (
            f"ERROR: File '{path}' does not exist. Use write_file to create a new file."
        )
    return _tool_write_file(args)


def _tool_glob(args: dict[str, Any]) -> str:
    """glob 패턴으로 파일 목록을 탐색하고 줄바꿈으로 구분된 경로를 반환한다."""
    pattern = args.get("pattern", "")
    base = args.get("base_path", str(SAFE_BASE))
    if not _is_safe_path(base):
        return f"ERROR: Blocked - path '{base}' is outside working directory"
    try:
        matches = sorted(Path(base).glob(pattern))
        return "\n".join(str(m) for m in matches) if matches else "(no matches)"
    except Exception as exc:
        return f"ERROR: {exc}"


def _grep_file(file: Path, regex: re.Pattern) -> list[str]:
    """단일 파일에서 정규식 매칭 라인을 반환한다."""
    if file.stat().st_size > MAX_FILE_SIZE:
        return []
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
        return [
            f"{file}:{i}: {line}"
            for i, line in enumerate(lines, 1)
            if regex.search(line)
        ]
    except (UnicodeDecodeError, OSError):
        return []


def _tool_grep(args: dict[str, Any]) -> str:
    """정규식으로 파일 내용을 검색하고 '파일:라인: 내용' 형식으로 반환한다."""
    pattern = args.get("pattern", "")
    path = args.get("path", str(SAFE_BASE))
    recursive = args.get("recursive", True)
    if not _is_safe_path(path):
        return f"ERROR: Blocked - path '{path}' is outside working directory"
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        return f"ERROR: Invalid regex - {exc}"
    target = Path(path)
    files = target.rglob("*") if recursive else target.glob("*")
    results: list[str] = []
    for file in files:
        if file.is_file():
            results.extend(_grep_file(file, regex))
    return "\n".join(results) if results else "(no matches)"


# ──────────────────────────────────────────
# Tool Dispatcher
# ──────────────────────────────────────────

_STATIC_DISPATCH: dict[str, Any] = {
    "run_python": _tool_run_python,
    "write_file": _tool_write_file,
    "edit_file": _tool_edit_file,
    "glob": _tool_glob,
    "grep": _tool_grep,
}


def build_dispatch(perm_config: PermissionConfig) -> dict[str, Any]:
    """PermissionConfig를 주입한 전체 tool dispatch 딕셔너리를 반환한다."""
    return {
        **_STATIC_DISPATCH,
        "bash": partial(tool_bash, config=perm_config),
    }


def execute_tool(name: str, args: dict[str, Any], dispatch: dict[str, Any]) -> str:
    """Tool 이름과 인자를 받아 실행하고 결과 문자열을 반환한다."""
    handler = dispatch.get(name)
    if handler is None:
        return f"ERROR: Unknown tool '{name}'"
    return handler(args)


# ──────────────────────────────────────────
# Message Management
# ──────────────────────────────────────────


def trim_messages(messages: list[dict], max_turns: int = 20) -> list[dict]:
    """system 메시지를 보존하고 최근 max_turns 개의 메시지만 유지한다."""
    system = [m for m in messages if m["role"] == "system"]
    rest = [m for m in messages if m["role"] != "system"]
    return system + rest[-max_turns:]


def trim_by_tokens(
    messages: list[dict],
    token_count: int,
    threshold: int = TOKEN_THRESHOLD,
) -> list[dict]:
    """token_count가 threshold를 초과하면 오래된 메시지를 제거해 컨텍스트를 줄인다.

    제거 우선순위:
    1. 가장 오래된 tool 메시지 (grep/bash 결과 등 용량 큼)
    2. 가장 오래된 assistant 메시지
    3. 최후 수단: trim_messages(max_turns=10) 슬라이딩 윈도우 폴백
    system 메시지는 항상 보존한다.
    """
    if token_count is None or token_count <= threshold:
        return messages

    result = list(messages)

    # 1순위: 오래된 tool 메시지 제거
    for i, msg in enumerate(result):
        if msg["role"] == "tool":
            result.pop(i)
            console.print(
                f"[dim][Token] {token_count:,} > {threshold:,} — tool 메시지 trim[/]"
            )
            return result

    # 2순위: 오래된 assistant 메시지 제거
    for i, msg in enumerate(result):
        if msg["role"] == "assistant":
            result.pop(i)
            console.print(
                f"[dim][Token] {token_count:,} > {threshold:,} — assistant 메시지 trim[/]"
            )
            return result

    # 최후 수단: 슬라이딩 윈도우 폴백
    console.print(
        f"[dim][Token] {token_count:,} > {threshold:,} — 슬라이딩 윈도우 폴백[/]"
    )
    return trim_messages(result, max_turns=10)


# ──────────────────────────────────────────
# Stream Helpers
# ──────────────────────────────────────────


def _accumulate_tool_calls(msg: dict[str, Any], accumulated: dict[int, dict]) -> None:
    """스트림 chunk에서 tool_calls를 누적한다 (분산 전송 대응)."""
    for tc in msg.get("tool_calls") or []:
        idx = tc.get("index", len(accumulated))
        if idx not in accumulated:
            accumulated[idx] = {"name": "", "arguments": ""}
        fn = tc.get("function") or {}
        accumulated[idx]["name"] += fn.get("name", "")
        raw_args = fn.get("arguments", "")
        if isinstance(raw_args, dict):
            accumulated[idx]["arguments"] = raw_args
        else:
            accumulated[idx]["arguments"] += raw_args


def _process_tool_calls(
    accumulated: dict[int, dict],
    messages: list[dict],
    dispatch: dict[str, Any],
) -> None:
    """누적된 tool_calls를 실행하고 결과를 messages에 추가한다."""
    for tc in accumulated.values():
        tool_name = tc["name"]
        raw_args = tc["arguments"]
        if isinstance(raw_args, dict):
            args: dict[str, Any] = raw_args
        else:
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError as exc:
                args = {}
                console.print(f"[red]Tool argument parse error: {exc}[/]")
        console.print(f"\n[bold yellow][Tool Call][/] {tool_name}({args})")
        result = execute_tool(tool_name, args, dispatch)
        console.print(f"[bold cyan][Tool Result][/] {result}\n")
        messages.append({"role": "tool", "content": result, "name": tool_name})


def _stream_response(stream: Any) -> tuple[str, str, dict[int, dict], int]:
    """스트림을 소비하며 content, thinking, tool_calls, prompt_eval_count를 반환한다.

    prompt_eval_count는 done=True인 마지막 chunk에서만 유효하게 제공된다.
    모델/버전에 따라 None일 수 있으므로 0으로 기본값 처리한다.
    """
    assistant_content = ""
    thinking_content = ""
    accumulated_tool_calls: dict[int, dict] = {}
    prompt_eval_count: int = 0

    with Live(console=console, refresh_per_second=10) as live:
        for chunk in stream:
            msg = chunk.get("message") or {}
            thinking = msg.get("thinking") or ""
            if thinking:
                thinking_content += thinking
            content = msg.get("content") or ""
            if content:
                assistant_content += content
                live.update(Markdown(assistant_content))
            _accumulate_tool_calls(msg, accumulated_tool_calls)
            # done=True 인 마지막 chunk에서 토큰 수 캡처
            if chunk.get("done"):
                prompt_eval_count = chunk.get("prompt_eval_count") or 0

    return (
        assistant_content,
        thinking_content,
        accumulated_tool_calls,
        prompt_eval_count,
    )


# ──────────────────────────────────────────
# Agentic Loop
# ──────────────────────────────────────────


def run_agentic_loop(
    messages: list[dict],
    client: Client,
    dispatch: dict[str, Any],
    model: str = "qwen3-coder-next:latest",
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """tool_calls가 없어질 때까지 모델을 반복 호출하는 agentic loop."""
    final_content = ""
    for _ in range(max_iterations):
        stream = client.chat(
            model=model,
            messages=messages,
            tools=TOOLS,
            stream=True,
        )
        assistant_content, thinking_content, tool_calls, token_count = _stream_response(
            stream
        )
        if thinking_content:
            console.print(
                Panel(thinking_content, title="[dim]Thinking[/]", style="dim")
            )
        messages.append({"role": "assistant", "content": assistant_content})
        if token_count:
            console.print(
                f"[dim][Token] {token_count:,} / 128k "
                f"({'⚠️ ' if token_count > TOKEN_THRESHOLD else ''}used)[/]"
            )
            messages = trim_by_tokens(messages, token_count)
        if not tool_calls:
            final_content = assistant_content
            break
        _process_tool_calls(tool_calls, messages, dispatch)
    else:
        console.print("[bold red][경고] 최대 반복 횟수(10) 초과[/]")
    return final_content


class ConnectionInfo(NamedTuple):
    """Ollama 서버 연결 정보. 서브에이전트에 직렬화하여 전달한다."""

    host: str
    cf_client_id: str
    cf_client_secret: str


def _build_full_system_prompt(base: str, memory: SessionMemory) -> str:
    """base prompt에 메모리 컨텍스트를 추가한 system prompt를 반환한다."""
    ctx = memory.to_context_string()
    return f"{base}\n\n{ctx}" if ctx else base


def _handle_memory_command(user: str, memory: SessionMemory) -> bool:
    """'/memory' 명령이면 처리하고 True를 반환한다. 일반 입력이면 False."""
    if not user.startswith("/memory"):
        return False
    parts = user.split(maxsplit=2)
    sub = parts[1] if len(parts) > 1 else ""

    if sub == "list":
        entries = memory.all()
        if not entries:
            console.print("[dim]저장된 메모리가 없습니다.[/]")
        for e in entries:
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            console.print(f"  [cyan]{e.id[:8]}[/] {e.content}{tag_str}")
    elif sub == "add" and len(parts) > 2:
        text = parts[2]
        tags = re.findall(r"#(\w+)", text)
        content = re.sub(r"#\w+", "", text).strip()
        entry = memory.add(content, tags)
        console.print(f"[green]Memory saved:[/] {entry.id[:8]} — {entry.content}")
    elif sub == "search" and len(parts) > 2:
        results = memory.search(parts[2])
        if not results:
            console.print("[dim]검색 결과 없음[/]")
        for e in results:
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            console.print(f"  [cyan]{e.id[:8]}[/] {e.content}{tag_str}")
    elif sub == "clear":
        count = memory.clear()
        console.print(f"[yellow]Memory cleared ({count} entries)[/]")
    else:
        console.print(
            "[dim]Usage: /memory add <text> [#tag] | "
            "/memory list | /memory search <q> | /memory clear[/]"
        )
    return True


def _handle_plan_command(
    user: str,
    client: Client,
    model: str,
    base_prompt: str,
) -> bool:
    """'/plan <task>' 명령이면 plan 모드로 실행하고 True를 반환한다."""
    if not user.startswith("/plan "):
        return False
    task = user[len("/plan ") :].strip()
    if not task:
        console.print("[dim]Usage: /plan <task description>[/]")
        return True
    run_plan(task, client, model, base_prompt)
    return True


def _parse_subagent_input(
    raw: str,
    default_model: str,
) -> list[tuple[str, str]]:
    """subagent 커맨드 텍스트를 파싱해 (task, model) 목록을 반환한다.

    지원 문법:
      --model <name> task1 | task2   → 전체 동일 모델 지정
      @model task1 | @model2 task2   → 태스크별 개별 모델 지정
      task1 | task2                  → default_model 사용
    """
    global_model = default_model
    flag_match = re.match(r"--model\s+(\S+)\s*(.*)", raw, re.DOTALL)
    if flag_match:
        global_model = flag_match.group(1)
        raw = flag_match.group(2).strip()

    result: list[tuple[str, str]] = []
    for text in (t.strip() for t in raw.split("|") if t.strip()):
        per_match = re.match(r"@(\S+)\s+(.*)", text, re.DOTALL)
        if per_match:
            result.append((per_match.group(2).strip(), per_match.group(1)))
        else:
            result.append((text, global_model))
    return result


def list_available_models(client: Client) -> set[str] | None:
    """올라마 서버에서 사용 가능한 모델 목록을 반환한다. 서버 오류 시 None."""
    try:
        response = client.list()
        return {m.model for m in response.models}
    except Exception:
        return None


def _is_model_available(model: str, available: set[str]) -> bool:
    """태그 포함/생략 둘 다 허용하여 모델 존재 여부를 확인한다.

    'llama3' 입력 시 'llama3:latest' 도 검색한다.
    """
    return model in available or f"{model}:latest" in available


def _handle_subagent_command(
    user: str,
    conn: ConnectionInfo,
    client: Client,
    model: str,
    base_prompt: str,
) -> bool:
    """'/subagent task1 | task2' 명령이면 병렬 실행하고 True를 반환한다."""
    if not user.startswith("/subagent "):
        return False
    raw = user[len("/subagent ") :].strip()
    parsed = _parse_subagent_input(raw, model)
    if not parsed:
        console.print(
            "[dim]Usage: /subagent [--model <m>] <task1> | [@<m>] <task2> | ...[/]"
        )
        return True
    available = list_available_models(client)
    if available is not None:
        invalid = [m for _, m in parsed if not _is_model_available(m, available)]
        if invalid:
            console.print(f"[red]알 수 없는 모델: {invalid}[/]")
            console.print(f"[dim]사용 가능: {', '.join(sorted(available))}[/]")
            return True
    tasks = [
        SubagentTask(
            name=f"agent-{i + 1}",
            task=task,
            model=task_model,
            host=conn.host,
            cf_client_id=conn.cf_client_id,
            cf_client_secret=conn.cf_client_secret,
            system_prompt=base_prompt,
        )
        for i, (task, task_model) in enumerate(parsed)
    ]
    console.print(
        Panel(f"[bold magenta]SUBAGENTS[/] — {len(tasks)}개 병렬 실행", style="magenta")
    )
    results = run_subagents(tasks)
    for task_obj, (name, result) in zip(tasks, results):
        console.print(
            Panel(result, title=f"[magenta]{name}[/] [dim]{task_obj.model}[/]")
        )
    return True


def _auto_save_session(messages: list[dict]) -> None:
    """대화 히스토리를 .agents/sessions/ 에 타임스탬프 파일명으로 저장한다."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SESSION_DIR / f"{ts}.json"
    save_session(messages, path)
    console.print(f"[dim][Session] 저장됨: {path}[/]")


def main() -> None:
    """대화형 agentic loop 진입점."""
    agent_config = load_config()
    console.print(
        f"[dim][Config] model={agent_config.model} | "
        f"mode={agent_config.permission_mode.value} | "
        f"threshold={agent_config.token_threshold:,}[/]"
    )
    conn = ConnectionInfo(
        host=os.getenv("OLLAMA_HOST") or agent_config.ollama_host,
        cf_client_id=os.getenv("CF_ACCESS_CLIENT_ID")
        or agent_config.cf_access_client_id,
        cf_client_secret=os.getenv("CF_ACCESS_CLIENT_SECRET")
        or agent_config.cf_access_client_secret,
    )
    client = Client(
        host=conn.host,
        headers={
            "CF-Access-Client-Id": conn.cf_client_id,
            "CF-Access-Client-Secret": conn.cf_client_secret,
        },
    )
    perm_config = PermissionConfig(
        mode=agent_config.permission_mode,
        deny_patterns=agent_config.deny_patterns,
    )
    dispatch = build_dispatch(perm_config)
    base_prompt = build_system_prompt(agent_config.agents_md_path)
    memory = SessionMemory()
    messages: list[dict] = [{"role": "system", "content": base_prompt}]

    while True:
        user = input("\nYou: ")
        if user.lower() in ["exit", "quit"]:
            _auto_save_session(messages)
            break
        if _handle_memory_command(user, memory):
            messages[0]["content"] = _build_full_system_prompt(base_prompt, memory)
            continue
        if _handle_plan_command(user, client, agent_config.model, base_prompt):
            continue
        if _handle_subagent_command(
            user, conn, client, agent_config.model, base_prompt
        ):
            continue
        messages[0]["content"] = _build_full_system_prompt(base_prompt, memory)
        messages.append({"role": "user", "content": user})
        messages = trim_messages(messages)
        console.print("\n[bold green]Agent:[/]")
        run_agentic_loop(
            messages,
            client,
            dispatch,
            model=agent_config.model,
            max_iterations=agent_config.max_iterations,
        )


if __name__ == "__main__":
    main()
