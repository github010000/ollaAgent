# ollaAgent

A local LLM agent powered by [ollama](https://ollama.com) — with persistent memory, plan mode, and parallel subagents.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Features

| Feature | Description |
|---------|-------------|
| **Agent Loop** | Iterative tool-calling loop with `run_python`, `run_bash`, `read_file`, `write_file`, `list_files` |
| **Persistent Memory** | JSON-backed memory with `/memory add/list/search/clear` commands |
| **Session Saving** | Conversation history auto-saved to `.agents/sessions/` on exit |
| **Plan Mode** | `/plan <task>` — generates a structured step-by-step plan without executing tools |
| **Subagents** | `/subagent` — runs multiple ollama instances in parallel via `multiprocessing.Pool` |
| **Permission Control** | Configurable allow/deny patterns for bash commands |
| **Cloudflare Access** | Supports CF-Access headers for tunneled ollama endpoints |

## Requirements

- Python 3.11+
- [ollama](https://ollama.com) running locally (or via Cloudflare Access)
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone https://github.com/github010000/ollaAgent
cd ollaAgent
uv sync
```

## Usage

```bash
# Start the agent (default model: qwen3-coder-next:latest)
ollaagent

# With a specific model
ollaagent --model qwen2.5-coder:7b

# With a remote ollama host
ollaagent --host https://your-ollama.example.com
```

## Built-in Commands

| Command | Description |
|---------|-------------|
| `/plan <task>` | Generate a step-by-step plan (no execution) |
| `/subagent` | Run tasks in parallel across multiple ollama instances |
| `/memory add <text>` | Add an entry to persistent memory |
| `/memory list` | List all memory entries |
| `/memory search <query>` | Search memory by keyword |
| `/memory clear` | Clear all memory entries |
| `/exit` | Exit the agent |

## Subagent Usage

Single model across all tasks:
```
/subagent
> --model llama3:8b task one | task two | task three
```

Per-task model assignment:
```
> @qwen2.5-coder:7b write a sorting algorithm | @llama3:8b explain it
```

## Configuration

Create a `.env` file in the project root:

```env
OLLAMA_HOST=http://localhost:11434
CF_ACCESS_CLIENT_ID=
CF_ACCESS_CLIENT_SECRET=
```

## Project Structure

```
ollaAgent/
├── agent.py          # Main agent loop & CLI entry point
├── memory.py         # Persistent memory (JSON)
├── plan_mode.py      # Plan-only mode (tools=[])
├── subagent.py       # Parallel subagents via multiprocessing
├── tool_bash.py      # Bash tool with permission control
├── permissions.py    # Allow/deny pattern matching
├── config_loader.py  # YAML config & system prompt builder
└── ollama_client.py  # Ollama client factory
```

## Running Tests

```bash
uv run pytest
```

## License

MIT — see [LICENSE](LICENSE) for details.
