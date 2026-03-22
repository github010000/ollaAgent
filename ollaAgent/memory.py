from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

MEMORY_PATH = Path.cwd() / ".agents" / "memory.json"
SESSION_DIR = Path.cwd() / ".agents" / "sessions"


# ──────────────────────────────────────────
# Data Model
# ──────────────────────────────────────────


class MemoryEntry(BaseModel):
    """단일 영구 메모리 항목."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ──────────────────────────────────────────
# Session Memory (JSON 기반)
# ──────────────────────────────────────────


class SessionMemory:
    """JSON 파일 기반 영구 메모리. /memory 커맨드로 관리한다."""

    VERSION = "1"

    def __init__(self, path: Path = MEMORY_PATH) -> None:
        self._path = path
        self._entries: list[MemoryEntry] = []
        self._load()

    def _load(self) -> None:
        """JSON 파일에서 메모리를 로드한다. 파일 없거나 파싱 오류 시 빈 상태로 시작."""
        if not self._path.exists():
            return
        try:
            with self._path.open(encoding="utf-8") as fh:
                data = json.load(fh)
            self._entries = [MemoryEntry(**e) for e in data.get("entries", [])]
        except (json.JSONDecodeError, KeyError, ValueError):
            self._entries = []

    def save(self) -> None:
        """현재 메모리를 JSON 파일에 저장한다."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.VERSION,
            "entries": [e.model_dump() for e in self._entries],
        }
        with self._path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    def add(self, content: str, tags: list[str] | None = None) -> MemoryEntry:
        """새 메모리 항목을 추가하고 즉시 저장한다."""
        entry = MemoryEntry(content=content, tags=tags or [])
        self._entries.append(entry)
        self.save()
        return entry

    def search(self, query: str) -> list[MemoryEntry]:
        """content 또는 tags에서 query를 포함하는 항목을 반환한다."""
        q = query.lower()
        return [
            e
            for e in self._entries
            if q in e.content.lower() or any(q in t.lower() for t in e.tags)
        ]

    def all(self) -> list[MemoryEntry]:
        """전체 메모리 항목을 반환한다."""
        return list(self._entries)

    def clear(self) -> int:
        """전체 메모리를 초기화하고 삭제된 개수를 반환한다."""
        count = len(self._entries)
        self._entries = []
        self.save()
        return count

    def to_context_string(self) -> str:
        """메모리 항목을 system prompt에 주입할 텍스트로 변환한다."""
        if not self._entries:
            return ""
        lines = ["## Persistent Memory"]
        for i, e in enumerate(self._entries, 1):
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"{i}. {e.content}{tag_str}")
        return "\n".join(lines)


# ──────────────────────────────────────────
# Session Save / Load
# ──────────────────────────────────────────


def save_session(messages: list[dict], path: Path) -> None:
    """대화 히스토리를 JSON 파일로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": "1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_session(path: Path) -> list[dict]:
    """저장된 세션 JSON에서 messages를 로드한다. 실패 시 빈 리스트 반환."""
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("messages", [])
    except (json.JSONDecodeError, KeyError):
        return []
