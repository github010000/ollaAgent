import json

import pytest
from ollaAgent.memory import MemoryEntry, SessionMemory, load_session, save_session

# ──────────────────────────────────────────
# Unit Tests: MemoryEntry
# ──────────────────────────────────────────


class TestMemoryEntry:

    def test_defaults_populated(self):
        entry = MemoryEntry(content="test")
        assert len(entry.id) == 36  # UUID format
        assert entry.content == "test"
        assert entry.tags == []
        assert entry.created_at != ""

    def test_custom_tags(self):
        entry = MemoryEntry(content="hello", tags=["a", "b"])
        assert entry.tags == ["a", "b"]


# ──────────────────────────────────────────
# Unit Tests: SessionMemory
# ──────────────────────────────────────────


class TestSessionMemoryAdd:

    def test_add_returns_entry(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        entry = mem.add("remember this")
        assert entry.content == "remember this"
        assert entry.tags == []

    def test_add_with_tags(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        entry = mem.add("tagged", tags=["x", "y"])
        assert entry.tags == ["x", "y"]

    def test_add_persists_to_disk(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = SessionMemory(path=path)
        mem.add("persisted")
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data["entries"]) == 1


class TestSessionMemorySearch:

    def test_search_by_content(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        mem.add("python is cool")
        mem.add("java is verbose")
        results = mem.search("python")
        assert len(results) == 1
        assert results[0].content == "python is cool"

    def test_search_by_tag(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        mem.add("entry one", tags=["alpha"])
        mem.add("entry two", tags=["beta"])
        results = mem.search("alpha")
        assert len(results) == 1
        assert results[0].tags == ["alpha"]

    def test_search_case_insensitive(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        mem.add("Hello World")
        assert len(mem.search("hello")) == 1
        assert len(mem.search("WORLD")) == 1

    def test_search_no_match(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        mem.add("something")
        assert mem.search("nothing") == []


class TestSessionMemoryClear:

    def test_clear_removes_all(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        mem.add("a")
        mem.add("b")
        count = mem.clear()
        assert count == 2
        assert mem.all() == []

    def test_clear_persists(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = SessionMemory(path=path)
        mem.add("x")
        mem.clear()
        mem2 = SessionMemory(path=path)
        assert mem2.all() == []


class TestSessionMemoryPersistence:

    def test_reload_from_disk(self, tmp_path):
        path = tmp_path / "mem.json"
        mem = SessionMemory(path=path)
        mem.add("will survive", tags=["tag1"])
        mem2 = SessionMemory(path=path)
        assert len(mem2.all()) == 1
        assert mem2.all()[0].content == "will survive"
        assert mem2.all()[0].tags == ["tag1"]

    def test_missing_file_gives_empty(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "nonexistent.json")
        assert mem.all() == []

    def test_invalid_json_gives_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{invalid json}")
        mem = SessionMemory(path=path)
        assert mem.all() == []


class TestSessionMemoryContextString:

    def test_empty_returns_empty_string(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        assert mem.to_context_string() == ""

    def test_includes_all_entries(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        mem.add("first item")
        mem.add("second item", tags=["important"])
        ctx = mem.to_context_string()
        assert "first item" in ctx
        assert "second item" in ctx
        assert "important" in ctx

    def test_starts_with_header(self, tmp_path):
        mem = SessionMemory(path=tmp_path / "mem.json")
        mem.add("x")
        assert mem.to_context_string().startswith("## Persistent Memory")


# ──────────────────────────────────────────
# Unit Tests: save_session / load_session
# ──────────────────────────────────────────


class TestSaveLoadSession:

    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "sessions" / "s.json"
        messages = [{"role": "user", "content": "hi"}]
        save_session(messages, path)
        assert path.exists()

    def test_load_returns_messages(self, tmp_path):
        path = tmp_path / "s.json"
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ]
        save_session(messages, path)
        loaded = load_session(path)
        assert loaded == messages

    def test_load_missing_file_returns_empty(self, tmp_path):
        result = load_session(tmp_path / "nope.json")
        assert result == []

    def test_load_invalid_json_returns_empty(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json at all")
        result = load_session(path)
        assert result == []

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "sess.json"
        save_session([{"role": "user", "content": "x"}], path)
        assert path.exists()

    def test_saved_json_has_metadata(self, tmp_path):
        path = tmp_path / "s.json"
        save_session([], path)
        data = json.loads(path.read_text())
        assert data["version"] == "1"
        assert "saved_at" in data
