import pytest
import yaml
from ollaAgent.config_loader import (AgentConfig, _load_yaml, _merge,
                           build_system_prompt, load_config)
from ollaAgent.permissions import PermissionMode

# ──────────────────────────────────────────
# Unit Tests: _load_yaml
# ──────────────────────────────────────────


class TestLoadYaml:

    def test_returns_empty_when_file_missing(self, tmp_path):
        result = _load_yaml(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_loads_valid_yaml(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("model: test-model\nmax_iterations: 5\n")
        result = _load_yaml(path)
        assert result["model"] == "test-model"
        assert result["max_iterations"] == 5

    def test_returns_empty_on_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("key: [unclosed")
        result = _load_yaml(path)
        assert result == {}

    def test_returns_empty_on_non_dict_yaml(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n")
        result = _load_yaml(path)
        assert result == {}


# ──────────────────────────────────────────
# Unit Tests: _merge
# ──────────────────────────────────────────


class TestMerge:

    def test_override_replaces_base(self):
        base = {"model": "a", "threshold": 80000}
        override = {"model": "b"}
        result = _merge(base, override)
        assert result["model"] == "b"
        assert result["threshold"] == 80000

    def test_none_value_not_applied(self):
        base = {"model": "a"}
        override = {"model": None}
        result = _merge(base, override)
        assert result["model"] == "a"

    def test_empty_override_unchanged(self):
        base = {"model": "a", "threshold": 80000}
        result = _merge(base, {})
        assert result == base


# ──────────────────────────────────────────
# Integration Tests: load_config
# ──────────────────────────────────────────


class TestLoadConfig:

    def _write_yaml(self, path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            yaml.dump(data, fh)

    def test_defaults_when_no_files(self, tmp_path):
        """모든 config 파일 없으면 AgentConfig 기본값 반환."""
        config = load_config(
            global_path=tmp_path / "global.yaml",
            project_path=tmp_path / "project.yaml",
            local_path=tmp_path / "local.yaml",
        )
        assert config == AgentConfig()

    def test_project_overrides_global(self, tmp_path):
        """project config가 global 값을 교체한다."""
        self._write_yaml(
            tmp_path / "global.yaml",
            {"model": "global-model", "max_iterations": 10},
        )
        self._write_yaml(
            tmp_path / "project.yaml",
            {"model": "project-model"},
        )
        config = load_config(
            global_path=tmp_path / "global.yaml",
            project_path=tmp_path / "project.yaml",
            local_path=tmp_path / "local.yaml",
        )
        assert config.model == "project-model"
        assert config.max_iterations == 10  # global 값 유지

    def test_local_overrides_project(self, tmp_path):
        """local config가 project 값을 최종 교체한다."""
        self._write_yaml(
            tmp_path / "global.yaml",
            {"permission_mode": "prompt"},
        )
        self._write_yaml(
            tmp_path / "project.yaml",
            {"permission_mode": "auto"},
        )
        self._write_yaml(
            tmp_path / "local.yaml",
            {"permission_mode": "deny"},
        )
        config = load_config(
            global_path=tmp_path / "global.yaml",
            project_path=tmp_path / "project.yaml",
            local_path=tmp_path / "local.yaml",
        )
        assert config.permission_mode == PermissionMode.DENY

    def test_partial_override_keeps_rest(self, tmp_path):
        """일부 키만 오버라이드 시 나머지는 상위 config 유지."""
        self._write_yaml(
            tmp_path / "global.yaml",
            {"model": "base-model", "token_threshold": 80000, "max_iterations": 10},
        )
        self._write_yaml(
            tmp_path / "project.yaml",
            {"token_threshold": 60000},  # 이것만 교체
        )
        config = load_config(
            global_path=tmp_path / "global.yaml",
            project_path=tmp_path / "project.yaml",
            local_path=tmp_path / "local.yaml",
        )
        assert config.model == "base-model"  # global 유지
        assert config.token_threshold == 60000  # project 교체
        assert config.max_iterations == 10  # global 유지

    def test_missing_file_skipped_without_error(self, tmp_path):
        """없는 파일은 에러 없이 skip된다."""
        self._write_yaml(tmp_path / "global.yaml", {"max_iterations": 7})
        config = load_config(
            global_path=tmp_path / "global.yaml",
            project_path=tmp_path / "nonexistent_project.yaml",
            local_path=tmp_path / "nonexistent_local.yaml",
        )
        assert config.max_iterations == 7

    def test_global_auto_created_when_missing(self, tmp_path):
        """global config 파일 없으면 기본값으로 자동 생성한다."""
        global_path = tmp_path / ".agents" / "config.yaml"
        assert not global_path.exists()
        load_config(
            global_path=global_path,
            project_path=tmp_path / "p.yaml",
            local_path=tmp_path / "l.yaml",
        )
        assert global_path.exists()
        data = _load_yaml(global_path)
        assert "model" in data


# ──────────────────────────────────────────
# Unit Tests: build_system_prompt
# ──────────────────────────────────────────


class TestBuildSystemPrompt:

    def test_returns_base_when_file_missing(self):
        result = build_system_prompt("/nonexistent/path/AGENT.md")
        assert "expert coder" in result
        assert len(result.split("\n")) < 5  # prepend 없음

    def test_prepends_file_content(self, tmp_path):
        md = tmp_path / "AGENT.md"
        md.write_text("# Project Rules\nAlways use type hints.")
        result = build_system_prompt(str(md))
        assert "# Project Rules" in result
        assert "expert coder" in result
        assert result.index("# Project Rules") < result.index("expert coder")

    def test_separator_between_md_and_base(self, tmp_path):
        md = tmp_path / "AGENT.md"
        md.write_text("# Rules")
        result = build_system_prompt(str(md))
        assert "---" in result
