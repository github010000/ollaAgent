# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.1.3] - 2026-03-22

### Fixed
- `tool_calls: null`, `message: null`, `content: null`, `thinking: null` 등 실제 API 응답의 null 필드로 인한 `TypeError` 방어 처리 (`agent.py`)
- `entries: null`, `messages: null` JSON 필드로 인한 memory 로드 오류 방어 처리 (`memory.py`)

### Tests
- `_make_chunk` 헬퍼에 `null_*` 플래그 추가 — 실제 API null 응답 재현 가능
- null API 응답 엣지 케이스 5종 추가 (138 → 143 tests)

---

## [0.1.2] - 2026-03-22

### Fixed
- `_accumulate_tool_calls`에서 `tool_calls` 키가 존재하지만 값이 `null`인 경우 `TypeError` 발생 수정
  - `msg.get("tool_calls", [])` → `msg.get("tool_calls") or []`

---

## [0.1.1] - 2026-03-22

### Added
- `AgentConfig`에 `ollama_host`, `cf_access_client_id`, `cf_access_client_secret` 필드 추가
- `~/.agents/config.yaml`만으로 `.env` 없이 원격 ollama 서버 접속 가능
- `.env.sample` 생성 (용도별 주석 포함)

### Fixed
- `OLLAMA_HOST` 하드코딩 제거 → `os.getenv("OLLAMA_HOST") or agent_config.ollama_host` 우선순위 적용
- README / README_ko 환경변수명 오류 수정: `CF_CLIENT_ID` → `CF_ACCESS_CLIENT_ID`

---

## [0.1.0] - 2026-03-22

### Added
- 최초 릴리즈
- 에이전트 루프 (`run_python`, `run_bash`, `read_file`, `write_file`, `list_files`)
- JSON 기반 영구 메모리 (`/memory add/list/search/clear`)
- 플랜 모드 (`/plan`)
- 병렬 서브에이전트 (`/subagent`, `multiprocessing.Pool`)
- 권한 제어 (`allow/deny` 패턴)
- Cloudflare Access 헤더 지원
- YAML 계층 설정 (`global < project < local`)
