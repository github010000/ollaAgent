# Changelog

All notable changes to this project will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [0.1.9] - 2026-03-23

### Fixed
- 스트리밍 중 plain text와 Markdown이 이중 출력되던 문제 — `Live(transient=True)` 적용으로 스트림 종료 시 plain text 자동 삭제

### Tests
- `test_markdown_printed_once_after_stream` — Markdown 1회만 출력 검증
- `test_markdown_not_printed_when_no_content` — content 없으면 Markdown 미출력 검증
- `test_live_created_with_transient_true` — `transient=True` 설정 검증

---

## [0.1.8] - 2026-03-23

### Fixed
- 스트리밍 실시간성 개선 — `Live(auto_refresh=False)` + `live.refresh()` 매 chunk 즉시 출력
- 스트림 완료 후 `Markdown` 최종 렌더링으로 가독성 확보

### Refactored
- `_stream_response` → `stream_response` (public 노출, 직접 테스트 가능)

### Tests
- `TestStreamResponse` 8종 추가 — content/thinking/tool_calls/token 반환값 정확성 검증

---

## [0.1.7] - 2026-03-22

### Fixed
- 빈 입력 시 LLM 호출 방지 (`strip()` 후 빈 문자열 skip)
- 멀티라인 입력 지원 — 줄 끝 `\` 입력 시 다음 줄 이어쓰기 (`... ` 프롬프트)

### Refactored
- 입력 처리 로직을 `_read_user_input()` 함수로 추출 (테스트 가능)

### Tests
- `TestReadUserInput` 8종 추가 (단일/빈/공백/멀티라인/EOF/Ctrl+C)

---

## [0.1.6] - 2026-03-22

### Fixed
- `/exit`, `/quit` 커맨드가 LLM으로 전달되던 문제 — exit 매칭에 `/exit`, `/quit` 추가
- `Ctrl+C` (`KeyboardInterrupt`) 시 traceback 출력 → 세션 저장 후 정상 종료
- `Ctrl+D` (EOF) 입력 시 비정상 종료 → 정상 종료 처리
- 한글 입력 중 삭제/수정 시 글자 깨짐 — `readline` import로 터미널 편집 정상화

---

## [0.1.4] - 2026-03-22

### Added
- `--version` / `-v` 플래그 지원 (`ollaagent --version`)
- `--model` / `-m` CLI 인수 지원 (config 기본값 오버라이드)
- `--host` CLI 인수 지원 (env/config보다 우선)

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
