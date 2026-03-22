# CLAUDE.md — ollaAgent

This file provides guidance to Claude Code when working in this repository.

---

## Project Overview

**ollaAgent** — ollama 기반 로컬 LLM 에이전트.
PyPI: https://pypi.org/project/ollaagent/
GitHub: https://github.com/github010000/ollaAgent

### 핵심 모듈

| 파일 | 역할 |
|------|------|
| `ollaAgent/agent.py` | 메인 에이전트 루프 & CLI 진입점 (`uv run ollaagent`) |
| `ollaAgent/memory.py` | JSON 기반 영구 메모리 (`/memory` 커맨드) |
| `ollaAgent/plan_mode.py` | 플랜 전용 모드 — `tools=[]`, `/plan` 커맨드 |
| `ollaAgent/subagent.py` | `multiprocessing.Pool` 병렬 서브에이전트, `/subagent` 커맨드 |
| `ollaAgent/tool_bash.py` | bash 도구 (권한 제어 포함) |
| `ollaAgent/permissions.py` | 허용/거부 패턴 매칭 |
| `ollaAgent/config_loader.py` | YAML 설정 & 시스템 프롬프트 빌더 |
| `ollaAgent/ollama_client.py` | ollama 클라이언트 팩토리 |

### 내장 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/plan <태스크>` | 도구 실행 없이 단계별 계획 생성 |
| `/subagent` | 여러 태스크 병렬 실행 |
| `/memory add/list/search/clear` | 영구 메모리 관리 |
| `/exit` | 에이전트 종료 |

### 서브에이전트 모델 지정

```
--model <name> task1 | task2     # 전체 모델 지정
@model1 task1 | @model2 task2    # 태스크별 모델 지정
```

---

## Code Standards & Safety (Priority)

- **Strict Type Safety**: 모든 함수에 Type Hint 필수 (Pydantic v2 준수)
- **Security**: eval/exec 금지, 환경변수 사용 철저 (`.env`)
- **Performance**: 루프 내 N+1 금지, `multiprocessing` 활용
- **Formatting**: 수정 후 반드시 `black` 및 `isort` 실행

## Auto-Validation Loop (필수)

파일 수정/생성 후 **사용자 확인 없이** 자동 실행:

1. `black` 적용
2. `isort` 적용
3. `uv run pytest` 실행

결과 즉시 출력:
- 성공: `"Auto-validation: black & isort applied → pytest passed"`
- 실패: `"pytest failed: [상세] → 대장, 확인 부탁드립니다."`

---

## Environment

```bash
uv sync          # 의존성 설치
uv run ollaagent # 실행
uv run pytest    # 테스트
uv build         # 빌드 (.whl, .tar.gz)
uv publish       # PyPI 배포 (UV_PUBLISH_TOKEN 필요)
```

`.env` 설정:
```env
OLLAMA_HOST=http://localhost:11434
CF_CLIENT_ID=
CF_CLIENT_SECRET=
PYPI_TOKEN=pypi-...
```

## Deployment

```bash
# 버전 변경 후 배포
# 1. pyproject.toml version 업데이트
# 2. git tag
git tag v0.1.0 && git push origin v0.1.0
# 3. 빌드 & 배포
uv build
UV_PUBLISH_TOKEN=$(grep PYPI_TOKEN .env | cut -d= -f2) uv publish
```

---

## Conventions

- **호칭**: 사용자를 '대장'으로 인식
- **응답**: 해결책 위주, 간결하게
- **The 30 Commandments**: 코드 생성/수정 시 자동 체크 (보안/성능/가독성/타입/아키텍처)
