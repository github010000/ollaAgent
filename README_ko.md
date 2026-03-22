# ollaAgent

[ollama](https://ollama.com) 기반 로컬 LLM 에이전트 — 영구 메모리, 플랜 모드, 병렬 서브에이전트를 지원합니다.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 주요 기능

| 기능 | 설명 |
|------|------|
| **에이전트 루프** | `run_python`, `run_bash`, `read_file`, `write_file`, `list_files` 도구를 활용한 반복 실행 루프 |
| **영구 메모리** | JSON 파일 기반 메모리. `/memory add/list/search/clear` 명령으로 관리 |
| **세션 저장** | 종료 시 대화 히스토리를 `.agents/sessions/`에 자동 저장 |
| **플랜 모드** | `/plan <태스크>` — 도구 실행 없이 단계별 계획만 생성 |
| **서브에이전트** | `/subagent` — `multiprocessing.Pool`로 여러 ollama 인스턴스를 병렬 실행 |
| **권한 제어** | bash 명령에 대한 허용/거부 패턴 설정 |
| **Cloudflare Access** | 터널링된 ollama 엔드포인트에 CF-Access 헤더 지원 |

## 요구사항

- Python 3.11 이상
- [ollama](https://ollama.com) 로컬 실행 중 (또는 Cloudflare Access 경유)
- [uv](https://docs.astral.sh/uv/)

## 설치

```bash
git clone https://github.com/github010000/ollaAgent
cd ollaAgent
uv sync
```

## 실행

```bash
# 에이전트 시작 (기본 모델: qwen3-coder-next:latest)
ollaagent

# 모델 지정
ollaagent --model qwen2.5-coder:7b

# 원격 ollama 호스트 지정
ollaagent --host https://your-ollama.example.com
```

## 내장 명령어

| 명령어 | 설명 |
|--------|------|
| `/plan <태스크>` | 단계별 계획 생성 (도구 실행 없음) |
| `/subagent` | 여러 태스크를 병렬로 실행 |
| `/memory add <내용>` | 영구 메모리에 항목 추가 |
| `/memory list` | 전체 메모리 목록 출력 |
| `/memory search <키워드>` | 키워드로 메모리 검색 |
| `/memory clear` | 전체 메모리 삭제 |
| `/exit` | 에이전트 종료 |

## 서브에이전트 사용법

단일 모델로 여러 태스크 병렬 실행:
```
/subagent
> --model llama3:8b 태스크1 | 태스크2 | 태스크3
```

태스크별 모델 지정:
```
> @qwen2.5-coder:7b 정렬 알고리즘 작성 | @llama3:8b 코드 설명
```

## 환경 설정

`.env` 파일을 `ollaagent`를 실행하는 디렉토리에 생성합니다:

```env
OLLAMA_HOST=http://localhost:11434
CF_ACCESS_CLIENT_ID=
CF_ACCESS_CLIENT_SECRET=
```

## 프로젝트 구조

```
ollaAgent/
├── agent.py          # 메인 에이전트 루프 & CLI 진입점
├── memory.py         # 영구 메모리 (JSON)
├── plan_mode.py      # 플랜 전용 모드 (tools=[])
├── subagent.py       # multiprocessing 기반 병렬 서브에이전트
├── tool_bash.py      # 권한 제어가 포함된 bash 도구
├── permissions.py    # 허용/거부 패턴 매칭
├── config_loader.py  # YAML 설정 & 시스템 프롬프트 빌더
└── ollama_client.py  # ollama 클라이언트 팩토리
```

## 테스트 실행

```bash
uv run pytest
```

## 라이선스

MIT — 자세한 내용은 [LICENSE](LICENSE)를 참조하세요.
