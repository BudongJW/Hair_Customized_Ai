# Hair Customized AI

Spring Boot, Expo React Native, Python FastAPI 기반 AI 가상 헤어 피팅 프로젝트입니다.

## 한 번에 실행하기

먼저 PostgreSQL은 pgAdmin/로컬 앱에서 실행되어 있어야 합니다.

그 다음 프로젝트 루트에서 아래 명령을 실행하면 Spring Boot, Python AI worker, Expo가 각각 새 PowerShell 창으로 열립니다.

```powershell
cd C:\Users\aaron\Project\Hair_Customized_Ai
.\scripts\start-dev.ps1
```

실행 계획만 확인하려면:

```powershell
.\scripts\start-dev.ps1 -DryRun
```

## 환경변수

Spring Boot를 IntelliJ가 아니라 스크립트로 실행하려면 루트에 `.env.backend` 파일이 필요합니다. 형식은 `.env.backend.example`을 기준으로 작성하면 됩니다.

```text
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-northeast-2
APP_STORAGE_MODE=s3
APP_STORAGE_BUCKET=hair-customized-ai-aaron-dev
APP_STORAGE_PUBLIC_BASE_URL=https://hair-customized-ai-aaron-dev.s3.ap-northeast-2.amazonaws.com
APP_QUEUE_PROVIDER=http
APP_AI_WORKER_BASE_URL=http://localhost:8000
APP_AI_WORKER_BACKEND_BASE_URL=http://localhost:8080
```

Python AI worker는 `ai/.env`를 읽습니다.

```text
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-northeast-2
APP_STORAGE_BUCKET=hair-customized-ai-aaron-dev
AI_WORKER_BACKEND_BASE_URL=http://localhost:8080
FACE_LANDMARKER_MODEL_PATH=C:\Users\aaron\Project\Hair_Customized_Ai\ai\models\face_landmarker.task
HAIR_SEGMENTER_MODEL_PATH=C:\Users\aaron\Project\Hair_Customized_Ai\ai\models\hair_segmenter.tflite
```

모델 경로는 생략해도 기본값으로 `ai/models` 아래 파일을 사용합니다. `scripts/run-ai.ps1`는 Face Landmarker와 Hair Segmenter 모델 파일이 없으면 자동으로 다운로드합니다.

AWS 키는 코드, frontend `.env`, Git에 올리지 마세요. 실제 `.env.backend`, `ai/.env`, `frontend/.env`, `ai/models/*.task`, `ai/models/*.tflite`는 `.gitignore`에 제외되어 있습니다.

## 개별 실행

각 서비스를 따로 실행해야 할 때는 아래 스크립트를 사용합니다.

```powershell
.\scripts\run-backend.ps1
.\scripts\run-ai.ps1
.\scripts\run-expo.ps1
```

## AI 단계

현재 Python worker는 MediaPipe Face Landmarker로 얼굴 기준점을 잡고, MediaPipe HairSegmenter로 헤어모델 사진의 머리카락 mask를 생성합니다. 헤어 피팅에서는 저장된 얼굴 위치/기울기 정보를 기준으로 분리된 헤어레이어를 맞춰 합성합니다.
