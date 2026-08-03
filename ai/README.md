# Hair Customized AI Worker

FastAPI 기반 AI 처리 서버입니다. Spring Boot가 얼굴 프로필/피팅 작업 ID를 전달하면, worker가 S3 이미지를 읽고 처리 결과를 다시 S3와 DB에 저장합니다.

## 현재 AI 단계

1. MediaPipe Face Landmarker로 내 얼굴의 실제 3D 얼굴 랜드마크를 추출합니다.
2. 추출한 `faceBox`, `keyLandmarks`, `pose`, 478개 랜드마크를 `landmarksJson`으로 DB에 저장합니다.
3. 헤어 피팅 시 저장된 얼굴 랜드마크를 기준으로 머리 위치와 기울기를 맞춥니다.
4. 헤어모델 사진에서도 얼굴 랜드마크를 추출하고, 눈/코/입/얼굴 중심부 보호 마스크를 만들어 헤어마스크에서 제외합니다.
5. 헤어모델 사진의 머리카락 후보 영역을 분리해 내 얼굴 사진 위에 합성합니다.

아직 전용 헤어 세그멘테이션/생성형 합성 모델은 붙지 않았습니다. 현재 단계는 MediaPipe 얼굴 랜드마크 기반으로 눈/얼굴 특징이 헤어레이어에 섞이는 문제를 줄이는 방식입니다.

## 실행

루트에서 전체 실행:

```powershell
cd C:\Users\aaron\Project\Hair_Customized_Ai
.\scripts\start-dev.ps1
```

AI worker만 실행:

```powershell
cd C:\Users\aaron\Project\Hair_Customized_Ai
.\scripts\run-ai.ps1
```

`run-ai.ps1`는 필요한 Python 패키지를 설치하고, `ai/models/face_landmarker.task`가 없으면 MediaPipe 공식 모델 파일을 다운로드합니다.

## 환경변수

`ai/.env`:

```text
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-northeast-2
APP_STORAGE_BUCKET=hair-customized-ai-aaron-dev
AI_WORKER_BACKEND_BASE_URL=http://localhost:8080
FACE_LANDMARKER_MODEL_PATH=C:\Users\aaron\Project\Hair_Customized_Ai\ai\models\face_landmarker.task
```

AWS 키는 코드, frontend `.env`, Git에 넣지 마세요.
