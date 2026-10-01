# Hair Customized AI Worker

## Texture-preserving fitting (current default)

See [INPAINTING.md](INPAINTING.md) for the `texture` workflow, saved masks,
thin-plate-spline alignment, pixel-preservation guarantees and the optional remote
GPU refinement boundary. Use `AI_FITTING_MODE=prepare` to inspect inputs without a
result, or `legacy` only to compare the older heuristic compositor.

## Hair removal (bald canvas)

See [HAIR_REMOVAL.md](HAIR_REMOVAL.md): skull-aware hair removal that fills hair outside the
skull with background and hair inside it with shaded scalp skin, instead of painting the whole
hair mask with skin colour ("mushroom head", leftover bangs). Not wired into `hair_transfer.py`
yet. Try it on a photo with `python tools/hair_removal_demo.py`.

FastAPI 기반 AI 처리 서버입니다. Spring Boot가 얼굴 프로필/피팅 작업 ID를 전달하면, worker가 S3 이미지를 읽고 처리 결과를 다시 S3와 DB에 저장합니다.

## 현재 AI 단계

1. MediaPipe Face Landmarker로 내 얼굴의 실제 3D 얼굴 랜드마크를 추출합니다.
2. 추출한 `faceBox`, `keyLandmarks`, `pose`, 478개 랜드마크를 `landmarksJson`으로 DB에 저장합니다.
3. 헤어 피팅 시 저장된 얼굴 랜드마크를 기준으로 눈 중심, 이마 anchor, 얼굴 폭, 기울기를 계산합니다.
4. MediaPipe HairSegmenter로 헤어모델 사진의 머리카락 segmentation mask를 생성합니다.
5. 헤어모델 사진에서도 얼굴 랜드마크를 추출하고, 눈/코/입/얼굴 중심부 보호 마스크를 만들어 헤어마스크에서 제외합니다.
6. 헤어모델의 헤어 crop 내부 anchor를 사용자 이마 anchor에 맞추고, 모델-사용자 기울기 차이만큼 회전 보정합니다.
7. 분리된 헤어레이어를 내 얼굴 사진 위에 합성합니다.

현재 단계는 MediaPipe의 공식 HairSegmenter와 Face Landmarker를 함께 사용합니다. 생성형 합성 모델은 아직 붙지 않았고, segmentation mask와 landmark-based transform을 조합하는 방식입니다.

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

`run-ai.ps1`는 필요한 Python 패키지를 설치하고, `ai/models/face_landmarker.task`, `ai/models/hair_segmenter.tflite`가 없으면 MediaPipe 공식 모델 파일을 다운로드합니다.

## 환경변수

`ai/.env`:

```text
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-northeast-2
APP_STORAGE_BUCKET=hair-customized-ai-aaron-dev
AI_WORKER_BACKEND_BASE_URL=http://localhost:8080
FACE_LANDMARKER_MODEL_PATH=C:\Users\aaron\Project\Hair_Customized_Ai\ai\models\face_landmarker.task
HAIR_SEGMENTER_MODEL_PATH=C:\Users\aaron\Project\Hair_Customized_Ai\ai\models\hair_segmenter.tflite
```

AWS 키는 코드, frontend `.env`, Git에 넣지 마세요.
