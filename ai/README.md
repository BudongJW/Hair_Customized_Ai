# Hair Customized AI Worker

## Texture-preserving fitting (current default)

See [INPAINTING.md](INPAINTING.md) for the `texture` workflow, saved masks,
thin-plate-spline alignment, pixel-preservation guarantees and the optional remote
GPU refinement boundary. Use `AI_FITTING_MODE=prepare` to inspect inputs without a
result, or `legacy` only to compare the older heuristic compositor.

## 바로 써보기 (데모 페이지)

AI worker를 실행한 뒤(`scripts/run-ai.ps1`) 브라우저에서 http://localhost:8000/demo 를 열면,
내 얼굴 사진과 헤어모델 사진을 올려 바로 합성 결과를 볼 수 있습니다. S3·백엔드 없이 동작하고
사진은 저장되지 않습니다. 같은 기능을 `POST /api/v1/demo/hair-transfer`(JSON, base64 이미지)로도 호출할 수 있습니다.

앱의 실제 피팅 작업에서 새 파이프라인을 쓰려면 `ai/.env`에 `AI_FITTING_MODE=skull`을 넣습니다.
결과, 머리 제거 결과(`bald-canvas.png`), 마스크들이 기존 결과 화면이 읽는 키 그대로 S3에 저장됩니다.
`app/hair_transfer.py`가 없는 체크아웃에서는 `texture`/`prepare` 모드가 명확한 오류로 실패하고,
`skull`/`legacy` 모드는 동작합니다.

## Hair removal and hair alignment

See [HAIR_REMOVAL.md](HAIR_REMOVAL.md): skull-aware hair removal that fills hair outside the
skull with background and hair inside it with shaded scalp skin, instead of painting the whole
hair mask with skin colour ("mushroom head", leftover bangs). `app/hair_alignment.py` then warps
the hair model's hair so its skull lands on the user's skull. `AI_FITTING_MODE=skull` and the
demo page use them directly; the texture mode (`hair_transfer.py`) does not use them yet. Try them
with `python tools/hair_removal_demo.py` and `python tools/hair_transfer_demo.py`.

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
