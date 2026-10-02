# 머리 제거와 새 머리 맞추기 — `app/hair_removal.py`, `app/hair_alignment.py`

헤어 합성 전에 사용자 사진에서 기존 머리카락을 지우는 단계입니다. 결과 화면의
"머리 제거 결과"(`bald-canvas.png`)에 해당합니다.

## 기존 결과가 이상했던 이유

| 증상 (결과 화면) | 원인 |
| --- | --- |
| 피부색 "버섯 머리" — 지운 자리가 원래 헤어스타일 실루엣 그대로 피부색 | 머리카락 마스크 **전체**를 피부색으로 칠함. 머리숱(볼륨)은 두개골보다 훨씬 크기 때문에, 두개골 바깥에 있던 머리카락은 피부가 아니라 **배경**으로 채워야 함 |
| 단색 스티커처럼 보이고 경계가 그대로 보임 | 음영·질감·경계 보정 없이 단색으로 채움 |
| 눈썹 위치에 검은 막대가 남음 | 눈썹을 랜드마크로 "보호"함. 앞머리가 눈썹을 덮고 있으면 보호 영역 안의 앞머리가 그대로 남음 |
| 앞머리 끝/잔머리가 남음 | 세그멘테이션 임계값 경계의 얇은 머리카락이 마스크에 안 잡힘 |

추가로 확인할 것: MediaPipe `ImageSegmenter` 결과의 `numpy_view()`는 segmenter 메모리를
가리킵니다. `with ... as segmenter:` 블록을 빠져나온 뒤에 읽으면 마스크가 깨질 수 있습니다
(실제로 큰 사각형 모양의 가짜 "머리카락"이 생기는 것을 확인함). 블록 안에서 `np.array(..., copy=True)`로
복사해야 합니다. 저장소에 없는 `hair_segmenter.py`도 같은 패턴인지 확인해 주세요.

## 새 방식

1. **세그멘테이션**: MediaPipe selfie multiclass(머리카락/얼굴 피부/몸 피부 구분) + 기존 hair segmenter를 합침.
2. **두개골 추정**: 얼굴 랜드마크(얼굴 윤곽, 10번 이마, 152번 턱)로 대머리 윤곽을 추정.
   위쪽 반타원 + 얼굴 윤곽의 convex hull. 짧은 머리면 머리카락 윗부분 높이로 정수리를 제한.
3. **지울 영역**: 머리에 붙어 있는 머리카락만(떨어진 오검출 제거) → 잔머리까지 약간 팽창 →
   이마 위 앞머리 사이 틈/구멍 메우기 → 경계의 어두운 머리카락 끝 추가.
   눈·코·입·수염(코 아래 얼굴 안쪽)은 보호. **눈썹은 보호하지 않음.**
4. **두 영역으로 분리해서 채우기**
   - 두개골 **밖** 머리카락 → 배경. 머리 전체를 가린 채 inpainting해서 "머리 뒤 배경"만 복원
     (LaMa ONNX가 있으면 LaMa, 없으면 push-pull 보간. 단색 증명사진 배경은 push-pull로도 충분).
   - 두개골 **안** 머리카락 → 두피. 보이는 얼굴 피부색·좌우 조명으로 구 형태 음영을 그리고,
     실제 피부와 맞닿는 경계에서 색 차이를 부드럽게 보정(push-pull), 피부 노이즈 추가.
5. **가려졌던 눈썹 다시 그리기**: 눈썹 랜드마크 영역의 35% 이상을 지웠다면 눈썹을 다시 그림.
6. 이마 쪽 경계는 넓게 페더링해서 원래 헤어라인이 선으로 남지 않게 합성.

CPU 4코어 기준 900×1125 한 장에 약 5초(LaMa 포함), LaMa 없이 약 3초.

## 사용법

```python
from app.config import get_settings
from app.hair_removal import default_inpainter, remove_hair, segment_head

settings = get_settings()
# portrait: 랜드마크를 뽑은 것과 같은 크기/크롭의 PIL 이미지
# landmarks: MediaPipe 478개 랜드마크 ([{"x":..,"y":..}, ...], 0~1 정규화)
segmentation = segment_head(
    portrait,
    hair_segmenter_model_path=settings.hair_segmenter_model_path,
    multiclass_model_path=settings.selfie_multiclass_model_path,
)
removal = remove_hair(
    portrait,
    landmarks,
    segmentation=segmentation,
    inpainter=default_inpainter(settings.lama_model_path),
)
removal.image          # bald canvas (RGB)
removal.removal_mask   # 실제로 다시 칠한 영역
removal.artifacts()    # {"bald-canvas.png": ("image/png", bytes), ...}
```

`hair_transfer.py`(texture 모드)에 연결할 때:
- `bald-canvas.png`를 만드는 부분을 `remove_hair(...).image`로 교체.
- 합성 수정 영역(edit mask)은 `removal.removal_mask` ∪ (새 헤어 마스크)로 잡으면 됨.
  새 헤어가 덮는 부분은 bald canvas 품질과 무관하고, 새 헤어가 안 덮는 부분만 bald canvas가 보임.
- 랜드마크는 반드시 `remove_hair`에 넣는 이미지와 같은 좌표계(같은 리사이즈/크롭)여야 함.

모델 파일은 `scripts/run-ai.ps1`가 자동으로 받습니다
(`selfie_multiclass_256x256.tflite`, `lama_fp32.onnx` ~170MB, `AI_SKIP_LAMA=1`이면 LaMa 생략).

## 데모 / 테스트

```bash
cd ai
python tools/hair_removal_demo.py my_photo.jpg --out out/     # 원본 | 기존 방식 | 새 방식 | 마스크
python -m unittest discover tests                            # 모델 없이 도는 테스트
```

데모 패널의 마스크: 빨강 = 배경으로 채운 영역, 초록 = 두피로 채운 영역, 파랑 = 보호 영역, 노란 선 = 추정 두개골.

## 2단계: 새 머리 맞추기 — `app/hair_alignment.py`

머리를 지운 뒤 헤어모델의 머리를 올리는 단계입니다. 레거시 합성기는 잘라낸 머리를 얼굴 폭 비율로만
키워서 이마에 붙이기 때문에, 새 머리가 작은 모자처럼 떠서 두피가 드러나곤 했습니다.

1. **헤어모델 머리 추출**: 1단계와 같은 세그멘테이션으로 부드러운 머리 마스크(alpha)를 만들고,
   머리 가장자리에 섞인 배경색을 걷어냄(밝은 배경의 흰 테두리 방지). 헤어라인 쪽 경계는 살짝 흐리게.
2. **두개골 기준 정렬**: 헤어모델과 사용자 각각의 두개골 곡선 + 이마·관자놀이 윤곽 + 눈꼬리 + 턱
   (31개 대응점)을 thin-plate spline으로 맞춤. 헤어모델의 두피 경계가 사용자의 두피 경계에 정확히 놓이고,
   바깥 머리 볼륨은 그 비율대로 따라감. 변형이 뒤집히면 affine으로 대체.
3. **헤어모델 사진에서 빌려오기**: 같은 변형으로 사용자 사진에 없던 것을 가져옴.
   - **귀**: 사용자 귀가 옛 머리에 가려져 있었다면, 헤어모델의 보이는 귀를 사용자 피부색으로 바꿔서 붙임
     (짧은 머리로 바꿨을 때 "귀 없는 머리"가 되는 문제).
   - **이마 피부 결**: 새 헤어라인 아래 그려낸 이마에 헤어모델 이마의 미세한 피부 결(모공·결)만 옮김.
     헤어모델의 음영까지 옮기면 관자놀이 회색 얼룩, 헤어라인 흰 테두리가 생겨서 결만 사용.
4. **합성**: bald canvas 위에 올리고, 머리 아래 약한 그림자, 노출 차이 보정(색상은 유지),
   단발처럼 머리와 볼 사이에 생기는 좁은 틈은 어두운 뒷머리 색으로 채움.

```python
from app.hair_alignment import transfer_hair

result = transfer_hair(user_portrait, user_landmarks, model_portrait, model_landmarks,
                       hair_segmenter_model_path=..., multiclass_model_path=...,
                       inpainter=default_inpainter(settings.lama_model_path))
result.image                 # 최종 합성
result.metadata["warnings"]  # 예: REFERENCE_HAIR_CROPPED_TOP (헤어모델 머리가 사진 밖으로 잘림)
result.artifacts()           # result.png, warped-hair-layer.png, bald-canvas.png, ...
```

데모: `python tools/hair_transfer_demo.py 내사진.jpg 헤어모델.jpg --out out/`, 또는 worker 실행 후
http://localhost:8000/demo 에서 사진을 올려 확인. 앱에서는 `AI_FITTING_MODE=skull`로 사용.

헤어모델 사진에서 머리가 위/옆으로 잘려 있으면 그 부분은 복원할 수 없어 일직선으로 잘린 모양이 됩니다.
`warnings`로 감지하니, 앱에서 "머리 전체가 나온 사진"을 다시 요청하는 데 쓰면 됩니다.

## 유사 서비스 비교 (HairFastGAN)

같은 7쌍(합성 인물 사진)을 HairFastGAN(NeurIPS 2024, 공개 코드·가중치)으로도 돌려 비교함.
공식 HF Space는 추론 서버가 내려가 있어, 공개 코드와 가중치를 받아 CPU로 실행함.

| | 이 파이프라인 | HairFastGAN |
| --- | --- | --- |
| 얼굴 | 원본 픽셀 그대로 | StyleGAN으로 다시 그려 주름·이목구비·표정이 바뀜 |
| 출력 | 원본 구도·해상도 | 1024×1024 정렬된 얼굴 크롭만 |
| 헤어스타일 모양 | 헤어모델 모양 그대로 | 곱슬 숏컷이 단발처럼 흐트러지는 등 모양이 바뀜 |
| 이마·헤어라인 | 그려낸 이마가 회색으로 뜨고 헤어라인이 오려 붙인 듯함 → 아래에서 개선 | 자연스럽게 이어짐 |
| 속도·크기 (CPU) | 쌍당 약 10초, 모델 약 200MB | 쌍당 약 130초, 가중치 약 6GB(GPU 권장) |

비교 후 반영한 개선:
- 그려낸 이마가 실제 얼굴보다 어둡고 회색빛이던 원인은, 옛 머리 그림자 속의 어두운 경계 색이
  두피 안쪽까지 퍼진 것. 안쪽으로 가는 경계 보정을 제한하고(`SEAM_FAR_WEIGHT`, `SEAM_FAR_LIMIT`),
  경계에서 떨어진 안쪽은 실제 얼굴 피부(눈 아래~코끝) 색에 맞춤(`_anchor_scalp_tone`).
- 새 헤어라인이 이마 위에서 끝나는 곳은 머리 투명도를 몇 mm에 걸쳐 옅게 함(`_fade_hairline`).
  접촉 그림자는 흐린 투명도끼리의 차이로 계산해 헤어라인에 검은 테두리가 생기지 않게 함.
- 피부 결을 옮기는 영역 경계를 흐리게 하고 결의 세기를 낮춤.

## 남은 한계 개선 (어깨선, 이마 하이라이트, 색온도)

- **어깨 위 번짐**: 긴 머리가 덮던 어깨 위를 LaMa가 배경·옷·머리 색이 섞인 안개로 채우던 문제.
  배경이 단색에 가까우면(배경 밝기 MAD ≤ `PLAIN_BACKGROUND_MAX_SPREAD`) 보이는 어깨선을 가려진 구간까지
  이어서, 그 위는 매끈한 배경으로 다시 채우고 아래(옷)는 LaMa 결과를 유지함. 얼굴 옆 두개골 바깥은
  정면 사진에서 항상 배경이므로 함께 채움. 무늬 있는 배경은 기존대로 LaMa에 맡김.
- **평평한 이마**: 사용자 얼굴(콧등·광대)이 얼마나 번들거리는지 재서 같은 수준의 하이라이트를 이마
  가운데에 넣음. 무광 피부면 0.
- **머리 색온도**: 두 사진의 배경이 무채색(회색·흰색)이면 배경 색 차이로 조명 색을 추정해 새 머리에
  일부(70%, 최대 ±12%) 반영. 색 있는 벽이나 검은 배경은 조명 정보가 없으므로 그대로 둠.

## 실제 사진으로 확인하며 고친 점 (증명사진 + 헤어모델 3장)

- **헤어모델 사진이 잘리던 문제**: worker가 헤어모델 사진도 사용자 사진처럼 4:5로 가운데를 잘라서, 세로로 긴
  사진은 머리 윗부분이, 가로로 긴 사진은 옆머리가 잘려 나갔음. 헤어모델 사진은 머리만 빌려 오므로 이제 자르지
  않고 크기만 제한함(긴 변 1600px).
- **증명사진에서 새 머리가 위로 잘리던 문제**: 증명사진은 머리 위 여백이 거의 없어서, 원래보다 풍성한 머리를
  올리면 사진 위 끝에서 잘림. 새 머리를 위로 늘린 캔버스에 미리 변형해 보고 모자라는 만큼 사진을 아래로 내린 뒤
  (아래쪽 가슴 부분이 그만큼 잘림) 위에 생긴 띠는 바로 아래 배경으로 이어 채움(`transfer_hair(..., headroom=True)`,
  skull 모드와 데모는 켜져 있음). 메타데이터 `headroomRows`에 내린 줄 수가 기록됨.
- **머리 위 끝의 검은 띠**: 단발과 볼 사이 같은 좁은 틈을 뒷머리 색으로 칠하는 단계가, 사진 위 끝과 새 머리
  사이도 "틈"으로 보고 칠하던 버그. 정수리 위쪽은 칠하지 않도록 고침.
- **얼굴 옆에 떠 있는 머리 조각**: 고개를 돌린 헤어모델 사진에서는 반대편 귀 뒤 머리가 변형 후 턱 옆에 따로
  떨어진 조각으로 남았음. 두상을 덮는 머리와 이어지지 않고 멀리 떨어진 조각은 지움(`strayHairPixelCount`).
- **고개를 돌린 헤어모델**: 정면 기준으로 맞추므로 머리가 한쪽으로 치우침. 헤어모델(또는 사용자) 얼굴이 15°
  넘게 돌아가 있으면 `REFERENCE_FACE_TURNED`(`USER_FACE_TURNED`) 경고를 남기고 데모 페이지에 안내함.

## 생성형 보정 (선택) — `app/generative_refine.py`

절차적으로 그린 부분(지운 머리 밑의 이마·관자놀이, 긴 머리가 덮던 귀·목·어깨, 새 헤어라인과 이마가 만나는 띠)만
Stable Diffusion 1.5 inpainting으로 다시 그립니다. 처음부터 새로 그리지 않고 지금 결과에서 출발하므로
(img2img 강도 0.75~0.85) 두상·헤어라인 모양과 피부 톤은 파이프라인이 정한 대로 두고, 사진다운 음영·질감·
머리카락이 피부로 잦아드는 경계만 생성 모델이 채웁니다. 눈·코·입과 옮겨 온 머리카락 몸통은 마스크에서 빠지고
나머지 픽셀은 보정 전 결과에서 그대로 붙여 넣기 때문에, HairFastGAN처럼 얼굴이 바뀌지 않습니다.
머리 위쪽 배경은 LaMa가 이미 잘 채우고, 생성 모델이 그곳에 머리카락을 지어내는 경우가 있어 제외합니다.

- 모델: `Lykon/absolute-reality-1.6525-inpainting`(SD 1.5 사실적 인물 파인튜닝, CreativeML OpenRAIL-M).
  `AI_GENERATIVE_MODEL`로 바꿀 수 있고, 기본 `stable-diffusion-v1-5/stable-diffusion-inpainting`도 동작하지만
  피부가 더 밋밋하게 나옴. 서버에서 실행되므로 사진이 외부로 나가지 않음.
- 프리셋(`AI_GENERATIVE_REFINE`):

  | 값 | 방식 | 4코어 CPU (측정) |
  | --- | --- | --- |
  | `off` (기본) | 보정 없음 | — |
  | `fast` | LCM-LoRA 8스텝, 강도 0.85 | 약 50초 |
  | `quality` | DPM-Solver++ 20스텝, CFG 5, 강도 0.75 | 약 2.5분 |

  GPU에서는 수 초 수준으로 예상(이 환경에 GPU가 없어 측정하지 못함). `quality`가 피부 결과 헤어라인이 더
  선명하지만 잔머리를 지어내는 경우가 조금 더 많음.

- 설치: `ai/.venv/Scripts/python -m pip install -r requirements-generative.txt` (NVIDIA GPU면 torch를
  pytorch.org 안내대로 CUDA 빌드로 먼저 설치). 첫 실행 때 모델(약 2GB)을 Hugging Face에서 받음.
  `AI_GENERATIVE_DEVICE`는 기본 `auto`(cuda → mps → cpu).
- 패키지가 없으면 skull 모드 작업은 보정 없이 완료되고 `GENERATIVE_REFINE_UNAVAILABLE` 경고가 남음.
  데모 페이지에서 보정을 고르면 503으로 설치 안내를 돌려줌.
- 산출물: `result.png`가 보정본이 되고, 보정 전 결과는 `unrefined-result.png`로 함께 저장됨.
  `skull-transfer.json`의 `generativeRefine`에 프리셋, 다시 그린 픽셀 수, 걸린 시간이 기록됨.
- 데모 페이지(`/demo`)의 "생성형 보정" 선택(끄기/빠르게/고품질)으로 바로 비교 가능.

## 한계와 다음 단계

- 머리카락에 가려졌던 **귀**는 헤어모델의 귀를 빌려 쓰므로 사용자 본인의 귀와는 다름.
- 단색이 아닌 배경(야외, 무늬 벽)에서 긴 머리를 지우면 어깨 위 복원은 여전히 LaMa 추정이라 뭉개질 수 있음.
- 정면 사진 기준. 고개를 많이 돌린 사진은 두개골 추정 오차가 커짐.
- 생성형 보정을 끄면 두피·이마는 절차적으로 그린 것이라 주름·굴곡 같은 세부가 없음(LaMa로 이마를 다시
  채우는 방법도 시험했지만 차이가 거의 없었음). 두피를 먼저 채우고 배경만 LaMa에 맡기는 방식도 시험했지만,
  LaMa가 두개골 둘레에 머리카락 테두리를 새로 그려서 더 나빴음.
- 생성형 보정은 CPU에서 느리고(약 1~3분), 시드에 따라 이마 음영이 조금씩 달라짐. 가끔 배경 쪽에 잔머리
  한두 가닥을 그리기도 함.
- Gemini 무료 키로는 이미지 생성·편집이 불가(이미지 출력 모델은 모두 유료 전용). 아래 생성 모델 방식에
  Gemini를 쓰려면 유료 등급이 필요하고, 사용자 얼굴 사진이 외부로 전송되는 점도 고려해야 함.
- 머리카락 몸통 자체(질감·볼륨)는 헤어모델 사진을 그대로 옮기므로, 헤어모델 사진이 흐리거나 잘려 있으면
  결과도 그대로 따라감. 이 부분까지 생성으로 바꾸는 방식(HairFastGAN, Stable-Hair)은 얼굴이 바뀌는 단점이 있음.
