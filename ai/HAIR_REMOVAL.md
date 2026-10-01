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

데모: `python tools/hair_transfer_demo.py 내사진.jpg 헤어모델.jpg --out out/`

헤어모델 사진에서 머리가 위/옆으로 잘려 있으면 그 부분은 복원할 수 없어 일직선으로 잘린 모양이 됩니다.
`warnings`로 감지하니, 앱에서 "머리 전체가 나온 사진"을 다시 요청하는 데 쓰면 됩니다.

## 한계와 다음 단계

- 머리카락에 가려졌던 **귀·목·어깨**는 실제로 복원할 수 없음(LaMa가 추정). 긴 머리 → 짧은 머리 합성에서 티가 남.
- 정면 사진 기준. 고개를 많이 돌린 사진은 두개골 추정 오차가 커짐.
- 두피는 절차적으로 그린 것이라 사진 수준의 사실감은 아님. 새 헤어라인 아래로 보이는 이마가 얼굴보다
  약간 밝고 균일한 색으로 보임(LaMa로 이마를 다시 채우는 방법도 시험했지만 차이가 거의 없었음).
- 긴 머리를 짧은 머리로 바꾸면 목·어깨 주변의 복원이 뭉개짐. 두피를 먼저 채우고 배경만 LaMa에
  맡기는 방식도 시험했지만, LaMa가 두개골 둘레에 머리카락 테두리를 새로 그려서 더 나빴음.
- 빌려온 귀는 헤어모델의 귀 모양이므로 사용자 본인의 귀와는 다름.
- Gemini 무료 키로는 이미지 생성·편집이 불가(이미지 출력 모델은 모두 유료 전용). 아래 생성 모델 방식에
  Gemini를 쓰려면 유료 등급이 필요하고, 사용자 얼굴 사진이 외부로 전송되는 점도 고려해야 함.
- 더 자연스럽게 하려면 생성 모델이 필요:
  - Stable Diffusion inpainting(“bald head” 프롬프트)이나 Stable-Hair의 bald converter에
    이 모듈의 `skull_mask` / `removal_mask`를 inpainting 마스크로 넘기는 방식이 가장 현실적.
  - 또는 HairFastGAN처럼 bald 단계 없이 latent 공간에서 헤어를 바꾸는 방식.
