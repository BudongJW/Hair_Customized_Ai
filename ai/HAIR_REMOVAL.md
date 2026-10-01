# 머리 제거 (bald canvas) — `app/hair_removal.py`

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

## 한계와 다음 단계

- 머리카락에 가려졌던 **귀·목·어깨**는 실제로 복원할 수 없음(LaMa가 추정). 긴 머리 → 짧은 머리 합성에서 티가 남.
- 정면 사진 기준. 고개를 많이 돌린 사진은 두개골 추정 오차가 커짐.
- 두피는 절차적으로 그린 것이라 사진 수준의 사실감은 아님. 더 자연스럽게 하려면 GPU 생성 모델이 필요:
  - Stable Diffusion inpainting(“bald head” 프롬프트)이나 Stable-Hair의 bald converter에
    이 모듈의 `skull_mask` / `removal_mask`를 inpainting 마스크로 넘기는 방식이 가장 현실적.
  - 또는 HairFastGAN처럼 bald 단계 없이 latent 공간에서 헤어를 바꾸는 방식.
