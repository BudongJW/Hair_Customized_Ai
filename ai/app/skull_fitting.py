"""Run the skull-aligned hair transfer on raw photo bytes (used by AI_FITTING_MODE=skull and the demo page)."""

from __future__ import annotations

from functools import lru_cache

from .config import Settings
from .face_landmarker import analyze_with_mediapipe
from .hair_alignment import HairTransferResult, transfer_hair
from .hair_removal import Inpainter, default_inpainter
from .image_processing import PORTRAIT_SIZE, _open_image, _portrait_cover


class FaceNotFoundError(ValueError):
    pass


def run_skull_transfer(face_bytes: bytes, reference_bytes: bytes, settings: Settings) -> HairTransferResult:
    """Crop both photos to the worker's portrait frame and put the reference hairstyle on the user."""

    user = _portrait_cover(_open_image(face_bytes), PORTRAIT_SIZE)
    hair_model = _portrait_cover(_open_image(reference_bytes), PORTRAIT_SIZE)
    user_face = analyze_with_mediapipe(user, settings.face_landmarker_model_path)
    if user_face is None:
        raise FaceNotFoundError("내 얼굴 사진에서 얼굴을 찾지 못했습니다. 정면 사진을 사용해 주세요.")
    model_face = analyze_with_mediapipe(hair_model, settings.face_landmarker_model_path)
    if model_face is None:
        raise FaceNotFoundError("헤어모델 사진에서 얼굴을 찾지 못했습니다. 얼굴이 보이는 사진을 사용해 주세요.")

    return transfer_hair(
        user,
        user_face.landmarks,
        hair_model,
        model_face.landmarks,
        hair_segmenter_model_path=settings.hair_segmenter_model_path,
        multiclass_model_path=settings.selfie_multiclass_model_path,
        inpainter=_inpainter(settings.lama_model_path),
    )


@lru_cache(maxsize=2)
def _inpainter(lama_model_path: str) -> Inpainter:
    # Loading the LaMa session takes about a second; reuse it across requests.
    return default_inpainter(lama_model_path)
