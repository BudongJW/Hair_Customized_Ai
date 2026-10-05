"""Run the skull-aligned hair transfer on raw photo bytes (used by AI_FITTING_MODE=skull and the demo page)."""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache

from PIL import Image

from .config import Settings
from .face_landmarker import analyze_with_mediapipe
from .generative_refine import DiffusionInpainter, generative_refine_available, refine_transfer
from .hair_alignment import HairTransferResult, transfer_hair
from .hair_removal import Inpainter, default_inpainter
from .image_processing import PORTRAIT_SIZE, _open_image, _portrait_cover

logger = logging.getLogger(__name__)

# The hair model photo only lends its hair, so it keeps its own framing (a tall or wide photo
# cropped to the portrait frame loses the top or sides of the hair); only its size is capped.
REFERENCE_MAX_SIDE = 1600
# Hair is warped as if both photos were frontal; past this head turn the result is skewed.
TURNED_FACE_DEGREES = 15.0


class FaceNotFoundError(ValueError):
    pass


class GenerativeRefineUnavailableError(RuntimeError):
    pass


def run_skull_transfer(
    face_bytes: bytes,
    reference_bytes: bytes,
    settings: Settings,
    *,
    refine: str | None = None,
    require_refine: bool = False,
) -> HairTransferResult:
    """Crop the user photo to the worker's portrait frame and put the reference hairstyle on it.

    ``refine`` overrides ``settings.generative_refine`` ("off", "fast" or "quality"). Without the
    generative packages the unrefined result is returned with a warning, or, with
    ``require_refine``, GenerativeRefineUnavailableError is raised.
    """

    user = _portrait_cover(_open_image(face_bytes), PORTRAIT_SIZE)
    hair_model = _limit_size(_open_image(reference_bytes), REFERENCE_MAX_SIDE)
    user_face = analyze_with_mediapipe(user, settings.face_landmarker_model_path)
    if user_face is None:
        raise FaceNotFoundError("내 얼굴 사진에서 얼굴을 찾지 못했습니다. 정면 사진을 사용해 주세요.")
    model_face = analyze_with_mediapipe(hair_model, settings.face_landmarker_model_path)
    if model_face is None:
        raise FaceNotFoundError("헤어모델 사진에서 얼굴을 찾지 못했습니다. 얼굴이 보이는 사진을 사용해 주세요.")

    result = transfer_hair(
        user,
        user_face.landmarks,
        hair_model,
        model_face.landmarks,
        hair_segmenter_model_path=settings.hair_segmenter_model_path,
        multiclass_model_path=settings.selfie_multiclass_model_path,
        inpainter=_inpainter(settings.lama_model_path),
        headroom=True,
    )
    warnings = list(result.metadata.get("warnings", []))
    if abs(model_face.yaw_degrees) > TURNED_FACE_DEGREES:
        warnings.append("REFERENCE_FACE_TURNED")
    if abs(user_face.yaw_degrees) > TURNED_FACE_DEGREES:
        warnings.append("USER_FACE_TURNED")
    result = replace(result, metadata={**result.metadata, "warnings": warnings})

    preset = settings.generative_refine if refine is None else refine
    return apply_generative_refine(result, preset, settings, require=require_refine)


def apply_generative_refine(
    result: HairTransferResult, preset: str, settings: Settings, *, require: bool = False
) -> HairTransferResult:
    if preset == "off":
        return result
    if not generative_refine_available():
        if require:
            raise GenerativeRefineUnavailableError(
                "생성형 보정에 필요한 패키지가 없습니다. ai/requirements-generative.txt를 설치해 주세요."
            )
        logger.warning("AI_GENERATIVE_REFINE=%s but torch/diffusers are not installed; skipping", preset)
        warnings = [*result.metadata.get("warnings", []), "GENERATIVE_REFINE_UNAVAILABLE"]
        return replace(result, metadata={**result.metadata, "warnings": warnings})
    refiner = _refiner(preset, settings.generative_model, settings.generative_device)
    return refine_transfer(result, refiner, preset=preset)


def _limit_size(image: Image.Image, max_side: int) -> Image.Image:
    scale = max_side / max(image.size)
    if scale >= 1.0:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


@lru_cache(maxsize=2)
def _inpainter(lama_model_path: str) -> Inpainter:
    # Loading the LaMa session takes about a second; reuse it across requests.
    return default_inpainter(lama_model_path)


@lru_cache(maxsize=1)
def _refiner(preset: str, model: str, device: str) -> DiffusionInpainter:
    # The diffusion model takes seconds to load and ~4 GB of memory; keep one.
    return DiffusionInpainter(preset=preset, model=model, device=device)
