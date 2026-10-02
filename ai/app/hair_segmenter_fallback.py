"""Stand-in for ``app/hair_segmenter.py``.

``image_processing.py`` imports ``app.hair_segmenter``, but that module is not committed to
the repository, so the worker could not start from a clean checkout. This provides the
same two names on top of the shared MediaPipe helper. Delete it once the real module is
in the repository (``image_processing.py`` prefers the real one when it exists).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .hair_removal import _resize_probability, _run_mediapipe_segmenter

DEFAULT_CONFIDENCE_THRESHOLD = 0.35


@dataclass(frozen=True)
class HairSegmentationResult:
    mask: Image.Image
    source: str
    confidence_threshold: float


def segment_hair_with_mediapipe(
    image: Image.Image,
    model_path: str | None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> HairSegmentationResult | None:
    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
    confidences = _run_mediapipe_segmenter(rgb, model_path)
    if confidences is None:
        return None
    probability = _resize_probability(confidences[-1], rgb.shape[1], rgb.shape[0])
    mask = np.where(probability >= confidence_threshold, probability, 0.0)
    return HairSegmentationResult(
        mask=Image.fromarray((mask * 255).astype(np.uint8), "L"),
        source="mediapipe-hair-segmenter-fallback",
        confidence_threshold=confidence_threshold,
    )
