"""Landmark-guided hair removal that produces the "bald canvas" for hair fitting.

Why this exists
---------------
Filling the *hair mask* with a flat skin colour produces a skin-coloured copy of the
old hairstyle's silhouette (the "mushroom head"), because hair volume extends well
beyond the skull. Protecting eyebrows by landmark also keeps any bangs lying on top
of them, which shows up as black bars over the eyes.

This module splits the removed hair into two very different problems:

* hair **outside** the estimated skull  -> becomes background (inpainted behind the head)
* hair **inside** the estimated skull   -> becomes scalp/forehead skin (shaded dome that is
  colour-matched to the visible face along the seam)

Eyebrows that were hidden under bangs are redrawn from the landmark polygon instead
of being "protected" (protecting them keeps the bangs that cover them).

Only numpy/Pillow/MediaPipe are required. If ``onnxruntime`` and a LaMa ONNX model are
available, the background is inpainted with LaMa; otherwise a smooth push-pull fill is
used, which is good enough for plain studio/ID-photo backgrounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .config import DEFAULT_LAMA_MODEL_PATH, DEFAULT_SELFIE_MULTICLASS_MODEL_PATH

FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
# Upper contour (outer -> inner) followed by lower contour (inner -> outer).
LEFT_BROW_INDICES = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW_INDICES = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSE_INDICES = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 64, 294, 48, 278, 129, 358, 49, 279]
FOREHEAD_TOP_INDEX = 10
CHIN_INDEX = 152
NOSE_TIP_INDEX = 1
JAW_LEFT_INDEX = 172
JAW_RIGHT_INDEX = 397
# Neck half-width relative to half the distance between the jaw angles.
NECK_WIDTH_SCALE = 0.80
# Robust luminance spread (MAD, 0-255) under which the background counts as plain enough to
# be rebuilt as a smooth field instead of being left to the inpainter.
PLAIN_BACKGROUND_MAX_SPREAD = 10.0
# Forehead highlight = scale * (how much brighter the face's 97th percentile is than its
# median), capped. Matte skin gets none, shiny skin a visible sheen.
FOREHEAD_HIGHLIGHT_SCALE = 0.45
FOREHEAD_HIGHLIGHT_MAX = 0.08
# How much of the seam colour offset reaches the middle of the synthetic scalp.
SEAM_FAR_WEIGHT = 0.35
SEAM_FAR_LIMIT = 15.0

# Skull model, expressed in face units (see _estimate_skull):
#   width  = face-oval width * SKULL_WIDTH_SCALE
#   vertex = landmark 10 minus face height (landmark 10 -> chin) * SKULL_TOP_SCALE
#   widest = landmark 10 plus face height * SKULL_CENTER_OFFSET
SKULL_WIDTH_SCALE = 1.04
SKULL_TOP_SCALE = 0.36
SKULL_CENTER_OFFSET = 0.12

# Selfie multiclass categories: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 others.
MULTICLASS_BACKGROUND = 0
MULTICLASS_HAIR = 1
MULTICLASS_BODY_SKIN = 2
MULTICLASS_FACE_SKIN = 3
MULTICLASS_CLOTHES = 4

LAMA_SIZE = 512
METADATA_VERSION = "hair-removal-v1-skull-split"

Inpainter = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class HeadSegmentation:
    """Per-pixel probabilities in [0, 1] at image resolution."""

    hair: np.ndarray
    face_skin: np.ndarray | None = None
    body_skin: np.ndarray | None = None
    source: str = "provided"
    background: np.ndarray | None = None
    clothes: np.ndarray | None = None


@dataclass(frozen=True)
class HairRemovalResult:
    image: Image.Image
    hair_mask: Image.Image
    removal_mask: Image.Image
    skull_mask: Image.Image
    scalp_fill_mask: Image.Image
    background_fill_mask: Image.Image
    protection_mask: Image.Image
    metadata: dict = field(default_factory=dict)

    def artifacts(self) -> dict[str, tuple[str, bytes]]:
        return {
            "bald-canvas.png": ("image/png", _encode_png(self.image)),
            "hair-removal-mask.png": ("image/png", _encode_png(self.removal_mask)),
            "skull-mask.png": ("image/png", _encode_png(self.skull_mask)),
            "scalp-fill-mask.png": ("image/png", _encode_png(self.scalp_fill_mask)),
            "background-fill-mask.png": ("image/png", _encode_png(self.background_fill_mask)),
        }


@dataclass(frozen=True)
class _FaceFrame:
    """Face-aligned coordinates: u runs along the eye line, v runs down the face."""

    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray

    def to_local(self, points: np.ndarray) -> np.ndarray:
        delta = points - self.origin
        return np.stack([delta @ self.axis_u, delta @ self.axis_v], axis=-1)

    def to_image(self, local: np.ndarray) -> np.ndarray:
        return self.origin + local[..., :1] * self.axis_u + local[..., 1:2] * self.axis_v


@dataclass(frozen=True)
class _Skull:
    center_u: float
    center_v: float
    radius_u: float
    radius_v: float
    top_v: float
    face_width: float
    face_height: float
    vertex_ratio: float
    vertex_clamped: bool
    polygon: np.ndarray  # image coordinates


def remove_hair(
    image: Image.Image,
    landmarks: Sequence[dict] | Sequence[Sequence[float]] | np.ndarray,
    *,
    segmentation: HeadSegmentation | None = None,
    hair_segmenter_model_path: str | None = None,
    multiclass_model_path: str | None = None,
    inpainter: Inpainter | None = None,
    redraw_hidden_eyebrows: bool = True,
) -> HairRemovalResult:
    """Return the portrait with its existing hair removed.

    ``landmarks`` are the 478 MediaPipe face landmarks, either as the ``[{"x", "y"}, ...]``
    dicts stored in ``landmarksJson`` (normalised to ``image``) or ``(N, 2+)`` pairs/array of
    normalised coordinates.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    points = _landmark_pixels(landmarks, width, height)
    frame = _face_frame(points)

    if segmentation is None:
        segmentation = segment_head(
            image,
            hair_segmenter_model_path=hair_segmenter_model_path,
            multiclass_model_path=multiclass_model_path,
        )
    if inpainter is None:
        inpainter = default_inpainter()

    local_grid = _local_grid(frame, width, height)
    skull = _estimate_skull(frame, points, segmentation)
    unit = skull.face_width

    hair = np.clip(segmentation.hair.astype(np.float32), 0.0, 1.0)
    protection = _protection_mask(points, (width, height), unit)
    beard_zone = _beard_zone(frame, points, local_grid, (width, height))
    skull_mask = _rasterize(skull.polygon, (width, height))

    # Only hair attached to the head counts: drops stray false positives and keeps
    # visible eyebrows that the segmenter sometimes labels as hair.
    head_seed = (hair > 0.5) & _dilate(skull_mask, radius=0.08 * unit) & ~beard_zone
    hair_region = _connected_to((hair > 0.15) & ~beard_zone, head_seed)

    # Grow the hair mask so soft strand tips and colour halos are removed as well.
    removal = _dilate(hair_region, radius=0.025 * unit) | _dilate(hair_region & (hair > 0.5), radius=0.04 * unit)
    removal &= ~protection
    removal &= ~beard_zone
    removal = _close_forehead_gaps(removal, rgb, skull_mask, protection, frame, points, local_grid, unit)
    if not removal.any():
        # Already bald (or no hair detected): nothing to synthesise.
        return HairRemovalResult(
            image=image.convert("RGB"),
            hair_mask=_mask_image(hair),
            removal_mask=_mask_image(removal),
            skull_mask=_mask_image(skull_mask),
            scalp_fill_mask=_mask_image(removal),
            background_fill_mask=_mask_image(removal),
            protection_mask=_mask_image(protection),
            metadata={"version": METADATA_VERSION, "segmentationSource": segmentation.source, "removedPixelCount": 0},
        )

    scalp_region = removal & skull_mask
    background_region = removal & ~skull_mask

    background = _fill_background(rgb, skull_mask | removal, inpainter, unit)
    background, silhouette_pixels = _restore_body_silhouette(
        background, rgb, background_region, segmentation, points, unit
    )
    background, neck_pixels = _rebuild_neck(
        background, rgb, background_region, segmentation.body_skin, points, frame, local_grid, unit
    )

    skin_probability = segmentation.face_skin
    if skin_probability is None:
        skin_probability = _fallback_skin_probability(points, (width, height))
    dome = _render_scalp(rgb, skull, local_grid, skin_probability, removal, protection, points, frame, unit)
    dome = _harmonize_seam(dome, rgb, removal, protection, skin_probability, segmentation.body_skin, unit)
    dome, forehead_gain = _anchor_scalp_tone(
        dome, rgb, removal, scalp_region, protection, skin_probability, frame, points, local_grid, unit
    )

    brows_redrawn: list[str] = []
    if redraw_hidden_eyebrows:
        dome, brows_redrawn = _redraw_hidden_eyebrows(dome, rgb, points, removal, hair, frame, local_grid, unit)

    skull_alpha = _blur(skull_mask.astype(np.float32), 0.8)[..., None]
    fill = dome * skull_alpha + background * (1.0 - skull_alpha)
    removal_alpha = _blur(removal.astype(np.float32), 1.2)
    # Feather into the real skin so the old hairline does not survive as a hard edge.
    # Eyes and visible eyebrows keep a sharp boundary.
    brows = _rasterize(points[LEFT_BROW_INDICES], (width, height)) | _rasterize(points[RIGHT_BROW_INDICES], (width, height))
    skin_side = skull_mask & ~removal & ~_dilate(protection | brows, radius=0.02 * unit)
    feather = np.clip(_blur(removal.astype(np.float32), max(2.0, 0.02 * unit)) * 2.0, 0.0, 1.0) * 0.85
    removal_alpha = np.where(skin_side, np.maximum(removal_alpha, feather), removal_alpha)[..., None]
    result = rgb * (1.0 - removal_alpha) + fill * removal_alpha

    metadata = {
        "version": METADATA_VERSION,
        "segmentationSource": segmentation.source,
        "inpainter": getattr(inpainter, "name", type(inpainter).__name__),
        "skull": {
            "centerU": skull.center_u,
            "centerV": skull.center_v,
            "radiusU": skull.radius_u,
            "radiusV": skull.radius_v,
            "faceWidth": skull.face_width,
            "faceHeight": skull.face_height,
            "vertexRatio": skull.vertex_ratio,
            "vertexClampedByHair": skull.vertex_clamped,
        },
        "removedPixelCount": int(removal.sum()),
        "scalpFillPixelCount": int(scalp_region.sum()),
        "backgroundFillPixelCount": int(background_region.sum()),
        "rebuiltNeckPixelCount": neck_pixels,
        "rebuiltBackgroundPixelCount": silhouette_pixels,
        "scalpToneGain": forehead_gain,
        "redrawnEyebrows": brows_redrawn,
    }
    return HairRemovalResult(
        image=Image.fromarray(np.clip(result + 0.5, 0, 255).astype(np.uint8), "RGB"),
        hair_mask=_mask_image(hair),
        removal_mask=_mask_image(removal),
        skull_mask=_mask_image(skull_mask),
        scalp_fill_mask=_mask_image(scalp_region),
        background_fill_mask=_mask_image(background_region),
        protection_mask=_mask_image(protection),
        metadata=metadata,
    )


# --------------------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------------------


def segment_head(
    image: Image.Image,
    *,
    hair_segmenter_model_path: str | None = None,
    multiclass_model_path: str | None = None,
) -> HeadSegmentation:
    """Hair/skin probabilities from MediaPipe.

    The selfie multiclass model separates hair, face skin and body skin, which is what the
    seam colour-matching needs. The dedicated hair segmenter is merged in when available
    because it is often better at thin strands.
    """

    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
    height, width = rgb.shape[:2]
    hair_parts: list[np.ndarray] = []
    face_skin = body_skin = background = clothes = None
    sources: list[str] = []

    multiclass_path = multiclass_model_path or str(DEFAULT_SELFIE_MULTICLASS_MODEL_PATH)
    confidences = _run_mediapipe_segmenter(rgb, multiclass_path)
    if confidences is not None and len(confidences) > MULTICLASS_FACE_SKIN:
        hair_parts.append(confidences[MULTICLASS_HAIR])
        body_skin = confidences[MULTICLASS_BODY_SKIN]
        face_skin = confidences[MULTICLASS_FACE_SKIN]
        background = confidences[MULTICLASS_BACKGROUND]
        if len(confidences) > MULTICLASS_CLOTHES:
            clothes = confidences[MULTICLASS_CLOTHES]
        sources.append("mediapipe-selfie-multiclass")

    if hair_segmenter_model_path:
        confidences = _run_mediapipe_segmenter(rgb, hair_segmenter_model_path)
        if confidences is not None:
            hair_parts.append(confidences[-1])
            sources.append("mediapipe-hair-segmenter")

    if not hair_parts:
        raise RuntimeError(
            "No hair segmentation model is available. Set HAIR_SEGMENTER_MODEL_PATH or "
            "SELFIE_MULTICLASS_MODEL_PATH."
        )

    hair = np.maximum.reduce([_resize_probability(part, width, height) for part in hair_parts])

    def resized(probability: np.ndarray | None) -> np.ndarray | None:
        return None if probability is None else _resize_probability(probability, width, height)

    return HeadSegmentation(
        hair=hair,
        face_skin=resized(face_skin),
        body_skin=resized(body_skin),
        source="+".join(sources),
        background=resized(background),
        clothes=resized(clothes),
    )


def _run_mediapipe_segmenter(rgb: np.ndarray, model_path: str | None) -> list[np.ndarray] | None:
    if not model_path or not Path(model_path).exists():
        return None

    import mediapipe as mp

    vision = mp.tasks.vision
    options = vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        output_confidence_masks=True,
        output_category_mask=False,
    )
    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        # numpy_view() aliases memory owned by the segmenter; copy before it is closed.
        masks = [np.array(mask.numpy_view(), dtype=np.float32, copy=True).squeeze() for mask in result.confidence_masks]
    return masks or None


def _resize_probability(probability: np.ndarray, width: int, height: int) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float32).squeeze(), 0.0, 1.0)
    if probability.shape == (height, width):
        return probability
    resized = Image.fromarray((probability * 255).astype(np.uint8), "L").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------


def _landmark_pixels(landmarks: Sequence[dict] | Sequence[Sequence[float]] | np.ndarray, width: int, height: int) -> np.ndarray:
    if len(landmarks) and isinstance(landmarks[0], dict):
        normalized = np.array([[float(point["x"]), float(point["y"])] for point in landmarks], dtype=np.float32)
    else:
        normalized = np.asarray(landmarks, dtype=np.float32)[:, :2]
    if len(normalized) < 468:
        raise ValueError("hair removal needs the full MediaPipe face mesh (468+ landmarks)")
    return normalized * np.array([width, height], dtype=np.float32)


def _face_frame(points: np.ndarray) -> _FaceFrame:
    left_eye = points[LEFT_EYE_INDICES].mean(axis=0)
    right_eye = points[RIGHT_EYE_INDICES].mean(axis=0)
    axis_u = right_eye - left_eye
    axis_u = axis_u / max(float(np.linalg.norm(axis_u)), 1e-6)
    if axis_u[0] < 0:
        axis_u = -axis_u
    axis_v = np.array([-axis_u[1], axis_u[0]], dtype=np.float32)
    return _FaceFrame(origin=(left_eye + right_eye) / 2.0, axis_u=axis_u, axis_v=axis_v)


def _local_grid(frame: _FaceFrame, width: int, height: int) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return frame.to_local(np.stack([xs, ys], axis=-1))


def _estimate_skull(frame: _FaceFrame, points: np.ndarray, segmentation: HeadSegmentation) -> _Skull:
    """Approximate the bald head outline: an upper half-ellipse joined to the face oval."""

    local = frame.to_local(points)
    oval = local[FACE_OVAL_INDICES]
    u_min, u_max = float(oval[:, 0].min()), float(oval[:, 0].max())
    face_width = u_max - u_min
    forehead_v = float(local[FOREHEAD_TOP_INDEX, 1])
    face_height = float(local[CHIN_INDEX, 1]) - forehead_v

    center_u = (u_min + u_max) / 2.0
    center_v = forehead_v + face_height * SKULL_CENTER_OFFSET
    top_v = forehead_v - face_height * SKULL_TOP_SCALE

    # A bald head can never be taller than the head silhouette (hair + face). For short
    # hair this pulls the vertex down to just below the top of the hair.
    estimated_top_v = top_v
    silhouette_top = _silhouette_top_v(frame, segmentation, center_u, face_width)
    if silhouette_top is not None:
        top_v = max(top_v, silhouette_top + face_height * 0.03)
        top_v = min(top_v, forehead_v - face_height * 0.12)

    radius_u = face_width / 2.0 * SKULL_WIDTH_SCALE
    radius_v = center_v - top_v
    angles = np.linspace(np.pi, 2.0 * np.pi, 72)
    upper = np.stack([center_u + radius_u * np.cos(angles), center_v + radius_v * np.sin(angles)], axis=-1)
    hull = _convex_hull(np.concatenate([upper, oval]))
    return _Skull(
        center_u=center_u,
        center_v=center_v,
        radius_u=radius_u,
        radius_v=radius_v,
        top_v=top_v,
        face_width=face_width,
        face_height=face_height,
        vertex_ratio=(forehead_v - top_v) / face_height,
        vertex_clamped=bool(top_v > estimated_top_v + 1e-3),
        polygon=frame.to_image(hull),
    )


def _silhouette_top_v(
    frame: _FaceFrame,
    segmentation: HeadSegmentation,
    center_u: float,
    face_width: float,
) -> float | None:
    hair = segmentation.hair > 0.5
    if not hair.any():
        return None
    ys, xs = np.nonzero(hair)
    local = frame.to_local(np.stack([xs, ys], axis=-1).astype(np.float32))
    central = np.abs(local[:, 0] - center_u) < face_width * 0.30
    if central.sum() < 20:
        return None
    # A robust "top of hair" that ignores a few flyaway pixels.
    return float(np.percentile(local[central, 1], 0.5))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(map(tuple, points.tolist()))
    if len(pts) <= 2:
        return np.array(pts, dtype=np.float32)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=np.float32)


def _protection_mask(points: np.ndarray, size: tuple[int, int], unit: float) -> np.ndarray:
    """Eyes, nose and mouth are never edited. Eyebrows are deliberately *not* protected:
    when bangs cover them, protecting the landmark polygon keeps the bangs."""

    mask = np.zeros((size[1], size[0]), dtype=bool)
    for indices, padding in (
        (LEFT_EYE_INDICES, 0.035),
        (RIGHT_EYE_INDICES, 0.035),
        (NOSE_INDICES, 0.02),
        (MOUTH_INDICES, 0.03),
    ):
        hull = _convex_hull(points[indices])
        mask |= _dilate(_rasterize(hull, size), radius=padding * unit)
    return mask


def _close_forehead_gaps(
    removal: np.ndarray,
    rgb: np.ndarray,
    skull_mask: np.ndarray,
    protection: np.ndarray,
    frame: _FaceFrame,
    points: np.ndarray,
    local_grid: np.ndarray,
    unit: float,
) -> np.ndarray:
    """Bangs are where segmentation is weakest: thin strands, gaps showing skin, low
    confidence holes. Above the eyebrows everything between strands is re-synthesised
    anyway, so close the mask there instead of leaving dark fragments behind."""

    local = frame.to_local(points)
    brow_top_v = float(min(local[LEFT_BROW_INDICES, 1].min(), local[RIGHT_BROW_INDICES, 1].min()))
    above_brows = skull_mask & (local_grid[..., 1] < brow_top_v - unit * 0.01) & ~protection

    radius = 0.06 * unit
    closed = _blur(_dilate(removal, radius).astype(np.float32), radius / 2.0) > 0.977
    removal = removal | (closed & above_brows)

    # Enclosed holes (skin islands surrounded by hair) inside the head.
    outside_seed = protection | ~_dilate(skull_mask, radius=0.02 * unit)
    reachable = _connected_to(~removal, outside_seed & ~removal)
    removal = removal | (~reachable & skull_mask & ~protection)

    # Dark strand tips just outside the mask on the forehead.
    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    near = _dilate(removal, radius=0.05 * unit) & ~removal & above_brows
    if near.any():
        skin_reference = np.percentile(luminance[near], 75)
        removal = removal | (near & (luminance < skin_reference * 0.72))
    return removal


def _beard_zone(
    frame: _FaceFrame,
    points: np.ndarray,
    local_grid: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Facial hair below the nose belongs to the face, not to the hairstyle."""

    oval = _rasterize(points[FACE_OVAL_INDICES], size)
    nose_tip_v = float(frame.to_local(points[NOSE_TIP_INDEX][None])[0, 1])
    return oval & (local_grid[..., 1] > nose_tip_v)


# --------------------------------------------------------------------------------------
# Filling
# --------------------------------------------------------------------------------------


def _fill_background(rgb: np.ndarray, head_region: np.ndarray, inpainter: Inpainter, unit: float) -> np.ndarray:
    """Inpaint what is *behind* the head.

    The whole head (skull + hair) is masked, not just the hair outside the skull, so the
    inpainter only sees real background/body pixels and cannot smear hair or skin outwards.
    """

    hole = _dilate(head_region, radius=0.02 * unit)
    return inpainter(rgb, hole).astype(np.float32)


def _restore_body_silhouette(
    background: np.ndarray,
    rgb: np.ndarray,
    background_region: np.ndarray,
    segmentation: HeadSegmentation,
    points: np.ndarray,
    unit: float,
) -> tuple[np.ndarray, int]:
    """Long hair over the shoulders leaves a hole the inpainter fills with a haze of
    background, clothes and hair colours. On a plain background (ID-style photos) the
    answer is simple: above the shoulder line it is background, below it is clothes.
    The hidden part of the shoulder line is interpolated from the columns where it is
    visible; above it the background is rebuilt as a smooth field, below it the
    inpainted clothes are kept."""

    if segmentation.background is None or not background_region.any():
        return background, 0
    height, width = background_region.shape
    hidden = _dilate(background_region, radius=0.01 * unit)
    known_background = (segmentation.background > 0.8) & ~hidden
    if known_background.sum() < 500:
        return background, 0

    rows, cols = np.nonzero(background_region)
    margin = int(0.3 * unit)
    window = (slice(max(0, rows.min() - margin), rows.max() + margin), slice(max(0, cols.min() - margin), cols.max() + margin))
    nearby = (rgb[window] @ np.array([0.299, 0.587, 0.114], dtype=np.float32))[known_background[window]]
    if nearby.size < 200:
        return background, 0
    if 1.4826 * float(np.median(np.abs(nearby - np.median(nearby)))) > PLAIN_BACKGROUND_MAX_SPREAD:
        return background, 0

    body = np.zeros_like(background_region)
    for probability in (segmentation.clothes, segmentation.body_skin):
        if probability is not None:
            body |= probability > 0.5
    jaw_y = int(max(points[JAW_LEFT_INDEX, 1], points[JAW_RIGHT_INDEX, 1]))
    below_jaw = np.arange(height)[:, None] >= jaw_y
    visible_body = body & ~hidden & below_jaw
    has_body = visible_body.any(axis=0)
    top = np.where(has_body, visible_body.argmax(axis=0), height)

    # A column's shoulder line is known when visible background sits right above it, or
    # when the column holds neither body nor removed hair (pure background).
    columns = np.arange(width)
    probe = np.clip(top - max(2, int(0.015 * unit)), 0, height - 1)
    above_is_background = (segmentation.background[probe, columns] > 0.5) & ~hidden[probe, columns]
    hidden_below = (background_region & below_jaw).any(axis=0)
    known = (has_body & above_is_background) | (~has_body & ~hidden_below)
    if known.sum() < 0.1 * width or not hidden_below.any():
        return background, 0
    shoulder = np.interp(columns, columns[known], top[known].astype(np.float32))

    # Beside the face (above the jaw) removed hair outside the skull is always background
    # in a frontal portrait; below the jaw only what lies above the shoulder line is.
    rebuild = background_region & (np.arange(height)[:, None] < shoulder[None, :])
    if not rebuild.any():
        return background, 0
    smooth = _push_pull(rgb, known_background.astype(np.float32))
    alpha = np.clip(_blur(rebuild.astype(np.float32), max(1.0, 0.004 * unit)), 0.0, 1.0)
    alpha = np.where(background_region, alpha, 0.0)[..., None]
    return background * (1.0 - alpha) + smooth * alpha, int(rebuild.sum())


def _rebuild_neck(
    background: np.ndarray,
    rgb: np.ndarray,
    background_region: np.ndarray,
    body_skin: np.ndarray | None,
    points: np.ndarray,
    frame: _FaceFrame,
    local_grid: np.ndarray,
    unit: float,
) -> tuple[np.ndarray, int]:
    """Long hair hanging beside the neck leaves a large hole that the inpainter smears into
    a mix of neck, background and clothes. Inside a neck band below the jaw, continue the
    visible neck skin instead, so the neck keeps a clean outline against the background."""

    if body_skin is None:
        return background, 0
    local = frame.to_local(points)
    left_u, right_u = float(local[JAW_LEFT_INDEX, 0]), float(local[JAW_RIGHT_INDEX, 0])
    jaw_v = max(float(local[JAW_LEFT_INDEX, 1]), float(local[JAW_RIGHT_INDEX, 1]))
    chin_v = float(local[CHIN_INDEX, 1])
    center_u = (left_u + right_u) / 2.0
    half_width = (right_u - left_u) / 2.0 * NECK_WIDTH_SCALE
    u, v = local_grid[..., 0], local_grid[..., 1]
    widening = 1.0 + np.clip((v - chin_v) / unit, 0.0, None) * 0.25
    band = (v > jaw_v - 0.05 * unit) & (np.abs(u - center_u) < half_width * widening)

    visible_neck = (body_skin > 0.6) & band & ~background_region
    if visible_neck.sum() < 0.002 * unit * unit:
        return background, 0
    # The neck ends at the collar: no lower than the visible neck skin reaches.
    neck_bottom_v = float(np.percentile(v[visible_neck], 99))
    band &= v < neck_bottom_v + 0.02 * unit
    hidden = background_region & band
    if not hidden.any():
        return background, 0

    skin = _push_pull(rgb, visible_neck.astype(np.float32))
    skin += _skin_grain(rgb, visible_neck, unit)[..., None]
    alpha = np.clip(_blur(hidden.astype(np.float32), max(1.0, 0.008 * unit)), 0.0, 1.0)
    alpha = np.where(band, alpha, 0.0)[..., None]
    return background * (1.0 - alpha) + skin * alpha, int(hidden.sum())


def _render_scalp(
    rgb: np.ndarray,
    skull: _Skull,
    local_grid: np.ndarray,
    skin_probability: np.ndarray,
    removal: np.ndarray,
    protection: np.ndarray,
    points: np.ndarray,
    frame: _FaceFrame,
    unit: float,
) -> np.ndarray:
    """Shaded skin dome covering the skull, lit like the visible face."""

    u = local_grid[..., 0]
    v = local_grid[..., 1]
    local_points = frame.to_local(points)
    nose_tip_v = float(local_points[NOSE_TIP_INDEX, 1])

    brows = _rasterize(points[LEFT_BROW_INDICES], (rgb.shape[1], rgb.shape[0])) | _rasterize(
        points[RIGHT_BROW_INDICES], (rgb.shape[1], rgb.shape[0])
    )
    skin_samples = (
        (skin_probability > 0.75)
        & ~_dilate(removal, radius=0.02 * unit)
        & ~protection
        & ~_dilate(brows, radius=0.02 * unit)
        & (v < nose_tip_v)
    )
    if skin_samples.sum() < 50:
        skin_samples = (skin_probability > 0.5) & ~removal & ~protection

    if skin_samples.sum() >= 50:
        colors = rgb[skin_samples]
        sample_u = u[skin_samples]
        base = np.median(colors, axis=0)
        # Left/right lighting gradient, fitted robustly on the visible face skin.
        design = np.stack([np.ones_like(sample_u), (sample_u - skull.center_u) / skull.radius_u], axis=-1)
        coef, *_ = np.linalg.lstsq(design, colors - base, rcond=None)
        slope = np.clip(coef[1], -90.0, 90.0)
    else:
        base = np.array([214.0, 178.0, 158.0], dtype=np.float32)
        slope = np.zeros(3, dtype=np.float32)

    du = np.clip((u - skull.center_u) / skull.radius_u, -1.2, 1.2)
    dv = np.minimum(v - skull.center_v, 0.0) / skull.radius_v
    radial = np.clip(du**2 + dv**2, 0.0, 1.0)
    normal_z = np.sqrt(1.0 - radial)
    shade = 1.0 - 0.30 * (1.0 - normal_z) ** 1.1
    highlight_v = skull.top_v + skull.radius_v * 0.38
    highlight = 0.02 * np.exp(-(du / 0.55) ** 2 - ((v - highlight_v) / (skull.radius_v * 0.35)) ** 2)

    dome = (base + du[..., None] * slope) * shade[..., None] + 255.0 * highlight[..., None]
    dome *= (1.0 + _forehead_highlight(rgb, skin_samples, local_points, u, v, skull))[..., None]
    dome += _skin_grain(rgb, skin_samples, unit)[..., None]
    return dome


def _forehead_highlight(
    rgb: np.ndarray,
    skin_samples: np.ndarray,
    local_points: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    skull: _Skull,
) -> np.ndarray:
    """A real forehead catches the same light as the nose bridge and cheekbones, which
    makes it read as a curved surface. Measure how shiny the user's face is and put a
    matching soft highlight on the middle of the drawn forehead (0 for matte skin)."""

    if skin_samples.sum() < 200:
        return np.zeros_like(u)
    luminance = rgb[skin_samples] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    median = max(float(np.median(luminance)), 1.0)
    shine = (float(np.percentile(luminance, 97)) - median) / median
    strength = float(np.clip(FOREHEAD_HIGHLIGHT_SCALE * shine, 0.0, FOREHEAD_HIGHLIGHT_MAX))

    forehead_v = float(local_points[FOREHEAD_TOP_INDEX, 1])
    brow_top_v = float(min(local_points[LEFT_BROW_INDICES, 1].min(), local_points[RIGHT_BROW_INDICES, 1].min()))
    center_v = forehead_v + 0.45 * (brow_top_v - forehead_v)
    spread_u = 0.20 * skull.face_width
    spread_v = 0.30 * max(brow_top_v - forehead_v, 0.05 * skull.face_height) + 0.04 * skull.face_height
    return strength * np.exp(-(((u - skull.center_u) / spread_u) ** 2) - ((v - center_v) / spread_v) ** 2)


def _skin_grain(rgb: np.ndarray, skin_samples: np.ndarray, unit: float) -> np.ndarray:
    """Fine noise with the same strength as the real skin/sensor grain.

    Without it the dome looks plastic next to the photo. Noise (rather than a tiled skin
    patch) avoids visible repetition.
    """

    gray = rgb.mean(axis=2)
    if skin_samples.sum() < 50:
        return np.zeros_like(gray)
    fine = max(0.8, unit * 0.002)
    high_pass = gray - _blur(gray, fine * 2.0)
    samples = high_pass[skin_samples]
    # Robust spread: wrinkles and pores should not inflate the grain.
    strength = float(np.clip(1.4826 * np.median(np.abs(samples - np.median(samples))), 0.4, 3.5))

    rng = np.random.default_rng(0)
    noise = _blur(rng.normal(0.0, 1.0, size=gray.shape).astype(np.float32), fine * 0.5)
    noise /= max(float(noise.std()), 1e-6)
    return noise * strength * 0.7


def _anchor_scalp_tone(
    dome: np.ndarray,
    rgb: np.ndarray,
    removal: np.ndarray,
    scalp_region: np.ndarray,
    protection: np.ndarray,
    skin_probability: np.ndarray,
    frame: _FaceFrame,
    points: np.ndarray,
    local_grid: np.ndarray,
    unit: float,
) -> tuple[np.ndarray, list[float] | None]:
    """Pull the middle of the synthetic scalp/forehead to the colour of the user's real
    upper-face skin (between the eyes and the nose tip).

    Compared with full-image generators (e.g. HairFastGAN), the drawn forehead under a new
    hairline read as a grey, darker patch. Seams keep their exact match; the correction
    fades in away from them.
    """

    local = frame.to_local(points)
    eye_v = float(local[LEFT_EYE_INDICES + RIGHT_EYE_INDICES, 1].max())
    nose_v = float(local[NOSE_TIP_INDEX, 1])
    v = local_grid[..., 1]
    reference = (skin_probability > 0.8) & ~removal & ~protection & (v > eye_v) & (v < nose_v)
    forehead = scalp_region & (v > float(local[FOREHEAD_TOP_INDEX, 1]) - 0.15 * unit)
    if reference.sum() < 200 or forehead.sum() < 200:
        return dome, None

    gain = np.clip(np.median(rgb[reference], axis=0) / np.maximum(np.median(dome[forehead], axis=0), 1.0), 0.9, 1.2)
    seam = _dilate(removal, radius=max(2.0, unit * 0.012)) & ~removal & ~protection & (skin_probability > 0.6)
    weight = (1.0 - _proximity(seam, unit * 0.05)) if seam.any() else np.ones(removal.shape, dtype=np.float32)
    weight = np.clip(weight, 0.0, 1.0)[..., None]
    return dome * (1.0 + (gain - 1.0) * weight), [round(float(g), 3) for g in gain]


def _harmonize_seam(
    dome: np.ndarray,
    rgb: np.ndarray,
    removal: np.ndarray,
    protection: np.ndarray,
    face_skin: np.ndarray,
    body_skin: np.ndarray | None,
    unit: float,
) -> np.ndarray:
    """Shift the dome colour so it continues the real skin it touches.

    Only seams against skin constrain the dome; seams against background do not. The
    correction is a smooth membrane interpolated from the seam (push-pull), so there is no
    visible line where synthetic skin meets real skin.
    """

    skin = face_skin if body_skin is None else np.maximum(face_skin, body_skin)
    band = max(2.0, unit * 0.012)
    seam = _dilate(removal, radius=band) & ~removal & ~protection & (skin > 0.6)
    if seam.sum() < 10:
        return dome
    # Isolated dark seam pixels are leftover strand tips, not shadow; coherent dark areas
    # (a side-lit temple) survive this test because their neighbourhood is dark too.
    luminance = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    scale = unit * 0.05
    local_mean = _blur(luminance * seam, scale) / np.maximum(_blur(seam.astype(np.float32), scale), 1e-4)
    seam &= luminance > local_mean * 0.75
    if seam.sum() < 10:
        return dome

    # Generous limit: side-lit temples can be much darker than the median skin, but a
    # mis-segmented dark pixel should not drag the scalp to black.
    difference = np.clip(rgb - dome, -110.0, 60.0)
    smoothed = _blur(difference * seam[..., None], band) / np.maximum(_blur(seam.astype(np.float32), band), 1e-4)[..., None]
    correction = _push_pull(smoothed, seam.astype(np.float32))
    # The seam itself must match exactly. Away from it, skin along the seam is usually in
    # the old hair's shadow, so only a small, bounded part of that offset is carried into
    # the interior; otherwise the whole scalp (and the forehead under a new hairline)
    # turns greyer and darker than the face.
    far = np.clip(_blur(correction, unit * 0.06), -SEAM_FAR_LIMIT, SEAM_FAR_LIMIT) * SEAM_FAR_WEIGHT
    near = _proximity(seam, unit * 0.05)[..., None]
    return dome + correction * near + far * (1.0 - near)


def _redraw_hidden_eyebrows(
    dome: np.ndarray,
    rgb: np.ndarray,
    points: np.ndarray,
    removal: np.ndarray,
    hair: np.ndarray,
    frame: _FaceFrame,
    local_grid: np.ndarray,
    unit: float,
) -> tuple[np.ndarray, list[str]]:
    height, width = removal.shape
    hair_color = np.median(rgb[hair > 0.8], axis=0) if (hair > 0.8).sum() > 50 else np.array([40.0, 32.0, 28.0])
    redrawn: list[str] = []
    for name, indices in (("left", LEFT_BROW_INDICES), ("right", RIGHT_BROW_INDICES)):
        polygon = points[indices]
        brow = _rasterize(polygon, (width, height))
        if brow.sum() == 0 or (removal & brow).sum() < 0.35 * brow.sum():
            continue

        local = frame.to_local(polygon)
        inner_u = local[4, 0]
        outer_u = local[0, 0]
        along = np.clip((local_grid[..., 0] - outer_u) / (inner_u - outer_u + 1e-6), 0.0, 1.0)
        # Brows are densest a little inside the arch and fade at the tail and the head.
        density = 0.55 + 0.40 * np.sin(np.pi * np.clip(along * 0.85 + 0.1, 0.0, 1.0))

        rng = np.random.default_rng(len(redrawn) + 7)
        strokes = _blur(rng.normal(0.0, 1.0, size=(height, width)).astype(np.float32), max(0.7, unit * 0.002))
        strokes /= max(float(strokes.std()), 1e-6)
        strokes = np.clip(0.85 + strokes * 0.12, 0.6, 1.0)

        alpha = _blur(brow.astype(np.float32), max(1.0, unit * 0.006)) * density * strokes * 0.78
        color = 0.55 * hair_color + 0.45 * (dome * 0.55)
        dome = dome * (1.0 - alpha[..., None]) + color * alpha[..., None]
        redrawn.append(name)
    return dome, redrawn


def _fallback_skin_probability(points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    oval = _rasterize(points[FACE_OVAL_INDICES], size)
    return _blur(oval.astype(np.float32), 3.0)


# --------------------------------------------------------------------------------------
# Inpainting backends
# --------------------------------------------------------------------------------------


def default_inpainter(lama_model_path: str | None = None) -> Inpainter:
    """LaMa when its model and onnxruntime are available, otherwise the push-pull fill."""

    model_path = lama_model_path or str(DEFAULT_LAMA_MODEL_PATH)
    if Path(model_path).exists():
        try:
            return LamaOnnxInpainter(model_path)
        except Exception:
            pass
    return push_pull_inpaint


class LamaOnnxInpainter:
    """LaMa (big-lama) exported to ONNX, e.g. ``Carve/LaMa-ONNX`` ``lama_fp32.onnx``.

    Works on a square crop around the hole at 512x512, which is LaMa's native size.
    """

    name = "lama-onnx"

    def __init__(self, model_path: str):
        import onnxruntime

        self._session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    def __call__(self, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
        if not hole.any():
            return rgb.copy()
        height, width = hole.shape
        ys, xs = np.nonzero(hole)
        top, bottom, left, right = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        side = int(max(bottom - top, right - left) * 1.35)
        side = min(max(side, 64), max(height, width))
        cy, cx = (top + bottom) // 2, (left + right) // 2
        y0 = int(np.clip(cy - side // 2, 0, max(0, height - side)))
        x0 = int(np.clip(cx - side // 2, 0, max(0, width - side)))
        y1, x1 = min(height, y0 + side), min(width, x0 + side)

        crop = rgb[y0:y1, x0:x1]
        crop_hole = hole[y0:y1, x0:x1]
        image_in = np.asarray(
            Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).resize((LAMA_SIZE, LAMA_SIZE), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        mask_in = np.asarray(
            Image.fromarray(crop_hole.astype(np.uint8) * 255).resize((LAMA_SIZE, LAMA_SIZE), Image.Resampling.NEAREST),
            dtype=np.float32,
        ) / 255.0
        mask_in = (mask_in > 0.5).astype(np.float32)

        output = self._session.run(
            None,
            {
                "image": image_in.transpose(2, 0, 1)[None],
                "mask": mask_in[None, None],
            },
        )[0][0].transpose(1, 2, 0)
        if output.max() <= 1.5:
            output = output * 255.0
        restored = np.asarray(
            Image.fromarray(np.clip(output, 0, 255).astype(np.uint8)).resize((x1 - x0, y1 - y0), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )

        result = rgb.copy()
        region = result[y0:y1, x0:x1]
        region[crop_hole] = restored[crop_hole]
        # Pixels outside the crop that are still in the hole (huge masks) fall back to push-pull.
        leftover = hole.copy()
        leftover[y0:y1, x0:x1] = False
        if leftover.any():
            result = push_pull_inpaint(result, leftover)
        return result


def push_pull_inpaint(rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Smooth membrane-like fill from the surrounding pixels (no model required)."""

    filled = _push_pull(rgb.astype(np.float32), (~hole).astype(np.float32))
    result = rgb.astype(np.float32).copy()
    result[hole] = filled[hole]
    return result


push_pull_inpaint.name = "push-pull"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------
# Small numpy image helpers
# --------------------------------------------------------------------------------------


def _push_pull(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Scattered-data interpolation over an image pyramid.

    ``weights`` marks known pixels (0..1). Unknown pixels get a smooth blend of the nearest
    known values at increasingly coarse scales.
    """

    squeeze = values.ndim == 2
    if squeeze:
        values = values[..., None]
    values = values.astype(np.float32)
    weights = np.clip(weights.astype(np.float32), 0.0, 1.0)

    pyramid: list[tuple[np.ndarray, np.ndarray]] = []
    current_values, current_weights = values * weights[..., None], weights
    while True:
        normalized = current_values / np.maximum(current_weights, 1e-6)[..., None]
        pyramid.append((normalized, np.minimum(current_weights, 1.0)))
        if min(current_weights.shape) <= 2:
            break
        current_values, current_weights = _downsample(current_values), _downsample(current_weights)

    filled, _ = pyramid[-1]
    for normalized, level_weights in reversed(pyramid[:-1]):
        upsampled = _upsample(filled, level_weights.shape)
        alpha = level_weights[..., None]
        filled = normalized * alpha + upsampled * (1.0 - alpha)
    return filled[..., 0] if squeeze else filled


def _downsample(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    padded = np.pad(array, [(0, height % 2), (0, width % 2)] + [(0, 0)] * (array.ndim - 2), mode="edge")
    return (padded[0::2, 0::2] + padded[1::2, 0::2] + padded[0::2, 1::2] + padded[1::2, 1::2])


def _upsample(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    rows = np.clip((np.arange(height) + 0.5) / 2.0 - 0.5, 0, array.shape[0] - 1)
    cols = np.clip((np.arange(width) + 0.5) / 2.0 - 0.5, 0, array.shape[1] - 1)
    r0, c0 = np.floor(rows).astype(int), np.floor(cols).astype(int)
    r1, c1 = np.minimum(r0 + 1, array.shape[0] - 1), np.minimum(c0 + 1, array.shape[1] - 1)
    fr, fc = (rows - r0)[:, None, None], (cols - c0)[None, :, None]
    top = array[r0][:, c0] * (1 - fc) + array[r0][:, c1] * fc
    bottom = array[r1][:, c0] * (1 - fc) + array[r1][:, c1] * fc
    return top * (1 - fr) + bottom * fr


def _blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate Gaussian blur (three box passes per axis)."""

    if sigma <= 0.3:
        return array.astype(np.float32)
    factor = int(sigma // 4)
    if factor >= 2:
        # Large blurs lose nothing by running at reduced resolution.
        small = _blur(_mean_pool(array.astype(np.float32), factor), sigma / factor)
        return _resize_float(small, array.shape[:2])
    result = _blur_axis(array.astype(np.float32), sigma, axis=0)
    return _blur_axis(result, sigma, axis=1)


def _mean_pool(array: np.ndarray, factor: int) -> np.ndarray:
    height, width = array.shape[:2]
    pad = [(0, (-height) % factor), (0, (-width) % factor)] + [(0, 0)] * (array.ndim - 2)
    padded = np.pad(array, pad, mode="edge")
    shape = (padded.shape[0] // factor, factor, padded.shape[1] // factor, factor) + padded.shape[2:]
    return padded.reshape(shape).mean(axis=(1, 3))


def _resize_float(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    channels = array[..., None] if array.ndim == 2 else array
    resized = [
        np.asarray(Image.fromarray(np.ascontiguousarray(channels[..., c]), "F").resize((width, height), Image.Resampling.BILINEAR))
        for c in range(channels.shape[2])
    ]
    result = np.stack(resized, axis=-1).astype(np.float32)
    return result[..., 0] if array.ndim == 2 else result


def _blur_axis(array: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    radius = max(1, int(round(np.sqrt(12.0 * sigma * sigma / 3.0 + 1.0) / 2.0)))
    result = array.astype(np.float32)
    for _ in range(3):
        result = _box_axis(result, radius, axis)
    return result


def _box_axis(array: np.ndarray, radius: int, axis: int) -> np.ndarray:
    moved = np.moveaxis(array, axis, 0)
    padded = np.concatenate([np.repeat(moved[:1], radius + 1, axis=0), moved, np.repeat(moved[-1:], radius, axis=0)])
    cumulative = np.cumsum(padded, axis=0, dtype=np.float32)
    length = moved.shape[0]
    window = cumulative[2 * radius + 1 : 2 * radius + 1 + length] - cumulative[:length]
    return np.moveaxis(window / (2 * radius + 1), 0, axis)


def _dilate(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius < 0.5:
        return mask.astype(bool)
    # Blur-and-threshold dilation: cheap and roughly circular.
    return _blur(mask.astype(np.float32), radius / 2.0) > 0.023


def _proximity(mask: np.ndarray, scale: float) -> np.ndarray:
    """1 on the mask, falling off to 0 about three ``scale`` away (stepped approximation)."""

    rings = [_dilate(mask, radius=scale * factor) for factor in (0.5, 1.0, 2.0, 3.0)]
    return _blur(np.mean(rings, axis=0).astype(np.float32), scale * 0.5)


def _connected_to(mask: np.ndarray, seed: np.ndarray, factor: int = 4) -> np.ndarray:
    """Parts of ``mask`` 8-connected to ``seed`` (morphological reconstruction at 1/factor scale)."""

    height, width = mask.shape
    pad_h, pad_w = (-height) % factor, (-width) % factor
    small_mask = np.pad(mask, ((0, pad_h), (0, pad_w))).reshape(
        (height + pad_h) // factor, factor, (width + pad_w) // factor, factor
    ).any(axis=(1, 3))
    small_seed = np.pad(seed & mask, ((0, pad_h), (0, pad_w))).reshape(small_mask.shape[0], factor, small_mask.shape[1], factor).any(axis=(1, 3))

    grown = small_seed & small_mask
    for _ in range(small_mask.shape[0] * small_mask.shape[1]):
        expanded = grown.copy()
        expanded[1:, :] |= grown[:-1, :]
        expanded[:-1, :] |= grown[1:, :]
        expanded[:, 1:] |= grown[:, :-1]
        expanded[:, :-1] |= grown[:, 1:]
        expanded[1:, 1:] |= grown[:-1, :-1]
        expanded[:-1, :-1] |= grown[1:, 1:]
        expanded[1:, :-1] |= grown[:-1, 1:]
        expanded[:-1, 1:] |= grown[1:, :-1]
        expanded &= small_mask
        if np.array_equal(expanded, grown):
            break
        grown = expanded

    full = np.repeat(np.repeat(grown, factor, axis=0), factor, axis=1)[:height, :width]
    return mask & full


def _rasterize(polygon: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    canvas = Image.new("L", size, 0)
    ImageDraw.Draw(canvas).polygon([tuple(map(float, point)) for point in polygon], fill=255)
    return np.asarray(canvas) > 127


def _mask_image(mask: np.ndarray) -> Image.Image:
    if mask.dtype == bool:
        return Image.fromarray(mask.astype(np.uint8) * 255, "L")
    return Image.fromarray(np.clip(mask * 255.0 + 0.5, 0, 255).astype(np.uint8), "L")


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
