"""Skull-aligned hair transfer: put a reference hairstyle on the bald canvas.

The legacy compositor scales a cropped hair layer by face width and pins it at the
forehead, so the new hair often ends up as a small cap that leaves the scalp exposed.
Here the reference hair is warped so that the reference *skull* (and upper face outline)
lands exactly on the user's skull, using the same skull model as hair removal. The hair
volume around it follows smoothly through a thin-plate-spline warp.

Pipeline: remove_hair(target) -> extract_hair_layer(reference) -> align_hair_layer -> composite.

The same warp also carries over what the hair model's photo shows that the user's photo
cannot: real skin texture for the forehead under the new hairline, and the ears when the
user's own ears were hidden by their old hair (both recoloured to the user's skin).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Sequence

import numpy as np
from PIL import Image

from .hair_removal import (
    CHIN_INDEX,
    FACE_OVAL_INDICES,
    FOREHEAD_TOP_INDEX,
    LEFT_BROW_INDICES,
    RIGHT_BROW_INDICES,
    HairRemovalResult,
    HeadSegmentation,
    Inpainter,
    _beard_zone,
    _blur,
    _connected_to,
    _dilate,
    _estimate_skull,
    _face_frame,
    _landmark_pixels,
    _local_grid,
    _protection_mask,
    _rasterize,
    _Skull,
    default_inpainter,
    push_pull_inpaint,
    remove_hair,
    segment_head,
)

# Upper face outline (cheekbone -> temple -> forehead -> temple -> cheekbone): where the
# hairline sits, so it anchors the warp together with the skull curve.
UPPER_FACE_OUTLINE_INDICES = [234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454]
EYE_CORNER_INDICES = [33, 133, 362, 263]
SKULL_CURVE_SAMPLES = 9
TPS_REGULARIZATION = 1e-3
WARP_GRID_STEP = 6
NOSE_BOTTOM_INDEX = 2
LEFT_FACE_EDGE_INDEX = 234
RIGHT_FACE_EDGE_INDEX = 454
# White balance: only near-neutral backdrops (each channel within this of the mean) are
# read as light colour; the hair gets this fraction of the tint ratio, capped.
NEUTRAL_BACKDROP_MAX_TINT = 0.12
WHITE_BALANCE_STRENGTH = 0.7
WHITE_BALANCE_MAX_SHIFT = 0.12
METADATA_VERSION = "hair-transfer-v3-skull-tps"


@dataclass(frozen=True)
class HairLayer:
    """Reference hair in reference-image pixels: straight (un-premultiplied) RGB + alpha."""

    rgb: np.ndarray
    alpha: np.ndarray
    points: np.ndarray
    skull: _Skull
    skin_luminance: float | None
    cropped_edges: tuple[str, ...] = ()
    # What else the hair model's photo can lend the user, in reference pixels.
    source_rgb: np.ndarray | None = None
    skin_color: np.ndarray | None = None
    skin_texture: np.ndarray | None = None  # fine skin detail relative to local skin brightness
    skin_texture_valid: np.ndarray | None = None
    ear_alpha: np.ndarray | None = None


@dataclass(frozen=True)
class _Warp:
    """For every target pixel, where to read in the reference image."""

    source_x: np.ndarray
    source_y: np.ndarray
    info: dict

    def sample(self, image: np.ndarray) -> np.ndarray:
        channels = image[..., None] if image.ndim == 2 else image
        sampled = _sample_bilinear(channels.astype(np.float32), self.source_x, self.source_y)
        return sampled[..., 0] if image.ndim == 2 else sampled


@dataclass(frozen=True)
class HairTransferResult:
    image: Image.Image
    bald: HairRemovalResult
    warped_hair: Image.Image  # RGBA in target space
    reference_hair: Image.Image  # RGBA in reference space
    edit_mask: Image.Image | None = None  # every user pixel this transfer changed
    metadata: dict = field(default_factory=dict)

    def artifacts(self) -> dict[str, tuple[str, bytes]]:
        return {
            "result.png": ("image/png", _encode_png(self.image)),
            "warped-hair-layer.png": ("image/png", _encode_png(self.warped_hair)),
            "warped-hair-mask.png": ("image/png", _encode_png(self.warped_hair.getchannel("A"))),
            "hair-layer.png": ("image/png", _encode_png(self.reference_hair)),
            "hair-mask.png": ("image/png", _encode_png(self.reference_hair.getchannel("A"))),
            # Names the result screen already shows as "기존 머리 영역 / 합성 수정 영역 / 얼굴 보호 영역".
            "target-hair-mask.png": ("image/png", _encode_png(self.bald.hair_mask)),
            "face-protection-mask.png": ("image/png", _encode_png(self.bald.protection_mask)),
            **({"edit-mask.png": ("image/png", _encode_png(self.edit_mask))} if self.edit_mask is not None else {}),
            **self.bald.artifacts(),
        }


def transfer_hair(
    target: Image.Image,
    target_landmarks: Sequence[dict] | Sequence[Sequence[float]] | np.ndarray,
    reference: Image.Image,
    reference_landmarks: Sequence[dict] | Sequence[Sequence[float]] | np.ndarray,
    *,
    target_segmentation: HeadSegmentation | None = None,
    reference_segmentation: HeadSegmentation | None = None,
    hair_segmenter_model_path: str | None = None,
    multiclass_model_path: str | None = None,
    inpainter: Inpainter | None = None,
) -> HairTransferResult:
    """Put the reference hairstyle on the target person.

    Both images must be the ones their landmarks were computed on (same crop and size).
    """

    if inpainter is None:
        inpainter = default_inpainter()
    if target_segmentation is None:
        target_segmentation = segment_head(
            target, hair_segmenter_model_path=hair_segmenter_model_path, multiclass_model_path=multiclass_model_path
        )
    if reference_segmentation is None:
        reference_segmentation = segment_head(
            reference, hair_segmenter_model_path=hair_segmenter_model_path, multiclass_model_path=multiclass_model_path
        )

    bald = remove_hair(target, target_landmarks, segmentation=target_segmentation, inpainter=inpainter)
    layer = extract_hair_layer(reference, reference_landmarks, reference_segmentation)

    width, height = target.size
    target_points = _landmark_pixels(target_landmarks, width, height)
    target_frame = _face_frame(target_points)
    target_skull = _estimate_skull(target_frame, target_points, target_segmentation)
    warped_rgb, warped_alpha, warp = align_hair_layer(layer, target_points, target_skull, (width, height))

    canvas = np.asarray(bald.image, dtype=np.float32)
    target_rgb = np.asarray(target.convert("RGB"), dtype=np.float32)
    removal = np.asarray(bald.removal_mask) > 127
    scalp_fill = np.asarray(bald.scalp_fill_mask) > 127
    target_grid = _local_grid(target_frame, width, height)
    target_skin_color = _skin_color(target_rgb, target_segmentation, exclude=removal)

    canvas, forehead_pixels = _lend_skin_texture(
        canvas, layer, warp, scalp_fill, target_frame, target_points, target_grid, target_skull.face_width
    )
    canvas, ear_pixels = _lend_ears(
        canvas, layer, warp, removal, target_skin_color, target_frame, target_points, target_grid, target_skull.face_width
    )
    warped_alpha = _fade_hairline(warped_alpha, scalp_fill, target_skull.face_width)
    exposure = _exposure_match(layer, canvas, target_points, target_segmentation)
    white_balance = _white_balance_gain(target_rgb, target_segmentation, layer.source_rgb, reference_segmentation)
    hair_rgb = np.clip(warped_rgb * exposure * white_balance, 0.0, 255.0)
    canvas, back_hair_pixels = _fill_back_hair(canvas, hair_rgb, warped_alpha, target_skull, target_segmentation)
    result = _composite(canvas, hair_rgb, warped_alpha, target_skull.face_width)
    changed = np.abs(result - target_rgb).max(axis=2) > 2.0

    warped_rgba = np.dstack([hair_rgb, warped_alpha * 255.0])
    reference_rgba = np.dstack([layer.rgb, layer.alpha * 255.0])
    metadata = {
        "version": METADATA_VERSION,
        "warnings": [f"REFERENCE_HAIR_CROPPED_{edge.upper()}" for edge in layer.cropped_edges],
        "hairRemoval": bald.metadata,
        "warp": warp.info,
        "exposureFactor": exposure,
        "whiteBalanceGain": [round(float(g), 3) for g in white_balance],
        "backHairPixelCount": back_hair_pixels,
        "lentSkinTexturePixelCount": forehead_pixels,
        "lentEarPixelCount": ear_pixels,
        "referenceSkull": {"radiusU": layer.skull.radius_u, "radiusV": layer.skull.radius_v},
        "targetSkull": {"radiusU": target_skull.radius_u, "radiusV": target_skull.radius_v},
    }
    return HairTransferResult(
        image=Image.fromarray(np.clip(result + 0.5, 0, 255).astype(np.uint8), "RGB"),
        bald=bald,
        warped_hair=Image.fromarray(np.clip(warped_rgba + 0.5, 0, 255).astype(np.uint8), "RGBA"),
        reference_hair=Image.fromarray(np.clip(reference_rgba + 0.5, 0, 255).astype(np.uint8), "RGBA"),
        edit_mask=Image.fromarray(changed.astype(np.uint8) * 255, "L"),
        metadata=metadata,
    )


def extract_hair_layer(
    reference: Image.Image,
    reference_landmarks: Sequence[dict] | Sequence[Sequence[float]] | np.ndarray,
    segmentation: HeadSegmentation,
) -> HairLayer:
    """Soft hair matte of the reference with the background colour removed from its edges."""

    rgb = np.asarray(reference.convert("RGB"), dtype=np.float32)
    height, width = rgb.shape[:2]
    points = _landmark_pixels(reference_landmarks, width, height)
    frame = _face_frame(points)
    skull = _estimate_skull(frame, points, segmentation)
    unit = skull.face_width

    hair = np.clip(segmentation.hair.astype(np.float32), 0.0, 1.0)
    beard = _beard_zone(frame, points, _local_grid(frame, width, height), (width, height))
    skull_mask = _rasterize(skull.polygon, (width, height))
    seed = (hair > 0.5) & _dilate(skull_mask, radius=0.08 * unit) & ~beard
    attached = _connected_to((hair > 0.1) & ~beard, seed)
    if not attached.any():
        raise ValueError("no hair attached to the head was found in the reference photo")

    alpha = np.where(attached, _smoothstep(0.12, 0.85, hair), 0.0).astype(np.float32)
    if segmentation.face_skin is not None:
        # The segmenter cuts the hairline sharply; real hairlines thin out. Only lower the
        # alpha there (never grow it), so no reference skin is carried onto the user.
        near_face = _dilate(segmentation.face_skin > 0.5, radius=0.04 * unit)
        alpha = np.where(near_face, np.minimum(alpha, _blur(alpha, max(1.0, 0.008 * unit))), alpha)

    # Edge pixels are a mix of hair and whatever was behind it (often the studio
    # background). Estimate that backdrop and un-mix it, so a light background does not
    # leave a pale halo when the hair is placed on skin.
    behind = push_pull_inpaint(rgb, _dilate(alpha > 0.02, radius=0.01 * unit))
    foreground = behind + (rgb - behind) / np.maximum(alpha, 0.25)[..., None]
    foreground = np.where((alpha > 0.0)[..., None], np.clip(foreground, 0.0, 255.0), rgb)

    skin = segmentation.face_skin
    skin_luminance = None
    if skin is not None and (skin > 0.8).sum() > 200:
        skin_luminance = float(np.median(_luminance(rgb)[skin > 0.8]))

    grid = _local_grid(frame, width, height)
    texture, texture_valid = _skin_texture(rgb, points, segmentation, unit)
    return HairLayer(
        rgb=foreground,
        alpha=alpha,
        points=points,
        skull=skull,
        skin_luminance=skin_luminance,
        cropped_edges=_cropped_edges(alpha, unit),
        source_rgb=rgb,
        skin_color=_skin_color(rgb, segmentation, exclude=alpha > 0.1),
        skin_texture=texture,
        skin_texture_valid=texture_valid,
        ear_alpha=_ear_matte(points, frame, grid, segmentation, alpha, unit),
    )


def _cropped_edges(alpha: np.ndarray, unit: float) -> tuple[str, ...]:
    """Image borders the hairstyle runs into: the part beyond them is missing and will
    show up as a straight cut once the hair is scaled onto the user."""

    edges = {"top": alpha[0], "left": alpha[:, 0], "right": alpha[:, -1]}
    return tuple(name for name, line in edges.items() if (line > 0.5).sum() > 0.05 * unit)


def align_hair_layer(
    layer: HairLayer,
    target_points: np.ndarray,
    target_skull: _Skull,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, _Warp]:
    """Warp the hair layer so the reference skull and hairline land on the target's.

    Returns the warped straight RGB, its alpha, and the warp itself so other reference
    content (skin, ears) can be carried over the same way.
    """

    source_controls = _control_points(layer.points, layer.skull)
    target_controls = _control_points(target_points, target_skull)

    unit = target_skull.face_width
    origin = target_points[FOREHEAD_TOP_INDEX]
    normalized_targets = (target_controls - origin) / unit

    width, height = size
    grid_ys = np.arange(0, height + WARP_GRID_STEP, WARP_GRID_STEP, dtype=np.float32)
    grid_xs = np.arange(0, width + WARP_GRID_STEP, WARP_GRID_STEP, dtype=np.float32)
    gx, gy = np.meshgrid(grid_xs, grid_ys)
    queries = (np.stack([gx.ravel(), gy.ravel()], axis=-1) - origin) / unit

    method = "thin-plate-spline"
    mapped = _tps_map(normalized_targets, source_controls, queries)
    coarse = mapped.reshape(len(grid_ys), len(grid_xs), 2)
    if not _is_fold_free(coarse):
        method = "affine-fallback"
        mapped = _affine_map(normalized_targets, source_controls, queries)
        coarse = mapped.reshape(len(grid_ys), len(grid_xs), 2)

    # Grid node (i, j) sits exactly on pixel (j * step, i * step); interpolate between nodes.
    pixel_ys, pixel_xs = np.mgrid[0:height, 0:width].astype(np.float32)
    full = _sample_bilinear(coarse, pixel_xs / WARP_GRID_STEP, pixel_ys / WARP_GRID_STEP)
    premultiplied = np.dstack([layer.rgb * layer.alpha[..., None], layer.alpha])
    sampled = _sample_bilinear(premultiplied, full[..., 0], full[..., 1])
    alpha = np.clip(sampled[..., 3], 0.0, 1.0)
    rgb = sampled[..., :3] / np.maximum(alpha, 1e-4)[..., None]

    residual = _affine_map(normalized_targets, source_controls, (target_controls - origin) / unit) - source_controls
    info = {
        "method": method,
        "controlPointCount": int(len(target_controls)),
        "scale": float(target_skull.face_width / max(layer.skull.face_width, 1e-6)),
        "affineResidualPixels": float(np.sqrt((residual**2).sum(axis=1)).mean()),
    }
    return np.clip(rgb, 0.0, 255.0), alpha, _Warp(source_x=full[..., 0], source_y=full[..., 1], info=info)


def _control_points(points: np.ndarray, skull: _Skull) -> np.ndarray:
    frame = _face_frame(points)
    angles = np.linspace(np.pi, 2.0 * np.pi, SKULL_CURVE_SAMPLES)
    curve = np.stack(
        [skull.center_u + skull.radius_u * np.cos(angles), skull.center_v + skull.radius_v * np.sin(angles)],
        axis=-1,
    )
    return np.concatenate(
        [
            points[UPPER_FACE_OUTLINE_INDICES],
            points[EYE_CORNER_INDICES],
            points[[CHIN_INDEX]],
            frame.to_image(curve),
        ]
    ).astype(np.float64)


def _tps_kernel(distances: np.ndarray) -> np.ndarray:
    squared = distances**2
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = squared * np.log(squared)
    return np.nan_to_num(kernel)


def _tps_map(centers: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Thin-plate spline through (centers -> values), evaluated at queries."""

    count = len(centers)
    kernel = _tps_kernel(np.linalg.norm(centers[:, None] - centers[None], axis=-1))
    affine = np.hstack([np.ones((count, 1)), centers])
    system = np.zeros((count + 3, count + 3))
    system[:count, :count] = kernel + TPS_REGULARIZATION * np.eye(count)
    system[:count, count:] = affine
    system[count:, :count] = affine.T
    rhs = np.zeros((count + 3, 2))
    rhs[:count] = values
    solution = np.linalg.solve(system, rhs)

    query_kernel = _tps_kernel(np.linalg.norm(queries[:, None] - centers[None], axis=-1))
    return query_kernel @ solution[:count] + np.hstack([np.ones((len(queries), 1)), queries]) @ solution[count:]


def _affine_map(centers: np.ndarray, values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    design = np.hstack([np.ones((len(centers), 1)), centers])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return np.hstack([np.ones((len(queries), 1)), queries]) @ coefficients


def _is_fold_free(coarse_map: np.ndarray) -> bool:
    """A warp that flips orientation anywhere would mirror strands; refuse it."""

    dx = np.diff(coarse_map, axis=1)[:-1]
    dy = np.diff(coarse_map, axis=0)[:, :-1]
    determinant = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
    return bool(np.percentile(determinant, 0.5) > 0.0)


def _sample_bilinear(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample ``image`` at fractional pixel positions; outside the image reads as zero."""

    height, width = image.shape[:2]
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    fx = (xs - x0)[..., None]
    fy = (ys - y0)[..., None]
    padded = np.pad(image, ((1, 2), (1, 2), (0, 0)))
    cx0 = np.clip(x0 + 1, 0, width + 2)
    cy0 = np.clip(y0 + 1, 0, height + 2)
    cx1 = np.clip(x0 + 2, 0, width + 2)
    cy1 = np.clip(y0 + 2, 0, height + 2)
    top = padded[cy0, cx0] * (1 - fx) + padded[cy0, cx1] * fx
    bottom = padded[cy1, cx0] * (1 - fx) + padded[cy1, cx1] * fx
    return top * (1 - fy) + bottom * fy


def _exposure_match(
    layer: HairLayer,
    canvas: np.ndarray,
    target_points: np.ndarray,
    target_segmentation: HeadSegmentation,
) -> float:
    """Gently match the reference photo's exposure to the user's photo (hue untouched)."""

    skin = target_segmentation.face_skin
    if layer.skin_luminance is None or skin is None or (skin > 0.8).sum() < 200:
        return 1.0
    target_luminance = float(np.median(_luminance(canvas)[skin > 0.8]))
    ratio = target_luminance / max(layer.skin_luminance, 1.0)
    return float(np.clip(np.sqrt(ratio), 0.8, 1.25))


def _fade_hairline(alpha: np.ndarray, scalp_fill: np.ndarray, unit: float) -> np.ndarray:
    """Where the new hair ends on the drawn forehead, thin it out over a few millimetres
    instead of ending on a cut-out edge (full-image generators get this for free)."""

    if not scalp_fill.any():
        return alpha
    near_forehead = _dilate(scalp_fill & (alpha < 0.5), radius=0.03 * unit)
    softened = np.minimum(alpha, _blur(alpha, max(1.0, 0.012 * unit)))
    weight = np.clip(_blur(near_forehead.astype(np.float32), max(1.0, 0.01 * unit)), 0.0, 1.0)
    return alpha * (1.0 - weight) + softened * weight


def _skin_color(rgb: np.ndarray, segmentation: HeadSegmentation, exclude: np.ndarray) -> np.ndarray | None:
    if segmentation.face_skin is None:
        return None
    skin = (segmentation.face_skin > 0.8) & ~exclude
    if skin.sum() < 200:
        return None
    return np.median(rgb[skin], axis=0)


def _skin_texture(
    rgb: np.ndarray,
    points: np.ndarray,
    segmentation: HeadSegmentation,
    unit: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The hair model's fine skin detail (pores, grain) as a factor around 1, relative to
    the local skin brightness, so it can be re-applied to the user's own skin colour.

    Only the fine scale is kept: the hair model's shading (side light, temple falloff,
    bright skin right under the hairline) showed up as grey patches and pale outlines.
    """

    if segmentation.face_skin is None:
        return None, None
    height, width = rgb.shape[:2]
    features = _protection_mask(points, (width, height), unit)
    brows = _rasterize(points[LEFT_BROW_INDICES], (width, height)) | _rasterize(points[RIGHT_BROW_INDICES], (width, height))
    valid = ((segmentation.face_skin > 0.6) & ~_dilate(features | brows, radius=0.02 * unit)).astype(np.float32)
    if valid.sum() < 200:
        return None, None

    luminance = _luminance(rgb)
    smooth = _masked_mean(luminance, valid, max(1.0, 0.006 * unit))
    texture = 1.0 + 0.7 * np.clip((luminance - smooth) / np.maximum(smooth, 1.0), -0.06, 0.06)
    return texture * valid, valid


def _lend_skin_texture(
    canvas: np.ndarray,
    layer: HairLayer,
    warp: _Warp,
    scalp_fill: np.ndarray,
    frame,
    points: np.ndarray,
    grid: np.ndarray,
    unit: float,
) -> tuple[np.ndarray, int]:
    """Where the user's forehead had to be synthesised and the hair model shows real
    forehead skin, borrow its fine texture so the skin under the new hairline is not
    plastic-smooth."""

    if layer.skin_texture is None or not scalp_fill.any():
        return canvas, 0
    valid = warp.sample(layer.skin_texture_valid)
    # Where no hair-model skin was sampled the factor must stay neutral (1), not 0.
    texture = np.where(valid > 1e-3, warp.sample(layer.skin_texture) / np.maximum(valid, 1e-3), 1.0)

    local = frame.to_local(points)
    brow_top_v = float(min(local[LEFT_BROW_INDICES, 1].min(), local[RIGHT_BROW_INDICES, 1].min()))
    above_brows = np.clip((brow_top_v - 0.015 * unit - grid[..., 1]) / (0.03 * unit), 0.0, 1.0)
    weight = (
        np.clip(_blur(scalp_fill.astype(np.float32), max(1.0, 0.01 * unit)), 0.0, 1.0)
        * np.clip(_blur(np.clip((valid - 0.5) * 2.0, 0.0, 1.0), max(1.0, 0.02 * unit)), 0.0, 1.0)
        * above_brows
    )
    if weight.max() <= 0.0:
        return canvas, 0
    lent = canvas * np.where(weight > 0.0, texture, 1.0)[..., None]
    return canvas * (1.0 - weight[..., None]) + lent * weight[..., None], int((weight > 0.5).sum())


def _ear_zone(points: np.ndarray, frame, grid: np.ndarray, unit: float) -> np.ndarray:
    local = frame.to_local(points)
    brow_top_v = float(min(local[LEFT_BROW_INDICES, 1].min(), local[RIGHT_BROW_INDICES, 1].min()))
    vertical = (grid[..., 1] > brow_top_v - 0.04 * unit) & (grid[..., 1] < local[NOSE_BOTTOM_INDEX, 1] + 0.08 * unit)
    left_u = local[LEFT_FACE_EDGE_INDEX, 0]
    right_u = local[RIGHT_FACE_EDGE_INDEX, 0]
    left = (grid[..., 0] < left_u + 0.03 * unit) & (grid[..., 0] > left_u - 0.28 * unit)
    right = (grid[..., 0] > right_u - 0.03 * unit) & (grid[..., 0] < right_u + 0.28 * unit)
    return vertical & (left | right)


def _ear_matte(
    points: np.ndarray,
    frame,
    grid: np.ndarray,
    segmentation: HeadSegmentation,
    hair_alpha: np.ndarray,
    unit: float,
) -> np.ndarray | None:
    """Visible ears of the hair model: skin just outside the face outline at ear height."""

    skin = np.zeros(hair_alpha.shape, dtype=bool)
    for probability in (segmentation.face_skin, segmentation.body_skin):
        if probability is not None:
            skin |= probability > 0.5
    height, width = hair_alpha.shape
    oval = _rasterize(points[FACE_OVAL_INDICES], (width, height))
    ears = skin & _ear_zone(points, frame, grid, unit) & ~oval & (hair_alpha < 0.5)
    ears = _connected_to(ears, ears & _dilate(oval, radius=0.03 * unit))
    if ears.sum() < 0.001 * unit * unit:
        return None
    return np.clip(_blur(ears.astype(np.float32), 1.0), 0.0, 1.0) * (1.0 - hair_alpha)


def _lend_ears(
    canvas: np.ndarray,
    layer: HairLayer,
    warp: _Warp,
    removal: np.ndarray,
    target_skin_color: np.ndarray | None,
    frame,
    points: np.ndarray,
    grid: np.ndarray,
    unit: float,
) -> tuple[np.ndarray, int]:
    """If the user's ears were under their old hair, the bald canvas has none. Borrow the
    hair model's ears (recoloured to the user's skin); the new hair still goes on top."""

    if layer.ear_alpha is None or layer.skin_color is None or target_skin_color is None:
        return canvas, 0
    hidden = removal & _ear_zone(points, frame, grid, unit)
    if not hidden.any():
        return canvas, 0
    alpha = warp.sample(layer.ear_alpha) * np.clip(_blur(hidden.astype(np.float32), 1.5), 0.0, 1.0)
    if alpha.max() <= 0.05:
        return canvas, 0
    tint = np.clip(target_skin_color / np.maximum(layer.skin_color, 1.0), 0.6, 1.6)
    ears = warp.sample(layer.source_rgb) * tint
    # Ears pick up coloured light (backlight, a red sweater); pull their hue towards the
    # user's skin while keeping their own shading.
    shading = _luminance(ears) / max(float(_luminance(target_skin_color[None])[0]), 1.0)
    ears = np.clip(0.65 * ears + 0.35 * shading[..., None] * target_skin_color, 0.0, 255.0)
    return canvas * (1.0 - alpha[..., None]) + ears * alpha[..., None], int((alpha > 0.5).sum())


def _masked_mean(values: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    return _blur(values * weights, sigma) / np.maximum(_blur(weights, sigma), 1e-4)


def _fill_back_hair(
    canvas: np.ndarray,
    hair_rgb: np.ndarray,
    alpha: np.ndarray,
    skull: _Skull,
    segmentation: HeadSegmentation,
) -> tuple[np.ndarray, int]:
    """Narrow gaps enclosed by the new hair (e.g. between a bob and the cheek) would show
    the inpainted background. In a real photo you see the shadowed hair behind the head
    there, so paint those gaps with a dark version of the hair colour."""

    unit = skull.face_width
    core = alpha > 0.5
    if core.sum() < 100:
        return canvas, 0
    radius = 0.1 * unit
    closed = _blur(_dilate(core, radius).astype(np.float32), radius / 2.0) > 0.977
    height, width = alpha.shape
    head = _rasterize(skull.polygon, (width, height))
    skin = np.zeros_like(core)
    for probability in (segmentation.face_skin, segmentation.body_skin):
        if probability is not None:
            skin |= probability > 0.4
    gap = closed & ~core & ~head & ~_dilate(skin, radius=0.01 * unit)
    if not gap.any():
        return canvas, 0

    back_color = np.median(hair_rgb[alpha > 0.9], axis=0) * 0.45 if (alpha > 0.9).any() else np.zeros(3)
    weight = np.clip(_blur(gap.astype(np.float32), max(1.0, 0.01 * unit)), 0.0, 1.0)[..., None] * 0.92
    return canvas * (1.0 - weight) + back_color * weight, int(gap.sum())


def _white_balance_gain(
    target_rgb: np.ndarray,
    target_segmentation: HeadSegmentation,
    reference_rgb: np.ndarray | None,
    reference_segmentation: HeadSegmentation,
) -> np.ndarray:
    """Tint the borrowed hair with the colour of the user's light.

    A neutral (grey/white) backdrop shows the colour of the light it is lit with, so the
    ratio of the two backdrops' tints tells how to re-light the hair model's hair. A
    coloured or black backdrop tells nothing (paint, not light), so the hair is left as is.
    """

    target_tint = _neutral_backdrop_tint(target_rgb, target_segmentation)
    reference_tint = None if reference_rgb is None else _neutral_backdrop_tint(reference_rgb, reference_segmentation)
    if target_tint is None or reference_tint is None:
        return np.ones(3, dtype=np.float32)
    gain = (target_tint / reference_tint) ** WHITE_BALANCE_STRENGTH
    return np.clip(gain, 1.0 - WHITE_BALANCE_MAX_SHIFT, 1.0 + WHITE_BALANCE_MAX_SHIFT).astype(np.float32)


def _neutral_backdrop_tint(rgb: np.ndarray, segmentation: HeadSegmentation) -> np.ndarray | None:
    if segmentation.background is None:
        return None
    backdrop = segmentation.background > 0.9
    if backdrop.sum() < 2000:
        return None
    color = np.median(rgb[backdrop], axis=0)
    level = float(color.mean())
    if level < 40.0:
        return None
    tint = color / level
    if float(np.abs(tint - 1.0).max()) > NEUTRAL_BACKDROP_MAX_TINT:
        return None
    return tint


def _composite(canvas: np.ndarray, hair_rgb: np.ndarray, alpha: np.ndarray, unit: float) -> np.ndarray:
    # A soft contact shadow just below and around the hair keeps it from looking pasted on.
    shadow = _blur(alpha, max(1.5, unit * 0.02))
    shifted = np.zeros_like(shadow)
    offset = max(1, int(unit * 0.01))
    shifted[offset:] = shadow[:-offset]
    # Compare blurred with blurred: against the sharp alpha, a softened hairline would get a
    # dark rim right under its semi-transparent edge.
    darken = 1.0 - 0.45 * np.clip(shifted - shadow, 0.0, 1.0)
    shaded = canvas * darken[..., None]
    return shaded * (1.0 - alpha[..., None]) + hair_rgb * alpha[..., None]


def _smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    t = np.clip((values - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
