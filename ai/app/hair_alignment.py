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
# Headroom: backdrop kept above the new hair (face widths), and at most this fraction of the
# photo height is given up at the bottom to make room for it.
HEADROOM_MARGIN = 0.05
MAX_HEADROOM_SHIFT = 0.25
# Warped hair further than this (face widths) from the hair on the head is dropped.
STRAY_HAIR_DISTANCE = 0.05
# Turned hair model photos: past this relative head turn the skull correspondences come from
# the 3D landmarks, and past MIRROR_MIN_YAW the side the turned head hides is replaced by the
# visible side (blending the two would leave a ghost of the misplaced hidden-side hair).
POSE_CORRECTION_MIN_YAW = 10.0
MIRROR_MIN_YAW = 15.0
# Landmarks that move little with expression, for the 3D pose fit.
RIGID_LANDMARK_INDICES = [
    10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 33, 133, 362, 263, 70, 300, 105, 334,
    234, 454, 127, 356, 93, 323, 21, 251, 54, 284, 103, 332, 109, 338,
]
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
    points3d: np.ndarray | None = None  # landmarks with MediaPipe depth, in reference pixels


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
    unrefined_image: Image.Image | None = None  # the result before generative_refine, if it ran
    target_points: np.ndarray | None = None  # user landmarks in result pixels (moved by headroom)

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
            **(
                {"unrefined-result.png": ("image/png", _encode_png(self.unrefined_image))}
                if self.unrefined_image is not None
                else {}
            ),
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
    headroom: bool = False,
) -> HairTransferResult:
    """Put the reference hairstyle on the target person.

    Each image must be the one its landmarks were computed on; the two may differ in size.
    With ``headroom``, a target photo framed too tight for the new hairstyle is moved down
    (see ``add_headroom``) so the hair is not cut off by the top of the photo.
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

    layer = extract_hair_layer(reference, reference_landmarks, reference_segmentation)
    width, height = target.size
    target_points = _landmark_pixels(target_landmarks, width, height)
    target_skull = _estimate_skull(_face_frame(target_points), target_points, target_segmentation)
    headroom_rows = 0
    if headroom:
        target_points3d = _landmark_points3d(target_landmarks, width, height)
        headroom_rows = required_headroom(layer, target_points, target_skull, (width, height), target_points3d)
    if headroom_rows:
        target, target_landmarks = add_headroom(target, target_landmarks, headroom_rows, target_segmentation.hair > 0.3)
        target_segmentation = _shift_segmentation(target_segmentation, headroom_rows)
        target_points = _landmark_pixels(target_landmarks, width, height)
        target_skull = _estimate_skull(_face_frame(target_points), target_points, target_segmentation)

    bald = remove_hair(target, target_landmarks, segmentation=target_segmentation, inpainter=inpainter)
    target_frame = _face_frame(target_points)
    target_points3d = _landmark_points3d(target_landmarks, width, height)
    warped_rgb, warped_alpha, warp = align_hair_layer(
        layer, target_points, target_skull, (width, height), target_points3d=target_points3d
    )
    warped_alpha, stray_pixels = _drop_stray_hair(warped_alpha, target_skull)

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
        "headroomRows": headroom_rows,
        "strayHairPixelCount": stray_pixels,
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
        target_points=target_points,
    )


def required_headroom(
    layer: HairLayer,
    target_points: np.ndarray,
    target_skull: _Skull,
    size: tuple[int, int],
    target_points3d: np.ndarray | None = None,
) -> int:
    """Rows of backdrop the user photo lacks above the head for the new hairstyle.

    ID photos are framed tight: a taller style than the user's own would be cut off by the
    top of the photo. The hair is warped onto a canvas extended upwards to see where its
    top lands.
    """

    width, height = size
    probe = int(MAX_HEADROOM_SHIFT * height)
    forehead_y = float(target_points[FOREHEAD_TOP_INDEX, 1])
    canvas_height = int(probe + max(forehead_y, 1.0))  # nothing of interest below the forehead
    lowered = target_points + np.array([0.0, probe], dtype=np.float32)
    lowered3d = None if target_points3d is None else target_points3d + np.array([0.0, probe, 0.0])
    _rgb, alpha, _warp = align_hair_layer(layer, lowered, target_skull, (width, canvas_height), target_points3d=lowered3d)
    hair_rows = np.nonzero((alpha > 0.5).sum(axis=1) > max(3, 0.01 * width))[0]
    if not len(hair_rows):
        return 0
    top = float(hair_rows[0]) - probe
    margin = HEADROOM_MARGIN * target_skull.face_width
    return int(np.clip(np.ceil(margin - top), 0, probe))


def add_headroom(
    image: Image.Image,
    landmarks: Sequence[dict] | Sequence[Sequence[float]] | np.ndarray,
    rows: int,
    hair: np.ndarray | None = None,
) -> tuple[Image.Image, list]:
    """Move the photo down by ``rows`` (dropping the bottom rows) and continue the backdrop
    into the strip that opens at the top. ``hair`` (the user's own hair) is kept out of the
    fill so it does not grow upwards. Returns the new photo and its landmarks."""

    width, height = image.size
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    shifted = np.zeros_like(rgb)
    shifted[rows:] = rgb[: height - rows]
    # Continue the backdrop seen in a band just below the strip (not the face further down).
    known = np.zeros((height, width), dtype=bool)
    known[rows : rows + max(8, height // 20)] = True
    if hair is not None:
        moved = np.zeros_like(known)
        moved[rows:] = hair[: height - rows]
        known &= ~_dilate(moved, radius=max(2.0, 0.01 * width))
    if known.sum() < 50:  # hair across the whole top edge: use whatever is there
        known = np.zeros_like(known)
        known[rows:] = True
    # A smooth fill, not LaMa: with nothing above the strip LaMa paints dark noise there,
    # and ID photo backdrops are plain anyway.
    filled = push_pull_inpaint(shifted, ~known)
    shifted[:rows] = filled[:rows]

    points = _landmark_pixels(landmarks, width, height)
    points[:, 1] += rows
    depth = _landmark_points3d(landmarks, width, height)
    moved_landmarks = []
    for index, (x, y) in enumerate(points):
        point = {"index": index, "x": float(x / width), "y": float(y / height)}
        if depth is not None:
            point["z"] = float(depth[index, 2] / width)  # depth is in units of the width, unchanged by the shift
        moved_landmarks.append(point)
    return Image.fromarray(np.clip(shifted + 0.5, 0, 255).astype(np.uint8), "RGB"), moved_landmarks


def _shift_segmentation(segmentation: HeadSegmentation, rows: int) -> HeadSegmentation:
    def shift(probability: np.ndarray | None, fill: float) -> np.ndarray | None:
        if probability is None:
            return None
        moved = np.full_like(probability, fill)
        moved[rows:] = probability[: probability.shape[0] - rows]
        return moved

    return HeadSegmentation(
        hair=shift(segmentation.hair, 0.0),
        face_skin=shift(segmentation.face_skin, 0.0),
        body_skin=shift(segmentation.body_skin, 0.0),
        source=segmentation.source,
        background=shift(segmentation.background, 1.0),
        clothes=shift(segmentation.clothes, 0.0),
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
        points3d=_landmark_points3d(reference_landmarks, width, height),
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
    target_points3d: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, _Warp]:
    """Warp the hair layer so the reference skull and hairline land on the target's.

    Returns the warped straight RGB, its alpha, and the warp itself so other reference
    content (skin, ears) can be carried over the same way.

    When the hair model's head is turned, the skull outline estimated from their (narrow,
    foreshortened) face is in the wrong place. With MediaPipe depth for both faces, the
    user's skull outline is instead carried into the hair model's photo through the 3D
    head pose, and the side the turn hides is filled from the visible side.
    """

    source_controls = _control_points(layer.points, layer.skull)
    target_controls = _control_points(target_points, target_skull)
    pose = _relative_pose(layer.points3d, target_points3d)
    pose_info = None
    far_sign = 0.0
    if pose is not None and abs(pose.yaw) >= POSE_CORRECTION_MIN_YAW:
        # The outline of a frontal head is its widest section, about as deep as the face edge.
        depth = float(target_points3d[[LEFT_FACE_EDGE_INDEX, RIGHT_FACE_EDGE_INDEX], 2].mean())
        curve = target_controls[-SKULL_CURVE_SAMPLES:]
        lifted = np.column_stack([curve, np.full(len(curve), depth)])
        in_reference = pose.to_reference(lifted)
        source_controls[-SKULL_CURVE_SAMPLES:] = in_reference[:, :2]
        # The end of the outline that turned away from the camera is the hidden side.
        far_sign = 1.0 if in_reference[-1, 2] > in_reference[0, 2] else -1.0
        pose_info = {
            "yawDegrees": round(pose.yaw, 1),
            "scale": round(pose.scale, 3),
            "hiddenSide": "right" if far_sign > 0 else "left",
        }

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
    if far_sign and abs(pose.yaw) >= MIRROR_MIN_YAW:
        sampled = _mirror_hidden_side(sampled, target_points, target_skull, far_sign)
        pose_info["mirroredHiddenSide"] = True
    alpha = np.clip(sampled[..., 3], 0.0, 1.0)
    rgb = sampled[..., :3] / np.maximum(alpha, 1e-4)[..., None]

    residual = _affine_map(normalized_targets, source_controls, (target_controls - origin) / unit) - source_controls
    info = {
        "method": method,
        "controlPointCount": int(len(target_controls)),
        "scale": float(target_skull.face_width / max(layer.skull.face_width, 1e-6)),
        "affineResidualPixels": float(np.sqrt((residual**2).sum(axis=1)).mean()),
        **({"pose": pose_info} if pose_info else {}),
    }
    return np.clip(rgb, 0.0, 255.0), alpha, _Warp(source_x=full[..., 0], source_y=full[..., 1], info=info)


@dataclass(frozen=True)
class _Pose:
    """Similarity transform taking reference 3D landmarks onto the target's: t = s R r + o."""

    scale: float
    rotation: np.ndarray
    offset: np.ndarray
    yaw: float  # degrees the hair model's head is turned relative to the user's

    def to_reference(self, points: np.ndarray) -> np.ndarray:
        return ((points - self.offset) @ self.rotation) / self.scale


def _landmark_points3d(
    landmarks: Sequence[dict] | Sequence[Sequence[float]] | np.ndarray, width: int, height: int
) -> np.ndarray | None:
    """Landmarks with MediaPipe depth (same scale as x) in pixels, or None without depth."""

    if len(landmarks) and isinstance(landmarks[0], dict):
        if any("z" not in point for point in landmarks):
            return None
        values = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in landmarks], dtype=np.float64)
    else:
        values = np.asarray(landmarks, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < 3:
            return None
        values = values[:, :3]
    return values * np.array([width, height, width], dtype=np.float64)


def _relative_pose(reference3d: np.ndarray | None, target3d: np.ndarray | None) -> _Pose | None:
    if reference3d is None or target3d is None:
        return None
    source, destination = reference3d[RIGID_LANDMARK_INDICES], target3d[RIGID_LANDMARK_INDICES]
    source_center, destination_center = source.mean(axis=0), destination.mean(axis=0)
    covariance = (source - source_center).T @ (destination - destination_center)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.diag([1.0, 1.0, np.sign(np.linalg.det(vt.T @ u.T))])
    rotation = vt.T @ sign @ u.T
    spread = float(((source - source_center) ** 2).sum())
    if spread <= 0:
        return None
    scale = float((singular * np.diag(sign)).sum() / spread)
    offset = destination_center - scale * rotation @ source_center
    yaw = float(np.degrees(np.arctan2(rotation[0, 2], rotation[0, 0])))
    return _Pose(scale=scale, rotation=rotation, offset=offset, yaw=yaw)


def _mirror_hidden_side(premultiplied: np.ndarray, target_points: np.ndarray, skull: _Skull, far_sign: float) -> np.ndarray:
    """A turned hair model shows one side of their hair and hides the other; hairstyles are
    close to symmetric, so the hidden side gets the visible side mirrored across the face.
    The middle (bangs, parting) keeps what the photo shows."""

    height, width = premultiplied.shape[:2]
    frame = _face_frame(target_points)
    grid = _local_grid(frame, width, height)
    mirrored = frame.to_image(np.stack([2.0 * skull.center_u - grid[..., 0], grid[..., 1]], axis=-1))
    flipped = _sample_bilinear(premultiplied, mirrored[..., 0], mirrored[..., 1])
    side = (grid[..., 0] - skull.center_u) * far_sign / skull.face_width
    ramp = _smoothstep(0.12, 0.3, side)[..., None]
    return premultiplied * (1.0 - ramp) + flipped * ramp


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


def _drop_stray_hair(alpha: np.ndarray, skull: _Skull) -> tuple[np.ndarray, int]:
    """Pieces of the warped hair that land away from the head would float on the face or
    neck: hair behind the far ear of a hair model who turned their head, or a stray
    segmentation blob. Keep the hair connected to what covers the skull and anything near it."""

    core = alpha > 0.5
    height, width = alpha.shape
    main = _connected_to(core, core & _rasterize(skull.polygon, (width, height)))
    if not main.any():
        return alpha, 0
    near = _dilate(main, radius=STRAY_HAIR_DISTANCE * skull.face_width)
    stray = core & ~near
    if not stray.any():
        return alpha, 0
    cleaned = alpha.copy()
    cleaned[_dilate(stray, radius=max(2.0, 0.01 * skull.face_width)) & ~near] = 0.0
    return cleaned, int(stray.sum())


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
    # Above the crown there is only backdrop; there the closing would just bridge the
    # hair to the top edge of a tightly framed photo.
    gap[: max(0, int(skull.polygon[:, 1].min()))] = False
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
