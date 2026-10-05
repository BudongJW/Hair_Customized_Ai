from __future__ import annotations

from dataclasses import dataclass
import json
from io import BytesIO
from math import cos, radians, sin, sqrt

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from .face_landmarker import MediaPipeFaceAnalysis, analyze_with_mediapipe
try:
    from .hair_segmenter import HairSegmentationResult, segment_hair_with_mediapipe
except ImportError:  # hair_segmenter.py is not committed yet; see hair_segmenter_fallback.py
    from .hair_segmenter_fallback import HairSegmentationResult, segment_hair_with_mediapipe

PORTRAIT_SIZE = (900, 1125)

LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_INDICES = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87]
NOSE_INDICES = [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344]
LOWER_FACE_OVAL_INDICES = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]


@dataclass(frozen=True)
class PortraitGeometry:
    width: int
    height: int
    face_box: tuple[int, int, int, int]
    forehead_point: tuple[int, int] | None = None
    eye_center_point: tuple[int, int] | None = None
    eye_distance: float | None = None
    roll_degrees: float = 0.0
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    source: str = "estimated"

    @property
    def face_width(self) -> int:
        return self.face_box[2] - self.face_box[0]

    @property
    def face_height(self) -> int:
        return self.face_box[3] - self.face_box[1]

    @property
    def center_x(self) -> int:
        return (self.face_box[0] + self.face_box[2]) // 2

    @property
    def forehead_x(self) -> int:
        if self.forehead_point is not None:
            return self.forehead_point[0]
        return self.center_x

    @property
    def forehead_y(self) -> int:
        if self.forehead_point is not None:
            return self.forehead_point[1]
        return int(self.face_box[1] + self.face_height * 0.18)

    @property
    def eye_center_x(self) -> int:
        if self.eye_center_point is not None:
            return self.eye_center_point[0]
        return self.center_x

    @property
    def eye_center_y(self) -> int:
        if self.eye_center_point is not None:
            return self.eye_center_point[1]
        return int(self.face_box[1] + self.face_height * 0.38)

    @property
    def fitting_width(self) -> float:
        if self.eye_distance is not None and self.eye_distance > 0:
            return self.eye_distance * 2.15
        return float(self.face_width)


@dataclass(frozen=True)
class HairExtractionResult:
    layer: Image.Image
    mask: Image.Image
    crop_box: tuple[int, int, int, int]
    anchor_point: tuple[int, int]
    source_width: float
    metadata: dict


@dataclass(frozen=True)
class HairPlacementResult:
    image: Image.Image
    metadata: dict


@dataclass(frozen=True)
class TargetHairCleanupResult:
    image: Image.Image
    mask: Image.Image
    metadata: dict


@dataclass(frozen=True)
class HairFittingOutput:
    result_bytes: bytes
    hair_mask_bytes: bytes
    hair_layer_bytes: bytes
    metadata: dict


def analyze_face_image(image_bytes: bytes, profile_id: str, model_path: str | None = None) -> tuple[bytes, dict]:
    image = _open_image(image_bytes)
    width, height = image.size

    canvas = _portrait_cover(image, PORTRAIT_SIZE)
    canvas = canvas.filter(ImageFilter.SMOOTH_MORE)
    media_pipe_analysis = _try_analyze_with_mediapipe(canvas, model_path)
    geometry = _geometry_from_analysis(canvas.size, media_pipe_analysis) if media_pipe_analysis else _estimate_geometry(canvas)
    canvas = _draw_analysis_overlay(canvas, geometry, media_pipe_analysis)
    face_box = geometry.face_box
    w, h = canvas.size
    pose = {
        "yawDegrees": media_pipe_analysis.yaw_degrees if media_pipe_analysis else 0.0,
        "pitchDegrees": media_pipe_analysis.pitch_degrees if media_pipe_analysis else 0.0,
        "rollDegrees": media_pipe_analysis.roll_degrees if media_pipe_analysis else 0.0,
    }

    landmarks = {
        "source": media_pipe_analysis.source if media_pipe_analysis else "python-worker-estimated-portrait",
        "profileId": profile_id,
        "image": {"width": width, "height": height},
        "processedImage": {"width": w, "height": h},
        "faceBox": {
            "left": face_box[0] / w,
            "top": face_box[1] / h,
            "right": face_box[2] / w,
            "bottom": face_box[3] / h,
        },
        "pose": pose,
        "keyLandmarks": media_pipe_analysis.key_landmarks if media_pipe_analysis else _estimated_key_landmarks(geometry),
        "landmarks": media_pipe_analysis.landmarks if media_pipe_analysis else _estimated_landmarks(geometry),
        "landmarkCount": len(media_pipe_analysis.landmarks) if media_pipe_analysis else 5,
        "transformMatrix": media_pipe_analysis.transform_matrix if media_pipe_analysis else None,
    }

    return _encode_jpeg(canvas), landmarks


def compose_hair_fitting(
    face_bytes: bytes,
    reference_bytes: bytes,
    job_id: str,
    face_landmarks: dict | None = None,
    model_path: str | None = None,
    hair_segmenter_model_path: str | None = None,
) -> HairFittingOutput:
    target = _portrait_cover(_open_image(face_bytes), PORTRAIT_SIZE)
    reference = _portrait_cover(_open_image(reference_bytes), PORTRAIT_SIZE)

    target_geometry = _geometry_from_landmarks_payload(target.size, face_landmarks) or _estimate_geometry(target)
    reference_analysis = _try_analyze_with_mediapipe(reference, model_path)
    reference_geometry = _geometry_from_analysis(reference.size, reference_analysis) if reference_analysis else _estimate_geometry(reference)
    extraction = _extract_reference_hair(
        reference,
        reference_geometry,
        reference_analysis,
        hair_segmenter_model_path,
    )
    placement = _fit_hair_to_target(
        extraction,
        target_geometry,
        reference_geometry,
        target.size,
    )
    positioned_hair = placement.image
    target_cleanup = _suppress_target_existing_hair(
        target,
        target_geometry,
        hair_segmenter_model_path,
        positioned_hair,
    )

    result = target_cleanup.image.convert("RGBA")
    shadow = _hair_shadow(positioned_hair)
    result.alpha_composite(shadow)
    result.alpha_composite(positioned_hair)

    draw = ImageDraw.Draw(result, "RGBA")
    draw.rectangle((0, result.height - 46, result.width, result.height), fill=(15, 23, 42, 168))
    draw.text((24, result.height - 31), f"Hair fitting result {job_id}", fill=(255, 255, 255, 230))

    metadata = {
        "version": "hair-mask-v11-conservative-fit",
        "targetGeometry": _geometry_metadata(target_geometry),
        "referenceGeometry": _geometry_metadata(reference_geometry),
        "hairExtraction": extraction.metadata,
        "hairPlacement": placement.metadata,
        "targetHairCleanup": target_cleanup.metadata,
        "referencePhotoSuitability": _reference_photo_suitability(target_geometry, reference_geometry),
        "rollDeltaDegrees": target_geometry.roll_degrees - reference_geometry.roll_degrees,
        "yawDeltaDegrees": target_geometry.yaw_degrees - reference_geometry.yaw_degrees,
        "referenceLandmarkSource": reference_geometry.source,
        "targetLandmarkSource": target_geometry.source,
    }

    return HairFittingOutput(
        result_bytes=_encode_jpeg(result.convert("RGB")),
        hair_mask_bytes=_encode_png(extraction.mask.convert("L")),
        hair_layer_bytes=_encode_png(extraction.layer),
        metadata=metadata,
    )


def landmarks_json(landmarks: dict) -> str:
    return json.dumps(landmarks, ensure_ascii=False)


def _open_image(image_bytes: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")


def _portrait_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = ImageOps.exif_transpose(image).convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (int(image.width * scale), int(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = max(0, (resized.width - width) // 2)
    top = max(0, (resized.height - height) // 2)
    return resized.crop((left, top, left + width, top + height))


def _estimate_geometry(image: Image.Image) -> PortraitGeometry:
    width, height = image.size
    face_width = int(width * 0.48)
    face_height = int(height * 0.50)
    center_x = width // 2
    top = int(height * 0.25)
    left = center_x - face_width // 2
    return PortraitGeometry(
        width=width,
        height=height,
        face_box=(left, top, left + face_width, top + face_height),
    )


def _try_analyze_with_mediapipe(image: Image.Image, model_path: str | None) -> MediaPipeFaceAnalysis | None:
    if not model_path:
        return None

    try:
        return analyze_with_mediapipe(image, model_path)
    except Exception:
        return None


def _geometry_from_analysis(
    image_size: tuple[int, int],
    analysis: MediaPipeFaceAnalysis | None,
) -> PortraitGeometry | None:
    if analysis is None:
        return None

    width, height = image_size
    face_box = _normalized_box_to_pixels(analysis.face_box, width, height)
    forehead = analysis.key_landmarks.get("forehead_center")
    forehead_point = None
    if forehead:
        forehead_point = (int(forehead["x"] * width), int(forehead["y"] * height))
    eye_center_point, eye_distance = _eye_metrics_from_key_landmarks(analysis.key_landmarks, width, height)

    return PortraitGeometry(
        width=width,
        height=height,
        face_box=face_box,
        forehead_point=forehead_point,
        eye_center_point=eye_center_point,
        eye_distance=eye_distance,
        roll_degrees=analysis.roll_degrees,
        yaw_degrees=analysis.yaw_degrees,
        pitch_degrees=analysis.pitch_degrees,
        source=analysis.source,
    )


def _geometry_from_landmarks_payload(
    image_size: tuple[int, int],
    payload: dict | None,
) -> PortraitGeometry | None:
    if not payload:
        return None

    face_box = payload.get("faceBox")
    if not face_box:
        return None

    width, height = image_size
    key_landmarks = payload.get("keyLandmarks") or {}
    forehead = key_landmarks.get("forehead_center")
    pose = payload.get("pose") or {}

    forehead_point = None
    if forehead:
        forehead_point = (int(float(forehead["x"]) * width), int(float(forehead["y"]) * height))
    eye_center_point, eye_distance = _eye_metrics_from_key_landmarks(key_landmarks, width, height)

    return PortraitGeometry(
        width=width,
        height=height,
        face_box=_normalized_box_to_pixels(face_box, width, height),
        forehead_point=forehead_point,
        eye_center_point=eye_center_point,
        eye_distance=eye_distance,
        roll_degrees=float(pose.get("rollDegrees") or 0.0),
        yaw_degrees=float(pose.get("yawDegrees") or 0.0),
        pitch_degrees=float(pose.get("pitchDegrees") or 0.0),
        source=str(payload.get("source") or "stored-landmarks"),
    )


def _eye_metrics_from_key_landmarks(
    key_landmarks: dict,
    width: int,
    height: int,
) -> tuple[tuple[int, int] | None, float | None]:
    left_eye = _average_key_points(
        key_landmarks,
        ["left_eye_outer", "left_eye_inner"],
        width,
        height,
    )
    right_eye = _average_key_points(
        key_landmarks,
        ["right_eye_outer", "right_eye_inner"],
        width,
        height,
    )
    if left_eye is None or right_eye is None:
        return None, None

    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    distance = sqrt((right_eye[0] - left_eye[0]) ** 2 + (right_eye[1] - left_eye[1]) ** 2)
    return center, distance


def _average_key_points(
    key_landmarks: dict,
    names: list[str],
    width: int,
    height: int,
) -> tuple[int, int] | None:
    points = []
    for name in names:
        point = key_landmarks.get(name)
        if point:
            points.append((float(point["x"]) * width, float(point["y"]) * height))
    if not points:
        return None

    return (
        int(sum(point[0] for point in points) / len(points)),
        int(sum(point[1] for point in points) / len(points)),
    )


def _normalized_box_to_pixels(face_box: dict, width: int, height: int) -> tuple[int, int, int, int]:
    left = int(float(face_box["left"]) * width)
    top = int(float(face_box["top"]) * height)
    right = int(float(face_box["right"]) * width)
    bottom = int(float(face_box["bottom"]) * height)
    min_width = int(width * 0.25)
    min_height = int(height * 0.30)

    if right - left < min_width:
        center = (left + right) // 2
        left = center - min_width // 2
        right = center + min_width // 2
    if bottom - top < min_height:
        center = (top + bottom) // 2
        top = center - min_height // 2
        bottom = center + min_height // 2

    return (
        max(0, left),
        max(0, top),
        min(width, right),
        min(height, bottom),
    )


def _draw_analysis_overlay(
    image: Image.Image,
    geometry: PortraitGeometry,
    analysis: MediaPipeFaceAnalysis | None,
) -> Image.Image:
    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    line_width = max(2, image.width // 260)

    if analysis:
        for point in analysis.landmarks[::8]:
            x = int(point["x"] * image.width)
            y = int(point["y"] * image.height)
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(42, 157, 143, 150))
    else:
        draw.ellipse(geometry.face_box, outline=(42, 157, 143, 220), width=line_width)

    fx = geometry.forehead_x
    fy = geometry.forehead_y
    draw.line((fx - 36, fy, fx + 36, fy), fill=(232, 93, 117, 210), width=line_width)
    return overlay.convert("RGB")


def _estimated_key_landmarks(geometry: PortraitGeometry) -> dict:
    w = geometry.width
    h = geometry.height
    return {
        "left_eye_outer": {"index": -1, "x": 0.39, "y": 0.43, "z": 0.0},
        "left_eye_inner": {"index": -1, "x": 0.46, "y": 0.43, "z": 0.0},
        "right_eye_inner": {"index": -1, "x": 0.54, "y": 0.43, "z": 0.0},
        "right_eye_outer": {"index": -1, "x": 0.61, "y": 0.43, "z": 0.0},
        "nose_tip": {"index": -1, "x": 0.50, "y": 0.53, "z": 0.0},
        "mouth_left": {"index": -1, "x": 0.44, "y": 0.66, "z": 0.0},
        "mouth_right": {"index": -1, "x": 0.56, "y": 0.66, "z": 0.0},
        "forehead_center": {"index": -1, "x": geometry.forehead_x / w, "y": geometry.forehead_y / h, "z": 0.0},
    }


def _estimated_landmarks(geometry: PortraitGeometry) -> list[dict]:
    return list(_estimated_key_landmarks(geometry).values())


def _geometry_metadata(geometry: PortraitGeometry) -> dict:
    return {
        "source": geometry.source,
        "faceBox": {
            "left": geometry.face_box[0] / geometry.width,
            "top": geometry.face_box[1] / geometry.height,
            "right": geometry.face_box[2] / geometry.width,
            "bottom": geometry.face_box[3] / geometry.height,
        },
        "forehead": {
            "x": geometry.forehead_x / geometry.width,
            "y": geometry.forehead_y / geometry.height,
        },
        "eyeCenter": {
            "x": geometry.eye_center_x / geometry.width,
            "y": geometry.eye_center_y / geometry.height,
        },
        "eyeDistance": geometry.eye_distance,
        "fittingWidth": geometry.fitting_width,
        "rollDegrees": geometry.roll_degrees,
        "yawDegrees": geometry.yaw_degrees,
        "pitchDegrees": geometry.pitch_degrees,
    }


def _reference_photo_suitability(
    target_geometry: PortraitGeometry,
    reference_geometry: PortraitGeometry,
) -> dict:
    yaw_delta = reference_geometry.yaw_degrees - target_geometry.yaw_degrees
    roll_delta = reference_geometry.roll_degrees - target_geometry.roll_degrees
    pitch_delta = reference_geometry.pitch_degrees - target_geometry.pitch_degrees
    warnings = []

    if abs(yaw_delta) >= 18.0:
        warnings.append("REFERENCE_FACE_TOO_SIDEWAYS")
    elif abs(yaw_delta) >= 11.0:
        warnings.append("REFERENCE_FACE_SLIGHTLY_SIDEWAYS")

    if abs(roll_delta) >= 14.0:
        warnings.append("REFERENCE_HEAD_TILT_TOO_LARGE")
    elif abs(roll_delta) >= 8.0:
        warnings.append("REFERENCE_HEAD_TILT_NOTICEABLE")

    if abs(pitch_delta) >= 14.0:
        warnings.append("REFERENCE_FACE_UP_DOWN_ANGLE_TOO_LARGE")

    severe = any(warning.endswith("TOO_SIDEWAYS") or warning.endswith("TOO_LARGE") for warning in warnings)
    status = "needs_better_reference_photo" if severe else "usable_with_correction" if warnings else "good"

    return {
        "status": status,
        "needsBetterReferencePhoto": status == "needs_better_reference_photo",
        "warnings": warnings,
        "yawDeltaDegrees": yaw_delta,
        "rollDeltaDegrees": roll_delta,
        "pitchDeltaDegrees": pitch_delta,
        "recommendedYawMaxDegrees": 12.0,
        "recommendedRollMaxDegrees": 8.0,
    }


def _build_reference_protection_mask(
    size: tuple[int, int],
    geometry: PortraitGeometry,
    analysis: MediaPipeFaceAnalysis | None,
) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    if analysis is None:
        _draw_estimated_face_protection(draw, geometry)
        return mask.filter(ImageFilter.GaussianBlur(3))

    landmarks = analysis.landmarks
    _draw_landmark_hull(draw, landmarks, LEFT_EYE_INDICES, size, padding=18, fill=255)
    _draw_landmark_hull(draw, landmarks, RIGHT_EYE_INDICES, size, padding=18, fill=255)
    _draw_landmark_hull(draw, landmarks, MOUTH_INDICES, size, padding=20, fill=255)
    _draw_landmark_hull(draw, landmarks, NOSE_INDICES, size, padding=14, fill=230)
    _draw_lower_face_protection(draw, landmarks, geometry, size)
    return mask.filter(ImageFilter.GaussianBlur(3))


def _draw_estimated_face_protection(draw: ImageDraw.ImageDraw, geometry: PortraitGeometry) -> None:
    left, top, right, bottom = geometry.face_box
    face_w = geometry.face_width
    face_h = geometry.face_height
    draw.ellipse(
        (
            left + int(face_w * 0.10),
            top + int(face_h * 0.18),
            right - int(face_w * 0.10),
            bottom - int(face_h * 0.06),
        ),
        fill=210,
    )


def _draw_landmark_hull(
    draw: ImageDraw.ImageDraw,
    landmarks: list[dict],
    indices: list[int],
    size: tuple[int, int],
    padding: int,
    fill: int,
) -> None:
    points = [_landmark_point(landmarks, index, size) for index in indices if index < len(landmarks)]
    points = [point for point in points if point is not None]
    if len(points) < 3:
        return

    left = min(point[0] for point in points) - padding
    top = min(point[1] for point in points) - padding
    right = max(point[0] for point in points) + padding
    bottom = max(point[1] for point in points) + padding
    draw.ellipse((left, top, right, bottom), fill=fill)


def _draw_lower_face_protection(
    draw: ImageDraw.ImageDraw,
    landmarks: list[dict],
    geometry: PortraitGeometry,
    size: tuple[int, int],
) -> None:
    oval_points = [_landmark_point(landmarks, index, size) for index in LOWER_FACE_OVAL_INDICES if index < len(landmarks)]
    oval_points = [point for point in oval_points if point is not None]
    if len(oval_points) >= 3:
        forehead_cut = geometry.forehead_y + int(geometry.face_height * 0.18)
        lower_points = [(x, y) for x, y in oval_points if y >= forehead_cut]
        if len(lower_points) >= 3:
            left = min(point[0] for point in lower_points) - int(geometry.face_width * 0.08)
            right = max(point[0] for point in lower_points) + int(geometry.face_width * 0.08)
            bottom = max(point[1] for point in lower_points) + int(geometry.face_height * 0.04)
            draw.rounded_rectangle(
                (left, forehead_cut, right, bottom),
                radius=max(18, geometry.face_width // 9),
                fill=190,
            )
            return

    left, top, right, bottom = geometry.face_box
    draw.rounded_rectangle(
        (
            left + int(geometry.face_width * 0.10),
            geometry.forehead_y + int(geometry.face_height * 0.18),
            right - int(geometry.face_width * 0.10),
            bottom,
        ),
        radius=max(18, geometry.face_width // 9),
        fill=190,
    )


def _landmark_point(
    landmarks: list[dict],
    index: int,
    size: tuple[int, int],
) -> tuple[int, int] | None:
    if index >= len(landmarks):
        return None
    width, height = size
    point = landmarks[index]
    return int(point["x"] * width), int(point["y"] * height)


def _extract_reference_hair(
    reference: Image.Image,
    geometry: PortraitGeometry,
    analysis: MediaPipeFaceAnalysis | None,
    hair_segmenter_model_path: str | None,
) -> HairExtractionResult:
    rgba = reference.convert("RGBA")
    mask = Image.new("L", reference.size, 0)
    pixels = reference.load()
    mask_pixels = mask.load()
    background = _estimate_background_color(reference)
    protection_mask = _build_reference_protection_mask(reference.size, geometry, analysis)
    protection_pixels = protection_mask.load()
    segmentation = _try_segment_hair(reference, hair_segmenter_model_path)

    left, top, right, bottom = geometry.face_box
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.forehead_x
    head_top = max(0, int(geometry.forehead_y - face_h * 0.58))
    hair_bottom = min(reference.height, int(top + face_h * 0.46))
    hair_left = max(0, int(cx - face_w * 0.90))
    hair_right = min(reference.width, int(cx + face_w * 0.90))
    accepted_pixels = 0

    if segmentation is not None and segmentation.mask.getbbox() is not None:
        mask = segmentation.mask
        accepted_pixels = _count_non_zero_mask_pixels(mask)
    else:
        for y in range(head_top, hair_bottom):
            vertical_weight = _vertical_weight(y, head_top, hair_bottom)
            for x in range(hair_left, hair_right):
                if protection_pixels[x, y] > 128:
                    continue
                if not _inside_head_region(x, y, cx, geometry.forehead_y, face_w, face_h):
                    continue

                red, green, blue = pixels[x, y]
                if _looks_like_face_skin(red, green, blue) and _inside_face_core(x, y, geometry):
                    continue

                bg_distance = _rgb_distance((red, green, blue), background)
                darkness = 1.0 - ((red + green + blue) / (255 * 3))
                saturation = (max(red, green, blue) - min(red, green, blue)) / 255
                chroma_bias = _hair_chroma_score(red, green, blue)
                color_score = max(bg_distance / 96, darkness * 1.18, saturation * 0.72, chroma_bias)

                if bg_distance < 24 and color_score < 0.34:
                    continue

                alpha = int(242 * min(1.0, color_score) * vertical_weight)
                if alpha > 26:
                    mask_pixels[x, y] = alpha
                    accepted_pixels += 1

    mask = ImageChops.subtract(mask, protection_mask.filter(ImageFilter.GaussianBlur(2)))
    mask = _suppress_non_hair_pixels(reference, mask, geometry, background)
    mask = _remove_reference_face_artifacts(mask, geometry)
    mask = _apply_hair_region_prior(mask, geometry)
    mask = _refine_hair_mask(mask)
    mask = _suppress_non_hair_pixels(reference, mask, geometry, background)
    mask = _remove_reference_face_artifacts(mask, geometry)
    mask = _remove_small_mask_islands(mask)
    bbox = mask.getbbox()
    if bbox is None:
        bbox = (
            max(0, int(cx - face_w * 0.76)),
            max(0, int(geometry.forehead_y - face_h * 0.50)),
            min(reference.width, int(cx + face_w * 0.76)),
            min(reference.height, int(geometry.forehead_y + face_h * 0.22)),
        )
        fallback_mask = Image.new("L", reference.size, 0)
        fallback_draw = ImageDraw.Draw(fallback_mask)
        fallback_draw.ellipse(bbox, fill=210)
        mask = fallback_mask.filter(ImageFilter.GaussianBlur(7))

    rgba.putalpha(mask)
    final_bbox = mask.getbbox() or bbox
    anchor = (
        int(geometry.forehead_x - final_bbox[0]),
        int(geometry.forehead_y - final_bbox[1]),
    )
    source_width = max(float(final_bbox[2] - final_bbox[0]), 1.0)
    metadata = {
        "cropBox": {
            "left": final_bbox[0] / reference.width,
            "top": final_bbox[1] / reference.height,
            "right": final_bbox[2] / reference.width,
            "bottom": final_bbox[3] / reference.height,
        },
        "anchor": {
            "x": anchor[0] / max(1, final_bbox[2] - final_bbox[0]),
            "y": anchor[1] / max(1, final_bbox[3] - final_bbox[1]),
        },
        "sourceWidth": source_width,
        "referenceFittingWidth": geometry.fitting_width,
        "acceptedPixelCount": accepted_pixels,
        "backgroundColor": {"r": background[0], "g": background[1], "b": background[2]},
        "protectionMaskApplied": analysis is not None,
        "segmentationSource": segmentation.source if segmentation else "color-landmark-fallback",
        "segmentationConfidenceThreshold": segmentation.confidence_threshold if segmentation else None,
    }
    return HairExtractionResult(
        layer=rgba.crop(final_bbox),
        mask=mask,
        crop_box=final_bbox,
        anchor_point=anchor,
        source_width=source_width,
        metadata=metadata,
    )


def _fit_hair_to_target(
    extraction: HairExtractionResult,
    target_geometry: PortraitGeometry,
    reference_geometry: PortraitGeometry,
    canvas_size: tuple[int, int],
) -> HairPlacementResult:
    canvas_width, canvas_height = canvas_size
    style_width_ratio = extraction.source_width / max(1.0, float(reference_geometry.fitting_width))
    style_width_ratio = max(1.02, min(1.36, style_width_ratio))
    yaw_delta_degrees = reference_geometry.yaw_degrees - target_geometry.yaw_degrees
    crop_center_x = (float(extraction.crop_box[0]) + float(extraction.crop_box[2])) / 2.0
    crop_offset_strength = max(
        -1.0,
        min(1.0, ((crop_center_x - reference_geometry.forehead_x) / max(1.0, reference_geometry.face_width)) * 1.45),
    )
    yaw_strength = max(-1.0, min(1.0, (yaw_delta_degrees / 32.0) + (crop_offset_strength * 0.22)))
    yaw_magnitude = abs(yaw_strength)
    yaw_width_compensation = 1.0 + yaw_magnitude * 0.06
    style_volume_adjustment = 0.92 + ((style_width_ratio - 1.02) / 0.34) * 0.12
    target_width = min(
        canvas_width * 0.66,
        max(
            float(target_geometry.fitting_width) * 1.02,
            float(target_geometry.face_width) * 1.18,
        )
        * style_volume_adjustment
        * yaw_width_compensation,
    )
    max_target_height = min(canvas_height * 0.42, target_geometry.face_height * 0.78)
    width_scale = target_width / max(1.0, extraction.source_width)
    height_scale = max_target_height / max(1.0, extraction.layer.height)
    scale = max(0.24, min(1.05, width_scale, height_scale))
    hair = extraction.layer.copy()
    scaled_size = (
        max(1, int(hair.width * scale)),
        max(1, int(hair.height * scale)),
    )
    hair = hair.resize(scaled_size, Image.Resampling.LANCZOS)
    anchor_x = extraction.anchor_point[0] * scale
    anchor_y = extraction.anchor_point[1] * scale
    if yaw_magnitude >= 0.12:
        stretch_width = int(hair.width * (1.0 + yaw_magnitude * 0.08))
        if stretch_width > hair.width:
            stretch_ratio = stretch_width / hair.width
            hair = hair.resize((stretch_width, hair.height), Image.Resampling.LANCZOS)
            anchor_x *= stretch_ratio
    yaw_warp_metadata = {"enabled": False}
    if yaw_magnitude >= 0.08:
        hair, anchor_x, yaw_warp_metadata = _warp_hair_layer_for_yaw(
            hair,
            anchor_x,
            yaw_strength,
        )

    roll_delta = max(-22.0, min(22.0, target_geometry.roll_degrees - reference_geometry.roll_degrees))
    if abs(roll_delta) >= 1.2:
        original_size = hair.size
        hair = hair.rotate(-roll_delta, resample=Image.Resampling.BICUBIC, expand=True)
        anchor_x, anchor_y = _rotate_point_in_expanded_image(
            anchor_x,
            anchor_y,
            original_size,
            hair.size,
            -roll_delta,
        )

    anchor_y = max(hair.height * 0.52, min(anchor_y, hair.height * 0.72))
    vertical_offset = max(target_geometry.face_height * 0.075, hair.height * 0.07)
    yaw_shift = yaw_strength * target_geometry.face_width * 0.08
    pitch_shift = max(-target_geometry.face_height * 0.035, min(target_geometry.face_height * 0.035, reference_geometry.pitch_degrees * target_geometry.face_height * 0.0022))
    x = int(target_geometry.forehead_x - anchor_x + yaw_shift)
    y = int(target_geometry.forehead_y - anchor_y - vertical_offset + pitch_shift)
    lower_limit = int(target_geometry.eye_center_y + target_geometry.face_height * 0.16)
    bottom_excess = (y + hair.height) - lower_limit
    if bottom_excess > 0:
        y -= int(bottom_excess * 0.58)
    x = max(-hair.width // 6, min(canvas_width - hair.width + hair.width // 6, x))
    y = max(0, min(canvas_height - hair.height, y))

    positioned = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    positioned.alpha_composite(hair, (x, y))
    metadata = {
        "styleWidthRatio": style_width_ratio,
        "scale": scale,
        "targetWidth": target_width,
        "maxTargetHeight": max_target_height,
        "finalHairWidth": hair.width,
        "finalHairHeight": hair.height,
        "x": x,
        "y": y,
        "anchor": {
            "x": anchor_x / max(1, hair.width),
            "y": anchor_y / max(1, hair.height),
        },
        "rollDeltaDegrees": roll_delta,
        "referenceYawDegrees": reference_geometry.yaw_degrees,
        "targetYawDegrees": target_geometry.yaw_degrees,
        "yawDeltaDegrees": yaw_delta_degrees,
        "cropOffsetStrength": crop_offset_strength,
        "styleVolumeAdjustment": style_volume_adjustment,
        "referencePitchDegrees": reference_geometry.pitch_degrees,
        "yawWidthCompensation": yaw_width_compensation,
        "yawShiftPixels": yaw_shift,
        "pitchShiftPixels": pitch_shift,
        "yawWarp": yaw_warp_metadata,
    }
    return HairPlacementResult(image=positioned, metadata=metadata)


def _warp_hair_layer_for_yaw(
    hair: Image.Image,
    anchor_x: float,
    yaw_strength: float,
) -> tuple[Image.Image, float, dict]:
    width, height = hair.size
    if width < 8 or height < 8:
        return hair, anchor_x, {"enabled": False}

    yaw_strength = max(-1.0, min(1.0, yaw_strength))
    max_shift = width * min(0.16, abs(yaw_strength) * 0.13)
    if max_shift < 1.0:
        return hair, anchor_x, {"enabled": False}

    half_width = max(float(anchor_x), float(width) - float(anchor_x), 1.0)

    def source_x_for_destination(destination_x: float) -> float:
        normalized = max(-1.0, min(1.0, (destination_x - anchor_x) / half_width))
        center_weight = max(0.0, 1.0 - abs(normalized))
        side_weight = normalized * abs(normalized)
        shift = (-yaw_strength * max_shift * center_weight) + (yaw_strength * max_shift * 0.28 * side_weight)
        return max(0.0, min(float(width - 1), destination_x + shift))

    segment_count = 18
    mesh = []
    for index in range(segment_count):
        left = int(round(width * index / segment_count))
        right = int(round(width * (index + 1) / segment_count))
        if right <= left:
            continue

        source_left = source_x_for_destination(float(left))
        source_right = source_x_for_destination(float(right))
        if source_right <= source_left:
            source_right = min(float(width - 1), source_left + max(1.0, float(right - left)))
        mesh.append(
            (
                (left, 0, right, height),
                (
                    source_left,
                    0,
                    source_left,
                    height,
                    source_right,
                    height,
                    source_right,
                    0,
                ),
            )
        )

    warped = hair.transform(
        hair.size,
        Image.Transform.MESH,
        mesh,
        Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    metadata = {
        "enabled": True,
        "yawStrength": yaw_strength,
        "maxShiftPixels": max_shift,
        "segmentCount": segment_count,
    }
    return warped, anchor_x, metadata


def _hair_shadow(hair_layer: Image.Image) -> Image.Image:
    alpha = hair_layer.getchannel("A").filter(ImageFilter.GaussianBlur(11))
    shadow = Image.new("RGBA", hair_layer.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda value: int(value * 0.20)))
    return shadow


def _suppress_target_existing_hair(
    target: Image.Image,
    geometry: PortraitGeometry,
    hair_segmenter_model_path: str | None,
    positioned_hair: Image.Image,
) -> TargetHairCleanupResult:
    mask, source, threshold = _build_target_existing_hair_mask(
        target,
        geometry,
        hair_segmenter_model_path,
    )
    coverage_mask = _build_positioned_hair_cleanup_coverage(positioned_hair, geometry)
    coverage_constrained = coverage_mask.getbbox() is not None
    if coverage_constrained:
        mask = ImageChops.multiply(mask, coverage_mask)

    bbox = mask.getbbox()
    if bbox is None:
        return TargetHairCleanupResult(
            image=target,
            mask=mask,
            metadata={
                "enabled": False,
                "source": source,
                "segmentationConfidenceThreshold": threshold,
                "coverageConstrained": coverage_constrained,
                "removedPixelCount": 0,
            },
        )

    fill = _build_target_hair_replacement_fill(target, geometry)
    blend_mask = mask.filter(ImageFilter.GaussianBlur(4)).point(lambda value: min(170, int(value * 0.70)))
    cleaned = Image.composite(fill, target, blend_mask)
    cleaned = Image.composite(cleaned.filter(ImageFilter.SMOOTH_MORE), cleaned, blend_mask.point(lambda value: value // 3))

    return TargetHairCleanupResult(
        image=cleaned,
        mask=mask,
        metadata={
            "enabled": True,
            "source": source,
            "segmentationConfidenceThreshold": threshold,
            "coverageConstrained": coverage_constrained,
            "maskBox": {
                "left": bbox[0] / target.width,
                "top": bbox[1] / target.height,
                "right": bbox[2] / target.width,
                "bottom": bbox[3] / target.height,
            },
            "removedPixelCount": _count_non_zero_mask_pixels(mask),
        },
    )


def _build_positioned_hair_cleanup_coverage(
    positioned_hair: Image.Image,
    geometry: PortraitGeometry,
) -> Image.Image:
    alpha = positioned_hair.getchannel("A")
    if alpha.getbbox() is None:
        return Image.new("L", positioned_hair.size, 0)

    coverage = alpha.point(lambda value: 255 if value > 42 else 0)
    coverage = coverage.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(3))

    prior = Image.new("L", positioned_hair.size, 0)
    draw = ImageDraw.Draw(prior)
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.forehead_x
    draw.ellipse(
        (
            int(cx - face_w * 0.98),
            int(geometry.forehead_y - face_h * 0.70),
            int(cx + face_w * 0.98),
            int(geometry.eye_center_y + face_h * 0.16),
        ),
        fill=255,
    )
    draw.rounded_rectangle(
        (
            int(cx - face_w * 0.82),
            int(geometry.forehead_y - face_h * 0.18),
            int(cx + face_w * 0.82),
            int(geometry.eye_center_y + face_h * 0.18),
        ),
        radius=max(18, face_w // 8),
        fill=230,
    )

    return ImageChops.multiply(coverage, prior.filter(ImageFilter.GaussianBlur(5)))


def _build_target_existing_hair_mask(
    target: Image.Image,
    geometry: PortraitGeometry,
    hair_segmenter_model_path: str | None,
) -> tuple[Image.Image, str, float | None]:
    segmentation = _try_segment_hair(target, hair_segmenter_model_path)
    if segmentation is not None:
        mask = segmentation.mask
        source = segmentation.source
        threshold = segmentation.confidence_threshold
    else:
        mask = _estimated_target_existing_hair_mask(target, geometry)
        source = "geometry-color-fallback"
        threshold = None

    mask = _apply_target_hair_cleanup_prior(mask, geometry)
    mask = _protect_target_face_features(mask, geometry)
    mask = _refine_target_cleanup_mask(mask)
    mask = _protect_target_face_features(mask, geometry)
    return mask, source, threshold


def _estimated_target_existing_hair_mask(target: Image.Image, geometry: PortraitGeometry) -> Image.Image:
    mask = Image.new("L", target.size, 0)
    pixels = target.load()
    mask_pixels = mask.load()
    background = _estimate_background_color(target)
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.forehead_x
    top = max(0, int(geometry.forehead_y - face_h * 0.62))
    bottom = min(target.height, int(geometry.eye_center_y + face_h * 0.10))
    left = max(0, int(cx - face_w * 0.82))
    right = min(target.width, int(cx + face_w * 0.82))
    accepted_pixels = 0

    for y in range(top, bottom):
        for x in range(left, right):
            if not _inside_head_region(x, y, cx, geometry.forehead_y, face_w, face_h):
                continue

            red, green, blue = pixels[x, y]
            if _looks_like_face_skin(red, green, blue) and y > geometry.forehead_y:
                continue

            bg_distance = _rgb_distance((red, green, blue), background)
            darkness = 1.0 - ((red + green + blue) / (255 * 3))
            chroma_bias = _hair_chroma_score(red, green, blue)
            score = max(bg_distance / 110, darkness * 1.24, chroma_bias * 1.20)
            if score < 0.32:
                continue

            mask_pixels[x, y] = min(235, int(score * 235))
            accepted_pixels += 1

    if accepted_pixels < target.width * target.height * 0.002:
        draw = ImageDraw.Draw(mask)
        draw.ellipse(
            (
                int(cx - face_w * 0.72),
                int(geometry.forehead_y - face_h * 0.44),
                int(cx + face_w * 0.72),
                int(geometry.eye_center_y + face_h * 0.04),
            ),
            fill=205,
        )

    return mask


def _apply_target_hair_cleanup_prior(mask: Image.Image, geometry: PortraitGeometry) -> Image.Image:
    prior = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(prior)
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.forehead_x
    top = max(0, int(geometry.forehead_y - face_h * 0.70))
    upper_bottom = min(mask.height, int(geometry.eye_center_y + face_h * 0.10))
    side_bottom = min(mask.height, int(geometry.eye_center_y + face_h * 0.22))

    draw.ellipse(
        (
            int(cx - face_w * 0.92),
            top,
            int(cx + face_w * 0.92),
            upper_bottom,
        ),
        fill=255,
    )
    draw.rounded_rectangle(
        (
            int(cx - face_w * 0.78),
            int(geometry.forehead_y - face_h * 0.24),
            int(cx + face_w * 0.78),
            side_bottom,
        ),
        radius=max(18, face_w // 8),
        fill=230,
    )

    prior = prior.filter(ImageFilter.GaussianBlur(8))
    return ImageChops.multiply(mask, prior)


def _protect_target_face_features(mask: Image.Image, geometry: PortraitGeometry) -> Image.Image:
    protection = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(protection)
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.center_x
    face_bottom = geometry.face_box[3]

    draw.rounded_rectangle(
        (
            int(cx - face_w * 0.36),
            int(geometry.eye_center_y - face_h * 0.03),
            int(cx + face_w * 0.36),
            int(face_bottom + face_h * 0.04),
        ),
        radius=max(18, face_w // 9),
        fill=255,
    )
    draw.ellipse(
        (
            int(cx - face_w * 0.52),
            int(geometry.eye_center_y + face_h * 0.02),
            int(cx + face_w * 0.52),
            int(face_bottom + face_h * 0.02),
        ),
        fill=210,
    )

    return ImageChops.subtract(mask, protection.filter(ImageFilter.GaussianBlur(3)))


def _refine_target_cleanup_mask(mask: Image.Image) -> Image.Image:
    hard = mask.filter(ImageFilter.MedianFilter(5))
    hard = hard.filter(ImageFilter.MaxFilter(7))
    hard = hard.filter(ImageFilter.MinFilter(3))
    soft = hard.filter(ImageFilter.GaussianBlur(5))
    return soft.point(lambda value: 0 if value < 20 else min(245, int(value * 1.05)))


def _build_target_hair_replacement_fill(target: Image.Image, geometry: PortraitGeometry) -> Image.Image:
    skin = _estimate_skin_color(target, geometry)
    background = _estimate_background_color(target)
    fill = Image.new("RGB", target.size, background)
    draw = ImageDraw.Draw(fill)
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.center_x

    draw.ellipse(
        (
            int(cx - face_w * 0.60),
            int(geometry.forehead_y - face_h * 0.10),
            int(cx + face_w * 0.60),
            int(geometry.eye_center_y + face_h * 0.24),
        ),
        fill=skin,
    )
    draw.rounded_rectangle(
        (
            int(cx - face_w * 0.42),
            int(geometry.forehead_y - face_h * 0.04),
            int(cx + face_w * 0.42),
            int(geometry.eye_center_y + face_h * 0.16),
        ),
        radius=max(18, face_w // 10),
        fill=skin,
    )

    return fill.filter(ImageFilter.GaussianBlur(18))


def _estimate_skin_color(image: Image.Image, geometry: PortraitGeometry) -> tuple[int, int, int]:
    pixels = image.load()
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.center_x
    y_top = int(geometry.eye_center_y + face_h * 0.08)
    y_bottom = int(geometry.eye_center_y + face_h * 0.25)
    sample_boxes = [
        (
            int(cx - face_w * 0.34),
            y_top,
            int(cx - face_w * 0.10),
            y_bottom,
        ),
        (
            int(cx + face_w * 0.10),
            y_top,
            int(cx + face_w * 0.34),
            y_bottom,
        ),
        (
            int(cx - face_w * 0.16),
            int(geometry.eye_center_y + face_h * 0.22),
            int(cx + face_w * 0.16),
            int(geometry.eye_center_y + face_h * 0.34),
        ),
    ]
    samples: list[tuple[int, int, int]] = []
    for left, top, right, bottom in sample_boxes:
        for y in range(max(0, top), min(image.height, bottom)):
            for x in range(max(0, left), min(image.width, right)):
                red, green, blue = pixels[x, y]
                if _looks_like_face_skin(red, green, blue):
                    samples.append((red, green, blue))

    if not samples:
        return (232, 205, 188)

    return _median_rgb(samples)


def _median_rgb(samples: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    middle = len(samples) // 2
    return (
        sorted(color[0] for color in samples)[middle],
        sorted(color[1] for color in samples)[middle],
        sorted(color[2] for color in samples)[middle],
    )


def _try_segment_hair(
    image: Image.Image,
    model_path: str | None,
) -> HairSegmentationResult | None:
    try:
        result = segment_hair_with_mediapipe(image, model_path)
    except Exception:
        return None

    if result is None:
        return None

    bbox = result.mask.getbbox()
    if bbox is None:
        return None

    image_area = image.width * image.height
    hair_area = _count_non_zero_mask_pixels(result.mask)
    if hair_area < image_area * 0.006:
        return None

    return result


def _apply_hair_region_prior(mask: Image.Image, geometry: PortraitGeometry) -> Image.Image:
    prior = Image.new("L", mask.size, 0)
    draw = ImageDraw.Draw(prior)
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.forehead_x
    top = max(0, int(geometry.forehead_y - face_h * 0.78))
    upper_bottom = min(mask.height, int(geometry.forehead_y + face_h * 0.35))
    side_bottom = min(mask.height, int(geometry.face_box[3] + face_h * 0.12))

    draw.ellipse(
        (
            int(cx - face_w * 1.05),
            top,
            int(cx + face_w * 1.05),
            upper_bottom,
        ),
        fill=255,
    )
    draw.rounded_rectangle(
        (
            int(cx - face_w * 0.92),
            int(geometry.forehead_y - face_h * 0.22),
            int(cx + face_w * 0.92),
            side_bottom,
        ),
        radius=max(18, face_w // 7),
        fill=220,
    )

    prior = prior.filter(ImageFilter.GaussianBlur(8))
    return ImageChops.multiply(mask, prior)


def _suppress_non_hair_pixels(
    image: Image.Image,
    mask: Image.Image,
    geometry: PortraitGeometry,
    background: tuple[int, int, int],
) -> Image.Image:
    cleaned = mask.copy()
    pixels = image.load()
    mask_pixels = cleaned.load()
    bbox = cleaned.getbbox()
    if bbox is None:
        return cleaned

    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.center_x
    forehead_y = geometry.forehead_y
    eye_y = geometry.eye_center_y
    face_bottom = geometry.face_box[3]
    for y in range(max(0, bbox[1]), min(image.height, bbox[3])):
        for x in range(max(0, bbox[0]), min(image.width, bbox[2])):
            if mask_pixels[x, y] == 0:
                continue

            red, green, blue = pixels[x, y]
            brightness = (red + green + blue) / (255 * 3)
            saturation = (max(red, green, blue) - min(red, green, blue)) / 255
            bg_distance = _rgb_distance((red, green, blue), background)
            central_face = abs(x - cx) <= face_w * 0.58 and forehead_y - face_h * 0.04 <= y <= face_bottom
            lower_face = abs(x - cx) <= face_w * 0.72 and y >= eye_y + face_h * 0.08
            background_like = bg_distance < 52 and brightness > 0.42 and saturation < 0.22
            soft_neutral_patch = brightness > 0.40 and saturation < 0.13 and y >= forehead_y - face_h * 0.05
            lower_neutral_patch = brightness > 0.28 and saturation < 0.18 and y >= eye_y - face_h * 0.04 and abs(x - cx) <= face_w * 0.84

            if (_looks_like_face_skin(red, green, blue) and central_face) or lower_face or background_like or soft_neutral_patch or lower_neutral_patch:
                mask_pixels[x, y] = 0

    return cleaned


def _remove_small_mask_islands(mask: Image.Image) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return mask

    source_pixels = mask.load()
    width, height = mask.size
    visited = bytearray(width * height)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    min_x, min_y, max_x, max_y = bbox

    for start_y in range(min_y, max_y):
        for start_x in range(min_x, max_x):
            offset = start_y * width + start_x
            if visited[offset] or source_pixels[start_x, start_y] <= 18:
                continue

            stack = [(start_x, start_y)]
            visited[offset] = 1
            points: list[tuple[int, int]] = []
            while stack:
                x, y = stack.pop()
                points.append((x, y))
                for neighbor_y in range(max(min_y, y - 1), min(max_y, y + 2)):
                    row_offset = neighbor_y * width
                    for neighbor_x in range(max(min_x, x - 1), min(max_x, x + 2)):
                        neighbor_offset = row_offset + neighbor_x
                        if visited[neighbor_offset] or source_pixels[neighbor_x, neighbor_y] <= 18:
                            continue
                        visited[neighbor_offset] = 1
                        stack.append((neighbor_x, neighbor_y))

            components.append((len(points), points))

    if not components:
        return mask

    largest = max(size for size, _points in components)
    image_area = width * height
    keep_threshold = max(int(largest * 0.06), int(image_area * 0.0008), 160)
    result = Image.new("L", mask.size, 0)
    result_pixels = result.load()
    for size, points in components:
        if size < keep_threshold:
            continue
        for x, y in points:
            result_pixels[x, y] = source_pixels[x, y]

    return result.filter(ImageFilter.GaussianBlur(1))


def _remove_reference_face_artifacts(mask: Image.Image, geometry: PortraitGeometry) -> Image.Image:
    cleaned = mask.copy()
    draw = ImageDraw.Draw(cleaned)
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.center_x

    central_top = int(geometry.eye_center_y - face_h * 0.12)
    central_left = int(cx - face_w * 0.42)
    central_right = int(cx + face_w * 0.42)
    central_bottom = int(geometry.face_box[3] + face_h * 0.10)
    draw.rounded_rectangle(
        (
            max(0, central_left),
            max(0, central_top),
            min(mask.width, central_right),
            min(mask.height, central_bottom),
        ),
        radius=max(18, face_w // 8),
        fill=0,
    )

    lower_top = int(geometry.eye_center_y + face_h * 0.02)
    lower_left = int(cx - face_w * 0.70)
    lower_right = int(cx + face_w * 0.70)
    draw.rounded_rectangle(
        (
            max(0, lower_left),
            max(0, lower_top),
            min(mask.width, lower_right),
            min(mask.height, central_bottom),
        ),
        radius=max(16, face_w // 10),
        fill=0,
    )

    return cleaned


def _count_non_zero_mask_pixels(mask: Image.Image) -> int:
    histogram = mask.convert("L").histogram()
    return sum(histogram[1:])


def _rotate_point_in_expanded_image(
    x: float,
    y: float,
    original_size: tuple[int, int],
    rotated_size: tuple[int, int],
    degrees_value: float,
) -> tuple[float, float]:
    original_width, original_height = original_size
    rotated_width, rotated_height = rotated_size
    center_x = original_width / 2
    center_y = original_height / 2
    theta = radians(degrees_value)
    translated_x = x - center_x
    translated_y = y - center_y
    rotated_x = translated_x * cos(theta) - translated_y * sin(theta)
    rotated_y = translated_x * sin(theta) + translated_y * cos(theta)
    return (
        rotated_x + rotated_width / 2,
        rotated_y + rotated_height / 2,
    )


def _refine_hair_mask(mask: Image.Image) -> Image.Image:
    hard = mask.filter(ImageFilter.MedianFilter(5))
    hard = hard.filter(ImageFilter.MaxFilter(9))
    hard = hard.filter(ImageFilter.MinFilter(3))
    soft = hard.filter(ImageFilter.GaussianBlur(4))
    return soft.point(lambda value: 0 if value < 18 else min(255, int(value * 1.10)))


def _estimate_background_color(image: Image.Image) -> tuple[int, int, int]:
    width, height = image.size
    sample_points = []
    inset_x = max(4, width // 20)
    inset_y = max(4, height // 20)
    for x in (inset_x, width - inset_x - 1):
        for y in (inset_y, height - inset_y - 1):
            sample_points.append(image.getpixel((x, y)))
    return tuple(sum(color[channel] for color in sample_points) // len(sample_points) for channel in range(3))


def _inside_head_region(x: int, y: int, cx: int, face_top: int, face_w: int, face_h: int) -> bool:
    upper_cy = face_top + int(face_h * 0.02)
    rx = face_w * 0.84
    ry = face_h * 0.48
    upper_score = ((x - cx) / rx) ** 2 + ((y - upper_cy) / ry) ** 2
    side_band = abs(x - cx) < face_w * 0.72 and y < face_top + face_h * 0.42
    return upper_score <= 1.12 or side_band


def _inside_face_core(x: int, y: int, geometry: PortraitGeometry) -> bool:
    left, top, right, bottom = geometry.face_box
    cx = (left + right) / 2
    cy = top + geometry.face_height * 0.42
    rx = geometry.face_width * 0.35
    ry = geometry.face_height * 0.38
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


def _vertical_weight(y: int, top: int, bottom: int) -> float:
    if bottom <= top:
        return 1.0
    ratio = (y - top) / (bottom - top)
    if ratio < 0.72:
        return 1.0
    return max(0.22, 1.0 - ((ratio - 0.72) / 0.28) * 0.62)


def _looks_like_face_skin(red: int, green: int, blue: int) -> bool:
    return (
        red > 80
        and green > 45
        and blue > 28
        and red > blue
        and red >= green * 0.92
        and max(red, green, blue) - min(red, green, blue) > 12
    )


def _hair_chroma_score(red: int, green: int, blue: int) -> float:
    warm_brown = max(0, red - blue) / 255 * 0.34
    cool_dark = max(0, blue - red) / 255 * 0.22
    green_suppression = max(0, max(red, blue) - green) / 255 * 0.24
    return warm_brown + cool_dark + green_suppression


def _rgb_distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    return sqrt(sum((first[index] - second[index]) ** 2 for index in range(3)))


def _square_thumbnail(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size))
    result = Image.new("RGB", (size, size), "#EEF2F6")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    result.paste(image, (x, y))
    return result


def _encode_jpeg(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


def _encode_png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue()
