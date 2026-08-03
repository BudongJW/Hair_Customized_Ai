from __future__ import annotations

from dataclasses import dataclass
import json
from io import BytesIO
from math import sqrt

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

from .face_landmarker import MediaPipeFaceAnalysis, analyze_with_mediapipe

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
    roll_degrees: float = 0.0
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


@dataclass(frozen=True)
class HairExtractionResult:
    layer: Image.Image
    mask: Image.Image
    crop_box: tuple[int, int, int, int]
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
) -> HairFittingOutput:
    target = _portrait_cover(_open_image(face_bytes), PORTRAIT_SIZE)
    reference = _portrait_cover(_open_image(reference_bytes), PORTRAIT_SIZE)

    target_geometry = _geometry_from_landmarks_payload(target.size, face_landmarks) or _estimate_geometry(target)
    reference_analysis = _try_analyze_with_mediapipe(reference, model_path)
    reference_geometry = _geometry_from_analysis(reference.size, reference_analysis) if reference_analysis else _estimate_geometry(reference)
    extraction = _extract_reference_hair(reference, reference_geometry, reference_analysis)
    positioned_hair = _fit_hair_to_target(
        extraction.layer,
        target_geometry,
        target.size,
        source_roll_degrees=reference_geometry.roll_degrees,
    )

    result = target.convert("RGBA")
    shadow = _hair_shadow(positioned_hair)
    result.alpha_composite(shadow)
    result.alpha_composite(positioned_hair)

    draw = ImageDraw.Draw(result, "RGBA")
    draw.rectangle((0, result.height - 46, result.width, result.height), fill=(15, 23, 42, 168))
    draw.text((24, result.height - 31), f"Hair fitting result {job_id}", fill=(255, 255, 255, 230))

    metadata = {
        "version": "hair-mask-v3-landmark-protection",
        "targetGeometry": _geometry_metadata(target_geometry),
        "referenceGeometry": _geometry_metadata(reference_geometry),
        "hairExtraction": extraction.metadata,
        "rollDeltaDegrees": target_geometry.roll_degrees - reference_geometry.roll_degrees,
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

    return PortraitGeometry(
        width=width,
        height=height,
        face_box=face_box,
        forehead_point=forehead_point,
        roll_degrees=analysis.roll_degrees,
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

    return PortraitGeometry(
        width=width,
        height=height,
        face_box=_normalized_box_to_pixels(face_box, width, height),
        forehead_point=forehead_point,
        roll_degrees=float(pose.get("rollDegrees") or 0.0),
        source=str(payload.get("source") or "stored-landmarks"),
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
        "rollDegrees": geometry.roll_degrees,
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
) -> HairExtractionResult:
    rgba = reference.convert("RGBA")
    mask = Image.new("L", reference.size, 0)
    pixels = reference.load()
    mask_pixels = mask.load()
    background = _estimate_background_color(reference)
    protection_mask = _build_reference_protection_mask(reference.size, geometry, analysis)
    protection_pixels = protection_mask.load()

    left, top, right, bottom = geometry.face_box
    face_w = geometry.face_width
    face_h = geometry.face_height
    cx = geometry.forehead_x
    head_top = max(0, int(geometry.forehead_y - face_h * 0.58))
    hair_bottom = min(reference.height, int(top + face_h * 0.46))
    hair_left = max(0, int(cx - face_w * 0.90))
    hair_right = min(reference.width, int(cx + face_w * 0.90))
    accepted_pixels = 0

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
    mask = _refine_hair_mask(mask)
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
    metadata = {
        "cropBox": {
            "left": final_bbox[0] / reference.width,
            "top": final_bbox[1] / reference.height,
            "right": final_bbox[2] / reference.width,
            "bottom": final_bbox[3] / reference.height,
        },
        "acceptedPixelCount": accepted_pixels,
        "backgroundColor": {"r": background[0], "g": background[1], "b": background[2]},
        "protectionMaskApplied": analysis is not None,
    }
    return HairExtractionResult(
        layer=rgba.crop(final_bbox),
        mask=mask,
        crop_box=final_bbox,
        metadata=metadata,
    )


def _fit_hair_to_target(
    hair_layer: Image.Image,
    geometry: PortraitGeometry,
    canvas_size: tuple[int, int],
    source_roll_degrees: float = 0.0,
) -> Image.Image:
    canvas_width, canvas_height = canvas_size
    target_width = int(geometry.face_width * 1.62)
    target_height = int(geometry.face_height * 0.72)

    hair = hair_layer.copy()
    hair.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
    if hair.width < target_width * 0.82:
        scale = target_width / hair.width
        hair = hair.resize((target_width, int(hair.height * scale)), Image.Resampling.LANCZOS)

    roll_delta = max(-22.0, min(22.0, geometry.roll_degrees - source_roll_degrees))
    if abs(roll_delta) >= 1.2:
        hair = hair.rotate(-roll_delta, resample=Image.Resampling.BICUBIC, expand=True)

    x = int(geometry.forehead_x - hair.width / 2)
    y = int(geometry.forehead_y - hair.height * 0.62)
    x = max(-hair.width // 6, min(canvas_width - hair.width + hair.width // 6, x))
    y = max(0, min(canvas_height - hair.height, y))

    positioned = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    positioned.alpha_composite(hair, (x, y))
    return positioned


def _hair_shadow(hair_layer: Image.Image) -> Image.Image:
    alpha = hair_layer.getchannel("A").filter(ImageFilter.GaussianBlur(11))
    shadow = Image.new("RGBA", hair_layer.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda value: int(value * 0.20)))
    return shadow


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
