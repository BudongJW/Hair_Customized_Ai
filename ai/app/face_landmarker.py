from __future__ import annotations

import os
from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path

import numpy as np
from PIL import Image

MEDIA_PIPE_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
os.environ.setdefault("MPLCONFIGDIR", str(MEDIA_PIPE_CACHE_DIR))

KEY_LANDMARK_INDICES = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_eyebrow_outer": 70,
    "right_eyebrow_outer": 300,
    "mouth_left": 61,
    "mouth_right": 291,
    "forehead_center": 10,
    "left_face_edge": 234,
    "right_face_edge": 454,
}


@dataclass(frozen=True)
class MediaPipeFaceAnalysis:
    source: str
    face_box: dict
    landmarks: list[dict]
    key_landmarks: dict
    roll_degrees: float
    yaw_degrees: float
    pitch_degrees: float
    confidence: float
    transform_matrix: list[list[float]] | None


def analyze_with_mediapipe(image: Image.Image, model_path: str) -> MediaPipeFaceAnalysis | None:
    model = Path(model_path)
    if not model.exists():
        return None

    MEDIA_PIPE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    import mediapipe as mp

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        output_facial_transformation_matrixes=True,
    )

    rgb_image = image.convert("RGB")
    image_array = np.asarray(rgb_image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_array)

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    face_landmarks = result.face_landmarks[0]
    landmarks = [
        {
            "index": index,
            "x": _clamp01(landmark.x),
            "y": _clamp01(landmark.y),
            "z": float(landmark.z),
        }
        for index, landmark in enumerate(face_landmarks)
    ]
    key_landmarks = {
        name: landmarks[index]
        for name, index in KEY_LANDMARK_INDICES.items()
        if index < len(landmarks)
    }

    face_box = _face_box_from_landmarks(landmarks)
    roll = _estimate_roll(key_landmarks)
    yaw = _estimate_yaw(key_landmarks)
    pitch = _estimate_pitch(key_landmarks)
    transform_matrix = _first_transform_matrix(result)

    return MediaPipeFaceAnalysis(
        source="mediapipe-face-landmarker",
        face_box=face_box,
        landmarks=landmarks,
        key_landmarks=key_landmarks,
        roll_degrees=roll,
        yaw_degrees=yaw,
        pitch_degrees=pitch,
        confidence=1.0,
        transform_matrix=transform_matrix,
    )


def _face_box_from_landmarks(landmarks: list[dict]) -> dict:
    xs = [point["x"] for point in landmarks]
    ys = [point["y"] for point in landmarks]
    return {
        "left": max(0.0, min(xs)),
        "top": max(0.0, min(ys)),
        "right": min(1.0, max(xs)),
        "bottom": min(1.0, max(ys)),
    }


def _estimate_roll(key_landmarks: dict) -> float:
    left_eye = key_landmarks.get("left_eye_outer")
    right_eye = key_landmarks.get("right_eye_outer")
    if not left_eye or not right_eye:
        return 0.0
    return degrees(atan2(right_eye["y"] - left_eye["y"], right_eye["x"] - left_eye["x"]))


def _estimate_yaw(key_landmarks: dict) -> float:
    left_edge = key_landmarks.get("left_face_edge")
    right_edge = key_landmarks.get("right_face_edge")
    nose = key_landmarks.get("nose_tip")
    if not left_edge or not right_edge or not nose:
        return 0.0

    left_distance = max(0.001, nose["x"] - left_edge["x"])
    right_distance = max(0.001, right_edge["x"] - nose["x"])
    asymmetry = (left_distance - right_distance) / (left_distance + right_distance)
    return max(-35.0, min(35.0, asymmetry * 42.0))


def _estimate_pitch(key_landmarks: dict) -> float:
    forehead = key_landmarks.get("forehead_center")
    chin = key_landmarks.get("chin")
    nose = key_landmarks.get("nose_tip")
    if not forehead or not chin or not nose:
        return 0.0

    face_height = max(0.001, chin["y"] - forehead["y"])
    nose_ratio = (nose["y"] - forehead["y"]) / face_height
    return max(-25.0, min(25.0, (nose_ratio - 0.47) * 60.0))


def _first_transform_matrix(result) -> list[list[float]] | None:
    matrices = getattr(result, "facial_transformation_matrixes", None)
    if not matrices:
        return None

    matrix = matrices[0]
    if hasattr(matrix, "tolist"):
        return matrix.tolist()
    return [[float(value) for value in row] for row in matrix]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
