import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACE_LANDMARKER_MODEL_PATH = AI_ROOT / "models" / "face_landmarker.task"
DEFAULT_HAIR_SEGMENTER_MODEL_PATH = AI_ROOT / "models" / "hair_segmenter.tflite"
DEFAULT_SELFIE_MULTICLASS_MODEL_PATH = AI_ROOT / "models" / "selfie_multiclass_256x256.tflite"
DEFAULT_LAMA_MODEL_PATH = AI_ROOT / "models" / "lama_fp32.onnx"


@dataclass(frozen=True)
class Settings:
    aws_region: str
    bucket: str
    default_backend_base_url: str
    face_landmarker_model_path: str
    hair_segmenter_model_path: str
    fitting_mode: str = "texture"
    selfie_multiclass_model_path: str = str(DEFAULT_SELFIE_MULTICLASS_MODEL_PATH)
    lama_model_path: str = str(DEFAULT_LAMA_MODEL_PATH)


def get_settings() -> Settings:
    fitting_mode = os.getenv("AI_FITTING_MODE", "texture").strip().lower()
    if fitting_mode not in {"texture", "prepare", "legacy"}:
        raise ValueError("AI_FITTING_MODE must be texture, prepare, or legacy")
    return Settings(
        aws_region=os.getenv("AWS_REGION", "ap-northeast-2"),
        bucket=os.getenv("APP_STORAGE_BUCKET", "hair-customized-ai-aaron-dev"),
        default_backend_base_url=os.getenv("AI_WORKER_BACKEND_BASE_URL", "http://localhost:8080"),
        face_landmarker_model_path=os.getenv(
            "FACE_LANDMARKER_MODEL_PATH",
            str(DEFAULT_FACE_LANDMARKER_MODEL_PATH),
        ),
        hair_segmenter_model_path=os.getenv(
            "HAIR_SEGMENTER_MODEL_PATH",
            str(DEFAULT_HAIR_SEGMENTER_MODEL_PATH),
        ),
        fitting_mode=fitting_mode,
        selfie_multiclass_model_path=os.getenv(
            "SELFIE_MULTICLASS_MODEL_PATH",
            str(DEFAULT_SELFIE_MULTICLASS_MODEL_PATH),
        ),
        lama_model_path=os.getenv("LAMA_MODEL_PATH", str(DEFAULT_LAMA_MODEL_PATH)),
    )
