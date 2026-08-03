import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FACE_LANDMARKER_MODEL_PATH = AI_ROOT / "models" / "face_landmarker.task"


@dataclass(frozen=True)
class Settings:
    aws_region: str
    bucket: str
    default_backend_base_url: str
    face_landmarker_model_path: str


def get_settings() -> Settings:
    return Settings(
        aws_region=os.getenv("AWS_REGION", "ap-northeast-2"),
        bucket=os.getenv("APP_STORAGE_BUCKET", "hair-customized-ai-aaron-dev"),
        default_backend_base_url=os.getenv("AI_WORKER_BACKEND_BASE_URL", "http://localhost:8080"),
        face_landmarker_model_path=os.getenv(
            "FACE_LANDMARKER_MODEL_PATH",
            str(DEFAULT_FACE_LANDMARKER_MODEL_PATH),
        ),
    )
