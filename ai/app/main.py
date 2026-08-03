from __future__ import annotations

import json

from fastapi import FastAPI

from .backend_client import BackendClient
from .config import get_settings
from .image_processing import analyze_face_image, compose_hair_fitting, landmarks_json
from .s3_storage import S3Storage
from .schemas import FaceProfileJobRequest, HairFittingJobRequest, JobResponse

app = FastAPI(title="Hair Customized AI Worker")


@app.get("/health")
def health() -> dict:
    return {"status": "UP"}


@app.post("/api/v1/jobs/face-profile-preprocessing", response_model=JobResponse)
def preprocess_face_profile(request: FaceProfileJobRequest) -> JobResponse:
    settings = get_settings()
    backend = BackendClient(request.backend_base_url or settings.default_backend_base_url)
    storage = S3Storage(settings.bucket, settings.aws_region)

    try:
        backend.update_face_profile(request.profile_id, {"status": "PROCESSING"})
        profile = backend.get_face_profile(request.profile_id)
        source_bytes = storage.download_bytes(profile["originalImageObjectKey"])
        canvas_bytes, landmarks = analyze_face_image(
            source_bytes,
            request.profile_id,
            settings.face_landmarker_model_path,
        )
        pose = landmarks.get("pose") or {}

        bald_canvas_key = f"ai/face-profiles/{request.profile_id}/bald-canvas.jpg"
        storage.upload_bytes(bald_canvas_key, canvas_bytes)

        backend.update_face_profile(
            request.profile_id,
            {
                "status": "COMPLETED",
                "baldCanvasObjectKey": bald_canvas_key,
                "landmarksJson": landmarks_json(landmarks),
                "yawDegrees": pose.get("yawDegrees", 0.0),
                "pitchDegrees": pose.get("pitchDegrees", 0.0),
                "rollDegrees": pose.get("rollDegrees", 0.0),
                "failureReason": None,
            },
        )
        return JobResponse(accepted=True, aggregate_id=request.profile_id, status="COMPLETED")
    except Exception as exc:
        backend.update_face_profile(
            request.profile_id,
            {
                "status": "FAILED",
                "failureReason": str(exc)[:1000],
            },
        )
        return JobResponse(accepted=True, aggregate_id=request.profile_id, status="FAILED")


@app.post("/api/v1/jobs/hair-fitting", response_model=JobResponse)
def generate_hair_fitting(request: HairFittingJobRequest) -> JobResponse:
    settings = get_settings()
    backend = BackendClient(request.backend_base_url or settings.default_backend_base_url)
    storage = S3Storage(settings.bucket, settings.aws_region)

    try:
        backend.update_fitting_job(request.fitting_job_id, {"status": "PROCESSING"})
        job = backend.get_fitting_job(request.fitting_job_id)
        profile = backend.get_face_profile(job["profileId"])

        face_key = profile["originalImageObjectKey"]
        face_bytes = storage.download_bytes(face_key)
        reference_bytes = storage.download_bytes(job["referenceImageObjectKey"])
        face_landmarks = _parse_landmarks(profile.get("landmarksJson"))
        fitting_output = compose_hair_fitting(
            face_bytes,
            reference_bytes,
            request.fitting_job_id,
            face_landmarks,
            settings.face_landmarker_model_path,
        )

        result_key = f"ai/fitting-jobs/{request.fitting_job_id}/result.jpg"
        hair_mask_key = f"ai/fitting-jobs/{request.fitting_job_id}/hair-mask.png"
        hair_layer_key = f"ai/fitting-jobs/{request.fitting_job_id}/hair-layer.png"
        storage.upload_bytes(result_key, fitting_output.result_bytes)
        storage.upload_bytes(hair_mask_key, fitting_output.hair_mask_bytes, "image/png")
        storage.upload_bytes(hair_layer_key, fitting_output.hair_layer_bytes, "image/png")

        hair_design_id = job.get("hairDesignId")
        if not hair_design_id:
            hair_design = backend.upsert_hair_design(
                {
                    "userId": job["userId"],
                    "sourceFittingJobId": request.fitting_job_id,
                    "status": "COMPLETED",
                    "referenceImageObjectKey": job["referenceImageObjectKey"],
                    "hairMaskObjectKey": hair_mask_key,
                    "hairLayerObjectKey": hair_layer_key,
                    "previewImageObjectKey": hair_layer_key,
                    "metadataJson": json.dumps(fitting_output.metadata, ensure_ascii=False),
                    "failureReason": None,
                }
            )
            hair_design_id = hair_design["id"]

        backend.update_fitting_job(
            request.fitting_job_id,
            {
                "status": "COMPLETED",
                "resultImageObjectKey": result_key,
                "hairMaskObjectKey": hair_mask_key,
                "hairLayerObjectKey": hair_layer_key,
                "hairDesignId": hair_design_id,
                "failureReason": None,
            },
        )
        return JobResponse(accepted=True, aggregate_id=request.fitting_job_id, status="COMPLETED")
    except Exception as exc:
        try:
            job = backend.get_fitting_job(request.fitting_job_id)
            if not job.get("hairDesignId"):
                backend.upsert_hair_design(
                    {
                        "userId": job["userId"],
                        "sourceFittingJobId": request.fitting_job_id,
                        "status": "FAILED",
                        "referenceImageObjectKey": job["referenceImageObjectKey"],
                        "failureReason": str(exc)[:1000],
                    }
                )
        except Exception:
            pass
        backend.update_fitting_job(
            request.fitting_job_id,
            {
                "status": "FAILED",
                "failureReason": str(exc)[:1000],
            },
        )
        return JobResponse(accepted=True, aggregate_id=request.fitting_job_id, status="FAILED")


def _parse_landmarks(raw_landmarks: str | None) -> dict | None:
    if not raw_landmarks:
        return None

    try:
        payload = json.loads(raw_landmarks)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None
