from __future__ import annotations

import json

from fastapi import FastAPI

from .backend_client import BackendClient
from .config import get_settings
from .hair_transfer import (
    SCHEMA_VERSION,
    PreparedHairTransfer,
    compose_texture_hair_transfer,
    prepare_hair_transfer,
    with_reference_hair_artifacts,
)
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
        if settings.fitting_mode in {"prepare", "texture"}:
            prepared = prepare_hair_transfer(
                face_bytes,
                reference_bytes,
                face_landmarker_model_path=settings.face_landmarker_model_path,
                hair_segmenter_model_path=settings.hair_segmenter_model_path,
            )
            if settings.fitting_mode == "texture" and job.get("hairDesignId"):
                prepared = _reuse_saved_hair_design(prepared, job["hairDesignId"], backend, storage)
            if settings.fitting_mode == "prepare":
                return _prepare_fitting_job(
                    request.fitting_job_id, job, prepared, backend, storage
                )
            return _complete_texture_fitting_job(
                request.fitting_job_id, job, prepared, backend, storage
            )
        face_landmarks = _parse_landmarks(profile.get("landmarksJson"))
        fitting_output = compose_hair_fitting(
            face_bytes,
            reference_bytes,
            request.fitting_job_id,
            face_landmarks,
            settings.face_landmarker_model_path,
            settings.hair_segmenter_model_path,
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


def _prepare_fitting_job(job_id, job, prepared, backend, storage) -> JobResponse:
    prefix = f"ai/fitting-jobs/{job_id}"
    keys = {}
    for filename, (content_type, data) in prepared.artifacts().items():
        keys[filename] = storage.upload_bytes(f"{prefix}/{filename}", data, content_type)

    manifest = {
        **prepared.metadata,
        "jobId": job_id,
        "artifacts": keys,
        "sourceObjects": {
            "profileId": job["profileId"],
            "referenceImageObjectKey": job["referenceImageObjectKey"],
        },
    }
    manifest_key = storage.upload_bytes(
        f"{prefix}/preparation.json", json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
        "application/json",
    )
    # Save diagnostics before updating the reusable design so failures remain inspectable.
    diagnostics = {
        "hairMaskObjectKey": keys["hair-mask.png"],
        "hairLayerObjectKey": keys["hair-layer.png"],
        "targetHairMaskObjectKey": keys["target-hair-mask.png"],
        "inpaintingMaskObjectKey": keys["inpainting-mask.png"],
        "faceProtectionMaskObjectKey": keys["face-protection-mask.png"],
        "pipelineManifestObjectKey": manifest_key,
    }
    backend.update_fitting_job(job_id, {"status": "PROCESSING", **diagnostics})
    hair_design_id = job.get("hairDesignId")
    if not hair_design_id:
        design = backend.upsert_hair_design({
            "userId": job["userId"],
            "sourceFittingJobId": job_id,
            "status": "COMPLETED",
            "referenceImageObjectKey": job["referenceImageObjectKey"],
            "hairMaskObjectKey": keys["hair-mask.png"],
            "hairLayerObjectKey": keys["hair-layer.png"],
            "previewImageObjectKey": keys["hair-layer.png"],
            "metadataJson": json.dumps({
                "schemaVersion": prepared.metadata["schemaVersion"],
                "imageSpace": prepared.metadata.get("imageSpace"),
                "referenceGeometry": prepared.metadata.get("referenceGeometry", {}),
                "referenceLandmarks": prepared.metadata.get("referenceLandmarks", []),
                "segmentation": (prepared.metadata.get("segmentation") or {}).get("reference", {}),
            }),
            "failureReason": None,
        })
        hair_design_id = design["id"]
    backend.update_fitting_job(job_id, {
        "status": "PREPARED", **diagnostics, "hairDesignId": hair_design_id, "failureReason": None,
    })
    return JobResponse(accepted=True, aggregate_id=job_id, status="PREPARED")


def _reuse_saved_hair_design(
    prepared: PreparedHairTransfer,
    hair_design_id: str,
    backend: BackendClient,
    storage: S3Storage,
) -> PreparedHairTransfer:
    design = backend.get_hair_design(hair_design_id)
    try:
        design_metadata = json.loads(design.get("metadataJson") or "{}")
    except json.JSONDecodeError:
        design_metadata = {}
    if design_metadata.get("schemaVersion") != SCHEMA_VERSION:
        # Older artifacts removed the face-protection polygon from the hair mask,
        # clipping bangs. Re-extract from the original reference instead.
        return prepared
    layer_key = design.get("hairLayerObjectKey")
    mask_key = design.get("hairMaskObjectKey")
    if not layer_key or not mask_key:
        return prepared
    try:
        return with_reference_hair_artifacts(
            prepared,
            storage.download_bytes(layer_key),
            storage.download_bytes(mask_key),
        )
    except ValueError:
        # Old legacy designs stored a differently cropped layer. Re-segmenting the
        # original reference is safer than stretching an incompatible artifact.
        return prepared


def _complete_texture_fitting_job(
    job_id,
    job,
    prepared: PreparedHairTransfer,
    backend: BackendClient,
    storage: S3Storage,
) -> JobResponse:
    texture_result = compose_texture_hair_transfer(prepared)
    prefix = f"ai/fitting-jobs/{job_id}"
    keys = {}
    artifacts = {**prepared.artifacts(), **texture_result.artifacts()}
    for filename, (content_type, data) in artifacts.items():
        keys[filename] = storage.upload_bytes(f"{prefix}/{filename}", data, content_type)

    manifest = {
        **prepared.metadata,
        "stage": "COMPLETED",
        "jobId": job_id,
        "textureTransfer": texture_result.metadata,
        "artifacts": keys,
        "sourceObjects": {
            "profileId": job["profileId"],
            "referenceImageObjectKey": job["referenceImageObjectKey"],
        },
    }
    manifest_key = storage.upload_bytes(
        f"{prefix}/texture-transfer.json",
        json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
        "application/json",
    )

    hair_design_id = job.get("hairDesignId")
    if not hair_design_id:
        design = backend.upsert_hair_design({
            "userId": job["userId"],
            "sourceFittingJobId": job_id,
            "status": "COMPLETED",
            "referenceImageObjectKey": job["referenceImageObjectKey"],
            "hairMaskObjectKey": keys["hair-mask.png"],
            "hairLayerObjectKey": keys["hair-layer.png"],
            "previewImageObjectKey": keys["hair-layer.png"],
            "metadataJson": json.dumps({
                "schemaVersion": prepared.metadata["schemaVersion"],
                "imageSpace": prepared.metadata.get("imageSpace"),
                "referenceGeometry": prepared.metadata.get("referenceGeometry", {}),
                "referenceLandmarks": prepared.metadata.get("referenceLandmarks", []),
                "segmentation": (prepared.metadata.get("segmentation") or {}).get("reference", {}),
            }, ensure_ascii=False),
            "failureReason": None,
        })
        hair_design_id = design["id"]

    backend.update_fitting_job(job_id, {
        "status": "COMPLETED",
        "resultImageObjectKey": keys["result.png"],
        "hairMaskObjectKey": keys["warped-hair-mask.png"],
        "hairLayerObjectKey": keys["warped-hair-layer.png"],
        "targetHairMaskObjectKey": keys["target-hair-mask.png"],
        "inpaintingMaskObjectKey": keys["edit-mask.png"],
        "faceProtectionMaskObjectKey": keys["face-protection-mask.png"],
        "pipelineManifestObjectKey": manifest_key,
        "hairDesignId": hair_design_id,
        "failureReason": None,
    })
    return JobResponse(accepted=True, aggregate_id=job_id, status="COMPLETED")


def _parse_landmarks(raw_landmarks: str | None) -> dict | None:
    if not raw_landmarks:
        return None

    try:
        payload = json.loads(raw_landmarks)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, dict) else None
