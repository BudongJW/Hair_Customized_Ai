"""AI_FITTING_MODE=skull job completion, with in-memory S3 and backend stand-ins.

Run from the ``ai`` directory: ``python -m unittest discover tests``
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import main  # noqa: E402
from app.hair_alignment import transfer_hair  # noqa: E402
from app.hair_removal import push_pull_inpaint  # noqa: E402
from test_hair_removal import _synthetic_portrait  # noqa: E402


class FakeStorage:
    def __init__(self):
        self.objects = {}

    def upload_bytes(self, object_key, data, content_type="image/jpeg"):
        self.objects[object_key] = (content_type, data)
        return object_key


class FakeBackend:
    def __init__(self):
        self.job_updates = []
        self.designs = []

    def upsert_hair_design(self, payload):
        self.designs.append(payload)
        return {"id": "design-1"}

    def update_fitting_job(self, job_id, payload):
        self.job_updates.append((job_id, payload))


class SkullModeJobTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image, landmarks, segmentation, _points, _hair = _synthetic_portrait()
        cls.transfer = transfer_hair(
            image,
            landmarks,
            image,
            landmarks,
            target_segmentation=segmentation,
            reference_segmentation=segmentation,
            inpainter=push_pull_inpaint,
        )

    def test_uploads_artifacts_and_points_the_job_at_them(self):
        storage, backend = FakeStorage(), FakeBackend()
        job = {"profileId": "profile-1", "userId": "user-1", "referenceImageObjectKey": "uploads/ref.jpg"}
        with mock.patch.object(main, "run_skull_transfer", return_value=self.transfer):
            response = main._complete_skull_fitting_job("job-1", job, b"face", b"ref", None, backend, storage)

        self.assertEqual(response.status, "COMPLETED")
        job_id, update = backend.job_updates[-1]
        self.assertEqual(job_id, "job-1")
        self.assertEqual(update["status"], "COMPLETED")
        self.assertEqual(update["hairDesignId"], "design-1")
        for field in (
            "resultImageObjectKey",
            "hairMaskObjectKey",
            "hairLayerObjectKey",
            "targetHairMaskObjectKey",
            "inpaintingMaskObjectKey",
            "faceProtectionMaskObjectKey",
            "pipelineManifestObjectKey",
        ):
            self.assertIn(update[field], storage.objects, field)
        # The result screen loads the bald canvas from this fixed key.
        self.assertIn("ai/fitting-jobs/job-1/bald-canvas.png", storage.objects)
        manifest = json.loads(storage.objects[update["pipelineManifestObjectKey"]][1])
        self.assertEqual(manifest["jobId"], "job-1")

    def test_existing_hair_design_is_kept(self):
        storage, backend = FakeStorage(), FakeBackend()
        job = {"profileId": "p", "userId": "u", "referenceImageObjectKey": "r.jpg", "hairDesignId": "saved-design"}
        with mock.patch.object(main, "run_skull_transfer", return_value=self.transfer):
            main._complete_skull_fitting_job("job-2", job, b"face", b"ref", None, backend, storage)
        self.assertEqual(backend.designs, [])
        self.assertEqual(backend.job_updates[-1][1]["hairDesignId"], "saved-design")


if __name__ == "__main__":
    unittest.main()
