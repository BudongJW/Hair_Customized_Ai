"""Model-free tests for app.hair_alignment.

Run from the ``ai`` directory: ``python -m unittest discover tests``
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.hair_alignment import _tps_map, transfer_hair  # noqa: E402
from app.hair_removal import LEFT_EYE_INDICES, HeadSegmentation, push_pull_inpaint  # noqa: E402
from test_hair_removal import BACKGROUND, _synthetic_portrait  # noqa: E402


def _scaled(image, landmarks, segmentation, factor):
    """The same synthetic person, shrunk around the image centre (a farther-away photo)."""

    width, height = image.size
    small_size = (int(width * factor), int(height * factor))
    offset = ((width - small_size[0]) // 2, (height - small_size[1]) // 2)

    scaled_image = Image.new("RGB", image.size, BACKGROUND)
    scaled_image.paste(image.resize(small_size, Image.Resampling.LANCZOS), offset)

    def scale_probability(probability):
        canvas = Image.new("L", image.size, 0)
        small = Image.fromarray((probability * 255).astype(np.uint8)).resize(small_size, Image.Resampling.BILINEAR)
        canvas.paste(small, offset)
        return np.asarray(canvas, dtype=np.float32) / 255.0

    scaled_landmarks = [
        [(x * width * factor + offset[0]) / width, (y * height * factor + offset[1]) / height] for x, y in landmarks
    ]
    scaled_segmentation = HeadSegmentation(
        hair=scale_probability(segmentation.hair),
        face_skin=scale_probability(segmentation.face_skin),
        source="synthetic-scaled",
    )
    return scaled_image, scaled_landmarks, scaled_segmentation


def _iou(first, second):
    return float((first & second).sum()) / float(max((first | second).sum(), 1))


class TransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image, cls.landmarks, cls.segmentation, cls.points, cls.hair_mask = _synthetic_portrait()

    def _transfer(self, reference, reference_landmarks, reference_segmentation):
        return transfer_hair(
            self.image,
            self.landmarks,
            reference,
            reference_landmarks,
            target_segmentation=self.segmentation,
            reference_segmentation=reference_segmentation,
            inpainter=push_pull_inpaint,
        )

    def test_same_photo_puts_the_hair_back_in_place(self):
        result = self._transfer(self.image, self.landmarks, self.segmentation)
        warped = np.asarray(result.warped_hair.getchannel("A")) > 127
        self.assertGreater(_iou(warped, self.hair_mask), 0.9)
        self.assertEqual(result.metadata["warp"]["method"], "thin-plate-spline")

    def test_hair_from_a_smaller_photo_is_scaled_onto_the_head(self):
        reference, landmarks, segmentation = _scaled(self.image, self.landmarks, self.segmentation, 0.75)
        result = self._transfer(reference, landmarks, segmentation)
        warped = np.asarray(result.warped_hair.getchannel("A")) > 127
        self.assertGreater(_iou(warped, self.hair_mask), 0.85)
        self.assertAlmostEqual(result.metadata["warp"]["scale"], 1 / 0.75, delta=0.05)

    def test_result_keeps_eyes_and_has_hair_where_the_hair_was(self):
        result = self._transfer(self.image, self.landmarks, self.segmentation)
        output = np.asarray(result.image, dtype=np.float32)
        original = np.asarray(self.image, dtype=np.float32)
        x, y = self.points[LEFT_EYE_INDICES].mean(axis=0).astype(int)
        np.testing.assert_allclose(output[y - 3 : y + 3, x - 3 : x + 3], original[y - 3 : y + 3, x - 3 : x + 3], atol=1.0)
        luminance = output @ np.array([0.299, 0.587, 0.114])
        self.assertLess(float(np.median(luminance[self.hair_mask])), 80.0)

    def test_warns_when_reference_hair_runs_off_the_photo(self):
        cropped = self._transfer(self.image, self.landmarks, self.segmentation)
        self.assertIn("REFERENCE_HAIR_CROPPED_TOP", cropped.metadata["warnings"])

        reference, landmarks, segmentation = _scaled(self.image, self.landmarks, self.segmentation, 0.75)
        whole = self._transfer(reference, landmarks, segmentation)
        self.assertEqual(whole.metadata["warnings"], [])

    def test_artifacts_include_result_and_bald_canvas(self):
        result = self._transfer(self.image, self.landmarks, self.segmentation)
        artifacts = result.artifacts()
        for name in ("result.png", "warped-hair-layer.png", "warped-hair-mask.png", "bald-canvas.png"):
            self.assertIn(name, artifacts)
            self.assertTrue(artifacts[name][1].startswith(b"\x89PNG"), name)


class ThinPlateSplineTest(unittest.TestCase):
    def test_interpolates_control_points(self):
        rng = np.random.default_rng(1)
        centers = rng.uniform(-1, 1, size=(12, 2))
        values = centers * 1.7 + np.array([3.0, -2.0]) + rng.normal(0, 0.05, size=(12, 2))
        mapped = _tps_map(centers, values, centers)
        np.testing.assert_allclose(mapped, values, atol=0.02)

    def test_reproduces_an_affine_map_exactly(self):
        centers = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.3], [0.2, 0.8]], dtype=np.float64)
        matrix = np.array([[1.2, 0.1], [-0.2, 0.9]])
        values = centers @ matrix.T + np.array([5.0, 7.0])
        queries = np.array([[2.0, -1.0], [0.3, 0.3]])
        np.testing.assert_allclose(_tps_map(centers, values, queries), queries @ matrix.T + np.array([5.0, 7.0]), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
