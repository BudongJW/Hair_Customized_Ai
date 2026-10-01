"""Model-free tests for app.hair_alignment.

Run from the ``ai`` directory: ``python -m unittest discover tests``
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import hair_alignment as ha  # noqa: E402
from app.hair_alignment import _tps_map, transfer_hair  # noqa: E402
from app.hair_removal import FACE_OVAL_INDICES, LEFT_EYE_INDICES, HeadSegmentation, push_pull_inpaint  # noqa: E402
from test_hair_removal import BACKGROUND, SKIN, _synthetic_portrait  # noqa: E402


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


class LendingTest(unittest.TestCase):
    """The hair model's photo can show what the user's photo hides (ears, forehead skin)."""

    @classmethod
    def setUpClass(cls):
        image, landmarks, segmentation, points, hair_mask = _synthetic_portrait()
        width, height = image.size
        oval = points[[234, 454]]
        unit = float(oval[1, 0] - oval[0, 0])
        ear_y = float((points[105][1] + points[2][1]) / 2)
        cls.ear_centers = [(oval[0, 0] - 0.07 * unit, ear_y), (oval[1, 0] + 0.07 * unit, ear_y)]

        def ears_mask():
            canvas = Image.new("L", image.size, 0)
            draw = ImageDraw.Draw(canvas)
            for x, y in cls.ear_centers:
                draw.ellipse((x - 0.06 * unit, y - 0.12 * unit, x + 0.06 * unit, y + 0.12 * unit), fill=255)
            return np.asarray(canvas) > 127

        ears = ears_mask()
        oval_canvas = Image.new("L", image.size, 0)
        ImageDraw.Draw(oval_canvas).polygon([tuple(p) for p in points[FACE_OVAL_INDICES]], fill=255)
        face = np.asarray(oval_canvas) > 127

        # Hair model: a short cap that leaves the forehead and both ears visible.
        cap = hair_mask & (np.arange(height)[:, None] < points[10][1] - 0.05 * unit)
        reference = np.asarray(image).copy()
        reference[hair_mask & ~cap] = BACKGROUND
        reference[(face | ears) & ~cap] = SKIN
        cls.reference = Image.fromarray(reference)
        cls.reference_landmarks = landmarks
        cls.reference_segmentation = HeadSegmentation(
            hair=cap.astype(np.float32), face_skin=((face | ears) & ~cap).astype(np.float32), source="synthetic"
        )

        # User: the old hair also hangs over both ears.
        side_hair = np.zeros_like(hair_mask)
        for x, _y in cls.ear_centers:
            side_hair[int(points[105][1]) : int(points[2][1] + 0.1 * unit), int(x - 0.1 * unit) : int(x + 0.1 * unit)] = True
        user_hair = hair_mask | side_hair
        target = np.asarray(image).copy()
        target[user_hair] = (38, 30, 26)
        cls.target = Image.fromarray(target)
        cls.target_landmarks = landmarks
        cls.target_segmentation = HeadSegmentation(
            hair=user_hair.astype(np.float32), face_skin=(face & ~user_hair).astype(np.float32), source="synthetic"
        )
        cls.result = transfer_hair(
            cls.target,
            cls.target_landmarks,
            cls.reference,
            cls.reference_landmarks,
            target_segmentation=cls.target_segmentation,
            reference_segmentation=cls.reference_segmentation,
            inpainter=push_pull_inpaint,
        )

    def test_hidden_ears_are_lent_from_the_hair_model(self):
        self.assertGreater(self.result.metadata["lentEarPixelCount"], 0)
        output = np.asarray(self.result.image, dtype=np.float32)
        for x, y in self.ear_centers:
            pixel = output[int(y), int(x)]
            self.assertLess(float(np.abs(pixel - np.array(SKIN)).max()), 45.0, (x, y, pixel))

    def test_forehead_skin_texture_is_lent(self):
        self.assertGreater(self.result.metadata["lentSkinTexturePixelCount"], 0)
        self.assertTrue(np.isfinite(np.asarray(self.result.image, dtype=np.float32)).all())


class SkinTextureEdgeTest(unittest.TestCase):
    def test_no_dark_rim_where_the_hair_model_shows_no_skin(self):
        # Regression: the feathered weight reached pixels with no sampled skin, where the
        # texture factor came out as 0 and painted a dark rim along the hairline.
        image, landmarks, _segmentation, points, _hair = _synthetic_portrait()
        width, height = image.size
        ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
        warp = ha._Warp(source_x=xs, source_y=ys, info={})
        valid = np.zeros((height, width), dtype=np.float32)
        valid[:, : width // 2] = 1.0
        layer = ha.HairLayer(
            rgb=np.zeros((height, width, 3), dtype=np.float32),
            alpha=np.zeros((height, width), dtype=np.float32),
            points=points,
            skull=None,
            skin_luminance=None,
            skin_texture=valid.copy(),  # a flat texture factor of 1 where valid
            skin_texture_valid=valid,
        )
        frame = ha._face_frame(points)
        canvas = np.full((height, width, 3), 200.0, dtype=np.float32)
        lent, _count = ha._lend_skin_texture(
            canvas, layer, warp, np.ones((height, width), dtype=bool), frame, points, ha._local_grid(frame, width, height), 400.0
        )
        self.assertGreater(float(lent.min()), 199.0)


class WhiteBalanceTest(unittest.TestCase):
    @staticmethod
    def _photo(backdrop):
        rgb = np.zeros((100, 100, 3), dtype=np.float32)
        rgb[:] = backdrop
        segmentation = HeadSegmentation(hair=np.zeros((100, 100), dtype=np.float32), background=np.ones((100, 100), dtype=np.float32))
        return rgb, segmentation

    def test_warm_light_warms_the_hair(self):
        warm, warm_segmentation = self._photo((240, 226, 205))
        neutral, neutral_segmentation = self._photo((228, 228, 228))
        gain = ha._white_balance_gain(warm, warm_segmentation, neutral, neutral_segmentation)
        self.assertGreater(gain[0], 1.0)
        self.assertLess(gain[2], 1.0)

    def test_coloured_or_black_backdrop_leaves_the_hair_alone(self):
        blue_wall, blue_segmentation = self._photo((90, 140, 230))
        black, black_segmentation = self._photo((10, 10, 12))
        neutral, neutral_segmentation = self._photo((228, 228, 228))
        np.testing.assert_allclose(ha._white_balance_gain(blue_wall, blue_segmentation, neutral, neutral_segmentation), 1.0)
        np.testing.assert_allclose(ha._white_balance_gain(black, black_segmentation, neutral, neutral_segmentation), 1.0)


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
