"""Model-free tests for app.generative_refine: the diffusion model is replaced by a stand-in.

Run from the ``ai`` directory: ``python -m unittest discover tests``
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import config, skull_fitting  # noqa: E402
from app.generative_refine import HAIRLINE_BAND, DiffusionInpainter, refine_transfer, refinement_mask  # noqa: E402
from app.hair_alignment import transfer_hair  # noqa: E402
from app.hair_removal import (  # noqa: E402
    FACE_OVAL_INDICES,
    LEFT_BROW_INDICES,
    LEFT_EYE_INDICES,
    MOUTH_INDICES,
    RIGHT_BROW_INDICES,
    RIGHT_EYE_INDICES,
    HeadSegmentation,
    _dilate,
    push_pull_inpaint,
)
from test_hair_removal import BACKGROUND, SKIN, _synthetic_portrait  # noqa: E402

MAGENTA = (255, 0, 255)


class RecordingGenerator:
    """Paints the masked pixels magenta and remembers what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, crop, mask, seed):
        self.calls.append((crop.size, mask.size, seed))
        painted = np.asarray(crop, dtype=np.uint8).copy()
        painted[np.asarray(mask) > 127] = MAGENTA
        return Image.fromarray(painted)


def _short_cap_transfer():
    """The mushroom-haired user from the removal tests gets a short cap that shows the forehead."""

    image, landmarks, segmentation, points, hair_mask = _synthetic_portrait()
    height = image.height
    oval = points[FACE_OVAL_INDICES]
    unit = float(oval[:, 0].max() - oval[:, 0].min())
    cap = hair_mask & (np.arange(height)[:, None] < points[10][1] - 0.05 * unit)
    face_canvas = Image.new("L", image.size, 0)
    ImageDraw.Draw(face_canvas).polygon([tuple(p) for p in oval], fill=255)
    face = np.asarray(face_canvas) > 127

    reference = np.asarray(image).copy()
    reference[hair_mask & ~cap] = BACKGROUND
    reference[face & ~cap] = SKIN
    result = transfer_hair(
        image,
        landmarks,
        Image.fromarray(reference),
        landmarks,
        target_segmentation=segmentation,
        reference_segmentation=HeadSegmentation(
            hair=cap.astype(np.float32), face_skin=(face & ~cap).astype(np.float32), source="synthetic"
        ),
        inpainter=push_pull_inpaint,
    )
    return result, landmarks, points, unit


class RefinementMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.landmarks, cls.points, cls.unit = _short_cap_transfer()
        cls.region, cls.mask_unit = refinement_mask(cls.result)

    def test_covers_the_drawn_forehead(self):
        alpha = np.asarray(self.result.warped_hair.getchannel("A"))
        scalp = np.asarray(self.result.bald.scalp_fill_mask) > 127
        drawn = scalp & (alpha < 64)
        self.assertGreater(drawn.sum(), 500, "fixture should draw forehead skin under the short cap")
        self.assertGreater(float(self.region[drawn].mean()), 0.9)
        self.assertAlmostEqual(self.mask_unit, self.unit, delta=0.05 * self.unit)

    def test_never_covers_eyes_or_mouth(self):
        for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES):
            x, y = self.points[indices].mean(axis=0).astype(int)
            self.assertFalse(self.region[y - 4 : y + 4, x - 4 : x + 4].any(), indices[:3])

    def test_protects_eyes_and_mouth_even_inside_drawn_skin(self):
        everywhere = Image.new("L", self.result.image.size, 255)
        result = replace(self.result, bald=replace(self.result.bald, scalp_fill_mask=everywhere, removal_mask=everywhere))
        region, _unit = refinement_mask(result)
        for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES, MOUTH_INDICES):
            x, y = self.points[indices].mean(axis=0).astype(int)
            self.assertFalse(region[y - 4 : y + 4, x - 4 : x + 4].any(), indices[:3])

    def test_leaves_the_backdrop_above_the_head_to_lama(self):
        removal = np.asarray(self.result.bald.removal_mask) > 127
        skull = np.asarray(self.result.bald.skull_mask) > 127
        brow_y = int(self.points[LEFT_BROW_INDICES + RIGHT_BROW_INDICES][:, 1].min())
        above = removal & ~skull
        above[brow_y:] = False
        self.assertGreater(above.sum(), 1000, "fixture should remove hair volume around the top of the head")
        hair_edge = _dilate(np.asarray(self.result.warped_hair.getchannel("A")) > 127, radius=HAIRLINE_BAND * self.unit + 2)
        self.assertFalse((self.region & above & ~hair_edge).any())

    def test_leaves_the_middle_of_the_new_hair_alone(self):
        alpha = np.asarray(self.result.warped_hair.getchannel("A")) > 250
        deep = ~_dilate(~alpha, radius=0.06 * self.unit)
        self.assertGreater(deep.sum(), 200)
        self.assertFalse((self.region & deep).any())


class RefineTransferTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.landmarks, cls.points, cls.unit = _short_cap_transfer()
        cls.generator = RecordingGenerator()
        cls.refined = refine_transfer(cls.result, cls.generator, preset="fast", max_side=256)
        cls.region, _unit = refinement_mask(cls.result)

    def test_repaints_only_the_region(self):
        before = np.asarray(self.result.image, dtype=np.float32)
        after = np.asarray(self.refined.image, dtype=np.float32)
        far = ~_dilate(self.region, radius=0.05 * self.unit)
        np.testing.assert_array_equal(after[far], before[far])
        core = ~_dilate(~self.region, radius=0.03 * self.unit)
        self.assertGreater(core.sum(), 200)
        distance = np.abs(after[core] - np.array(MAGENTA)).max(axis=1)
        self.assertLess(float(np.median(distance)), 40.0)

    def test_eyes_stay_put(self):
        before = np.asarray(self.result.image, dtype=np.float32)
        after = np.asarray(self.refined.image, dtype=np.float32)
        for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
            x, y = self.points[indices].mean(axis=0).astype(int)
            np.testing.assert_array_equal(after[y - 3 : y + 3, x - 3 : x + 3], before[y - 3 : y + 3, x - 3 : x + 3])

    def test_generator_gets_a_small_crop_in_multiples_of_8(self):
        (crop_size, mask_size, _seed), = self.generator.calls
        self.assertEqual(crop_size, mask_size)
        self.assertLessEqual(max(crop_size), 256)
        self.assertTrue(all(side % 8 == 0 for side in crop_size), crop_size)

    def test_keeps_the_unrefined_result_and_reports_the_step(self):
        self.assertIs(self.refined.unrefined_image, self.result.image)
        self.assertEqual(self.refined.metadata["generativeRefine"]["preset"], "fast")
        self.assertEqual(self.refined.metadata["generativeRefine"]["repaintedPixelCount"], int(self.region.sum()))
        artifacts = self.refined.artifacts()
        self.assertIn("unrefined-result.png", artifacts)
        self.assertNotIn("unrefined-result.png", self.result.artifacts())
        edit = np.asarray(self.refined.edit_mask) > 127
        core = ~_dilate(~self.region, radius=0.03 * self.unit)
        self.assertTrue(edit[core].all())


class SettingsTest(unittest.TestCase):
    def test_generative_refine_defaults_off_and_validates(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AI_GENERATIVE_REFINE", None)
            self.assertEqual(config.get_settings().generative_refine, "off")
        with mock.patch.dict(os.environ, {"AI_GENERATIVE_REFINE": "Quality"}):
            self.assertEqual(config.get_settings().generative_refine, "quality")
        with mock.patch.dict(os.environ, {"AI_GENERATIVE_REFINE": "max"}):
            with self.assertRaises(ValueError):
                config.get_settings()

    def test_unknown_preset_is_rejected_before_loading_anything(self):
        with self.assertRaises(ValueError):
            DiffusionInpainter(preset="max")


class WorkerFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result, cls.landmarks, _points, _unit = _short_cap_transfer()
        cls.settings = config.Settings(
            aws_region="", bucket="", default_backend_base_url="", face_landmarker_model_path="", hair_segmenter_model_path=""
        )

    def test_off_returns_the_result_as_is(self):
        self.assertIs(skull_fitting.apply_generative_refine(self.result, "off", self.settings), self.result)

    def test_missing_packages_skip_with_a_warning_or_fail_when_required(self):
        with mock.patch.object(skull_fitting, "generative_refine_available", return_value=False):
            skipped = skull_fitting.apply_generative_refine(self.result, "fast", self.settings)
            self.assertIs(skipped.image, self.result.image)
            self.assertIn("GENERATIVE_REFINE_UNAVAILABLE", skipped.metadata["warnings"])
            with self.assertRaises(skull_fitting.GenerativeRefineUnavailableError):
                skull_fitting.apply_generative_refine(self.result, "fast", self.settings, require=True)

    def test_installed_packages_run_the_refiner(self):
        generator = RecordingGenerator()
        with (
            mock.patch.object(skull_fitting, "generative_refine_available", return_value=True),
            mock.patch.object(skull_fitting, "_refiner", return_value=generator) as factory,
        ):
            refined = skull_fitting.apply_generative_refine(self.result, "quality", self.settings)
        factory.assert_called_once_with("quality", self.settings.generative_model, "auto")
        self.assertEqual(len(generator.calls), 1)
        self.assertEqual(refined.metadata["generativeRefine"]["preset"], "quality")


if __name__ == "__main__":
    unittest.main()
