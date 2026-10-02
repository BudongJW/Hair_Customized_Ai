"""Model-free tests for app.hair_removal.

Run from the ``ai`` directory: ``python -m unittest discover tests``
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

from app.hair_removal import (  # noqa: E402
    FACE_OVAL_INDICES,
    LEFT_BROW_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_BROW_INDICES,
    RIGHT_EYE_INDICES,
    HeadSegmentation,
    _connected_to,
    _push_pull,
    push_pull_inpaint,
    remove_hair,
)

FIXTURE = Path(__file__).parent / "fixtures" / "frontal_face_landmarks.json"
BACKGROUND = (226, 228, 232)
SKIN = (222, 182, 160)
HAIR = (38, 30, 26)


def _load_landmarks():
    payload = json.loads(FIXTURE.read_text())
    width, height = payload["imageSize"]
    return payload["landmarks"], (width, height)


def _synthetic_portrait():
    """A plain-background portrait with a voluminous "mushroom" haircut and bangs."""

    landmarks, size = _load_landmarks()
    width, height = size
    points = np.array(landmarks, dtype=np.float32) * np.array(size, dtype=np.float32)

    image = Image.new("RGB", size, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.polygon([tuple(p) for p in points[FACE_OVAL_INDICES]], fill=SKIN)
    for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
        draw.polygon([tuple(p) for p in points[indices]], fill=(60, 45, 40))

    oval = points[FACE_OVAL_INDICES]
    face_left, face_right = oval[:, 0].min(), oval[:, 0].max()
    face_width = face_right - face_left
    brow_y = min(points[LEFT_BROW_INDICES][:, 1].min(), points[RIGHT_BROW_INDICES][:, 1].min())

    hair = Image.new("L", size, 0)
    hair_draw = ImageDraw.Draw(hair)
    # Hair volume far wider and taller than the skull: the case that produced the mushroom.
    hair_draw.ellipse(
        (
            face_left - face_width * 0.35,
            points[10][1] - face_width * 0.75,
            face_right + face_width * 0.35,
            brow_y + face_width * 0.05,
        ),
        fill=255,
    )
    hair_mask = np.asarray(hair) > 127
    pixels = np.asarray(image).copy()
    pixels[hair_mask] = HAIR
    image = Image.fromarray(pixels)

    face_skin = np.zeros((height, width), dtype=np.float32)
    face_draw_image = Image.new("L", size, 0)
    ImageDraw.Draw(face_draw_image).polygon([tuple(p) for p in oval], fill=255)
    face_skin[(np.asarray(face_draw_image) > 127) & ~hair_mask] = 1.0

    segmentation = HeadSegmentation(hair=hair_mask.astype(np.float32), face_skin=face_skin, source="synthetic")
    return image, landmarks, segmentation, points, hair_mask


class HairRemovalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image, cls.landmarks, cls.segmentation, cls.points, cls.hair_mask = _synthetic_portrait()
        cls.result = remove_hair(
            cls.image,
            cls.landmarks,
            segmentation=cls.segmentation,
            inpainter=push_pull_inpaint,
        )
        cls.output = np.asarray(cls.result.image, dtype=np.float32)
        cls.removal = np.asarray(cls.result.removal_mask) > 127
        cls.skull = np.asarray(cls.result.skull_mask) > 127

    def test_output_matches_input_geometry(self):
        self.assertEqual(self.result.image.size, self.image.size)
        self.assertEqual(self.result.image.mode, "RGB")

    def test_all_hair_is_removed(self):
        luminance = self.output @ np.array([0.299, 0.587, 0.114])
        self.assertLess(float((luminance[self.hair_mask] < 90).mean()), 0.01)

    def test_hair_outside_skull_becomes_background(self):
        outside = self.hair_mask & ~self.skull
        self.assertGreater(outside.sum(), 1000, "fixture should have hair volume outside the skull")
        error = np.abs(self.output[outside] - np.array(BACKGROUND)).max(axis=1)
        self.assertLess(float(np.percentile(error, 95)), 12.0)

    def test_hair_inside_skull_becomes_skin(self):
        inside = self.hair_mask & self.skull
        median = np.median(self.output[inside], axis=0)
        self.assertLess(float(np.abs(median - np.array(SKIN)).max()), 40.0)

    def test_bald_head_is_not_the_hair_silhouette(self):
        # The old flat fill painted the whole hair silhouette with skin ("mushroom head").
        changed_to_skin = np.abs(self.output - np.array(SKIN)).max(axis=2) < 45
        self.assertLess(
            float((changed_to_skin & self.hair_mask).sum()),
            0.8 * float(self.hair_mask.sum()),
        )
        skin_columns = np.nonzero(changed_to_skin.any(axis=0))[0]
        hair_columns = np.nonzero(self.hair_mask.any(axis=0))[0]
        self.assertLess(skin_columns.max() - skin_columns.min(), hair_columns.max() - hair_columns.min())

    def test_eyes_are_untouched(self):
        eye_center = self.points[LEFT_EYE_INDICES].mean(axis=0).astype(int)
        x, y = eye_center
        original = np.asarray(self.image, dtype=np.float32)[y - 3 : y + 3, x - 3 : x + 3]
        np.testing.assert_allclose(self.output[y - 3 : y + 3, x - 3 : x + 3], original, atol=1.0)

    def test_pixels_away_from_hair_are_untouched(self):
        original = np.asarray(self.image, dtype=np.float32)
        far = ~self.removal
        far[: self.image.height // 2] = False  # skin-side feathering only happens near the hairline
        np.testing.assert_allclose(self.output[far], original[far], atol=1.0)

    def test_hidden_eyebrows_are_redrawn(self):
        self.assertEqual(sorted(self.result.metadata["redrawnEyebrows"]), ["left", "right"])

    def test_artifacts_are_png(self):
        for name, (content_type, data) in self.result.artifacts().items():
            self.assertEqual(content_type, "image/png", name)
            self.assertTrue(data.startswith(b"\x89PNG"), name)


class LongHairOnPlainBackgroundTest(unittest.TestCase):
    """Long hair over the shoulders on a plain backdrop: above the shoulder line the hole
    must become clean background, not a blend of backdrop and clothes colours."""

    CLOTHES = (70, 60, 110)

    @classmethod
    def setUpClass(cls):
        image, landmarks, segmentation, points, hair_mask = _synthetic_portrait()
        width, height = image.size
        oval = points[FACE_OVAL_INDICES]
        unit = float(oval[:, 0].max() - oval[:, 0].min())
        cls.shoulder_y = int(points[152][1] + 0.35 * unit)

        pixels = np.asarray(image).copy()
        clothes = np.zeros((height, width), dtype=bool)
        clothes[cls.shoulder_y :, :] = True
        pixels[clothes] = cls.CLOTHES

        # Two curtains of hair from the temples down over the shoulders.
        curtains = np.zeros_like(clothes)
        for x in (oval[:, 0].min() - 0.18 * unit, oval[:, 0].max() + 0.02 * unit):
            curtains[int(points[10][1]) : int(cls.shoulder_y + 0.3 * unit), int(x) : int(x + 0.16 * unit)] = True
        hair = hair_mask | curtains
        pixels[hair] = HAIR
        cls.curtains = curtains & ~hair_mask

        face = segmentation.face_skin > 0.5
        background = ~(hair | clothes | face)
        cls.result = remove_hair(
            Image.fromarray(pixels),
            landmarks,
            segmentation=HeadSegmentation(
                hair=hair.astype(np.float32),
                face_skin=segmentation.face_skin,
                source="synthetic",
                background=background.astype(np.float32),
                clothes=(clothes & ~hair).astype(np.float32),
            ),
            inpainter=push_pull_inpaint,
        )

    def test_background_above_the_shoulders_is_restored(self):
        output = np.asarray(self.result.image, dtype=np.float32)
        skull = np.asarray(self.result.skull_mask) > 127
        region = self.curtains & ~skull
        region[self.shoulder_y - 8 :, :] = False
        self.assertGreater(region.sum(), 1000)
        error = np.abs(output[region] - np.array(BACKGROUND)).max(axis=1)
        self.assertLess(float(np.percentile(error, 95)), 8.0)
        self.assertGreater(self.result.metadata["rebuiltBackgroundPixelCount"], 0)

    def test_clothes_below_the_shoulder_line_are_not_painted_as_background(self):
        output = np.asarray(self.result.image, dtype=np.float32)
        region = self.curtains.copy()
        region[: self.shoulder_y + 12, :] = False
        distance_to_background = np.abs(output[region] - np.array(BACKGROUND)).max(axis=1)
        self.assertGreater(float(np.median(distance_to_background)), 60.0)


class ForeheadHighlightTest(unittest.TestCase):
    def test_matte_skin_gets_no_highlight(self):
        from app.hair_removal import _estimate_skull, _face_frame, _forehead_highlight, _landmark_pixels, _local_grid

        image, landmarks, segmentation, _points, _hair = _synthetic_portrait()
        width, height = image.size
        rgb = np.asarray(image, dtype=np.float32)
        points = _landmark_pixels(landmarks, width, height)
        frame = _face_frame(points)
        grid = _local_grid(frame, width, height)
        skull = _estimate_skull(frame, points, segmentation)
        flat_skin = segmentation.face_skin > 0.5  # one flat colour: nothing shinier than the median
        highlight = _forehead_highlight(rgb, flat_skin, frame.to_local(points), grid[..., 0], grid[..., 1], skull)
        self.assertEqual(float(np.abs(highlight).max()), 0.0)


class NoHairTest(unittest.TestCase):
    def test_bald_input_is_returned_unchanged(self):
        image, landmarks, segmentation, _points, _hair = _synthetic_portrait()
        bald = HeadSegmentation(hair=np.zeros_like(segmentation.hair), face_skin=segmentation.face_skin)
        result = remove_hair(image, landmarks, segmentation=bald, inpainter=push_pull_inpaint)
        np.testing.assert_array_equal(np.asarray(result.image), np.asarray(image))
        self.assertEqual(result.metadata["removedPixelCount"], 0)


class HelperTest(unittest.TestCase):
    def test_push_pull_interpolates_between_known_values(self):
        values = np.zeros((64, 64), dtype=np.float32)
        weights = np.zeros((64, 64), dtype=np.float32)
        weights[:, :8] = 1.0
        weights[:, -8:] = 1.0
        values[:, -8:] = 100.0
        filled = _push_pull(values, weights)
        row = filled[32]
        self.assertTrue(np.all(np.diff(row) >= -1e-3))
        self.assertAlmostEqual(float(row[0]), 0.0, places=3)
        self.assertAlmostEqual(float(row[-1]), 100.0, places=3)

    def test_connected_to_drops_islands(self):
        mask = np.zeros((80, 80), dtype=bool)
        mask[10:20, 10:20] = True
        mask[50:60, 50:60] = True
        seed = np.zeros_like(mask)
        seed[12, 12] = True
        kept = _connected_to(mask, seed)
        self.assertEqual(int(kept.sum()), 100)
        self.assertFalse(kept[55, 55])


if __name__ == "__main__":
    unittest.main()
