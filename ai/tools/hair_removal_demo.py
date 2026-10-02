"""Run hair removal on local photos and write side-by-side debug panels.

Usage (from the ``ai`` directory)::

    python tools/hair_removal_demo.py photo1.jpg photo2.jpg --out out/

Panels: original | flat-skin fill of the hair mask (old behaviour) | new bald canvas | masks
(red = background fill, green = scalp fill, blue = protected, yellow line = skull estimate).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

from app.config import get_settings  # noqa: E402
from app.face_landmarker import analyze_with_mediapipe  # noqa: E402
from app.hair_removal import (  # noqa: E402
    _blur,
    _dilate,
    _rasterize,
    LEFT_BROW_INDICES,
    LEFT_EYE_INDICES,
    RIGHT_BROW_INDICES,
    RIGHT_EYE_INDICES,
    default_inpainter,
    push_pull_inpaint,
    remove_hair,
    segment_head,
)

PORTRAIT_SIZE = (900, 1125)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--out", default="out")
    parser.add_argument("--no-lama", action="store_true", help="use the push-pull fallback instead of LaMa")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inpainter = push_pull_inpaint if args.no_lama else default_inpainter(settings.lama_model_path)

    for path in args.images:
        portrait = _portrait_cover(Image.open(path), PORTRAIT_SIZE)
        analysis = analyze_with_mediapipe(portrait, settings.face_landmarker_model_path)
        if analysis is None:
            print(f"{path}: no face found")
            continue

        started = time.perf_counter()
        segmentation = segment_head(
            portrait,
            hair_segmenter_model_path=settings.hair_segmenter_model_path,
            multiclass_model_path=settings.selfie_multiclass_model_path,
        )
        result = remove_hair(portrait, analysis.landmarks, segmentation=segmentation, inpainter=inpainter)
        elapsed = time.perf_counter() - started

        flat = _flat_skin_fill(portrait, analysis.landmarks, segmentation.hair)
        panel = _panel([portrait, flat, result.image, _mask_overlay(portrait, result)])
        name = Path(path).stem
        panel.save(out_dir / f"{name}_panel.jpg", quality=90)
        result.image.save(out_dir / f"{name}_bald.png")
        skull = result.metadata["skull"]
        print(
            f"{name}: {elapsed:.2f}s inpainter={result.metadata['inpainter']} "
            f"vertexRatio={skull['vertexRatio']:.3f} clamped={skull['vertexClampedByHair']} "
            f"brows={result.metadata['redrawnEyebrows']}"
        )


def _flat_skin_fill(portrait: Image.Image, landmarks: list[dict], hair: np.ndarray) -> Image.Image:
    """Approximation of the current pipeline: paint the hair mask with the median skin colour."""

    rgb = np.asarray(portrait, dtype=np.float32)
    size = portrait.size
    points = np.array([[p["x"], p["y"]] for p in landmarks], dtype=np.float32) * np.array(size, dtype=np.float32)
    protect = np.zeros(rgb.shape[:2], dtype=bool)
    for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES, LEFT_BROW_INDICES, RIGHT_BROW_INDICES):
        protect |= _dilate(_rasterize(points[indices], size), 8)
    mask = (hair > 0.5) & ~protect
    cheek = rgb[int(points[50][1]) - 10 : int(points[50][1]) + 10, int(points[50][0]) - 10 : int(points[50][0]) + 10]
    skin = np.median(cheek.reshape(-1, 3), axis=0)
    alpha = _blur(mask.astype(np.float32), 2.0)[..., None]
    out = rgb * (1 - alpha) + skin * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _mask_overlay(portrait: Image.Image, result) -> Image.Image:
    base = np.asarray(portrait.convert("L").convert("RGB"), dtype=np.float32) * 0.55
    background = np.asarray(result.background_fill_mask) > 127
    scalp = np.asarray(result.scalp_fill_mask) > 127
    protection = np.asarray(result.protection_mask) > 127
    base[background] = base[background] * 0.4 + np.array([230, 60, 60]) * 0.6
    base[scalp] = base[scalp] * 0.4 + np.array([60, 200, 90]) * 0.6
    base[protection] = base[protection] * 0.4 + np.array([70, 110, 240]) * 0.6
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    skull = np.asarray(result.skull_mask) > 127
    edge = skull ^ _dilate(skull, 2.5)
    pixels = np.asarray(image).copy()
    pixels[edge] = (250, 220, 40)
    return Image.fromarray(pixels)


def _panel(images: list[Image.Image]) -> Image.Image:
    width = 360
    height = int(width * images[0].height / images[0].width)
    labels = ["original", "flat fill (old)", "new bald canvas", "masks"]
    sheet = Image.new("RGB", (width * len(images), height + 24), "white")
    draw = ImageDraw.Draw(sheet)
    for index, image in enumerate(images):
        sheet.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (index * width, 24))
        draw.text((index * width + 8, 6), labels[index], fill=(20, 20, 20))
    return sheet


def _portrait_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = size
    scale = max(width / image.width, height / image.height)
    resized = image.resize((int(image.width * scale + 0.5), int(image.height * scale + 0.5)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


if __name__ == "__main__":
    main()
