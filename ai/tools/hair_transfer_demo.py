"""Put a hair model's hairstyle on a user photo and write a debug panel.

Usage (from the ``ai`` directory)::

    python tools/hair_transfer_demo.py user.jpg hair_model.jpg --out out/
    python tools/hair_transfer_demo.py user.jpg hair_model.jpg --refine fast   # + generative touch-up

Panel: user photo | hair model | result | bald canvas | warped hair layer (| result before touch-up).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

AI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_ROOT))

from app.config import get_settings  # noqa: E402
from app.face_landmarker import analyze_with_mediapipe  # noqa: E402
from app.generative_refine import PRESETS, DiffusionInpainter, refine_transfer  # noqa: E402
from app.hair_alignment import transfer_hair  # noqa: E402
from app.hair_removal import default_inpainter, push_pull_inpaint  # noqa: E402

PORTRAIT_SIZE = (900, 1125)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("user_photo")
    parser.add_argument("hair_model_photo")
    parser.add_argument("--out", default="out")
    parser.add_argument("--no-lama", action="store_true", help="use the push-pull fallback instead of LaMa")
    parser.add_argument("--refine", choices=sorted(PRESETS), help="generative touch-up (needs requirements-generative.txt)")
    args = parser.parse_args()

    settings = get_settings()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    user = _portrait_cover(Image.open(args.user_photo), PORTRAIT_SIZE)
    hair_model = _portrait_cover(Image.open(args.hair_model_photo), PORTRAIT_SIZE)
    user_face = analyze_with_mediapipe(user, settings.face_landmarker_model_path)
    model_face = analyze_with_mediapipe(hair_model, settings.face_landmarker_model_path)
    if user_face is None or model_face is None:
        raise SystemExit("no face found in one of the photos")

    started = time.perf_counter()
    result = transfer_hair(
        user,
        user_face.landmarks,
        hair_model,
        model_face.landmarks,
        hair_segmenter_model_path=settings.hair_segmenter_model_path,
        multiclass_model_path=settings.selfie_multiclass_model_path,
        inpainter=push_pull_inpaint if args.no_lama else default_inpainter(settings.lama_model_path),
        headroom=True,
    )
    if args.refine:
        refiner = DiffusionInpainter(preset=args.refine, model=settings.generative_model, device=settings.generative_device)
        result = refine_transfer(result, refiner, preset=args.refine)
    elapsed = time.perf_counter() - started

    hair_preview = Image.new("RGB", result.warped_hair.size, (255, 255, 255))
    hair_preview.paste(result.warped_hair, mask=result.warped_hair.getchannel("A"))
    name = f"{Path(args.user_photo).stem}__{Path(args.hair_model_photo).stem}"
    panels = [user, hair_model, result.image, result.bald.image, hair_preview]
    if result.unrefined_image is not None:
        panels.append(result.unrefined_image)
    _panel(panels).save(out_dir / f"{name}_panel.jpg", quality=90)
    result.image.save(out_dir / f"{name}_result.png")
    print(f"{name}: {elapsed:.2f}s warp={result.metadata['warp']} warnings={result.metadata['warnings']}")


def _panel(images: list[Image.Image]) -> Image.Image:
    width = 300
    height = int(width * images[0].height / images[0].width)
    labels = ["user photo", "hair model", "result", "bald canvas", "warped hair", "before touch-up"]
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
