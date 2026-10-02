"""Optional generative touch-up of what the hair transfer had to invent.

The transfer (``hair_removal`` + ``hair_alignment``) never edits the user's eyes, nose or
mouth and pastes the hair model's real hair, but some pixels it has to make up: the
forehead and temples that were under the old hair, the ears, neck and shoulders where long
hair hung, and the strip where the new hairline meets that drawn skin. Those are smooth
guesses with visible seams. Here a Stable Diffusion 1.5 inpainting model repaints exactly
those pixels. It starts from our guess (img2img strength below 1), so the head shape,
hairline and skin tone stay what the pipeline decided; the model adds the shading,
texture and the soft hair-to-skin transition of a photo. Everything else is pasted back
from the unrefined result, so identity and the borrowed hair are untouched.

The model runs locally (photos never leave the server). It needs the packages in
``requirements-generative.txt`` and downloads about 2 GB of weights on first use.
"""

from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
from PIL import Image

from .config import DEFAULT_GENERATIVE_MODEL as DEFAULT_MODEL
from .hair_alignment import HairTransferResult
from .hair_removal import (
    FACE_OVAL_INDICES,
    LEFT_BROW_INDICES,
    RIGHT_BROW_INDICES,
    _blur,
    _dilate,
    _face_frame,
    _local_grid,
)

LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"

PROMPT = "professional portrait photo of a person, realistic skin texture, natural hairline, sharp focus, high detail"
NEGATIVE_PROMPT = "hat, cap, headband, hands, text, watermark, blurry, cartoon, painting, deformed"

MAX_SIDE = 768  # SD 1.5 is trained at 512-768; a head crop at this size keeps skin detail
CROP_PADDING = 0.15  # context around the repainted pixels, in face widths
HAIRLINE_BAND = 0.03  # how far on either side of the new hair edge is repainted
BODY_MAX_HAIR_ALPHA = 0.3  # rebuilt ears/neck/shoulders under at most this much new hair
PROTECTION_PADDING = 0.03
GENERATOR_MASK_GROWTH = 0.01  # the model's mask is a bit wider so it repaints across our seams
PASTE_FEATHER = 0.008


@dataclass(frozen=True)
class RefinePreset:
    sampler: str  # "lcm" (LCM-LoRA, few steps) or "dpm" (DPM-Solver++, classifier-free guidance)
    steps: int
    strength: float
    guidance: float


# On a 4-core CPU "fast" takes ~50 s and "quality" ~2.5 min (a GPU should need seconds).
# "quality" draws more skin detail and a crisper hairline but invents stray hairs more often.
PRESETS = {
    "fast": RefinePreset(sampler="lcm", steps=8, strength=0.85, guidance=1.0),
    "quality": RefinePreset(sampler="dpm", steps=20, strength=0.75, guidance=5.0),
}

# (crop, mask, seed) -> repainted crop of the same size. The mask is white where to repaint.
Generator = Callable[[Image.Image, Image.Image, int], Image.Image]


def refinement_mask(result: HairTransferResult) -> tuple[np.ndarray, float]:
    """Pixels the transfer synthesised (and so may be repainted), and the face width in pixels."""

    if result.target_points is None:
        raise ValueError("generative refinement needs a HairTransferResult made by transfer_hair")
    width, height = result.image.size
    points = result.target_points
    frame = _face_frame(points)
    local = frame.to_local(points)
    unit = float(np.ptp(local[FACE_OVAL_INDICES, 0]))
    brow_v = float(local[LEFT_BROW_INDICES + RIGHT_BROW_INDICES, 1].min())

    alpha = np.asarray(result.warped_hair.getchannel("A"), dtype=np.float32) / 255.0
    scalp = np.asarray(result.bald.scalp_fill_mask) > 127
    removal = np.asarray(result.bald.removal_mask) > 127
    skull = np.asarray(result.bald.skull_mask) > 127
    protection = _dilate(np.asarray(result.bald.protection_mask) > 127, radius=PROTECTION_PADDING * unit)

    hair = alpha > 0.5
    band = HAIRLINE_BAND * unit
    drawn_skin = scalp & ~hair
    hairline = _dilate(hair, band) & ~_erode(hair, band) & _dilate(scalp, band)
    # Below the brows the rebuilt pixels are ears, neck and shoulders, where the procedural fill
    # leaves seams. Above them it is backdrop around the head, which LaMa already fills well and
    # where the model tends to invent extra hair.
    below_brows = _local_grid(frame, width, height)[..., 1] > brow_v
    rebuilt_body = removal & ~skull & (alpha < BODY_MAX_HAIR_ALPHA) & below_brows
    return (drawn_skin | hairline | rebuilt_body) & ~protection, unit


def refine_transfer(
    result: HairTransferResult,
    generator: Generator,
    *,
    preset: str = "custom",
    max_side: int = MAX_SIDE,
    seed: int = 0,
) -> HairTransferResult:
    """Repaint the synthesised regions of ``result`` with ``generator`` and paste them back."""

    region, unit = refinement_mask(result)
    if not region.any():
        return result
    image = np.asarray(result.image, dtype=np.float32)
    height, width = region.shape

    ys, xs = np.nonzero(region)
    pad = int(CROP_PADDING * unit)
    x0, x1 = max(0, int(xs.min()) - pad), min(width, int(xs.max()) + 1 + pad)
    y0, y1 = max(0, int(ys.min()) - pad), min(height, int(ys.max()) + 1 + pad)
    crop_width, crop_height = x1 - x0, y1 - y0
    scale = min(1.0, max_side / max(crop_width, crop_height))
    size = (_multiple_of_8(crop_width * scale), _multiple_of_8(crop_height * scale))

    crop = result.image.crop((x0, y0, x1, y1)).resize(size, Image.Resampling.LANCZOS)
    grown = _dilate(region, GENERATOR_MASK_GROWTH * unit)[y0:y1, x0:x1]
    mask = Image.fromarray(grown.astype(np.uint8) * 255, "L").resize(size, Image.Resampling.BILINEAR)

    started = time.perf_counter()
    repainted = generator(crop, mask, seed).convert("RGB")
    elapsed = time.perf_counter() - started
    repainted = np.asarray(repainted.resize((crop_width, crop_height), Image.Resampling.LANCZOS), dtype=np.float32)

    weight = np.clip(_blur(region[y0:y1, x0:x1].astype(np.float32), max(1.0, PASTE_FEATHER * unit)), 0.0, 1.0)
    refined = image.copy()
    refined[y0:y1, x0:x1] = image[y0:y1, x0:x1] * (1.0 - weight[..., None]) + repainted * weight[..., None]
    refined_image = Image.fromarray(np.clip(refined + 0.5, 0, 255).astype(np.uint8), "RGB")

    edit_mask = np.asarray(result.edit_mask) > 127 if result.edit_mask is not None else np.zeros_like(region)
    edit_mask = edit_mask | (np.abs(refined - image).max(axis=2) > 2.0)
    metadata = {
        **result.metadata,
        "generativeRefine": {
            "preset": preset,
            "repaintedPixelCount": int(region.sum()),
            "crop": [x0, y0, x1, y1],
            "generatorSize": list(size),
            "seconds": round(elapsed, 2),
        },
    }
    return replace(
        result,
        image=refined_image,
        unrefined_image=result.image,
        edit_mask=Image.fromarray(edit_mask.astype(np.uint8) * 255, "L"),
        metadata=metadata,
    )


class DiffusionInpainter:
    """SD 1.5 inpainting through diffusers. Loads lazily; keep one instance per process."""

    def __init__(self, preset: str = "fast", model: str = DEFAULT_MODEL, device: str = "auto"):
        if preset not in PRESETS:
            raise ValueError(f"unknown generative refine preset {preset!r}; use one of {sorted(PRESETS)}")
        self.preset_name = preset
        self.preset = PRESETS[preset]
        self.model = model
        self.device = device
        self._pipe = None

    def __call__(self, crop: Image.Image, mask: Image.Image, seed: int) -> Image.Image:
        import torch

        pipe = self._pipeline()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        output = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            image=crop,
            mask_image=mask,
            width=crop.width,
            height=crop.height,
            num_inference_steps=self.preset.steps,
            strength=self.preset.strength,
            guidance_scale=self.preset.guidance,
            generator=generator,
        )
        return output.images[0]

    def _pipeline(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionInpaintPipeline

        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        # Half precision is slow or unsupported on CPU; the fp16 weights are just a smaller download.
        dtype = torch.float32 if device == "cpu" else torch.float16
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            self.model, variant="fp16", torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
        if self.preset.sampler == "lcm":
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            pipe.load_lora_weights(LCM_LORA)
            pipe.fuse_lora()
        else:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++", solver_type="midpoint", use_karras_sigmas=True
            )
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe.to(device)
        return self._pipe


def generative_refine_available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in ("torch", "diffusers", "peft"))


def _erode(mask: np.ndarray, radius: float) -> np.ndarray:
    return ~_dilate(~mask, radius)


def _multiple_of_8(value: float) -> int:
    return max(64, int(round(value / 8.0)) * 8)
