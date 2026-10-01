"""Local try-it page for the skull-aligned hair transfer.

Open http://localhost:8000/demo while the worker runs, pick a face photo and a hair model
photo, and the page shows the result. Nothing is stored: the photos only live for the
request. Needs no S3 or backend, only the model files the worker already uses.
"""

from __future__ import annotations

import base64
import binascii
import time
from io import BytesIO

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from .config import get_settings
from .skull_fitting import FaceNotFoundError, run_skull_transfer

router = APIRouter()

MAX_IMAGE_BYTES = 12 * 1024 * 1024


class DemoTransferRequest(BaseModel):
    user_photo: str = Field(alias="userPhoto", description="data URL or base64 of the user's face photo")
    hair_model_photo: str = Field(alias="hairModelPhoto", description="data URL or base64 of the hair model photo")


@router.get("/demo", response_class=HTMLResponse, include_in_schema=False)
def demo_page() -> str:
    return DEMO_PAGE


@router.post("/api/v1/demo/hair-transfer")
def demo_hair_transfer(request: DemoTransferRequest) -> dict:
    face_bytes = _decode(request.user_photo, "내 얼굴 사진")
    reference_bytes = _decode(request.hair_model_photo, "헤어모델 사진")
    started = time.perf_counter()
    try:
        result = run_skull_transfer(face_bytes, reference_bytes, get_settings())
    except FaceNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="이미지 파일을 읽지 못했습니다. JPG나 PNG 사진을 사용해 주세요.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"사진을 처리하지 못했습니다: {exc}") from exc

    hair_preview = Image.new("RGB", result.warped_hair.size, (255, 255, 255))
    hair_preview.paste(result.warped_hair, mask=result.warped_hair.getchannel("A"))
    return {
        "result": _data_url(result.image),
        "baldCanvas": _data_url(result.bald.image),
        "warpedHair": _data_url(hair_preview),
        "warnings": result.metadata.get("warnings", []),
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "warp": result.metadata.get("warp", {}),
    }


def _decode(value: str, label: str) -> bytes:
    payload = value.split(",", 1)[1] if value.startswith("data:") else value
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"{label}을(를) 읽지 못했습니다.") from exc
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=422, detail=f"{label}이(가) 비어 있거나 너무 큽니다.")
    return data


def _data_url(image: Image.Image) -> str:
    output = BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")


DEMO_PAGE = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>헤어 합성 데모</title>
<style>
  :root { --ink: #1f2933; --muted: #616e7c; --line: #d9e2ec; --accent: #2a6f97; --bg: #f5f7fa; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
         color: var(--ink); background: var(--bg); }
  main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 48px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  p.lead { margin: 0 0 20px; color: var(--muted); font-size: 14px; }
  .inputs, .outputs { display: grid; gap: 12px; }
  .inputs { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .outputs { grid-template-columns: repeat(3, minmax(0, 1fr)); margin-top: 20px; }
  .card { background: #fff; border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
  .card h2 { font-size: 13px; margin: 0 0 8px; color: var(--muted); }
  .frame { aspect-ratio: 4 / 5; background: #eef2f6; border-radius: 8px; overflow: hidden;
           display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 13px; }
  .frame img { width: 100%; height: 100%; object-fit: cover; }
  input[type=file] { margin-top: 8px; width: 100%; font-size: 13px; }
  button { margin-top: 16px; padding: 12px 20px; border: 0; border-radius: 8px; background: var(--accent);
           color: #fff; font-size: 15px; font-weight: 700; cursor: pointer; }
  button:disabled { opacity: .5; cursor: progress; }
  #status { margin-top: 12px; font-size: 14px; color: var(--muted); min-height: 20px; }
  #status.error { color: #b42318; }
  @media (max-width: 720px) { .outputs { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<main>
  <h1>헤어 합성 데모</h1>
  <p class="lead">정면 얼굴 사진과 원하는 헤어스타일의 사진을 고르면, 기존 머리를 지우고 새 머리를 맞춰 올립니다. 사진은 저장되지 않습니다.</p>
  <div class="inputs">
    <div class="card"><h2>내 얼굴 사진</h2><div class="frame" id="userPreview">사진을 선택하세요</div>
      <input type="file" id="userPhoto" accept="image/*"></div>
    <div class="card"><h2>헤어모델 사진</h2><div class="frame" id="modelPreview">사진을 선택하세요</div>
      <input type="file" id="modelPhoto" accept="image/*"></div>
  </div>
  <button id="run" disabled>합성하기</button>
  <div id="status"></div>
  <div class="outputs">
    <div class="card"><h2>합성 결과</h2><div class="frame" id="result"></div></div>
    <div class="card"><h2>머리 제거 결과</h2><div class="frame" id="bald"></div></div>
    <div class="card"><h2>맞춘 새 머리</h2><div class="frame" id="hair"></div></div>
  </div>
</main>
<script>
const photos = {};
const WARNINGS = { REFERENCE_HAIR_CROPPED_TOP: "헤어모델 사진에서 머리 윗부분이 잘려 있어 정수리가 평평하게 보일 수 있습니다.",
                   REFERENCE_HAIR_CROPPED_LEFT: "헤어모델 사진에서 머리 왼쪽이 잘려 있습니다.",
                   REFERENCE_HAIR_CROPPED_RIGHT: "헤어모델 사진에서 머리 오른쪽이 잘려 있습니다." };
function show(id, src) { document.getElementById(id).innerHTML = src ? '<img alt="" src="' + src + '">' : ""; }
function shrink(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      const scale = Math.min(1, 1600 / Math.max(image.width, image.height));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(image.width * scale); canvas.height = Math.round(image.height * scale);
      canvas.getContext("2d").drawImage(image, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(image.src);
      resolve(canvas.toDataURL("image/jpeg", 0.92));
    };
    image.onerror = reject;
    image.src = URL.createObjectURL(file);
  });
}
for (const [inputId, previewId, key] of [["userPhoto", "userPreview", "user"], ["modelPhoto", "modelPreview", "model"]]) {
  document.getElementById(inputId).addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    photos[key] = await shrink(file);
    show(previewId, photos[key]);
    document.getElementById("run").disabled = !(photos.user && photos.model);
  });
}
document.getElementById("run").addEventListener("click", async () => {
  const button = document.getElementById("run"), status = document.getElementById("status");
  button.disabled = true; status.className = ""; status.textContent = "합성 중입니다… (보통 10초 내외)";
  try {
    const response = await fetch("/api/v1/demo/hair-transfer", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userPhoto: photos.user, hairModelPhoto: photos.model }) });
    const body = await response.json();
    if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : "처리하지 못했습니다.");
    show("result", body.result); show("bald", body.baldCanvas); show("hair", body.warpedHair);
    const notes = body.warnings.map((w) => WARNINGS[w] || w);
    status.textContent = "완료 (" + body.elapsedSeconds + "초)" + (notes.length ? " · " + notes.join(" ") : "");
  } catch (error) {
    status.className = "error"; status.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""
