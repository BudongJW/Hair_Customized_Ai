$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Hair AI - Python Worker"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$aiDir = Join-Path $root "ai"
$python = Join-Path $aiDir ".venv\Scripts\python.exe"
$requirements = Join-Path $aiDir "requirements.txt"
$depsStamp = Join-Path $aiDir ".venv\.requirements-installed"
$modelDir = Join-Path $aiDir "models"
$faceLandmarkerModelPath = Join-Path $modelDir "face_landmarker.task"
$faceLandmarkerModelUrl = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
$hairSegmenterModelPath = Join-Path $modelDir "hair_segmenter.tflite"
$hairSegmenterModelUrl = "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/latest/hair_segmenter.tflite"
$selfieMulticlassModelPath = Join-Path $modelDir "selfie_multiclass_256x256.tflite"
$selfieMulticlassModelUrl = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
$lamaModelPath = Join-Path $modelDir "lama_fp32.onnx"
$lamaModelUrl = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"

Set-Location $aiDir
$env:MPLCONFIGDIR = Join-Path $aiDir ".cache\matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

$shouldInstall = -not (Test-Path -LiteralPath $depsStamp)
if (-not $shouldInstall) {
    $shouldInstall = (Get-Item -LiteralPath $requirements).LastWriteTimeUtc -gt (Get-Item -LiteralPath $depsStamp).LastWriteTimeUtc
}

if ($shouldInstall) {
    Write-Host "Installing Python worker dependencies..." -ForegroundColor Cyan
    & $python -m pip install -r requirements.txt
    New-Item -ItemType File -Path $depsStamp -Force | Out-Null
}

if (-not (Test-Path -LiteralPath $faceLandmarkerModelPath)) {
    Write-Host "Downloading MediaPipe Face Landmarker model..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    Invoke-WebRequest -Uri $faceLandmarkerModelUrl -OutFile $faceLandmarkerModelPath
}

if (-not (Test-Path -LiteralPath $hairSegmenterModelPath)) {
    Write-Host "Downloading MediaPipe Hair Segmenter model..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    Invoke-WebRequest -Uri $hairSegmenterModelUrl -OutFile $hairSegmenterModelPath
}

if (-not (Test-Path -LiteralPath $selfieMulticlassModelPath)) {
    Write-Host "Downloading MediaPipe Selfie Multiclass model..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    Invoke-WebRequest -Uri $selfieMulticlassModelUrl -OutFile $selfieMulticlassModelPath
}

# LaMa (~170MB) only improves the background behind removed hair. Set AI_SKIP_LAMA=1 to
# skip it; hair removal then falls back to a smooth fill that suits plain backgrounds.
if (-not $env:AI_SKIP_LAMA -and -not (Test-Path -LiteralPath $lamaModelPath)) {
    Write-Host "Downloading LaMa inpainting model (~170MB)..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path $modelDir | Out-Null
    Invoke-WebRequest -Uri $lamaModelUrl -OutFile $lamaModelPath
}

if (-not $env:FACE_LANDMARKER_MODEL_PATH) {
    $env:FACE_LANDMARKER_MODEL_PATH = $faceLandmarkerModelPath
}

if (-not $env:HAIR_SEGMENTER_MODEL_PATH) {
    $env:HAIR_SEGMENTER_MODEL_PATH = $hairSegmenterModelPath
}

if (-not $env:SELFIE_MULTICLASS_MODEL_PATH) {
    $env:SELFIE_MULTICLASS_MODEL_PATH = $selfieMulticlassModelPath
}

if (-not $env:LAMA_MODEL_PATH) {
    $env:LAMA_MODEL_PATH = $lamaModelPath
}

Write-Host "Starting Python AI worker on http://localhost:8000" -ForegroundColor Cyan
& $python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
