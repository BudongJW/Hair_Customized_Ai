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

if (-not $env:FACE_LANDMARKER_MODEL_PATH) {
    $env:FACE_LANDMARKER_MODEL_PATH = $faceLandmarkerModelPath
}

if (-not $env:HAIR_SEGMENTER_MODEL_PATH) {
    $env:HAIR_SEGMENTER_MODEL_PATH = $hairSegmenterModelPath
}

Write-Host "Starting Python AI worker on http://localhost:8000" -ForegroundColor Cyan
& $python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
