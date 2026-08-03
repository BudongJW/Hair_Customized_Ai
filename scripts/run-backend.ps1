$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Hair AI - Spring Boot"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$dotenv = Join-Path $PSScriptRoot "import-dotenv.ps1"
$backendEnv = Join-Path $root ".env.backend"

Set-Location $root

if (Test-Path -LiteralPath $backendEnv) {
    & $dotenv -Path $backendEnv
} else {
    Write-Host "No .env.backend found. S3 mode needs AWS variables in this shell." -ForegroundColor Yellow
}

if (-not $env:APP_QUEUE_PROVIDER) {
    $env:APP_QUEUE_PROVIDER = "http"
}
if (-not $env:APP_AI_WORKER_BASE_URL) {
    $env:APP_AI_WORKER_BASE_URL = "http://localhost:8000"
}
if (-not $env:APP_AI_WORKER_BACKEND_BASE_URL) {
    $env:APP_AI_WORKER_BACKEND_BASE_URL = "http://localhost:8080"
}

Write-Host "Starting Spring Boot on http://localhost:8080" -ForegroundColor Cyan
.\gradlew.bat bootRun
