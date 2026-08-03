$ErrorActionPreference = "Stop"
$Host.UI.RawUI.WindowTitle = "Hair AI - Expo"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$frontendDir = Join-Path $root "frontend"

Set-Location $frontendDir

if (-not (Test-Path -LiteralPath "node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
    npm install --no-audit --no-fund
}

Write-Host "Starting Expo. Scan the QR code with Expo Go." -ForegroundColor Cyan
node node_modules\expo\bin\cli start -c
