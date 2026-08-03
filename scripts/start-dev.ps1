param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$scripts = @(
    @{ Name = "Spring Boot"; Path = Join-Path $PSScriptRoot "run-backend.ps1" },
    @{ Name = "Python Worker"; Path = Join-Path $PSScriptRoot "run-ai.ps1" },
    @{ Name = "Expo"; Path = Join-Path $PSScriptRoot "run-expo.ps1" }
)

Write-Host "Starting Hair Customized AI local stack..." -ForegroundColor Cyan
Write-Host "Root: $root"

foreach ($script in $scripts) {
    Write-Host "Opening $($script.Name) terminal..." -ForegroundColor Green
    if ($DryRun) {
        Write-Host "  $($script.Path)"
        continue
    }

    Start-Process powershell.exe -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        $script.Path
    )
    Start-Sleep -Seconds 1
}

Write-Host "Done. Close each opened terminal or press Ctrl+C inside it to stop that service." -ForegroundColor Cyan
