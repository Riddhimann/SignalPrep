$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendIndex = Join-Path $projectRoot "frontend\dist\index.html"

if (-not (Test-Path -LiteralPath $frontendIndex)) {
    Write-Host "Building the React frontend..." -ForegroundColor Cyan
    Push-Location (Join-Path $projectRoot "frontend")
    try {
        npm.cmd install
        npm.cmd run build
    }
    finally {
        Pop-Location
    }
}

Write-Host "SignalPrep is starting at http://127.0.0.1:8000" -ForegroundColor Green
python -m uvicorn interview_coach.api.app:app --host 127.0.0.1 --port 8000
