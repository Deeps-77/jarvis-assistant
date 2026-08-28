param([string]$Mode)

$ErrorActionPreference = "Stop"
$PROJECT  = $PSScriptRoot
$py       = Join-Path $PROJECT ".venv\Scripts\python.exe"
$cudaBins = @(
    Join-Path $PROJECT ".venv\Lib\site-packages\nvidia\cublas\bin"
    Join-Path $PROJECT ".venv\Lib\site-packages\nvidia\cudnn\bin"
)
$env:PATH = ($cudaBins -join ";") + ";" + $env:PATH

if (-not (Test-Path $py)) {
    Write-Error "Virtual environment not found at $py. Run 'uv sync' first."
    exit 1
}

if (-not $Mode) {
    Write-Host "`nJarvis launcher - choose a mode:"
    Write-Host "  1) Bot  (main.py  - Telegram)"
    Write-Host "  2) Web  (app.py   - Chainlit)"
    Write-Host "  3) Both (bot + web in separate windows)"
    $choice = Read-Host "Enter 1, 2 or 3"
    $Mode = @{ "1" = "bot"; "2" = "web"; "3" = "both" }[$choice]
}

switch ($Mode) {
    "bot"  { & $py (Join-Path $PROJECT "main.py") }
    "web"  { & $py (Join-Path $PROJECT "app.py") }
    "both" {
        Start-Process -FilePath $py -ArgumentList "main.py" -WorkingDirectory $PROJECT
        Start-Process -FilePath $py -ArgumentList "app.py"  -WorkingDirectory $PROJECT
        Write-Host "Launched bot + web in separate windows. Close them to stop."
    }
    default { Write-Error "Unknown mode: $Mode (use bot/web/both)"; exit 1 }
}
