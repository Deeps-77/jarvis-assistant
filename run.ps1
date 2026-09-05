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
    Write-Host "  1) Bot   (main.py    - Telegram)"
    Write-Host "  2) Web   (app.py     - chat UI :8000)"
    Write-Host "  3) Code  (code_ui.py - code assistant :8500)"
    Write-Host "  4) All   (bot + web + code in separate windows)"
    $choice = Read-Host "Enter 1-4"
    $Mode = @{ "1" = "bot"; "2" = "web"; "3" = "code"; "4" = "all" }[$choice]
}

switch ($Mode) {
    "bot"  { & $py (Join-Path $PROJECT "main.py") }
    "web"  { & $py (Join-Path $PROJECT "app.py") }
    "code" { & $py (Join-Path $PROJECT "code_ui.py") }
    "both" {
        Start-Process -FilePath $py -ArgumentList "main.py" -WorkingDirectory $PROJECT
        Start-Process -FilePath $py -ArgumentList "app.py"  -WorkingDirectory $PROJECT
        Write-Host "Launched bot + web in separate windows. Close them to stop."
    }
    "all" {
        Start-Process -FilePath $py -ArgumentList "main.py" -WorkingDirectory $PROJECT
        Start-Process -FilePath $py -ArgumentList "app.py"  -WorkingDirectory $PROJECT
        Start-Process -FilePath $py -ArgumentList "code_ui.py" -WorkingDirectory $PROJECT
        Write-Host "Launched bot + web + code in separate windows. Close them to stop."
    }
    default { Write-Error "Unknown mode: $Mode (use bot/web/code/both/all)"; exit 1 }
}
