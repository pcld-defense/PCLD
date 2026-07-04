# Create .venv and install pinned dependencies on Windows.
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 [-Cuda]
param([switch]$Cuda)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

py -3.10 -m venv .venv
& .venv\Scripts\python.exe -m pip install --upgrade pip
if ($Cuda) {
    & .venv\Scripts\pip.exe install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
}
& .venv\Scripts\pip.exe install -e .

Write-Host ""
Write-Host "Done. Activate with:  .venv\Scripts\Activate.ps1"