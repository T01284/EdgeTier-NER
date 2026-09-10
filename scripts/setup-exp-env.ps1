#Requires -Version 5.1
<#
.SYNOPSIS
  Create Python venv and install experiment dependencies (CPU / local prep).
#>
param(
    [switch]$GpuExtras
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $RepoRoot ".venv"

Write-Host "=== EdgeFS NER Experiment Environment ==="

if (-not (Test-Path $VenvPath)) {
    Write-Host "[..] Creating venv at $VenvPath"
    python -m venv $VenvPath
}

$Python = Join-Path $VenvPath "Scripts\python.exe"
$Pip = Join-Path $VenvPath "Scripts\pip.exe"

& $Python -m pip install --upgrade pip setuptools wheel
& $Pip install -e "$RepoRoot"
& $Pip install -r (Join-Path $RepoRoot "requirements\base.txt")

if ($GpuExtras) {
    Write-Host "[WARN] GPU torch must match remote CUDA — edit requirements/gpu.txt then pip install -r"
}

Write-Host ""
Write-Host "Activate:  .\.venv\Scripts\Activate.ps1"
Write-Host "Smoke test:"
Write-Host "  python scripts/prepare_datasets.py --config configs/dataset/toy.yaml"
Write-Host "  python scripts/train_full.py --config configs/train/full_toy.yaml --device cpu"
