#Requires -Version 5.1
param(
    [switch]$SkipPrepare,
    [string]$Config = "configs/train/full_conll2003.yaml"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install paramiko pyyaml -q

$pyArgs = @("--config", $Config)
if ($SkipPrepare) { $pyArgs += "--skip-prepare" }
& $Py (Join-Path $PSScriptRoot "remote_start_train.py") @pyArgs
