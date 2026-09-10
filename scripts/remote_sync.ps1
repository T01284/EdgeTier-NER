#Requires -Version 5.1
param(
    [switch]$Setup,
    [switch]$SyncOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) { $Py = "python" }

& $Py -m pip install paramiko pyyaml -q
$argsList = @()
if ($Setup) { $argsList += "--setup" }
if ($SyncOnly) { $argsList += "--sync-only" }
& $Py (Join-Path $PSScriptRoot "remote_sync.py") @argsList
