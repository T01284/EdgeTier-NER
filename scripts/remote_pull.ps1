#Requires -Version 5.1
param(
    [string]$LocalDir = "outputs/runs/conll2003_full",
    [string]$RemoteDir = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install paramiko pyyaml -q

$argsList = @("--local-dir", $LocalDir)
if ($RemoteDir) { $argsList += @("--remote-dir", $RemoteDir) }
& $Py (Join-Path $PSScriptRoot "remote_pull.py") @argsList
