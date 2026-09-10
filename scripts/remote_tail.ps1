#Requires -Version 5.1
param([int]$Lines = 40)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root ".venv\Scripts\python.exe"
& $Py -m pip install paramiko pyyaml -q
& $Py (Join-Path $PSScriptRoot "remote_tail.py") --lines $Lines
