#Requires -Version 5.1
<#
.SYNOPSIS
  Sync project to remote GPU server via rsync/scp (fill configs/remote/gpu_server.yaml first).
#>
param(
    [string]$RemoteConfig = "configs/remote/gpu_server.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$CfgPath = Join-Path $RepoRoot $RemoteConfig

if (-not (Test-Path $CfgPath)) {
    Write-Host "Remote config not found: $CfgPath"
    Write-Host "Copy configs/remote/gpu_server.yaml.example -> configs/remote/gpu_server.yaml"
    exit 1
}

# Minimal YAML parse without extra deps
$lines = Get-Content $CfgPath
$cfg = @{}
foreach ($line in $lines) {
    if ($line -match '^\s*([a-zA-Z0-9_]+):\s*(.+)\s*$') {
        $cfg[$Matches[1]] = $Matches[2].Trim()
    }
}

$hostName = $cfg["host"]
$user = $cfg["user"]
$port = if ($cfg["port"]) { $cfg["port"] } else { "22" }
$remoteDir = $cfg["project_dir"]

if ($hostName -match "YOUR_" -or $user -match "YOUR_") {
    Write-Host "Please edit $CfgPath with real SSH host/user before syncing."
    exit 1
}

$dest = "${user}@${hostName}:${remoteDir}/"
Write-Host "Syncing $RepoRoot -> $dest"

if (Get-Command rsync -ErrorAction SilentlyContinue) {
    rsync -avz --exclude ".venv" --exclude "outputs" --exclude "data/raw" --exclude "paper/build" `
        -e "ssh -p $port" "$RepoRoot/" $dest
}
elseif (Get-Command scp -ErrorAction SilentlyContinue) {
    Write-Host "rsync not found; using scp for src/configs/scripts only"
    scp -r -P $port "$RepoRoot/src" "$RepoRoot/configs" "$RepoRoot/scripts" "$RepoRoot/pyproject.toml" $dest
}
else {
    Write-Host "Install OpenSSH client or rsync. Alternatively use git push on remote."
    exit 1
}

Write-Host "Done. SSH and run remote setup:"
Write-Host "  ssh -p $port ${user}@${hostName}"
Write-Host "  cd $remoteDir && python3 -m venv .venv && source .venv/bin/activate"
Write-Host "  pip install -r requirements/gpu.txt && pip install -e ."
