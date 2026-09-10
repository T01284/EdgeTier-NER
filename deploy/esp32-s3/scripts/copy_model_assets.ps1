# Copy ESP-PPQ output and deploy manifest to PlatformIO project
param(
    [string]$EspdlPath = "",
    [string]$DeployDir = "outputs/export/deploy"
)

$Root = Split-Path -Parent $PSScriptRoot
$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

if ($EspdlPath -and (Test-Path $EspdlPath)) {
    Copy-Item $EspdlPath (Join-Path $Models "edgefs_model.espdl") -Force
    Write-Host "Copied ESPDL -> models/edgefs_model.espdl"
}
else {
    Write-Host "No ESPDL provided. Place quantized model at models/edgefs_model.espdl manually."
}

$deploySrc = Join-Path (Split-Path -Parent (Split-Path -Parent $Root)) $DeployDir
if (Test-Path $deploySrc) {
    Copy-Item "$deploySrc/*" $Models -Recurse -Force
    Write-Host "Copied deploy manifest assets -> models/"
}
