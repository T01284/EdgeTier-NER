#Requires -Version 5.1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    pio run -e esp32-s3-n16r8-stub -t upload
}
finally {
    Pop-Location
}
