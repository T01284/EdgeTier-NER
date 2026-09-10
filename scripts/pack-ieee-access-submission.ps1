#Requires -Version 5.1
<#
.SYNOPSIS
  Sync submission-flat/ and rebuild ieee-access-manuscript-source.zip
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Flat = Join-Path $Root "paper/ieee-access/submission-flat"
$TexDir = Join-Path $Root "paper/ieee-access"
$Tables = Join-Path $Root "paper/tables"
$Figures = Join-Path $Root "paper/figures"
$ZipPath = Join-Path $Root "paper/submission/ieee-access-manuscript-source.zip"

foreach ($pair in @(
    @("main.tex"), @("main.bbl"), @("main.pdf"), @("ieeeaccess.cls")
)) {
    $name = $pair[0]
    Copy-Item (Join-Path $TexDir $name) (Join-Path $Flat $name) -Force
}
Copy-Item (Join-Path $Tables "*.tex") (Join-Path $Flat "tables") -Force
Copy-Item (Join-Path $Figures "fig-tradeoff.tex") (Join-Path $Flat "figures") -Force

$py = @"
import os, zipfile
src = r"$Flat"
zip_path = r"$ZipPath"
exclude_ext = {'.aux', '.log', '.blg', '.out'}
exclude_names = {'.gitkeep'}
if os.path.exists(zip_path):
    os.remove(zip_path)
count = 0
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for dirpath, _, filenames in os.walk(src):
        for fn in filenames:
            if fn in exclude_names:
                continue
            if os.path.splitext(fn)[1].lower() in exclude_ext:
                continue
            full = os.path.join(dirpath, fn)
            arc = os.path.relpath(full, src).replace('\\\\', '/')
            zf.write(full, arc)
            count += 1
print(f'[OK] {zip_path} ({count} files, {os.path.getsize(zip_path)} bytes)')
"@
python -c $py
