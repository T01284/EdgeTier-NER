#Requires -Version 5.1
<#
.SYNOPSIS
  Build IEEE Access manuscript PDF from paper/ieee-access/main.tex
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$TexDir = Join-Path $Root "paper/ieee-access"
Set-Location $TexDir

foreach ($ext in @("aux", "bbl", "blg", "out", "log")) {
    Remove-Item "main.$ext" -ErrorAction SilentlyContinue
}

pdflatex -interaction=nonstopmode main.tex | Out-Null
if ($LASTEXITCODE -ne 0) { throw "pdflatex pass 1 failed (see main.log)" }
bibtex main | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
if ($LASTEXITCODE -ne 0) { throw "pdflatex final pass failed (see main.log)" }
if (Select-String -Path main.log -Pattern "^!" -Quiet) {
    throw "main.log still contains LaTeX errors"
}

$dest = Join-Path $Root "paper/submission/ieee-access-manuscript-flat.pdf"
Copy-Item (Join-Path $TexDir "main.pdf") $dest -Force
Write-Host "[OK] $dest"

& (Join-Path $Root "scripts/pack-ieee-access-submission.ps1")
