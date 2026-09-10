#Requires -Version 5.1
<#
.SYNOPSIS
  Push a single orphan commit to GitHub (experiment code only, no manuscript).

.DESCRIPTION
  Creates one root commit on an orphan branch with Shu Wang as the sole author,
  excludes manuscript / venue / IDE paths, force-pushes to github:main, then
  returns to the original branch. Use this after recreating an empty GitHub repo.

.EXAMPLE
  .\scripts\publish-to-github.ps1
  .\scripts\publish-to-github.ps1 -DryRun
#>
param(
    [switch]$DryRun,
    [string]$GithubBranch = "main",
    [string]$GithubUrl = "https://github.com/T01284/EdgeTier-NER.git",
    [string]$OrphanBranch = "github-public-release",
    [string]$AuthorName = "Shu Wang",
    [string]$AuthorEmail = "shu.wang@hljtyw.com"
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$ExcludeFromGithub = @(
    "paper",
    "NER_Edge_Paper_Handbook.md",
    "STAGE-TODO.md",
    "cover_letter_MICPRO.docx",
    "`$build",
    ".cursor",
    ".vscode",
    ".idea"
)

Write-Host "EdgeTier-NER -> GitHub (orphan public snapshot)"
Write-Host "Excluded: $($ExcludeFromGithub -join ', ')"
Write-Host ""

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git not found"
}

$origBranch = (git rev-parse --abbrev-ref HEAD).Trim()
$remotes = git remote 2>$null
if ($remotes -notcontains "github") {
    Write-Host "Adding remote 'github' -> $GithubUrl"
    if (-not $DryRun) {
        git remote add github $GithubUrl
    }
}

if ($DryRun) {
    Write-Host "[dry-run] Would create orphan branch $OrphanBranch, commit, push to github:$GithubBranch"
    exit 0
}

if (git branch --list $OrphanBranch) {
    git branch -D $OrphanBranch | Out-Null
}
git checkout --orphan $OrphanBranch
if ($LASTEXITCODE -ne 0) {
    throw "failed to create orphan branch $OrphanBranch"
}

try {
    git reset --mixed
    if ($LASTEXITCODE -ne 0) {
        throw "git reset failed"
    }

    git add -A
    foreach ($path in $ExcludeFromGithub) {
        $full = Join-Path $RepoRoot $path
        if (Test-Path $full) {
            git rm -r --cached --ignore-unmatch $path 2>$null | Out-Null
        }
    }

    $env:GIT_AUTHOR_NAME = $AuthorName
    $env:GIT_AUTHOR_EMAIL = $AuthorEmail
    $env:GIT_COMMITTER_NAME = $AuthorName
    $env:GIT_COMMITTER_EMAIL = $AuthorEmail

    git commit -m @"
Public release: EdgeTier-NER source, firmware adapters, and evaluation artifacts.

Reproduces the operator-constrained NER export audit
(CoNLL-2003 / CLUENER deploy bundles, cited deploy/results JSON summaries).
"@

    if ($LASTEXITCODE -ne 0) {
        throw "commit failed"
    }

    git push -u github "HEAD:${GithubBranch}" --force
    if ($LASTEXITCODE -ne 0) {
        throw "git push failed"
    }
    Write-Host "[OK] https://github.com/T01284/EdgeTier-NER ($GithubBranch)"
}
finally {
    Remove-Item Env:GIT_AUTHOR_NAME -ErrorAction SilentlyContinue
    Remove-Item Env:GIT_AUTHOR_EMAIL -ErrorAction SilentlyContinue
    Remove-Item Env:GIT_COMMITTER_NAME -ErrorAction SilentlyContinue
    Remove-Item Env:GIT_COMMITTER_EMAIL -ErrorAction SilentlyContinue
    git checkout $origBranch 2>$null | Out-Null
    git branch -D $OrphanBranch 2>$null | Out-Null
}

Write-Host "Back on branch '$origBranch'. paper/ remains in your working tree and on private remot