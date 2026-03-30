Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-RepoRoot {
    try {
        $root = git rev-parse --show-toplevel 2>$null
        if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($root)) {
            return $root.Trim()
        }
    }
    catch {
        # Fallback unten.
    }

    return (Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")).Path
}

try {
    $repoRoot = Resolve-RepoRoot
    $hookTemplateSh = Join-Path $repoRoot ".githooks\pre-commit"
    $hookTemplatePs1 = Join-Path $repoRoot ".githooks\pre-commit.ps1"
    $targetHookDir = Join-Path $repoRoot ".git\hooks"
    $targetHookSh = Join-Path $targetHookDir "pre-commit"
    $targetHookPs1 = Join-Path $targetHookDir "pre-commit.ps1"

    if (-not (Test-Path $hookTemplateSh)) {
        throw "Hook-Template fehlt: $hookTemplateSh"
    }
    if (-not (Test-Path $hookTemplatePs1)) {
        throw "PowerShell-Hook-Template fehlt: $hookTemplatePs1"
    }

    if (-not (Test-Path $targetHookDir)) {
        New-Item -ItemType Directory -Path $targetHookDir -Force | Out-Null
    }

    Copy-Item -Path $hookTemplateSh -Destination $targetHookSh -Force
    Copy-Item -Path $hookTemplatePs1 -Destination $targetHookPs1 -Force

    Write-Host "[install-hook] Hooks installiert:"
    Write-Host "  - $targetHookSh"
    Write-Host "  - $targetHookPs1"
    Write-Host "[install-hook] Test (Git-Hook): .git/hooks/pre-commit"
    Write-Host "[install-hook] Test (PowerShell): powershell -ExecutionPolicy Bypass -File .git/hooks/pre-commit.ps1"
}
catch {
    Write-Host ("[install-hook] FEHLER: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
}
