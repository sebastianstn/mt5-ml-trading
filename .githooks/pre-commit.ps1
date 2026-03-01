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

    # Fallback: von .githooks oder .git\hooks nach oben laufen.
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $candidate1 = Resolve-Path (Join-Path $scriptDir "..") -ErrorAction SilentlyContinue
    if ($candidate1 -and (Test-Path (Join-Path $candidate1.Path "reports\doc_drift_guard.py"))) {
        return $candidate1.Path
    }

    $candidate2 = Resolve-Path (Join-Path $scriptDir "..\..") -ErrorAction SilentlyContinue
    if ($candidate2 -and (Test-Path (Join-Path $candidate2.Path "reports\doc_drift_guard.py"))) {
        return $candidate2.Path
    }

    throw "Projektwurzel konnte nicht bestimmt werden."
}

function Resolve-Python([string]$RepoRoot) {
    $candidates = @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot "venv\Scripts\python.exe")
    )

    foreach ($cand in $candidates) {
        if (Test-Path $cand) {
            return @{ Type = "path"; Value = $cand }
        }
    }

    if (Get-Command python -ErrorAction SilentlyContinue) {
        return @{ Type = "cmd"; Value = "python" }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @{ Type = "py"; Value = "py" }
    }

    throw "Kein Python-Interpreter gefunden (.venv, venv, python, py)."
}

try {
    $repoRoot = Resolve-RepoRoot
    $guardScript = Join-Path $repoRoot "reports\doc_drift_guard.py"

    if (-not (Test-Path $guardScript)) {
        throw "Guard-Skript nicht gefunden: $guardScript"
    }

    Write-Host "[pre-commit] Starte Doc-Drift-Guard ..."

    $pyInfo = Resolve-Python -RepoRoot $repoRoot
    if ($pyInfo.Type -eq "path") {
        & $pyInfo.Value $guardScript
    }
    elseif ($pyInfo.Type -eq "py") {
        & py -3 $guardScript
    }
    else {
        & python $guardScript
    }

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host "[pre-commit] Doc-Drift-Guard erfolgreich."
}
catch {
    Write-Host ("[pre-commit] FEHLER: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
}
