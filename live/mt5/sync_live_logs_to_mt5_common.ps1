param(
    # Quelle: Ordner mit Python-Logs (live_trader.py schreibt standardmäßig in ./logs)
    [string]$SourceDir = "C:\Users\Sebastian Setnescu\mt5_trading\logs",

    # Symbole laut aktueller Betriebsstrategie (2 aktive Paare)
    [string[]]$Symbols = @("USDCAD", "USDJPY"),

    # Optionales Ziel (wenn leer, wird MT5 Common Files automatisch ermittelt)
    [string]$TargetDir = "",

    # Dauerlauf aktivieren (für Live-Betrieb empfohlen)
    [switch]$Continuous,

    # Intervall für Dauerlauf in Sekunden
    [int]$IntervalSec = 5,

    # Testmodus: zeigt nur an, was kopiert würde
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Mt5CommonFiles {
    # Standardpfad für MT5 Common Files unter Windows
    $candidate = Join-Path $env:APPDATA "MetaQuotes\Terminal\Common\Files"
    if (Test-Path $candidate) {
        return $candidate
    }

    throw "MT5 Common Files nicht gefunden: $candidate"
}

function Ensure-Directory([string]$PathValue) {
    if (-not (Test-Path $PathValue)) {
        New-Item -ItemType Directory -Path $PathValue -Force | Out-Null
    }
}

function Copy-IfNeeded(
    [string]$Src,
    [string]$Dst,
    [switch]$IsDryRun
) {
    if (-not (Test-Path $Src)) {
        Write-Host ("[SYNC] FEHLT: {0}" -f $Src) -ForegroundColor Yellow
        return
    }

    $copy = $true
    if (Test-Path $Dst) {
        $srcItem = Get-Item $Src
        $dstItem = Get-Item $Dst

        # Nur kopieren wenn Quelle neuer ist oder Dateigröße abweicht
        if ($srcItem.LastWriteTimeUtc -le $dstItem.LastWriteTimeUtc -and $srcItem.Length -eq $dstItem.Length) {
            $copy = $false
        }
    }

    if (-not $copy) {
        Write-Host ("[SYNC] Unverändert: {0}" -f (Split-Path $Src -Leaf)) -ForegroundColor DarkGray
        return
    }

    if ($IsDryRun) {
        Write-Host ("[SYNC] DRY-RUN -> {0} => {1}" -f $Src, $Dst) -ForegroundColor Cyan
        return
    }

    Copy-Item -Path $Src -Destination $Dst -Force
    Write-Host ("[SYNC] Kopiert: {0}" -f (Split-Path $Src -Leaf)) -ForegroundColor Green
}

function Sync-Once(
    [string]$SrcDir,
    [string]$DstDir,
    [string[]]$Pairs,
    [switch]$IsDryRun
) {
    foreach ($sym in $Pairs) {
        $src = Join-Path $SrcDir ("{0}_live_trades.csv" -f $sym)
        $dst = Join-Path $DstDir ("{0}_live_trades.csv" -f $sym)
        Copy-IfNeeded -Src $src -Dst $dst -IsDryRun:$IsDryRun
    }
}

try {
    if (-not (Test-Path $SourceDir)) {
        throw "SourceDir nicht gefunden: $SourceDir"
    }

    # Zielordner auflösen (manuell oder automatisch)
    if ([string]::IsNullOrWhiteSpace($TargetDir)) {
        $TargetDir = Resolve-Mt5CommonFiles
    }

    Ensure-Directory -PathValue $TargetDir

    if ($IntervalSec -lt 1) {
        $IntervalSec = 1
    }

    Write-Host "[SYNC] Quelle: $SourceDir"
    Write-Host "[SYNC] Ziel:   $TargetDir"
    Write-Host ("[SYNC] Symbole: {0}" -f ($Symbols -join ", "))
    if ($DryRun) {
        Write-Host "[SYNC] Modus: DRY-RUN (keine Dateien werden kopiert)" -ForegroundColor Cyan
    }

    if ($Continuous) {
        Write-Host ("[SYNC] Dauerlauf aktiv (Intervall: {0}s). Stoppen mit Ctrl+C." -f $IntervalSec) -ForegroundColor Magenta
        while ($true) {
            Sync-Once -SrcDir $SourceDir -DstDir $TargetDir -Pairs $Symbols -IsDryRun:$DryRun
            Start-Sleep -Seconds $IntervalSec
        }
    }
    else {
        Sync-Once -SrcDir $SourceDir -DstDir $TargetDir -Pairs $Symbols -IsDryRun:$DryRun
        Write-Host "[SYNC] Fertig." -ForegroundColor Green
    }
}
catch {
    Write-Host ("[SYNC] FEHLER: {0}" -f $_.Exception.Message) -ForegroundColor Red
    exit 1
}
