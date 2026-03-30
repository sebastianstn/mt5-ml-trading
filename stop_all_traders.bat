@echo off
REM =============================================================================
REM stop_all_traders.bat – Beendet laufende MT5-ML-Trader-Prozesse
REM
REM Ausführen auf Windows 11 Laptop:
REM     Doppelklick auf diese Datei ODER in PowerShell: .\stop_all_traders.bat
REM
REM Verhalten:
REM     - Schließt bekannte Trader-CMD-Fenster (v4/v5, baseline/shadow)
REM     - Beendet python.exe-Prozesse mit "live_trader.py" in der CommandLine
REM =============================================================================

setlocal EnableDelayedExpansion

echo ================================================
echo   MT5 ML-Trading - Stoppe Trader Prozesse
echo ================================================
echo.

echo [INFO] Stoppe bekannte Trader-Fenster (Titel-basiert)...

REM Shadow-Fenster
call :kill_window "MT5-Shadow-USDCAD-v4*"
call :kill_window "MT5-Shadow-USDJPY-v5*"

REM Baseline-Fenster (alt + neu)
call :kill_window "MT5-Trader-USDCAD*"
call :kill_window "MT5-Trader-USDJPY*"
call :kill_window "MT5-Trader-USDCAD-v4*"
call :kill_window "MT5-Trader-USDJPY-v4*"

REM Demo-/Paper-/Testphase-Fenster
call :kill_window "MT5-Demo-USDCAD*"
call :kill_window "MT5-Demo-USDJPY*"
call :kill_window "MT5-Testphase-USDCAD*"
call :kill_window "MT5-Testphase-USDJPY*"
call :kill_window "MT5-Paper-USDCAD*"
call :kill_window "MT5-Paper-USDJPY*"

REM Generischer Fallback: alle MT5-* Fensterprozesse
for /f "usebackq delims=" %%L in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$wins = @(Get-Process).Where({ $_.MainWindowTitle -like 'MT5-*' }); if (-not $wins -or $wins.Count -eq 0) { Write-Output '[INFO] Keine weiteren MT5-* Fenster gefunden.' } else { foreach ($p in $wins) { try { $null = taskkill /PID $p.Id /T /F 2>&1; Write-Output ('[OK] Fensterprozessbaum beendet: PID=' + $p.Id + ' Titel=' + $p.MainWindowTitle) } catch { Write-Output ('[WARN] Fensterprozess nicht beendet: PID=' + $p.Id + ' - ' + $_.Exception.Message) } } }"`) do (
    echo   %%L
)

echo.
echo [INFO] Stoppe live_trader.py Prozesse (CommandLine-basiert)...
for /f "usebackq delims=" %%L in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$targets = @(Get-CimInstance Win32_Process).Where({ $_.CommandLine -and $_.CommandLine -match 'live_trader\.py' }); if (-not $targets -or $targets.Count -eq 0) { Write-Output '[INFO] Keine live_trader.py Prozesse gefunden.' } else { $killed = @{}; foreach ($t in $targets) { if (-not $killed[$t.ProcessId]) { $killed[$t.ProcessId] = $true; try { $null = taskkill /PID $t.ProcessId /T /F 2>&1; Write-Output ('[OK] live_trader Prozessbaum beendet: PID=' + $t.ProcessId) } catch { Write-Output ('[WARN] live_trader Prozess nicht beendet: PID=' + $t.ProcessId + ' - ' + $_.Exception.Message) } } } }"`) do (
    echo   %%L
)

set "REMAIN=0"
for /f %%R in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "@(Get-CimInstance Win32_Process).Where({ $_.CommandLine -match 'live_trader\.py' }).Count"') do set "REMAIN=%%R"

echo.
if not defined REMAIN set "REMAIN=0"

if %REMAIN% GTR 0 (
    echo [WARNUNG] Es laufen noch %REMAIN% live_trader.py Prozess(e).
    echo Bitte offene Trader-Fenster manuell mit Ctrl+C stoppen und Skript erneut ausfuehren.
) else (
    echo ✅ Fertig. Alle live_trader.py Prozesse sind beendet.
    echo Danach kannst du sauber mit start_shadow_compare.bat neu starten.
)

echo.
echo Tipp: Falls weiterhin Prozesse haengen, PowerShell als Administrator starten und stop_all_traders.bat erneut ausfuehren.
echo.
pause
endlocal
exit /b 0

:kill_window
REM Kein /T (Tree-Kill) hier – verhindert Race Condition, bei der cmd.exe
REM sich selbst im Prozessbaum killt und das Sprungziel verliert.
REM Kind-Prozesse (Python) werden in der zweiten Phase sauber per CommandLine beendet.
taskkill /F /FI "WINDOWTITLE eq %~1" >nul 2>&1
if not errorlevel 1 (
    echo   [OK] Fenster %~1 beendet.
)
exit /b 0
