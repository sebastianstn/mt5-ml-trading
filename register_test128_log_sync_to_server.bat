@echo off
REM =============================================================================
REM register_test128_log_sync_to_server.bat
REM
REM Zweck:
REM   Registriert auf dem Windows-Laptop einen Scheduled Task, der die
REM   Test-128-Logs regelmaessig auf den Linux-Server kopiert.
REM
REM Ziel auf Linux-Server:
REM   /mnt/1Tb-Data/XGBoost-LightGBM/logs/paper_test128/
REM
REM Wichtig:
REM   - Als Administrator ausfuehren empfohlen
REM   - OpenSSH/Keys muessen fuer den Laptop -> Server Sync funktionieren
REM =============================================================================

setlocal

set "BASE_DIR=C:\Users\Sebastian Setnescu\mt5_trading"
set "POWERSHELL_EXE=powershell.exe"
set "REGISTER_SCRIPT=%BASE_DIR%\scripts\windows_register_live_log_sync_task.ps1"
set "TEST128_LOG_DIR=%BASE_DIR%\logs\paper_test128"
set "LINUX_TARGET_DIR=/mnt/1Tb-Data/XGBoost-LightGBM/logs/paper_test128"
set "TASK_NAME=MT5_Sync_Test128_Logs_To_Linux"

if not exist "%BASE_DIR%" (
    echo [FEHLER] Projektordner nicht gefunden: %BASE_DIR%
    pause
    exit /b 1
)

if not exist "%REGISTER_SCRIPT%" (
    echo [FEHLER] Register-Skript nicht gefunden: %REGISTER_SCRIPT%
    pause
    exit /b 1
)

if not exist "%TEST128_LOG_DIR%" (
    mkdir "%TEST128_LOG_DIR%"
)

cd /d "%BASE_DIR%"

echo ========================================================
echo   Test-128 Log-Sync zum Linux-Server
echo   Quelle: %TEST128_LOG_DIR%
echo   Ziel:   %LINUX_TARGET_DIR%
echo   Task:   %TASK_NAME%
echo ========================================================
echo.

%POWERSHELL_EXE% -NoProfile -ExecutionPolicy Bypass -File "%REGISTER_SCRIPT%" ^
    -ProjectDir "%BASE_DIR%" ^
    -LocalLogsDir "%TEST128_LOG_DIR%" ^
    -LinuxLogsDir "%LINUX_TARGET_DIR%" ^
    -TaskName "%TASK_NAME%" ^
    -LinuxUser sebastian ^
    -LinuxHost 192.168.1.35 ^
    -Symbols "USDCAD,USDJPY" ^
    -WatchdogTimeframe "M5_TWO_STAGE" ^
    -WatchdogStaleFactor 1.5 ^
    -SyncCloses ^
    -RunHidden ^
    -RunNow

if errorlevel 1 (
    echo.
    echo [FEHLER] Task-Registrierung oder Sofort-Start fehlgeschlagen.
    echo Bitte PowerShell / diese BAT als Administrator starten und SSH-Keys pruefen.
    pause
    exit /b 1
)

echo.
echo [OK] Test-128 Log-Sync wurde registriert und direkt gestartet.
echo Linux-Zielordner: %LINUX_TARGET_DIR%
echo Watchdog-Artefakte: live_log_watchdog_latest.json / .csv
echo.
pause
exit /b 0
