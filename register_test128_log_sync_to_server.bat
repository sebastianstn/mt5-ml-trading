@echo off
REM =============================================================================
REM register_test128_log_sync_to_server.bat - Live-Log-Sync registrieren
REM
REM Kompatibilitaets-Name (historisch): "test128" im Dateinamen bleibt bewusst,
REM damit bestehende Deploy- und Runbook-Referenzen weiter funktionieren.
REM
REM Laeuft auf: Windows 11 Laptop (NICHT auf dem Linux-Server)
REM
REM Zweck:
REM   Registriert auf dem Windows-Laptop einen Scheduled Task, der aktive
REM   Live-Logs regelmaessig per SCP auf den Linux-Server kopiert.
REM =============================================================================

setlocal

for /f "tokens=2 delims=:." %%A in ('chcp') do set "PREV_CP=%%A"
set "PREV_CP=%PREV_CP: =%"
chcp 65001 >nul

set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"
set "POWERSHELL_EXE=powershell.exe"
set "REGISTER_SCRIPT=%BASE_DIR%\scripts\windows_register_live_log_sync_task.ps1"
set "ACTIVE_LOG_DIR=%BASE_DIR%\logs"
set "LINUX_TARGET_DIR=/mnt/1Tb-Data/XGBoost-LightGBM/logs"
set "TASK_NAME=MT5_Sync_Live_Logs_To_Linux"

if not exist "%BASE_DIR%" (
    echo [FEHLER] Projektordner nicht gefunden: %BASE_DIR%
    pause
    if defined PREV_CP chcp %PREV_CP% >nul
    exit /b 1
)

if not exist "%REGISTER_SCRIPT%" (
    echo [FEHLER] Register-Skript nicht gefunden: %REGISTER_SCRIPT%
    pause
    if defined PREV_CP chcp %PREV_CP% >nul
    exit /b 1
)

if not exist "%ACTIVE_LOG_DIR%" (
    mkdir "%ACTIVE_LOG_DIR%"
)

cd /d "%BASE_DIR%"

echo ========================================================
echo   Live-Log-Sync zum Linux-Server
echo   Quelle: %ACTIVE_LOG_DIR%
echo   Ziel:   %LINUX_TARGET_DIR%
echo   Task:   %TASK_NAME%
echo ========================================================
echo.

%POWERSHELL_EXE% -NoProfile -ExecutionPolicy Bypass -File "%REGISTER_SCRIPT%" ^
    -ProjectDir "%BASE_DIR%" ^
    -LocalLogsDir "%ACTIVE_LOG_DIR%" ^
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
    echo Bitte BAT/PowerShell als Administrator starten und SSH-Keys pruefen.
    pause
    if defined PREV_CP chcp %PREV_CP% >nul
    exit /b 1
)

echo.
echo [OK] Live-Log-Sync wurde registriert und direkt gestartet.
echo Linux-Zielordner: %LINUX_TARGET_DIR%
echo.
pause
if defined PREV_CP chcp %PREV_CP% >nul
exit /b 0
