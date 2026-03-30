@echo off
setlocal

REM =============================================================================
REM register_check_logs_auto.bat
REM
REM Zweck:
REM   Registriert eine automatische Check-Routine fuer check_logs_now.bat
REM   (stuendlich) inkl. Log-Historie auf dem Windows-Laptop.
REM =============================================================================

set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"
set "SCRIPT=%BASE_DIR%\scripts\windows_register_check_logs_task_v2.ps1"

if not exist "%SCRIPT%" (
    echo [FEHLER] Script nicht gefunden: %SCRIPT%
    pause
    exit /b 1
)

echo ========================================================
echo   Auto-Check Task fuer check_logs_now.bat
echo   Projekt: %BASE_DIR%
echo   Registrar: windows_register_check_logs_task_v2.ps1
echo ========================================================

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" -ProjectDir "%BASE_DIR%" -EveryMinutes 60 -RunNow
if errorlevel 1 (
    echo.
    echo [FEHLER] Registrierung fehlgeschlagen.
    pause
    exit /b 1
)

echo.
echo [OK] Auto-Check wurde registriert und gestartet.
echo Logs:
echo   %BASE_DIR%\logs\ops_checks\check_logs_last_run.log
echo   %BASE_DIR%\logs\ops_checks\check_logs_history.log
pause
exit /b 0
