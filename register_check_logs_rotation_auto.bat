@echo off
setlocal

REM =============================================================================
REM register_check_logs_rotation_auto.bat
REM
REM Zweck:
REM   Registriert die taegliche Rotation der Auto-Check-History (23:59).
REM =============================================================================

set "BASE_DIR=%~dp0"
if "%BASE_DIR:~-1%"=="\" set "BASE_DIR=%BASE_DIR:~0,-1%"
set "SCRIPT=%BASE_DIR%\scripts\windows_register_check_logs_rotation_task.ps1"

if not exist "%SCRIPT%" (
    echo [FEHLER] Script nicht gefunden: %SCRIPT%
    pause
    exit /b 1
)

echo ========================================================
echo   Rotation-Task fuer check_logs_history.log
echo   Projekt: %BASE_DIR%
echo ========================================================

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT%" -ProjectDir "%BASE_DIR%"
if errorlevel 1 (
    echo.
    echo [FEHLER] Registrierung der Rotation fehlgeschlagen.
    pause
    exit /b 1
)

echo.
echo [OK] Rotation wurde registriert (taeglich 23:59).
echo Archiv:
echo   %BASE_DIR%\logs\ops_checks\archive
pause
exit /b 0
